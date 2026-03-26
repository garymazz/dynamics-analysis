import os
import time
import h5py
import json
from cement import Controller, ex

# Import our decoupled core modules
from core.io_manager import IOManager, ConfigManager, get_schema_definition
from core.math_engine import run_sweeps_gpu_grouped
from core.hdf5_manager import save_to_hdf5, build_hierarchical_schema, SCRIPT_VERSION, SCHEMA_VERSION

class BaseController(Controller):
    class Meta:
        label = 'base'
        description = 'DMD GPU Profiler - Main Sweep and Forecasting Workflow'
        
        arguments = [
            (['-i', '--input'], {'help': 'Input file path (Excel or Parquet). Default: p_all.xlsx', 'default': 'p_all.xlsx'}),
            (['-o', '--output'], {'help': 'Base name for output files', 'default': None}),
            (['--hdf5-dir'], {'help': 'Directory for SVD/DMD HDF5 output', 'default': None}),
            (['--format'], {'help': 'Output format [parquet, xlsx, both]', 'choices': ['parquet', 'xlsx', 'both'], 'default': 'both'}),
            (['--keep-temp'], {'help': 'Retain the temporary CSV buffer', 'action': 'store_true', 'default': False}),
            (['--resume'], {'help': 'Resume analysis from last known good state', 'action': 'store_true', 'default': False}),
            (['--schema'], {'help': 'Print the output JSON schema and exit', 'action': 'store_true', 'default': False}),
            (['--channels'], {'help': 'List of channels to analyze', 'nargs': '+', 'default': ['S1', 'S2', 'S3', 'S4', 'S5']}),
            (['--start-row'], {'help': 'Starting row index (1-based)', 'type': int, 'default': 1}),
            (['--end-row'], {'help': 'Ending row index (1-based)', 'type': int, 'default': None}),
            (['--inc-start'], {'help': 'Iterate forwards by increasing start-row', 'action': 'store_true', 'default': False}),
            (['--dec-end'], {'help': 'Iterate backwards from end-row', 'action': 'store_true', 'default': False}),
            (['--train-rec'], {'help': 'Calculate training reconstruction error', 'action': 'store_true', 'default': False}),
            (['--min-stack'], {'help': 'Minimum stack size', 'type': int, 'default': 5}),
            (['--max-stack'], {'help': 'Maximum stack size', 'type': int, 'default': None}),
            (['--min-window'], {'help': 'Minimum window size', 'type': int, 'default': 20}),
            (['--max-window'], {'help': 'Maximum window size', 'type': int, 'default': 151}),
            (['--no-svd-hdf5'], {'help': 'Disable writing SVD/DMD HDF5 file', 'action': 'store_true', 'default': False}),
            (['--svd-store-hankel'], {'help': 'Store full Hankel H matrix', 'action': 'store_true', 'default': False}),
            (['--svd-no-modes'], {'help': 'Skip storing Phi and b arrays', 'action': 'store_true', 'default': False}),
            (['--svd-gpu'], {'help': 'Force SVD on GPU', 'action': 'store_true', 'default': False}),
            (['--hdf5-eigen'], {'help': 'Include Eigendecomposition', 'action': 'store_true', 'default': False}),
            (['--hdf5-dmd'], {'help': 'Include DMD Operator', 'action': 'store_true', 'default': False}),
            (['--hdf5-all'], {'help': 'Include all DMD/Eigen data', 'action': 'store_true', 'default': False}),
        ]

    @ex(hide=True)
    def default(self):
        args = self.app.pargs
        
        # Early intercept for Schema print
        if args.schema:
            print(json.dumps(get_schema_definition(args.channels), indent=2))
            return
            
        # 1. Output Naming & Resume Configuration Override
        cli_output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        if args.dec_end: cli_output_base += "_dec_end"
        elif args.inc_start: cli_output_base += "_inc_start"

        config_dict = None
        if args.resume:
            config_dict = ConfigManager.load_run_config_dict(cli_output_base)
            if config_dict is None:
                print("Resume requested but no valid config found; proceeding with current CLI arguments.")
            else:
                ignored_flags = ConfigManager.compute_ignored_cli_options(args, config_dict)
                if ignored_flags:
                    print("WARNING: --resume specified; ignoring CLI overrides for these options:")
                    for flag in ignored_flags:
                        print(f"  {flag}")
                args = ConfigManager.apply_config_to_args(config_dict, args)

        if args.hdf5_all:
            args.hdf5_eigen = True
            args.hdf5_dmd = True

        output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        if args.dec_end: output_base += "_dec_end"
        elif args.inc_start: output_base += "_inc_start"
        
        hdf5_base = os.path.join(args.hdf5_dir, os.path.basename(output_base)) if args.hdf5_dir else output_base
        svd_hdf5_path = f"{hdf5_base}_svd.hdf5"

        # 2. Instantiate IO Manager and Handle State
        io_mgr = IOManager(output_base=output_base, buffer_size=1)
        
        resume_state = None
        if args.resume and config_dict is not None:
            resume_state = io_mgr.determine_resume_state_granular(args.inc_start)
            if resume_state:
                print("Resuming analysis from:")
                print(f"  Row:   {resume_state['row']}")
                print(f"  Window:{resume_state['window']}")
                print(f"  Stack: {resume_state['stack']}")
                print("Restarting at Next Hyperparameter...")

        # Save config immediately before sweep
        ConfigManager.save_run_config(args, output_base)
        
        # Clear temp buffer if we are deliberately starting fresh
        if (not args.resume or config_dict is None) and os.path.exists(io_mgr.temp_file):
            os.remove(io_mgr.temp_file)
            
        # 3. Load Data
        print(f"=== GPU Profiler v{SCRIPT_VERSION} ===")
        try:
            df = io_mgr.load_data(args.input, args.channels)
            full_data_matrix = df.values.astype(float)
        except Exception as e:
            print(f"[Error] Data load failed: {e}")
            return

        # 4. HDF5 Lifecycle & Callback Setup
        hf_file = None
        hdf5_callback = None
        
        if not args.no_svd_hdf5:
            try:
                file_mode = "a" if args.resume else "w"
                hf_file = h5py.File(svd_hdf5_path, file_mode)
                
                if file_mode == "w" or "hierarchical_schema" not in hf_file.attrs:
                    hf_file.attrs["script_version"] = SCRIPT_VERSION
                    hf_file.attrs["schema_version"] = SCHEMA_VERSION
                    hf_file.attrs["input_file"] = args.input
                    hf_file.attrs["channels"] = json.dumps(args.channels)
                    
                    schema = build_hierarchical_schema(args.hdf5_dmd, args.hdf5_eigen, args.svd_no_modes, args.svd_store_hankel)
                    hf_file.attrs["hierarchical_schema"] = json.dumps(schema)
                    
                    dataset_list = []
                    for dsets in schema["representative_group_schema"]["datasets"].values():
                        dataset_list.extend(dsets.keys())
                    hf_file.attrs["data_sets"] = json.dumps(dataset_list)
                
                def _hdf5_injector(**kwargs):
                    save_to_hdf5(hf_file, **kwargs)
                hdf5_callback = _hdf5_injector
                
            except Exception as e:
                print(f"[Warning] Failed to initialize HDF5: {e}")

        # 5. Core Execution Setup
        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)
        
        end_row = args.end_row if args.end_row is not None else len(full_data_matrix) - 1
        
        if args.inc_start and args.max_stack is None:
            span = end_row - args.start_row
            args.max_stack = max(args.min_stack + 1, span // 2)
        elif args.max_stack is None:
            args.max_stack = 41
            
        stack_range = list(range(args.min_stack, args.max_stack))

        print(f"Target Input: {args.input}")
        print(f"Stack sweep range: {args.min_stack} to {args.max_stack - 1}")

        # 6. Execute Sweep Logic
        t0 = time.time()
        
        if args.inc_start:
            print(f"Starting Inc Start Sweep (Row {args.start_row} -> {end_row})...")
            curr_start = resume_state["row"] if (args.resume and resume_state) else args.start_row
            target_idx = end_row if end_row < len(full_data_matrix) else len(full_data_matrix) - 1
            ver_vals = full_data_matrix[target_idx, :]

            while curr_start < end_row:
                if abort_check(): break
                dataset = full_data_matrix[curr_start - 1 : target_idx]
                if len(dataset) < 2: break
                
                active_stacks = stack_range
                if args.resume and resume_state and curr_start == resume_state["row"]:
                    active_stacks = [s for s in stack_range if s > resume_state["stack"]]
                    if not active_stacks:
                        curr_start += 1
                        continue
                
                run_sweeps_gpu_grouped(
                    dataset, curr_start, end_row, ver_vals, args.train_rec, args.channels, active_stacks,
                    abort_check=abort_check, record_callback=io_mgr.record_callback, hdf5_callback=hdf5_callback,
                    svd_store_hankel=args.svd_store_hankel, svd_no_modes=args.svd_no_modes, svd_gpu=args.svd_gpu,
                    hdf5_dmd=args.hdf5_dmd, hdf5_eigen=args.hdf5_eigen
                )
                print(f"Processed Start Row {curr_start} ({time.time() - t0:.1f}s)")
                t0 = time.time()
                curr_start += 1
                
        else:
            lower_bound = max(99, args.start_row + args.min_window)
            curr_end = resume_state["row"] if (args.resume and resume_state) else end_row
            print(f"Starting Dec End Sweep (Row {curr_end} -> {lower_bound})...")
            
            while curr_end >= lower_bound:
                if abort_check(): break
                if curr_end >= len(full_data_matrix):
                    curr_end -= 1
                    continue
                
                dataset = full_data_matrix[args.start_row - 1 : curr_end]
                ver_vals = full_data_matrix[curr_end, :]

                active_stacks = stack_range
                if args.resume and resume_state and curr_end == resume_state["row"]:
                    active_stacks = [s for s in stack_range if s > resume_state["stack"]]
                    if not active_stacks:
                        curr_end -= 1
                        continue

                run_sweeps_gpu_grouped(
                    dataset, args.start_row, curr_end, ver_vals, args.train_rec, args.channels, active_stacks,
                    abort_check=abort_check, record_callback=io_mgr.record_callback, hdf5_callback=hdf5_callback,
                    svd_store_hankel=args.svd_store_hankel, svd_no_modes=args.svd_no_modes, svd_gpu=args.svd_gpu,
                    hdf5_dmd=args.hdf5_dmd, hdf5_eigen=args.hdf5_eigen
                )
                print(f"Processed End Row {curr_end} ({time.time() - t0:.1f}s)")
                t0 = time.time()
                curr_end -= 1

        # 7. Teardown
        if hf_file:
            hf_file.close()
            
        io_mgr.cleanup_and_merge(keep_temp=args.keep_temp, format_type=args.format)