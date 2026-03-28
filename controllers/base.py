import os
import time
import json
from cement import Controller, ex

# Import our decoupled core modules
from core.io_manager import IOManager, ConfigManager, get_schema_definition
from core.math_engine import run_sweeps_gpu_grouped
from core.hdf5_manager import save_stage_to_hdf5, generate_hdf5_dir_name, build_hierarchical_schema, SCRIPT_VERSION

class BaseController(Controller):
    class Meta:
        label = 'base'
        description = 'DMD GPU Profiler - Main Sweep and Forecasting Workflow'
        
        arguments = [
            (['-i', '--input'], {'help': 'Input file path (Excel or Parquet). Default: p_all.xlsx', 'default': 'p_all.xlsx'}),
            (['-o', '--output'], {'help': 'Base name for output files', 'default': None}),
            (['--hdf5-dir'], {'help': 'Directory for SVD/DMD HDF5 output', 'default': None}),
            (['--hdf5'], {'help': 'List of data types to write to HDF5 (hankle, svd, dmd-op, eigen, dmd-modes, dmd_amp, pred, all)', 'nargs': '+', 'default': []}),
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
            (['--svd-gpu'], {'help': 'Force SVD on GPU', 'action': 'store_true', 'default': False}),
        ]

    @ex(hide=True)
    def _default(self):
        args = self.app.pargs
        
        if args.schema:
            print(json.dumps(get_schema_definition(args.channels), indent=2))
            return
            
        # 1. Output Naming & Resume Configuration
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

        output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        if args.dec_end: output_base += "_dec_end"
        elif args.inc_start: output_base += "_inc_start"
        
        # Determine the ZFS Base Directory
        hdf5_base = os.path.join(args.hdf5_dir, os.path.basename(output_base)) if args.hdf5_dir else output_base

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

        ConfigManager.save_run_config(args, output_base)
        
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

        # 4. Atomic HDF5 Dispatch Setup
        hdf5_callback = None
        
        # Only initialize the HDF5 routing if the user specified targets in the list
        if args.hdf5:
            # Generate the schema string once
            schema_dict = build_hierarchical_schema()
            full_schema_json = json.dumps(schema_dict)
            
            # Setup the Master Directory
            hdf5_base_dir = generate_hdf5_dir_name(hdf5_base)
            os.makedirs(hdf5_base_dir, exist_ok=True)
            
            # Curate the routing callback for the engine
            def _hdf5_injector(data_class, d_start, d_end, w, s, payload):
                save_stage_to_hdf5(
                    base_dir=hdf5_base_dir,
                    data_class=data_class,
                    d_start=d_start,
                    d_end=d_end,
                    w=w,
                    s=s,
                    payload=payload,
                    full_schema_json=full_schema_json
                )
            hdf5_callback = _hdf5_injector

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
                    svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5
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
                    svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5
                )
                print(f"Processed End Row {curr_end} ({time.time() - t0:.1f}s)")
                t0 = time.time()
                curr_end -= 1

        # 7. Teardown
        io_mgr.cleanup_and_merge(keep_temp=args.keep_temp, format_type=args.format)