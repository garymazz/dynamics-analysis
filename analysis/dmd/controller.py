import os
import time
import json
import pandas as pd
from cement import Controller, ex

# --- UPDATED DOMAIN-DRIVEN IMPORTS ---
from analysis.dmd._main import run_sweeps_gpu_grouped, run_sweeps_gpu_batched, DEVICE
from utils.io._main import IOManager, ConfigManager, get_schema_definition
from utils.hdf5._main import HDF5Manager

class DMDController(Controller):
    class Meta:
        label = 'dmd'
        stacked_on = 'base'         # <-- Stacked on the root base controller
        stacked_type = 'embedded'   # <-- Attaches sub-command to base controller without creating a new CLI layer (i.e., `python main.py cluster ...` instead of `python main.py cluster cluster ...`)
        description = 'DMD GPU Profiler - Main Sweep and Forecasting Workflow'

    @ex(
        help='Run standard DMD hyperparameter sweeps and generate forecasts.',
        arguments=[
            (['-i', '--input'], {'help': 'Input file (.xlsx or .parquet)', 'required': True}),
            (['-o', '--output'], {'help': 'Output file prefix (default: sweep_results)', 'default': 'sweep_results'}),
            (['--channels'], {'help': 'Channels to analyze', 'nargs': '+', 'default': ['S1', 'S2', 'S3', 'S4', 'S5']}),
            (['--start-row'], {'help': 'Start row index', 'type': int, 'default': 1}),
            (['--end-row'], {'help': 'End row index (inclusive)', 'type': int, 'default': None}),
            (['--min-stack'], {'help': 'Minimum stack size', 'type': int, 'default': 5}),
            (['--max-stack'], {'help': 'Maximum stack size', 'type': int, 'default': None}),
            (['--min-window'], {'help': 'Minimum window size', 'type': int, 'default': 20}),
            (['--max-window'], {'help': 'Maximum window size', 'type': int, 'default': 151}),
            (['--inc-start'], {'help': 'Sweep forwards (Increment start row)', 'action': 'store_true'}),
            (['--dec-end'], {'help': 'Sweep backwards (Decrement end row)', 'action': 'store_true'}),
            (['--batch-mode'], {'help': 'Use massively parallel 3D Tensor batched evaluation', 'action': 'store_true'}),
            (['--format'], {'help': 'Output format (parquet, xlsx, both)', 'default': 'parquet'}),
            (['--resume'], {'help': 'Resume from the last saved state in _config.json', 'action': 'store_true'}),
            (['--keep-temp'], {'help': 'Keep temporary CSV files after completion', 'action': 'store_true'}),
            (['--schema'], {'help': 'Print the JSON output schema to the console and exit', 'action': 'store_true'}),
            (['--train-rec'], {'help': 'Compute reconstruction error on the training data', 'action': 'store_true'}),
            (['--svd-gpu'], {'help': 'Force SVD calculation on GPU', 'action': 'store_true'}),
            (['--dmd-lstsq'], {'help': 'Use exact Least-Squares solver instead of Pseudo-Inverse', 'action': 'store_true'}),
            (['--perf'], {'help': 'Track execution times. Use "con" to print to console.', 'nargs': '?', 'const': 'file', 'default': None}),
            (['--hdf5'], {'help': 'HDF5 save targets', 'nargs': '+', 'default': []}),
            (['--hdf5-dir'], {'help': 'Custom directory path for HDF5 output', 'default': None}),
        ]
    )
    def dmd(self):
        """Main execution logic for DMD sweeps."""
        args = self.app.pargs

        if args.schema:
            print(json.dumps(get_schema_definition(args.channels), indent=2))
            return

        print(f"Initializing DMD Profiler Workflow (Hardware: {DEVICE.upper()})")
        
        cli_output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        if args.dec_end: cli_output_base += "_dec_end"
        elif args.inc_start: cli_output_base += "_inc_start"

        config_dict = None
        if args.resume:
            config_dict = ConfigManager.load_run_config_dict(cli_output_base)
            if config_dict:
                ignored_flags = ConfigManager.compute_ignored_cli_options(args, config_dict)
                if ignored_flags:
                    print("WARNING: --resume specified; ignoring CLI overrides for these options:")
                    for flag in ignored_flags:
                        print(f"  {flag}")
                args = ConfigManager.apply_config_to_args(config_dict, args)

        output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        if args.dec_end: output_base += "_dec_end"
        elif args.inc_start: output_base += "_inc_start"

        io_mgr = IOManager(output_base)
        
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

        try:
            df_selected = IOManager.load_data(args.input, args.channels)
            full_data_matrix = df_selected.values.astype(float)
            data_len = len(full_data_matrix)
        except Exception as e:
            print(f"[Fatal Error] Data Loading Failed: {e}")
            return

        hdf5_mgr = None
        if args.hdf5:
            hdf5_mgr = HDF5Manager(output_base, custom_dir=args.hdf5_dir)
            
        end_row = args.end_row if args.end_row is not None else data_len - 1
        
        row_diff = end_row - args.start_row
        if row_diff < args.min_stack:
            print(f"\n[WARNING] Prediction Analysis can not be calculated")
            print(f"The data window (Start Row to End Row) is smaller than the Minimum Stack Size.")
            return

        if args.inc_start and args.max_stack is None:
            span = end_row - args.start_row
            args.max_stack = max(args.min_stack + 1, span // 2)
        elif args.max_stack is None:
            args.max_stack = 41

        # RESTORED: Exact stack generation logic from original base.py
        stack_range = list(range(args.min_stack, args.max_stack))

        print(f"Data Loaded: {data_len} rows | Target Range: [{args.start_row} to {end_row}]")
        print(f"Stack sweep range: {args.min_stack} to {args.max_stack - 1}")

        t0 = time.time()
        global_perf_data = []
        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)

        try:
            if args.batch_mode:
                print(f"\n[INFO] Starting High-Performance Batched Tensor Workflow...")
                fixed_w = args.max_window
                slice_start_idx = args.start_row - 1 - fixed_w
                
                if slice_start_idx < 0:
                    print(f"[ERROR] Cannot predict row {args.start_row} with a window of {fixed_w}.")
                    return
                    
                slice_end_idx = end_row 
                data_slice = full_data_matrix[slice_start_idx : slice_end_idx]
                
                results, perf_data = run_sweeps_gpu_batched(
                    full_data_matrix=data_slice,
                    w=fixed_w,
                    stack_list=stack_range,
                    channel_names=args.channels,
                    hdf5_callback=hdf5_mgr.save_group if hdf5_mgr else None,
                    hdf5_targets=args.hdf5,
                    global_start_row=args.start_row,
                    perf_mode=args.perf,
                    dmd_lstsq=args.dmd_lstsq
                )
                io_mgr.record_callback(results)
                if perf_data:
                    global_perf_data.extend(perf_data)
                
            elif args.inc_start:
                print(f"Starting Inc Start Sweep (Row {args.start_row} -> {end_row})...")
                curr_start = resume_state["row"] if (args.resume and resume_state) else args.start_row
                target_idx = end_row if end_row < len(full_data_matrix) else len(full_data_matrix) - 1
                ver_vals = full_data_matrix[target_idx, :]

                while curr_start < end_row:
                    if abort_check(): break
                    
                    # RESTORED: Exact slicing from original base.py
                    dataset = full_data_matrix[curr_start - 1 : target_idx]
                    if len(dataset) < 2: break
                    
                    active_stacks = stack_range
                    if args.resume and resume_state and curr_start == resume_state["row"]:
                        active_stacks = [s for s in stack_range if s > resume_state["stack"]]
                        if not active_stacks:
                            curr_start += 1
                            continue
                    
                    row_start_time = time.perf_counter()
                    
                    results, perf_data = run_sweeps_gpu_grouped(
                        dataset, curr_start, end_row, ver_vals, args.train_rec, args.channels, active_stacks,
                        abort_check=abort_check, record_callback=None, hdf5_callback=hdf5_mgr.save_group if hdf5_mgr else None,
                        svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5, dmd_lstsq=args.dmd_lstsq, perf_mode=args.perf
                    )
                    
                    io_mgr.record_callback(results)
                    row_duration = time.perf_counter() - row_start_time
                    
                    if perf_data:
                        for p in perf_data:
                            p["target_row"] = curr_start
                            p["direction"] = "inc_start"
                            p["total_row_time_s"] = row_duration
                        global_perf_data.extend(perf_data)

                    if not args.perf == 'con':
                        print(f"Processed Start Row {curr_start} ({row_duration:.2f}s)")
                        
                    curr_start += 1
                    
            else:
                # Exact dec_end logic from original base.py
                lower_bound = max(99, args.start_row + args.min_window)
                curr_end = resume_state["row"] if (args.resume and resume_state) else end_row
                print(f"Starting Dec End Sweep (Row {curr_end} -> {lower_bound})...")
                
                while curr_end >= lower_bound:
                    if abort_check(): break
                    if curr_end >= len(full_data_matrix):
                        curr_end -= 1
                        continue
                    
                    # Exact slicing from original base.py
                    dataset = full_data_matrix[args.start_row - 1 : curr_end]
                    ver_vals = full_data_matrix[curr_end, :]

                    active_stacks = stack_range
                    if args.resume and resume_state and curr_end == resume_state["row"]:
                        active_stacks = [s for s in stack_range if s > resume_state["stack"]]
                        if not active_stacks:
                            curr_end -= 1
                            continue

                    row_start_time = time.perf_counter()

                    results, perf_data = run_sweeps_gpu_grouped(
                        dataset, args.start_row, curr_end, ver_vals, args.train_rec, args.channels, active_stacks,
                        abort_check=abort_check, record_callback=None, hdf5_callback=hdf5_mgr.save_group if hdf5_mgr else None,
                        svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5, dmd_lstsq=args.dmd_lstsq, perf_mode=args.perf
                    )
                    
                    io_mgr.record_callback(results)
                    row_duration = time.perf_counter() - row_start_time
                    
                    if perf_data:
                        for p in perf_data:
                            p["target_row"] = curr_end
                            p["direction"] = "dec_end"
                            p["total_row_time_s"] = row_duration
                        global_perf_data.extend(perf_data)

                    if not args.perf == 'con':
                        print(f"Processed End Row {curr_end} ({row_duration:.2f}s)")
                        
                    curr_end -= 1

        except KeyboardInterrupt:
            print("\n[Sweep Paused] Saving data buffers to disk...")
        finally:
            io_mgr.cleanup_and_merge(keep_temp=args.keep_temp, format_type=args.format)
            
            if args.perf and global_perf_data:
                perf_df = pd.DataFrame(global_perf_data)
                perf_file_name = f"{os.path.basename(output_base)}_perf.parquet"
                perf_path = os.path.join(hdf5_mgr.base_dir, perf_file_name) if hdf5_mgr else perf_file_name
                perf_df.to_parquet(perf_path)