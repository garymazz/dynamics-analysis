import os
import time
import json
from cement import Controller, ex

# Import our decoupled core modules
from core.io_manager import IOManager, ConfigManager, get_schema_definition
from core.math_engine import run_sweeps_gpu_grouped, run_sweeps_gpu_batched
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
            (['--batch-mode'], {'help': 'Enable highly-optimized 3D tensor batch processing for a fixed window size', 'action': 'store_true', 'default': False}),
            (['--perf'], {'help': 'Enable performance monitoring. Use "--perf con" to also print to console', 'nargs': '?', 'const': 'file', 'default': None}),
            (['--dmd-lstsq'], {'help': 'Use least-squares (lstsq) solver instead of pseudo-inverse (pinv) for DMD modes', 'action': 'store_true', 'default': False}),
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

        # 4. Atomic HDF5 Dispatch Setup & Perf Setup
        hdf5_callback = None
        hdf5_base_dir = None
        
        if args.hdf5 or args.perf:
            hdf5_base_dir = generate_hdf5_dir_name(hdf5_base)
            os.makedirs(hdf5_base_dir, exist_ok=True)
            
        if args.hdf5:
            schema_dict = build_hierarchical_schema()
            full_schema_json = json.dumps(schema_dict)
            
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
        
        # --- NEW PRE-FLIGHT VALIDATION CHECK ---
        row_diff = end_row - args.start_row
        if row_diff < args.min_stack:
            start_lbl = "Default" if args.start_row == 1 else "User-Provided"
            end_lbl = "Default" if args.end_row is None else "User-Provided"
            stack_lbl = "Default" if args.min_stack == 5 else "User-Provided"
            
            print(f"\n[WARNING] Prediction Analysis can not be calculated")
            print(f"The data window (Start Row to End Row) is smaller than the Minimum Stack Size.")
            print(f"To perform this analysis, the Start to End row difference must be equal to or greater than the Minimum Stack Size.\n")
            print(f"Current Configuration:")
            print(f"  • Start Row: {args.start_row} ({start_lbl})")
            print(f"  • End Row: {end_row} ({end_lbl})")
            print(f"  • Row Difference: {row_diff}")
            print(f"  • Min Stack Size: {args.min_stack} ({stack_lbl})\n")
            print(f"Please adjust your --start-row, --end-row, or --min-stack arguments.")
            return
        # ---------------------------------------
        
        if args.inc_start and args.max_stack is None:
            span = end_row - args.start_row
            args.max_stack = max(args.min_stack + 1, span // 2)
        elif args.max_stack is None:
            args.max_stack = 41

        # RESTORED: Generate the list of stacks to sweep
        stack_range = list(range(args.min_stack, args.max_stack))

        print(f"Target Input: {args.input}")
        print(f"Stack sweep range: {args.min_stack} to {args.max_stack - 1}")

# 6. Execute Sweep Logic
        t0 = time.time()
        global_perf_data = []
        
        if args.batch_mode:
            print(f"\n[INFO] Starting High-Performance Batched Tensor Workflow...")
            
            fixed_w = args.max_window
            slice_start_idx = args.start_row - 1 - fixed_w
            
            if slice_start_idx < 0:
                print(f"[ERROR] Cannot predict row {args.start_row} with a window of {fixed_w}.")
                print(f"There are only {args.start_row - 1} historical rows available before it.")
                print(f"To use a window of {fixed_w}, your --start-row must be at least {fixed_w + 1}.")
                return
                
            slice_end_idx = end_row 
            print(f"[INFO] Processing predictions for rows {args.start_row} to {end_row} (Window: {fixed_w})...")
            
            data_slice = full_data_matrix[slice_start_idx : slice_end_idx]
            
            results, perf_data = run_sweeps_gpu_batched(
                full_data_matrix=data_slice,
                w=fixed_w,
                stack_list=stack_range,
                channel_names=args.channels,
                hdf5_callback=hdf5_callback,
                hdf5_targets=args.hdf5,
                global_start_row=args.start_row,
                perf_mode=args.perf,
                dmd_lstsq=args.dmd_lstsq
            )
            io_mgr.record_callback(results)
            if perf_data:
                global_perf_data.extend(perf_data)
            print(f"Batched processing complete. Generated {len(results)} records in ({time.time() - t0:.1f}s)")
            
        elif args.inc_start:
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
                
                if args.perf == 'con':
                    print(f"\n[Row Benchmark] Target Row: {curr_start} | Window: {len(dataset)} | Stacks to Execute: {len(active_stacks)}")
                
                row_start_time = time.perf_counter()
                
                results, perf_data = run_sweeps_gpu_grouped(
                    dataset, curr_start, end_row, ver_vals, args.train_rec, args.channels, active_stacks,
                    abort_check=abort_check, record_callback=io_mgr.record_callback, hdf5_callback=hdf5_callback,
                    svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5, dmd_lstsq=args.dmd_lstsq, perf_mode=args.perf
                )
                
                row_end_time = time.perf_counter()
                row_duration = row_end_time - row_start_time
                
                if perf_data:
                    for p in perf_data:
                        p["target_row"] = curr_start
                        p["direction"] = "inc_start"
                        p["total_row_time_s"] = row_duration
                    global_perf_data.extend(perf_data)
                    
                if args.perf == 'con':
                    s1 = sum(p.get("stage_1_hankel_s", 0) for p in perf_data) if perf_data else 0
                    s2 = sum(p.get("stage_2_svd_s", 0) for p in perf_data) if perf_data else 0
                    s3 = sum(p.get("stage_3_operator_s", 0) for p in perf_data) if perf_data else 0
                    s4 = sum(p.get("stage_4_modes_s", 0) for p in perf_data) if perf_data else 0
                    s5 = sum(p.get("stage_5_predict_s", 0) for p in perf_data) if perf_data else 0
                    s6 = sum(p.get("stage_6_hdf5_s", 0) for p in perf_data) if perf_data else 0
                    s7 = sum(p.get("stage_7_format_s", 0) for p in perf_data) if perf_data else 0
                    
                    print(f"  -> Stage 1 (Hankel Construction)  : {s1:.5f}s")
                    print(f"  -> Stage 2 (SVD Truncation)       : {s2:.5f}s")
                    print(f"  -> Stage 3 (Operator & Eigen)     : {s3:.5f}s")
                    print(f"  -> Stage 4 (Modes & Amplitudes)   : {s4:.5f}s")
                    print(f"  -> Stage 5 (Predictions)          : {s5:.5f}s")
                    print(f"  -> Stage 6 (HDF5 I/O Saving)      : {s6:.5f}s")
                    print(f"  -> Stage 7 (Format Tabular Recs)  : {s7:.5f}s")
                    print(f"  ================================================")
                    print(f"  -> Total Row Execution Period     : {row_duration:.5f}s")
                else:
                    print(f"Processed Start Row {curr_start} ({row_duration:.2f}s)")
                    
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

                if args.perf == 'con':
                    print(f"\n[Row Benchmark] Target Row: {curr_end} | Window: {len(dataset)} | Stacks to Execute: {len(active_stacks)}")

                row_start_time = time.perf_counter()

                results, perf_data = run_sweeps_gpu_grouped(
                    dataset, args.start_row, curr_end, ver_vals, args.train_rec, args.channels, active_stacks,
                    abort_check=abort_check, record_callback=io_mgr.record_callback, hdf5_callback=hdf5_callback,
                    svd_gpu=args.svd_gpu, hdf5_targets=args.hdf5, dmd_lstsq=args.dmd_lstsq, perf_mode=args.perf
                )
                
                row_end_time = time.perf_counter()
                row_duration = row_end_time - row_start_time
                
                if perf_data:
                    for p in perf_data:
                        p["target_row"] = curr_end
                        p["direction"] = "dec_end"
                        p["total_row_time_s"] = row_duration
                    global_perf_data.extend(perf_data)
                    
                if args.perf == 'con':
                    s1 = sum(p.get("stage_1_hankel_s", 0) for p in perf_data) if perf_data else 0
                    s2 = sum(p.get("stage_2_svd_s", 0) for p in perf_data) if perf_data else 0
                    s3 = sum(p.get("stage_3_operator_s", 0) for p in perf_data) if perf_data else 0
                    s4 = sum(p.get("stage_4_modes_s", 0) for p in perf_data) if perf_data else 0
                    s5 = sum(p.get("stage_5_predict_s", 0) for p in perf_data) if perf_data else 0
                    s6 = sum(p.get("stage_6_hdf5_s", 0) for p in perf_data) if perf_data else 0
                    s7 = sum(p.get("stage_7_format_s", 0) for p in perf_data) if perf_data else 0
                    
                    print(f"  -> Stage 1 (Hankel Construction)  : {s1:.5f}s")
                    print(f"  -> Stage 2 (SVD Truncation)       : {s2:.5f}s")
                    print(f"  -> Stage 3 (Operator & Eigen)     : {s3:.5f}s")
                    print(f"  -> Stage 4 (Modes & Amplitudes)   : {s4:.5f}s")
                    print(f"  -> Stage 5 (Predictions)          : {s5:.5f}s")
                    print(f"  -> Stage 6 (HDF5 I/O Saving)      : {s6:.5f}s")
                    print(f"  -> Stage 7 (Format Tabular Recs)  : {s7:.5f}s")
                    print(f"  ================================================")
                    print(f"  -> Total Row Execution Period     : {row_duration:.5f}s")
                else:
                    print(f"Processed End Row {curr_end} ({row_duration:.2f}s)")
                    
                curr_end -= 1

        # 7. Teardown
        io_mgr.cleanup_and_merge(keep_temp=args.keep_temp, format_type=args.format)
        
        # Save Performance Data to Parquet
        if args.perf and global_perf_data:
            import pandas as pd
            perf_df = pd.DataFrame(global_perf_data)
            perf_file_name = f"{os.path.basename(output_base)}_perf.parquet"
            perf_path = os.path.join(hdf5_base_dir, perf_file_name) if hdf5_base_dir else perf_file_name
            perf_df.to_parquet(perf_path)
            if args.perf == 'con':
                print(f"\n[INFO] Complete performance metrics saved to: {perf_path}")