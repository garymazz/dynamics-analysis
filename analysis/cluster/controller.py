import os
import json
import time
import pandas as pd
from cement import Controller, ex

# Import our decoupled core modules
from core.io_manager import IOManager
from core.cluster_engine import run_cluster_forecast_workflow, run_cluster_forecast_workflow_batched
from core.period_analysis import analyze_period_report

class ClusterController(Controller):
    class Meta:
        label = 'cluster'
        stacked_on = 'base'
        stacked_type = 'nested'
        description = 'Smart Mode Cluster-DMD Forecasting Workflow'
        
        # Native Cement argument parsing for the cluster workflow
        arguments = [
            (['-i', '--input'], {'help': 'Input file path (Excel or Parquet)', 'required': True}),
            (['-o', '--output'], {'help': 'Base name for output files', 'default': None}),
            (['--channels'], {'help': 'List of channels to analyze', 'nargs': '+', 'default': ['S1', 'S2', 'S3', 'S4', 'S5']}),
            
            # Smart Mode Specifics
            (['--error-threshold'], {'help': 'Target error percentage for finding needle windows', 'type': float, 'default': 1.0}),
            (['--forecast-row'], {'help': 'Specific row index to forecast. Defaults to N+1', 'type': int, 'default': None}),
            (['--ensemble-width'], {'help': 'Window width (+/-) for ensemble averaging', 'type': int, 'default': 5}),
            (['--sweep-input'], {'help': 'Path to a previous sweep file for Period Detection', 'default': None}),
            
            # Dimensions
            (['--min-stack'], {'help': 'Minimum stack size', 'type': int, 'default': 5}),
            (['--max-stack'], {'help': 'Maximum stack size', 'type': int, 'default': 41}),
            (['--min-window'], {'help': 'Minimum window size', 'type': int, 'default': 20}),
            (['--max-window'], {'help': 'Maximum window size', 'type': int, 'default': 151}),
            
            # Hardware, Profiling & Execution Modes
            (['--batch-mode'], {'help': 'Enable High-Performance Batched Tensor Workflow for optimization', 'action': 'store_true', 'default': False}),
            (['--svd-gpu'], {'help': 'Force SVD on GPU', 'action': 'store_true', 'default': False}),
            (['--perf'], {'help': 'Enable performance monitoring. Use "--perf con" to also print to console', 'nargs': '?', 'const': 'file', 'default': None}),
        ]

    @ex(hide=True)
    def _default(self):
        """CLI Routing for: python main.py cluster ..."""
        args = self.app.pargs
        
        # 1. Output Naming Logic
        output_base = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0]
        
        # 2. Instantiate IO Manager to load data
        io_mgr = IOManager(output_base=output_base)
        
        try:
            df = io_mgr.load_data(args.input, args.channels)
            full_data_matrix = df.values.astype(float)
        except Exception as e:
            print(f"[Error] Data load failed: {e}")
            return

        # 3. Optional Period Detection Check
        detected_period = None
        if args.sweep_input:
            print(f"Loading sweep data from {args.sweep_input} for Period Analysis...")
            try:
                if args.sweep_input.endswith(".parquet"):
                    sweep_df = pd.read_parquet(args.sweep_input)
                else:
                    sweep_df = pd.read_csv(args.sweep_input)

                # Use the primary channel for period detection
                primary_ch = args.channels[0]
                report = analyze_period_report(sweep_df, channel_name=primary_ch)
                
                if report.get("status") == "SUCCESS":
                    detected_period = report["period_integer"]
                    print(f"Detected Dominant Period: {detected_period}")
                else:
                    print(f"Period detection failed: {report.get('reason')}")
                    
            except Exception as e:
                print(f"[Error] Failed to load sweep data for period analysis: {e}")

        # 4. Setup Framework-Agnostic Hooks
        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)
        
        def _console_logger(msg):
            print(msg)

        # 5. Execute the Core Cluster Engine
        t0 = time.time()
        
        if args.batch_mode:
            print("\n[INFO] Routing to Batched Tensor Cluster Engine...")
            final_payload, perf_data = run_cluster_forecast_workflow_batched(
                full_data_matrix=full_data_matrix,
                channel_names=args.channels,
                error_threshold=args.error_threshold,
                min_window=args.min_window,
                max_window=args.max_window,
                min_stack=args.min_stack,
                max_stack=args.max_stack,
                ensemble_width=args.ensemble_width,
                forecast_row=args.forecast_row,
                detected_period=detected_period,
                abort_check=abort_check,
                svd_gpu=args.svd_gpu,
                log_callback=_console_logger,
                perf_mode=args.perf,
                validation_batch_size=20 # Expose this block size if you want later
            )
        else:
            print("\n[INFO] Routing to Sequential Cluster Engine...")
            final_payload, perf_data = run_cluster_forecast_workflow(
                full_data_matrix=full_data_matrix,
                channel_names=args.channels,
                error_threshold=args.error_threshold,
                min_window=args.min_window,
                max_window=args.max_window,
                min_stack=args.min_stack,
                max_stack=args.max_stack,
                ensemble_width=args.ensemble_width,
                forecast_row=args.forecast_row,
                detected_period=detected_period,
                abort_check=abort_check,
                svd_gpu=args.svd_gpu,
                log_callback=_console_logger,
                perf_mode=args.perf
            )
            
        total_runtime = time.time() - t0

        # 6. Save the results
        if final_payload:
            fname = f"{output_base}_forecast.json"
            try:
                with open(fname, "w") as f:
                    json.dump(final_payload, f, indent=2)
                print(f"\n[Success] Saved forecast payload to {fname}")
            except Exception as e:
                print(f"[Error] Failed to write forecast JSON: {e}")
        else:
            print("\n[Warning] Cluster workflow did not return a valid forecast.")

        # 7. Save Performance Data to Parquet
        if args.perf and perf_data:
            for p in perf_data:
                p["total_cluster_workflow_s"] = total_runtime
                p["cluster_execution_mode"] = "batched" if args.batch_mode else "sequential"
                
            perf_df = pd.DataFrame(perf_data)
            perf_file_name = f"{output_base}_cluster_perf.parquet"
            perf_df.to_parquet(perf_file_name)
            
            if args.perf == 'con':
                print(f"[INFO] Complete Cluster performance metrics saved to: {perf_file_name}")