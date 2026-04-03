import os
import pandas as pd
from cement import Controller, ex

# --- UPDATED DOMAIN-DRIVEN IMPORTS ---
from analysis.cluster._main import run_cluster_forecast_workflow, run_cluster_forecast_workflow_batched
from utils.io._main import IOManager
from meta_analysis.period._main import analyze_period_report

class ClusterController(Controller):
    class Meta:
        label = 'cluster'
        stacked_on = 'base'         # <-- Stacked on the root base controller 
        stacked_type = 'embedded'   # <-- Attaches sub-command to base controller without creating a new CLI layer (i.e., `python main.py cluster ...` instead of `python main.py cluster cluster ...`)
        description = 'Smart Mode: Automated Hyperparameter Clustering and Ensemble Forecasting'

    @ex(
        help='Run smart mode ensemble forecasting with dynamic window optimization.',
        arguments=[
            (['-i', '--input'], {'help': 'Input file (.xlsx or .parquet)', 'required': True}),
            (['-o', '--output'], {'help': 'Output file prefix (default: cluster_results)', 'default': 'cluster_results'}),
            (['--channels'], {'help': 'Channels to analyze', 'nargs': '+', 'default': ['S1', 'S2', 'S3', 'S4', 'S5']}),
            (['--error-threshold'], {'help': 'Target error % for finding optimal windows', 'type': float, 'default': 1.0}),
            (['--forecast-row'], {'help': 'Specific row to forecast (Defaults to N+1)', 'type': int, 'default': None}),
            (['--ensemble-width'], {'help': '+/- variance in window size for the ensemble cluster', 'type': int, 'default': 5}),
            (['--sweep-input'], {'help': 'Path to a previous sweep file for Period Detection', 'default': None}),
            (['--min-stack'], {'help': 'Minimum stack size', 'type': int, 'default': 5}),
            (['--max-stack'], {'help': 'Maximum stack size', 'type': int, 'default': 41}),
            (['--min-window'], {'help': 'Minimum window size', 'type': int, 'default': 20}),
            (['--max-window'], {'help': 'Maximum window size', 'type': int, 'default': 151}),
            (['--batch-mode'], {'help': 'Use massively parallel Batched Tensor evaluation for Optimization', 'action': 'store_true'}),
            (['--svd-gpu'], {'help': 'Force SVD calculation on GPU', 'action': 'store_true'}),
            (['--perf'], {'help': 'Track cluster execution times. Use "con" for console stream.', 'nargs': '?', 'const': 'file', 'default': None}),
        ]
    )
    def cluster(self):
        """CLI Routing for: python main.py cluster ..."""
        args = self.app.pargs

        io_mgr = IOManager(args.input, args.output, args.channels, 'parquet', keep_temp=False, resume=False)
        full_data, data_len = io_mgr.load_data()
        if full_data is None: return

        detected_period = None
        if args.sweep_input and os.path.exists(args.sweep_input):
            print(f"\n[Pre-Processing] Running Period Analysis on: {args.sweep_input}")
            sweep_df = pd.read_parquet(args.sweep_input) if args.sweep_input.endswith('.parquet') else pd.read_csv(args.sweep_input)
            primary_ch = args.channels[0]
            report = analyze_period_report(sweep_df, channel_name=primary_ch)
            if report.get("status") == "SUCCESS":
                detected_period = report["period_integer"]
                print(f" -> Dominant Period Detected: {detected_period} samples (Confidence: {report['confidence_pct']}%)\n")
            else:
                print(f" -> Period Analysis Failed: {report.get('reason')}. Defaulting to full spectrum scan.\n")

        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)
        
        # Route to the appropriate cluster engine logic
        if args.batch_mode:
            payload, perf_data = run_cluster_forecast_workflow_batched(
                full_data_matrix=full_data, channel_names=args.channels, error_threshold=args.error_threshold,
                min_window=args.min_window, max_window=args.max_window, min_stack=args.min_stack, max_stack=args.max_stack,
                ensemble_width=args.ensemble_width, forecast_row=args.forecast_row, detected_period=detected_period,
                abort_check=abort_check, svd_gpu=args.svd_gpu, perf_mode=args.perf
            )
        else:
            payload, perf_data = run_cluster_forecast_workflow(
                full_data_matrix=full_data, channel_names=args.channels, error_threshold=args.error_threshold,
                min_window=args.min_window, max_window=args.max_window, min_stack=args.min_stack, max_stack=args.max_stack,
                ensemble_width=args.ensemble_width, forecast_row=args.forecast_row, detected_period=detected_period,
                abort_check=abort_check, svd_gpu=args.svd_gpu, perf_mode=args.perf
            )

        if payload:
            io_mgr.export_cluster_forecast(payload)
            print(f"\nForecast Payload saved to {args.output}_forecast.json")
            
        if perf_data and args.perf:
            io_mgr.export_cluster_performance(perf_data, args.batch_mode)
            print(f"Cluster Performance Metrics saved to {args.output}_cluster_perf.parquet")