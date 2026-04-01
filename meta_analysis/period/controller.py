import os
import pandas as pd
from cement import Controller, ex

# --- UPDATED DOMAIN-DRIVEN IMPORTS ---
from meta_analysis.period._main import analyze_period_report, print_period_report

class PeriodController(Controller):
    class Meta:
        label = 'analysis'        # CLI command remains 'python main.py analysis'
        stacked_on = 'base'       # <-- Stacked on the root base controller
        stacked_type = 'embedded' # <-- Attaches sub-command to base controller without creating a new CLI layer (i.e., `python main.py cluster ...` instead of `python main.py cluster cluster ...`)
        description = 'Post-processing and data analysis tools'

    @ex(
        help='Run standalone Dominant Period Analysis on a sweep file.',
        arguments=[
            (['sweep_input'], {'help': 'Path to a previous sweep file (.parquet or .csv)'}),
            (['-c', '--channel'], {'help': 'Specific channel to analyze (default: S1)', 'default': 'S1'})
        ]
    )
    def analysis(self): # Execution is now explicitly named 'analysis'
        """CLI Routing for: python main.py analysis period <file> --channel S1"""
        args = self.app.pargs

        if not os.path.exists(args.sweep_input):
            print(f"[Error] File not found: {args.sweep_input}")
            return

        print(f"Loading sweep data from: {args.sweep_input}")
        try:
            if args.sweep_input.endswith('.parquet'):
                df = pd.read_parquet(args.sweep_input)
            elif args.sweep_input.endswith('.csv'):
                df = pd.read_csv(args.sweep_input)
            else:
                print("[Error] Unsupported file format. Please provide a .parquet or .csv file.")
                return
        except Exception as e:
            print(f"[Error] Failed to read file: {e}")
            return

        report = analyze_period_report(df, channel_name=args.channel)
        
        if "error" in report:
            print(f"[Error] {report['error']}")
        else:
            print_period_report(report)