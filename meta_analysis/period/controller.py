
import pandas as pd
from cement import Controller, ex
from core.period_analysis import analyze_period_report, print_period_report

class AnalysisController(Controller):
    class Meta:
        label = 'analysis'
        stacked_on = 'base'
        stacked_type = 'nested'
        description = 'Post-processing and data analysis tools'

    @ex(
        help='Run standalone Dominant Period Analysis on a sweep file.',
        arguments=[
            (['sweep_input'], {'help': 'Path to a previous sweep file (.parquet or .csv)'}),
            (['-c', '--channel'], {'help': 'Specific channel to analyze (default: S1)', 'default': 'S1'})
        ]
    )
    def period(self):
        """CLI Routing for: python main.py analysis period <file> --channel S1"""
        args = self.app.pargs
        print(f"=== Period Analysis Mode ===")
        print(f"Loading: {args.sweep_input}")
        
        try:
            if args.sweep_input.endswith(".parquet"):
                sweep_df = pd.read_parquet(args.sweep_input)
            else:
                sweep_df = pd.read_csv(args.sweep_input)

            report = analyze_period_report(sweep_df, channel_name=args.channel)
            print_period_report(report)
        except Exception as e:
            print(f"Error executing period analysis: {e}")