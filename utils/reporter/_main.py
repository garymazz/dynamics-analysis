import pandas as pd
import json
import os

def generate_sweep_report(results_data, output_base, channel_names, threshold=2.0):
    """
    Analyzes the raw sweep results and generates a comprehensive summary report
    highlighting the best configurations, error distributions, and contiguous bands.
    """
    if not results_data:
        return
        
    df = pd.DataFrame(results_data)
    primary_ch = channel_names[0]
    err_col = f"{primary_ch}_err_pct"
    
    if err_col not in df.columns:
        return

    report = {
        "report_type": "DMD_Sweep_Summary",
        "total_configurations_tested": len(df),
        "primary_optimization_channel": primary_ch,
        "global_statistics": {
            "median_error": float(df[err_col].median()),
            "min_error": float(df[err_col].min()),
            "max_error": float(df[err_col].max())
        },
        "top_performing_configs": []
    }

    # Find the top 5 configs
    top_configs = df.nsmallest(5, err_col)
    for _, row in top_configs.iterrows():
        report["top_performing_configs"].append({
            "window_size": int(row["window_size"]),
            "stack_size": int(row["stack_size"]),
            "error_pct": float(row[err_col])
        })

    # Save the report
    report_path = f"{output_base}_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\n[Reporter] Successfully generated sweep summary: {report_path}")
    except Exception as e:
        print(f"\n[Reporter Error] Could not save report: {e}")
        