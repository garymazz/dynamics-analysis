import numpy as np

def analyze_period_report(sweep_df, channel_name="S1"):
    """Standalone logic for detecting dominant periods in sweep data."""
    col_err = f"{channel_name}_err_pct"
    if col_err not in sweep_df.columns:
        return {"error": f"Channel {channel_name} not found"}

    df_win = sweep_df.groupby("window_size")[col_err].min().reset_index()
    df_win = df_win.sort_values("window_size")

    windows = df_win["window_size"].values
    errors = df_win[col_err].values

    needle_windows = []
    for i in range(1, len(errors) - 1):
        if errors[i] < errors[i - 1] and errors[i] < errors[i + 1]:
            if errors[i] < 5.0:
                needle_windows.append(windows[i])

    if len(needle_windows) < 2:
        return {
            "status": "FAILED",
            "reason": "Not enough needle points (<2) found.",
            "needles_found": list(needle_windows),
        }

    gaps = np.diff(needle_windows)
    period_est = float(np.median(gaps))

    valid_gaps = [g for g in gaps if abs(g - period_est) < 2.0]
    refined_period = float(np.mean(valid_gaps)) if valid_gaps else period_est
    confidence = (len(valid_gaps) / len(gaps)) * 100.0 if gaps.size > 0 else 0.0

    return {
        "status": "SUCCESS",
        "dominant_period": round(refined_period, 2),
        "period_integer": int(round(refined_period)),
        "confidence_pct": round(confidence, 1),
        "num_needles": len(needle_windows),
        "needles": [int(n) for n in needle_windows],
        "gaps": [float(g) for g in gaps],
    }

def print_period_report(report):
    """Outputs the period analysis report to the console."""
    print("\n=== Dominant Period Analysis Report ===")
    if report.get("status") == "FAILED":
        print("Status: FAILED")
        print(f"Reason: {report.get('reason')}")
        return

    print("Status: SUCCESS")
    print(f"Dominant Period: {report['dominant_period']} samples")
    print(f"Confidence: {report['confidence_pct']}%")
    print(f"Needles Detected: {report['num_needles']}")
    print(f"Needle Locations (Windows): {report['needles']}")
    print(f"Detected Gaps: {report['gaps']}")
    print("=======================================")