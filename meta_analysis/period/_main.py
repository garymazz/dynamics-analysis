import numpy as np

# ==========================================
# INDEPENDENT QUALITATIVE & QUANTITATIVE ANALYSIS
# ==========================================

def detect_needles_heuristic(windows: np.ndarray, errors: np.ndarray, max_error_threshold: float = 5.0) -> list:
    """
    Qualitative analysis: Identifies local minima (needles) based on curve shape.
    A point is a needle if its error is strictly lower than its immediate neighbors 
    and falls below the specified maximum error threshold.
    """
    needle_windows = []
    for i in range(1, len(errors) - 1):
        # Shape check: point must be lower than its neighbors
        if errors[i] < errors[i - 1] and errors[i] < errors[i + 1]:
            # Threshold check: absolute error must be reasonably low
            if errors[i] < max_error_threshold:
                needle_windows.append(windows[i])
    return needle_windows

def calculate_gap_statistics(needle_windows: list) -> tuple:
    """
    Quantitative analysis: Calculates period, median, and confidence metrics.
    Filters out extreme outliers to find the true resonant frequency.
    """
    if len(needle_windows) < 2:
        return None, 0.0, []
        
    gaps = np.diff(needle_windows)
    period_est = float(np.median(gaps))
    
    # Filter valid gaps (removing severe outliers that deviate by more than 2.0)
    valid_gaps = [g for g in gaps if abs(g - period_est) < 2.0]
    
    refined_period = float(np.mean(valid_gaps)) if valid_gaps else period_est
    confidence = (len(valid_gaps) / len(gaps)) * 100.0 if gaps.size > 0 else 0.0
    
    return refined_period, confidence, gaps


# ==========================================
# ORCHESTRATOR & REPORTING
# ==========================================

def analyze_period_report(sweep_df, channel_name="S1"):
    """
    Standalone logic for detecting dominant periods in sweep data.
    Decoupled to use independent qualitative and quantitative analysis functions.
    """
    col_err = f"{channel_name}_err_pct"
    if col_err not in sweep_df.columns:
        return {"error": f"Channel {channel_name} not found"}

    # Aggregate minimum errors per window
    df_win = sweep_df.groupby("window_size")[col_err].min().reset_index()
    df_win = df_win.sort_values("window_size")

    windows = df_win["window_size"].values
    errors = df_win[col_err].values

    # 1. Qualitative Analysis
    needle_windows = detect_needles_heuristic(windows, errors, max_error_threshold=5.0)

    if len(needle_windows) < 2:
        return {
            "status": "FAILED",
            "reason": "Not enough needle points (<2) found.",
            "needles_found": list(needle_windows),
        }

    # 2. Quantitative Analysis
    refined_period, confidence, gaps = calculate_gap_statistics(needle_windows)

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
    """
    Outputs the period analysis report to the console.
    """
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