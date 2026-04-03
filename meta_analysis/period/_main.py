"""
Independent Meta-Analysis: Period & Structural Pattern Discovery
Housed in: meta_analysis/period/_main.py

PURPOSE:
To identify the 'Natural Pulse' of a system by analyzing how prediction 
error 'needles' (local minima) cluster, distribute, and correlate across 
three distinct dimensions of the search space:
1. Data Row Range (Temporal Stability)
2. Stack Size (Model Order/Rank Complexity)
3. Window Size (Observation Scale/Frequency)

ALGORITHM BENEFITS:
- Noise Rejection: True physical resonances persist across different stack 
  sizes, while stochastic noise typically creates isolated error drops.
- Structural Validation: By correlating peak patterns across sweeps, the 
  system identifies the most robust "Natural Pulse" for forecasting.
- Range Discovery: Detects transient stability zones within single windows 
  that are otherwise obscured by global error averages.

DRAWBACKS:
- Computational Load: The Triple-Sweep (O(N*M*K)) is resource-intensive 
  and requires GPU acceleration for large datasets.
- Sensitivity: Heuristics for "needles" require careful tuning of the 
  % error thresholds to avoid missing narrow resonance bands.

WORKFLOW DESCRIPTION:
1. Feature Extraction: Identify 'needles' (local minima) in error surfaces.
2. Distribution Analysis: Calculate gap distances between needles to 
   find periodic consistency.
3. Structural Correlation: Compare needle "fingerprints" across different 
   stack and window ranges to ensure pattern persistence.
4. Judgment: Output a confidence-weighted period report.

IMPLEMENTATION NOTE: 
The intent behind this multi-layered schema is to provide traceability. 
If a forecast (Ensemble) is incorrect, the user can step back to this 
Period analysis to see if the natural pulse was weak, or to the Primary 
schema to check raw prediction errors for specific stack sizes.
"""

import numpy as np

def detect_intra_window_needle_ranges(window_data: np.ndarray, threshold: float = 0.01) -> list:
    """
    STEP INTENT: Find ranges of needle patterns within a single window.
    Detects sub-segments of low volatility where DMD fit is highest.
    """
    # Use L2 norm for multi-channel data to get a singular magnitude signal
    signal = np.linalg.norm(window_data, axis=0) if window_data.ndim > 1 else window_data
    
    # Calculate rolling variance to find "stability needles"
    local_var = np.array([np.var(signal[i:i+5]) for i in range(len(signal)-5)])
    
    ranges = []
    in_range = False
    start_idx = 0
    
    for i, var in enumerate(local_var):
        if var < threshold:
            if not in_range:
                start_idx = i
                in_range = True
        else:
            if in_range:
                ranges.append((start_idx, i))
                in_range = False
    return ranges

def calculate_gap_distribution(needle_indices: list) -> dict:
    """
    STEP INTENT: Quantify the 'Natural Pulse' spacing between accuracy needles.
    Analyzes gap distances over % error, stack size, and data row sweeps.
    """
    if len(needle_indices) < 2:
        return {"mean": 0.0, "std": 0.0, "distribution": []}
    
    gaps = np.diff(needle_indices)
    return {
        "mean": float(np.mean(gaps)),
        "std": float(np.std(gaps)),
        "distribution": gaps.tolist(),
        "is_periodic": np.std(gaps) < (0.1 * np.mean(gaps)) if np.mean(gaps) > 0 else False
    }

def calculate_peak_correlation(error_curve_a: list, error_curve_b: list) -> float:
    """
    STEP INTENT: Expose similar needle patterns between different sweeps.
    Correlates peak patterns over stack size and data row range sweeps.
    """
    if len(error_curve_a) != len(error_curve_b) or len(error_curve_a) < 2:
        return 0.0
    
    # Standardize to compare peak locations (shape) rather than absolute error
    a = (error_curve_a - np.mean(error_curve_a)) / (np.std(error_curve_a) + 1e-9)
    b = (error_curve_b - np.mean(error_curve_b)) / (np.std(error_curve_b) + 1e-9)
    
    return float(np.corrcoef(a, b)[0, 1])

def analyze_period_report(needle_data: dict, correlation_score: float) -> dict:
    """
    STEP INTENT: Final qualitative judgment on period consistency.
    Combines gap distribution and inter-sweep correlation.
    """
    confidence = correlation_score * (1.0 - (needle_data['std'] / (needle_data['mean'] + 1e-9)))
    
    return {
        "status": "Periodic" if confidence > 0.7 else "Stochastic",
        "confidence": max(0.0, float(confidence)),
        "mean_period": needle_data['mean'],
        "consistency": needle_data['is_periodic']
    }

def print_period_report(results: dict):
    """
    STEP INTENT: Present structural meta-analysis to the user.
    """
    print("\n" + "="*45)
    print("      QUALITATIVE STRUCTURAL PERIOD REPORT")
    print("="*45)
    print(f"Status:             {results['status']}")
    print(f"Confidence Score:   {results['confidence']:.2%}")
    print(f"Detected Mean Gap:  {results['mean_period']:.2f}")
    print(f"Periodic Integrity: {'HIGH' if results['consistency'] else 'LOW'}")
    print("="*45 + "\n")

