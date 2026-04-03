"""
Independent Meta-Analysis: Base Evaluator
Housed in: meta_analysis/evaluator/_main.py

This module provides the fundamental accuracy metrics and error dictionary 
generation used by all other meta-analysis functions.
"""

def calculate_prediction_errors(pred_val: float, target_val: float) -> dict:
    """
    Independent Quantitative Analysis: 
    Computes a standardized dictionary of continuous and integer-based error metrics.
    
    This function is a decoupled evaluator; it does not know about DMD 
    operators or window sizes, only raw values.
    """
    if target_val is None:
        return {}
        
    # Qualitative Heuristic: Integer-rounded comparison
    p_int = int(round(pred_val))
    t_int = int(target_val)
    
    # Quantitative Calculations: Continuous Errors
    err = abs(pred_val - target_val)
    # Ensure zero-division safety for percentage calculations
    pct = (err / max(1.0, abs(target_val))) * 100.0
    
    # Quantitative Calculations: Integer Errors
    err_int = abs(p_int - t_int)
    pct_int = (err_int / max(1.0, abs(t_int))) * 100.0
    
    return {
        "val_target": float(target_val),
        "pred_err": err,
        "err_pct": pct,
        "val_target_int": t_int,
        "pred_err_int": err_int,
        "err_pct_int": pct_int
    }