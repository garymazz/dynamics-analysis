"""
Independent Meta-Analysis: Ensemble Aggregator
Housed in: meta_analysis/ensemble/_main.py

This module performs statistical aggregation and uncertainty estimation 
on clusters of prediction results.
"""
import numpy as np

def calculate_ensemble_statistics(preds: list, target_val: float = None) -> dict:
    """
    Independent Quantitative Analysis: 
    Computes the statistical median and standard deviation (uncertainty) 
    from a collection of prediction values.
    """
    if not preds:
        return {}
    
    # Quantitative Aggregation
    med_pred = float(np.median(preds))
    std_pred = float(np.std(preds))
    
    stats = {
        "prediction_median": med_pred,
        "uncertainty_std": std_pred
    }
    
    # Optional Validation: Included if a ground-truth target is provided
    if target_val is not None:
        truth = float(target_val)
        stats["actual"] = truth
        stats["error"] = abs(med_pred - truth)
        
    return stats