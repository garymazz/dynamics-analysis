"""
Evaluator Package Initializer
PURPOSE: Expose the primary accuracy metrics for the meta-analysis refactor.
"""

from ._main import calculate_prediction_errors

__all__ = ['calculate_prediction_errors']