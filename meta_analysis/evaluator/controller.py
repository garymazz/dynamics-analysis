"""
Controller: Evaluator Orchestrator
Housed in: meta_analysis/evaluator/controller.py

PURPOSE:
To manage the generation and validation of first-order accuracy metadata. 
This controller translates raw prediction tensors into standardized 
error dictionaries used for downstream meta-analysis.

ALGORITHM BENEFITS:
- Standardization: Ensures every analysis module (Period, Cluster, Ensemble) 
  uses the exact same definition of 'Accuracy' and '% Error'.
- Deterministic Validation: Provides a fixed point of truth for model 
  fidelity before qualitative heuristics are applied.

DRAWBACKS:
- I/O Bound: Frequently accessing HDF5 or CSV files to compute errors 
  across massive sweeps can create a bottleneck if not batched.

WORKFLOW DESCRIPTION:
1. Input Retrieval: Pulls {ch}_pred_val and {ch}_val_target from storage.
2. Metric Computation: Executes calculate_prediction_errors() from _main.py.
3. Metadata Injection: Updates the primary record schema with err_pct 
   and err_pct_int fields.
4. Threshold Enforcement: Flags configurations that exceed user-defined 
   tolerances for early-exit optimization.
"""

from cement import Controller, expose
from ._main import calculate_prediction_errors

class EvaluatorController(Controller):
    class Meta:
        label = 'evaluator'
        stacked_on = 'base'
        stacked_type = 'embedded'
        description = "Manage quantitative accuracy metrics and error dictionary generation."
        
        arguments = [
            (['--threshold'], dict(help='Global error threshold for model validity', default=5.0, type=float)),
            (['--metric'], dict(help='Primary metric for evaluation (err_pct, err_pct_int)', default='err_pct')),
            (['--batch-size'], dict(help='Number of records to process per I/O cycle', default=1000, type=int)),
        ]

    @expose(hide=True)
    def evaluate(self):
        """
        STEP INTENT: Initialize the evaluation environment.
        """
        p = self.app.pargs
        self.app.log.info(f"Evaluator initialized with {p.metric} threshold: {p.threshold}%")

    def evaluate_record(self, pred_val, target_val):
        """
        STEP INTENT: Execute the base accuracy logic for a single data point.
        WORKFLOW: 
        1. Call independent evaluator logic.
        2. Apply thresholding 'judgment' based on controller settings.
        """
        metrics = calculate_prediction_errors(pred_val, target_val)
        
        # Add a Boolean flag for 'validity' based on CLI threshold
        if metrics:
            primary_err = metrics.get(self.app.pargs.metric, 100.0)
            metrics['is_valid'] = primary_err <= self.app.pargs.threshold
            
        return metrics

# IMPLEMENTATION NOTE: 
# The intent behind this multi-layered schema is to provide traceability. 
# If a forecast (Ensemble) is incorrect, the user can step back to the 
# Period analysis to see if the natural pulse was weak, or to this 
# Evaluator schema to check raw prediction errors for specific stack sizes.