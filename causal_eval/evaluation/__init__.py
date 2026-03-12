"""
AEGIS Causal Evaluation — Experiment orchestration and metrics.
"""

from .experiment import CausalExperiment, run_experiment
from .metrics import compute_metrics, format_results_table, save_results_csv
