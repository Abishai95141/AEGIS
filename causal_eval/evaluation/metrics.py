"""
Evaluation metrics for the causal evaluation experiment.

Computes:
  - Mean Absolute Error (MAE): average |τ̂_i - τ_i| across patients
  - 95th percentile error: worst-case accuracy
  - Confidence sequence coverage: fraction of patients where true τ_i
    is contained in ALL confidence sets (anytime-valid coverage)
  - Bias: average (τ̂_i - τ_i) — signed, detects systematic over/under-estimation

All metrics compare estimator output against finite-difference ground truth.
"""

import numpy as np
import csv
import os


def compute_metrics(tau_true, tau_estimated, confidence_bounds=None):
    """
    Compute evaluation metrics for one estimator under one proxy condition.

    Args:
        tau_true: dict mapping patient_id → true τ_i (from ground truth)
        tau_estimated: dict mapping patient_id → estimated τ̂_i
        confidence_bounds: dict mapping patient_id → (lower, upper) or None

    Returns:
        dict with:
            mae: mean absolute error
            p95_error: 95th percentile absolute error
            bias: mean signed error (τ̂ - τ)
            coverage: fraction of patients with true τ in confidence bounds
            n_patients: number of patients evaluated
            errors: array of per-patient absolute errors
    """
    # Align patient IDs
    common_ids = sorted(set(tau_true.keys()) & set(tau_estimated.keys()))
    assert len(common_ids) > 0, "No common patient IDs between true and estimated"

    errors = []
    signed_errors = []
    covered = []

    for pid in common_ids:
        true_val = tau_true[pid]
        est_val = tau_estimated[pid]
        err = abs(est_val - true_val)
        errors.append(err)
        signed_errors.append(est_val - true_val)

        if confidence_bounds is not None and pid in confidence_bounds:
            lo, hi = confidence_bounds[pid]
            covered.append(lo <= true_val <= hi)

    errors = np.array(errors)
    signed_errors = np.array(signed_errors)

    result = {
        'mae': float(np.mean(errors)),
        'p95_error': float(np.percentile(errors, 95)),
        'bias': float(np.mean(signed_errors)),
        'n_patients': len(common_ids),
        'errors': errors,
    }

    if len(covered) > 0:
        result['coverage'] = float(np.mean(covered))
    else:
        result['coverage'] = float('nan')

    return result


def format_results_table(all_results):
    """
    Format results into the required output table.

    Args:
        all_results: list of dicts, each with:
            estimator: name string
            proxy_condition: 'strong' or 'weak'
            mae: mean absolute error
            p95_error: 95th percentile error
            coverage: confidence sequence coverage
            bias: mean signed error

    Returns:
        table_str: formatted string for stdout
        rows: list of dicts for CSV output
    """
    header = (
        f"{'Estimator':<25} {'Proxy':<8} {'MAE':>10} {'P95 Err':>10} "
        f"{'Coverage':>10} {'Bias':>10}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]
    rows = []

    for r in all_results:
        cov_str = f"{r['coverage']:.3f}" if not np.isnan(r.get('coverage', float('nan'))) else "N/A"
        line = (
            f"{r['estimator']:<25} {r['proxy_condition']:<8} "
            f"{r['mae']:>10.4f} {r['p95_error']:>10.4f} "
            f"{cov_str:>10} {r['bias']:>+10.4f}"
        )
        lines.append(line)
        rows.append({
            'estimator': r['estimator'],
            'proxy_condition': r['proxy_condition'],
            'mae': r['mae'],
            'p95_error': r['p95_error'],
            'coverage': r.get('coverage', float('nan')),
            'bias': r['bias'],
        })

    lines.append(separator)
    return "\n".join(lines), rows


def save_results_csv(rows, filepath):
    """
    Save results table to CSV.

    Args:
        rows: list of dicts from format_results_table
        filepath: path to output CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ['estimator', 'proxy_condition', 'mae', 'p95_error', 'coverage', 'bias']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
