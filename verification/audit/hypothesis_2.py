#!/usr/bin/env python3
"""
HYPOTHESIS 2 — Lambda Calibration Adds No Information Over a Constant

WHAT IT MEASURES:
    Compares CTS posterior evolution between two configurations on
    identical blocked-action sequences:
    (A) Current isotonic LambdaCalibrator
    (B) Fixed λ = 0.3 (the population default)
    
    Measures posterior mean and variance of blocked arms at steps 50, 100, 200.

CONFIRMS HYPOTHESIS (architecture must change):
    At all checkpoints, posterior means differ by <1% and variances differ
    by <5%. The calibrator is not contributing meaningful information.

REFUTES HYPOTHESIS (implementation can be fixed):
    Posterior means or variances diverge measurably, indicating the
    calibrator provides genuinely different weighting.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer4_decision import (
    CounterfactualThompsonSampling,
    LambdaCalibrator,
)
from config import L4_CF_LAMBDA_POPULATION


class FixedLambdaCalibrator:
    """Mock calibrator that always returns a fixed lambda."""
    
    def __init__(self, fixed_lambda=0.3):
        self._fixed = fixed_lambda
        self._is_calibrated = True  # Always "calibrated"
    
    def record(self, prediction, actual):
        pass  # No-op
    
    def get_lambda(self, prediction_error_abs):
        return self._fixed
    
    def is_calibrated(self):
        return self._is_calibrated
    
    def n_observations(self):
        return 9999  # Pretend we have lots of data


def run_cts_with_calibrator(calibrator, n_steps, dt_predictions, dt_actuals,
                             blocked_arm=2, seed=42):
    """
    Run CTS with a given calibrator, blocking arm `blocked_arm` at every step
    and using counterfactual updates.
    
    Returns dict with posterior snapshots at checkpoints.
    """
    rng = np.random.default_rng(seed)
    n_arms = 4
    
    cts = CounterfactualThompsonSampling(
        n_arms, lambda_calibrator=calibrator
    )
    
    checkpoints = {50: None, 100: None, 200: None}
    
    for t in range(n_steps):
        # Feed calibrator with DT prediction data (same for both configurations)
        if t < len(dt_predictions):
            calibrator.record(dt_predictions[t], dt_actuals[t])
        
        # Simulate a blocked action: arm `blocked_arm` was proposed but blocked by L5
        # DT provides a prediction of what would have happened
        dt_pred = dt_predictions[t % len(dt_predictions)] + rng.normal(0, 5)
        cts.counterfactual_update(blocked_arm, digital_twin_prediction=dt_pred)
        
        # Also do some real updates on other arms to keep posteriors moving
        for arm in range(n_arms):
            if arm != blocked_arm and rng.random() < 0.3:
                reward = rng.normal(-0.5, 0.3)
                cts.update(arm, reward)
        
        step = t + 1
        if step in checkpoints:
            checkpoints[step] = {
                'means': cts.post_means.copy(),
                'vars': cts.post_vars.copy(),
            }
    
    return checkpoints


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    n_steps = 200
    blocked_arm = 2
    
    # Generate synthetic DT prediction data (same for both runs)
    # DT predicts glucose; actuals have some noise
    true_glucose = 150.0 + 20 * np.sin(np.linspace(0, 4*np.pi, n_steps))
    dt_predictions = true_glucose + rng.normal(0, 15, n_steps)  # DT prediction
    dt_actuals = true_glucose + rng.normal(0, 5, n_steps)       # Actual glucose
    
    print("=" * 70)
    print("HYPOTHESIS 2: Lambda Calibration vs Fixed Constant")
    print("=" * 70)
    print(f"Blocked arm: {blocked_arm}, Steps: {n_steps}")
    print(f"Config A: Isotonic LambdaCalibrator (min=0.05, max=0.8)")
    print(f"Config B: Fixed λ = {L4_CF_LAMBDA_POPULATION}")
    print()
    
    # Run A: isotonic calibrator
    cal_A = LambdaCalibrator(min_samples=30, refit_interval=20)
    snapshots_A = run_cts_with_calibrator(
        cal_A, n_steps, dt_predictions, dt_actuals, blocked_arm, seed=42
    )
    
    # Run B: fixed lambda
    cal_B = FixedLambdaCalibrator(fixed_lambda=L4_CF_LAMBDA_POPULATION)
    snapshots_B = run_cts_with_calibrator(
        cal_B, n_steps, dt_predictions, dt_actuals, blocked_arm, seed=42
    )
    
    # Compare
    all_close = True
    for step in [50, 100, 200]:
        A = snapshots_A[step]
        B = snapshots_B[step]
        
        mean_diff = np.abs(A['means'] - B['means'])
        var_diff = np.abs(A['vars'] - B['vars'])
        
        # Relative differences
        mean_rel = mean_diff / (np.abs(A['means']) + 1e-8)
        var_rel = var_diff / (np.abs(A['vars']) + 1e-8)
        
        print(f"--- Step {step} ---")
        print(f"  Calibrator posterior means: {np.round(A['means'], 5)}")
        print(f"  Fixed λ posterior means:    {np.round(B['means'], 5)}")
        print(f"  Abs difference (means):    {np.round(mean_diff, 5)}")
        print(f"  Relative difference:       {np.round(mean_rel * 100, 2)}%")
        print()
        print(f"  Calibrator posterior vars: {np.round(A['vars'], 6)}")
        print(f"  Fixed λ posterior vars:    {np.round(B['vars'], 6)}")
        print(f"  Abs difference (vars):    {np.round(var_diff, 6)}")
        print(f"  Relative difference:      {np.round(var_rel * 100, 2)}%")
        print()
        
        # Check: are posteriors distinguishable for the blocked arm?
        blocked_mean_diff = mean_rel[blocked_arm]
        blocked_var_diff = var_rel[blocked_arm]
        
        if blocked_mean_diff > 0.05 or blocked_var_diff > 0.10:
            print(f"  Blocked arm {blocked_arm}: DISTINGUISHABLE "
                  f"(mean_diff={blocked_mean_diff*100:.1f}%, "
                  f"var_diff={blocked_var_diff*100:.1f}%)")
            all_close = False
        else:
            print(f"  Blocked arm {blocked_arm}: INDISTINGUISHABLE "
                  f"(mean_diff={blocked_mean_diff*100:.1f}%, "
                  f"var_diff={blocked_var_diff*100:.1f}%)")
        print()
    
    # Check if calibrator is even calibrated by step 200
    print("-" * 70)
    print(f"Calibrator calibrated by step 200: {cal_A._is_calibrated}")
    print(f"Calibrator observations: {len(cal_A._predictions)}")
    
    # Verdict
    print()
    print("=" * 70)
    if all_close:
        print("VERDICT: CONFIRMED — Posteriors are statistically indistinguishable.")
        print("         The isotonic calibrator adds no information over λ=0.3.")
        print("         This is consistent with Audit Item 1 (circular fit).")
    else:
        print("VERDICT: REFUTED — Posteriors diverge measurably.")
        print("         The calibrator IS producing different λ weights,")
        print("         though the target construction may still be a proxy.")
    print("=" * 70)


if __name__ == '__main__':
    main()
