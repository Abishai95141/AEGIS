"""
AEGIS 3.0 Robust Validation — Layer 2 (Digital Twin) Tests

6 tests: 3 nominal + 3 robustness

Key insight: The UKF's value is in STATE ESTIMATION (after incorporating measurement),
not in open-loop prediction. Tests evaluate the right quantities.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.layer2_digital_twin import DigitalTwin, BergmanMinimalModel, AdaptiveConstrainedUKF
from simulator.patient import HovorkaPatientSimulator
from simulator.noise import SensorNoiseModel
from config import (L2_NOMINAL, L2_ROBUST, GLUCOSE_MIN, GLUCOSE_MAX,
                    REMOTE_INSULIN_MIN, REMOTE_INSULIN_MAX,
                    INSULIN_MIN, INSULIN_MAX)


def _generate_bergman_data(n_steps=500, seed=42):
    """Generate glucose from Bergman for nominal tests."""
    rng = np.random.RandomState(seed)
    bergman = BergmanMinimalModel()
    state = np.array([120.0, 0.01, 10.0])

    glucose, insulin_vals, carb_vals = [], [], []
    for i in range(n_steps):
        hour = (i * 5 / 60) % 24
        carbs = 0.0
        if abs(hour - 8) < 0.1: carbs = 40
        elif abs(hour - 13) < 0.1: carbs = 60
        elif abs(hour - 19) < 0.1: carbs = 50
        insulin = 0.3 + (2.0 if carbs > 10 else 0)

        state = bergman.rk4_step(state, insulin, carbs)
        state = np.clip(state, 
                        [GLUCOSE_MIN, REMOTE_INSULIN_MIN, INSULIN_MIN], 
                        [GLUCOSE_MAX, REMOTE_INSULIN_MAX, INSULIN_MAX])
        obs = state[0] + rng.randn() * 8
        glucose.append(np.clip(obs, GLUCOSE_MIN, GLUCOSE_MAX))
        insulin_vals.append(insulin)
        carb_vals.append(carbs)

    return np.array(glucose), np.array(insulin_vals), np.array(carb_vals)


def run_all_tests():
    results = []
    glucose, insulin, carbs = _generate_bergman_data(500, seed=42)

    # ── R-L2-01: State Estimation Quality (Nominal) ──
    # UKF should produce better state estimates than raw noisy measurements
    dt1 = DigitalTwin()
    estimate_errors = []
    raw_errors = []

    # Ground truth Bergman re-simulation
    bergman_gt = BergmanMinimalModel()
    gt_state = np.array([120.0, 0.01, 10.0])

    for i in range(len(glucose)):
        ins_val, carb_val = float(insulin[i]), float(carbs[i])
        gt_state = bergman_gt.rk4_step(gt_state, ins_val, carb_val)
        gt_state = np.clip(gt_state, 
                           [GLUCOSE_MIN, REMOTE_INSULIN_MIN, INSULIN_MIN], 
                           [GLUCOSE_MAX, REMOTE_INSULIN_MAX, INSULIN_MAX])
        g_true = gt_state[0]
        g_obs = float(glucose[i])

        dt1.predict(ins_val, carb_val)
        dt1.update(g_obs, ins_val, carb_val)
        estimate_errors.append((dt1.get_glucose() - g_true) ** 2)
        raw_errors.append((g_obs - g_true) ** 2)

    est_rmse = np.sqrt(np.mean(estimate_errors[-300:]))
    raw_rmse = np.sqrt(np.mean(raw_errors[-300:]))
    ratio = est_rmse / max(raw_rmse, 1.0)

    results.append({
        'test_id': 'R-L2-01',
        'name': 'State Estimation Quality',
        'type': 'nominal',
        'metrics': {'estimate_rmse': float(est_rmse), 'raw_rmse': float(raw_rmse),
                    'improvement_ratio': float(ratio)},
        'threshold': {'ratio_max': L2_NOMINAL['ude_rmse_ratio_max']},
        'passed': ratio <= L2_NOMINAL['ude_rmse_ratio_max'],
    })

    # ── R-L2-02: Constraint Satisfaction (Nominal) ──
    dt2 = DigitalTwin()
    violations = 0
    for i in range(len(glucose)):
        dt2.predict(float(insulin[i]), float(carbs[i]))
        dt2.update(float(glucose[i]), float(insulin[i]), float(carbs[i]))
        violations += dt2.check_constraint_violations()

    violation_rate = violations / max(len(glucose), 1)
    results.append({
        'test_id': 'R-L2-02',
        'name': 'Constraint Satisfaction',
        'type': 'nominal',
        'metrics': {'violations': violations, 'total': len(glucose),
                    'violation_rate': float(violation_rate)},
        'threshold': {'violation_rate': L2_NOMINAL['constraint_violation_rate']},
        'passed': violation_rate <= L2_NOMINAL['constraint_violation_rate'],
    })

    # ── R-L2-03: UKF Q Adaptation Direction (Nominal) ──
    # After stable period with good tracking, introduce sudden model mismatch.
    # Q should increase as innovation grows.
    ukf = AdaptiveConstrainedUKF()

    # Phase 1: Stable tracking — feed data close to model prediction
    for i in range(60):
        ukf.predict(0.5, 0)
        ukf.update(ukf.x_hat[0] + np.random.randn() * 5)

    # Q should now be near base (or slightly elevated)
    q_stable = ukf.Q[0, 0]
    q_base = ukf.Q_base[0, 0]

    # Phase 2: Sudden model mismatch — glucose jumps to 250
    # and stays there, but model predicts ~120
    for i in range(40):
        ukf.predict(0.5, 0)
        # Feed glucose far from model's prediction
        ukf.update(250 + np.random.randn() * 5)

    q_after = ukf.Q[0, 0]

    # The key: Q should have increased (or at least not decreased below base)
    q_increased = q_after > q_stable

    results.append({
        'test_id': 'R-L2-03',
        'name': 'UKF Q Adaptation Direction',
        'type': 'nominal',
        'metrics': {'q_base': float(q_base), 'q_stable': float(q_stable),
                    'q_after_mismatch': float(q_after),
                    'q_increased': q_increased},
        'threshold': {'Q_increases_on_mismatch': True},
        'passed': q_increased,
    })

    # ── R-L2-04: Model Mismatch (Robustness) ──
    sim = HovorkaPatientSimulator(seed=42)
    hov_data = sim.generate_dataset(n_days=2)
    hov_glucose = hov_data['glucose_mg_dl'].values
    hov_insulin = hov_data['insulin_bolus_u'].values
    hov_carbs = hov_data['carbs_g'].values

    dt4 = DigitalTwin()
    mismatch_errors = []
    for i in range(min(len(hov_glucose), 400)):
        dt4.predict(float(hov_insulin[i]), float(hov_carbs[i]))
        dt4.update(float(hov_glucose[i]), float(hov_insulin[i]), float(hov_carbs[i]))
        mismatch_errors.append((dt4.get_glucose() - hov_glucose[i]) ** 2)

    mismatch_rmse = np.sqrt(np.mean(mismatch_errors[-200:]))
    results.append({
        'test_id': 'R-L2-04',
        'name': 'Model Mismatch (Hovorka vs Bergman)',
        'type': 'robust',
        'metrics': {'mismatch_rmse': float(mismatch_rmse)},
        'threshold': {'rmse_max': L2_ROBUST['mismatch_rmse_max']},
        'passed': mismatch_rmse <= L2_ROBUST['mismatch_rmse_max'],
    })

    # ── R-L2-05: Unannounced Meal Detection (Robustness) ──
    # Track innovation magnitude; unannounced meal should produce 
    # sustained large innovations.
    ukf5 = AdaptiveConstrainedUKF()
    for i in range(50):
        ukf5.predict(0.5, 0)
        ukf5.update(120 + np.random.randn() * 3)

    # Record baseline innovation
    baseline_innov = np.mean(np.abs(ukf5.innovation_history[-20:]))

    # Unannounced meal: glucose rises
    for i in range(30):
        ukf5.predict(0.5, 0)
        meal_glucose = 120 + 60 * (1 - np.exp(-0.15 * i))
        ukf5.update(meal_glucose + np.random.randn() * 3)

    # Post-meal innovation should be noticeably larger
    post_innov = np.mean(np.abs(ukf5.innovation_history[-20:]))
    innov_ratio = post_innov / max(baseline_innov, 1.0)
    detected = innov_ratio > 1.5  # Innovation at least 1.5x baseline

    results.append({
        'test_id': 'R-L2-05',
        'name': 'Unannounced Meal Detection (Innovation)',
        'type': 'robust',
        'metrics': {'baseline_innovation': float(baseline_innov),
                    'post_meal_innovation': float(post_innov),
                    'innovation_ratio': float(innov_ratio),
                    'detected': detected},
        'threshold': {'innovation_ratio_min': 1.5},
        'passed': detected,
    })

    # ── R-L2-06: Sensor Noise Filtering (Robustness) ──
    noise_model = SensorNoiseModel(seed=42)
    dt6 = DigitalTwin()
    clean_glucose, clean_ins, clean_carbs = _generate_bergman_data(300, seed=99)
    noisy_glucose = noise_model.apply_gaussian_noise(clean_glucose, noise_std=20.0)

    # Ground truth re-simulation
    bergman_gt6 = BergmanMinimalModel()
    gt_state6 = np.array([120.0, 0.01, 10.0])
    est_errors, noisy_errors = [], []

    for i in range(len(noisy_glucose)):
        gt_state6 = bergman_gt6.rk4_step(gt_state6, float(clean_ins[i]), float(clean_carbs[i]))
        gt_state6 = np.clip(gt_state6, 
                            [GLUCOSE_MIN, REMOTE_INSULIN_MIN, INSULIN_MIN], 
                            [GLUCOSE_MAX, REMOTE_INSULIN_MAX, INSULIN_MAX])
        g_true = gt_state6[0]

        dt6.predict(float(clean_ins[i]), float(clean_carbs[i]))
        dt6.update(float(noisy_glucose[i]), float(clean_ins[i]), float(clean_carbs[i]))
        est_errors.append((dt6.get_glucose() - g_true) ** 2)
        noisy_errors.append((noisy_glucose[i] - g_true) ** 2)

    est_rmse = np.sqrt(np.mean(est_errors[-200:]))
    noisy_rmse = np.sqrt(np.mean(noisy_errors[-200:]))
    filter_improvement = est_rmse < noisy_rmse  # Filtering should help

    results.append({
        'test_id': 'R-L2-06',
        'name': 'Sensor Noise Filtering',
        'type': 'robust',
        'metrics': {'filtered_rmse': float(est_rmse),
                    'raw_noisy_rmse': float(noisy_rmse),
                    'filter_helps': filter_improvement},
        'threshold': {'filtered_better_than_raw': True},
        'passed': filter_improvement,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Layer 2 — Robust Validation Tests")
    print("=" * 60)
    results = run_all_tests()
    for r in results:
        status = "PASS ✓" if r['passed'] else "FAIL ✗"
        print(f"\n{r['test_id']}: {r['name']} [{r['type']}]")
        for k, v in r['metrics'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        print(f"  Status: {status}")

    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'=' * 60}")
    print(f"Layer 2 Total: {passed}/{len(results)} passed")
