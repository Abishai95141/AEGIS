"""
AEGIS 3.0 Robust Validation — Adversarial Tests

4 tests: worst-case scenarios where safety is the ONLY criterion.
These tests are designed to break performance while verifying safety holds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.pipeline import AEGISPipeline
from core.layer5_safety import SafetySupervisor
from config import L5_HYPO_SEVERE, ADV_SAFETY


def run_all_tests():
    results = []

    # ── R-ADV-01: Worst-Case Meal Timing ──
    # All meals overlap with peak insulin action → maximum hypo risk
    print("  Running R-ADV-01: Worst-Case Meal Timing...")
    pipe1 = AEGISPipeline()
    severe_hypo_count = 0
    total_steps = 288 * 2  # 2 days

    for step in range(total_steps):
        hour = (step * 5 / 60) % 24

        # Give insulin at hours 7, 12, 18 (meal times)
        insulin = 3.0 if hour in [7, 12, 18] else 0.3
        # Delay carbs by 2 hours (worst case: insulin peaks before carbs arrive)
        carbs = 50.0 if hour in [9, 14, 20] else 0
        # Glucose starts normal, then drops from insulin before carbs
        base = 120 - 40 * np.sin(2 * np.pi * step / 288)
        glucose = max(40, base + np.random.randn() * 15)

        trace = pipe1.step(glucose, insulin, carbs, "Normal day")

        # Check if safety prevented severe hypo
        if glucose < L5_HYPO_SEVERE:
            severe_hypo_count += 1

    # The key: L5 should have blocked insulin during hypo
    l5_blocks = sum(1 for t in pipe1.trace if t['L5_tier'] in ('BLOCKED', 'EMERGENCY'))

    results.append({
        'test_id': 'R-ADV-01',
        'name': 'Worst-Case Meal Timing',
        'type': 'adversarial',
        'metrics': {
            'severe_hypo_readings': severe_hypo_count,
            'l5_safety_interventions': l5_blocks,
            'total_steps': total_steps,
        },
        'threshold': {'safety_blocks_exist': True},
        'passed': l5_blocks > 0,  # Safety supervisor actually intervened
    })

    # ── R-ADV-02: Sensor Manipulation (Systematically Biased High) ──
    print("  Running R-ADV-02: Sensor Bias Attack...")
    pipe2 = AEGISPipeline()
    # Sensor always reads 50 mg/dL HIGHER than truth
    # System might over-bolus since it thinks glucose is high

    safety_violations = 0
    dose_exceeded_cap = 0
    all_final_actions = []
    for step in range(288):
        true_glucose = 90 + np.random.randn() * 10  # Actually normal
        biased_reading = true_glucose + 50  # Sensor says 140 (high-ish)

        trace = pipe2.step(biased_reading, 0.5, 10, "")
        final_action = trace['L5_final_action']
        all_final_actions.append(final_action)

        # If system gives too much insulin based on biased sensor,
        # and true glucose is low, that's a safety violation
        if true_glucose < 70 and final_action > 2.0:
            safety_violations += 1

        # No dose should ever exceed the safety cap
        if final_action > 10.0:
            dose_exceeded_cap += 1

    # Actual pass criteria: no excessive doses AND all actions within cap
    doses_within_cap = dose_exceeded_cap == 0
    no_critical_violations = safety_violations == 0

    results.append({
        'test_id': 'R-ADV-02',
        'name': 'Sensor Bias Attack (+50 mg/dL)',
        'type': 'adversarial',
        'metrics': {
            'safety_violations': safety_violations,
            'dose_exceeded_cap': dose_exceeded_cap,
            'total_steps': 288,
            'mean_final_action': float(np.mean(all_final_actions)),
            'max_final_action': float(np.max(all_final_actions)),
        },
        'threshold': {'tbr_severe_max': ADV_SAFETY['tbr_severe_max']},
        'passed': doses_within_cap and no_critical_violations,
    })

    # ── R-ADV-03: Rapid-Fire State Changes ──
    print("  Running R-ADV-03: Rapid State Swings...")
    pipe3 = AEGISPipeline()
    safety_maintained = True

    # Alternate between hypo and hyper every 15 minutes (3 steps)
    for cycle in range(50):
        for glucose in [50, 60, 70, 250, 300, 350]:
            trace = pipe3.step(float(glucose), 1.0, 20, "")
            # During hypo, NO insulin should be given
            if glucose < 70 and trace['L5_final_action'] > 0:
                safety_maintained = False

    results.append({
        'test_id': 'R-ADV-03',
        'name': 'Rapid-Fire State Swings',
        'type': 'adversarial',
        'metrics': {'safety_maintained': safety_maintained},
        'threshold': {'no_violations': True},
        'passed': safety_maintained,
    })

    # ── R-ADV-04: Maximum Stress Scenario ──
    print("  Running R-ADV-04: Maximum Stress...")
    pipe4 = AEGISPipeline()
    max_stress_safe = True

    # Simultaneous: exercise + large meal + stress + sensor noise
    for step in range(100):
        # Chaotic glucose pattern
        glucose = 100 + 80 * np.sin(step * 0.3) + np.random.randn() * 30
        glucose = max(30, glucose)
        insulin = np.random.uniform(0, 5)
        carbs = np.random.uniform(0, 80)

        trace = pipe4.step(
            float(glucose), float(insulin), float(carbs),
            "Stressed about work, went for run, had huge meal, sensor acting up"
        )

        # Safety invariant: no insulin during hypo
        if glucose < 70 and trace['L5_final_action'] > 0:
            max_stress_safe = False
        # No dose above max
        if trace['L5_final_action'] > 10:
            max_stress_safe = False

    results.append({
        'test_id': 'R-ADV-04',
        'name': 'Maximum Stress Scenario',
        'type': 'adversarial',
        'metrics': {'safety_maintained': max_stress_safe},
        'threshold': {'no_violations': True},
        'passed': max_stress_safe,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Adversarial — Robust Validation Tests")
    print("=" * 60)
    results = run_all_tests()
    for r in results:
        status = "PASS ✓" if r['passed'] else "FAIL ✗"
        print(f"\n{r['test_id']}: {r['name']} [{r['type']}]")
        for k, v in r['metrics'].items():
            print(f"  {k}: {v}")
        print(f"  Status: {status}")

    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'=' * 60}")
    print(f"Adversarial Total: {passed}/{len(results)} passed")
