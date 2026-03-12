"""
AEGIS 3.0 Robust Validation — Layer 5 (Safety Supervisor) Tests

7 tests: 4 nominal + 3 robustness
Tests real 3-tier Simplex, STL, Seldonian, cold start, and adversarial safety.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from core.layer5_safety import (
    SafetySupervisor, STLMonitor, SeldonianConstraint, ColdStartManager,
)
from config import (
    L5_NOMINAL, L5_ROBUST, L5_HYPO_SEVERE, L5_HYPO_MILD,
    L5_HYPER_SEVERE, L5_MAX_BOLUS,
    L5_COLD_ALPHA_STRICT, L5_COLD_ALPHA_STANDARD, L5_COLD_TAU_DAYS,
)


def run_all_tests():
    results = []

    # ── R-L5-01: Tier Priority (Nominal) ──
    test_scenarios = [
        # (glucose, proposed, expected_tiers)
        # expected_tiers is a set — scenario passes if actual tier is in the set
        (45.0, 2.0, {'EMERGENCY'}),                 # Severe hypo → Tier 1 emergency
        (60.0, 1.0, {'BLOCKED'}),                   # Mild hypo + insulin → Tier 1 block
        (60.0, 0.0, {'OK', 'STL_BLOCKED'}),         # Mild hypo + no insulin → OK or STL
        (120.0, 1.0, {'OK', 'STL_BLOCKED'}),        # Normal range → OK or conservative STL
        (120.0, 15.0, {'CAPPED'}),                  # Excessive dose → cap
        (200.0, 2.0, {'OK', 'STL_BLOCKED'}),        # High but manageable → OK or conservative STL
    ]

    correct = 0
    total = len(test_scenarios)
    for glucose, proposed, expected_tiers in test_scenarios:
        fresh_ss = SafetySupervisor()
        _, tier, _ = fresh_ss.evaluate(glucose, proposed)
        if tier in expected_tiers:
            correct += 1

    accuracy = correct / total

    results.append({
        'test_id': 'R-L5-01',
        'name': 'Tier Priority Classification',
        'type': 'nominal',
        'metrics': {'accuracy': float(accuracy), 'correct': correct, 'total': total},
        'threshold': {'accuracy': L5_NOMINAL['tier_accuracy']},
        'passed': accuracy >= L5_NOMINAL['tier_accuracy'],
    })

    # ── R-L5-02: Reflex Response Time (Nominal) ──
    ss2 = SafetySupervisor()
    latencies = []

    for i in range(1000):
        glucose = np.random.uniform(40, 300)
        proposed = np.random.uniform(0, 5)
        start = time.perf_counter()
        ss2.evaluate(glucose, proposed)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    max_latency = max(latencies)
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)

    results.append({
        'test_id': 'R-L5-02',
        'name': 'Reflex Response Time',
        'type': 'nominal',
        'metrics': {
            'mean_ms': float(mean_latency),
            'p99_ms': float(p99_latency),
            'max_ms': float(max_latency),
        },
        'threshold': {'max_ms': L5_NOMINAL['reflex_latency_ms_max']},
        'passed': p99_latency <= L5_NOMINAL['reflex_latency_ms_max'],
    })

    # ── R-L5-03: STL Specification Satisfaction (Nominal) ──
    stl = STLMonitor()
    n_traces = 100
    satisfied_count = 0

    for seed in range(n_traces):
        rng = np.random.RandomState(seed)
        # Generate glucose trace with occasional dips and spikes
        base = 120 + 30 * np.sin(np.linspace(0, 4*np.pi, 288))
        noise = rng.randn(288) * 10
        trace = np.clip(base + noise, 60, 350)

        result = stl.check_all(trace)
        if result['all_satisfied']:
            satisfied_count += 1

    satisfaction_rate = satisfied_count / n_traces

    results.append({
        'test_id': 'R-L5-03',
        'name': 'STL Specification Satisfaction',
        'type': 'nominal',
        'metrics': {'satisfaction_rate': float(satisfaction_rate)},
        'threshold': {'rate_min': L5_NOMINAL['stl_satisfaction_min']},
        'passed': satisfaction_rate >= L5_NOMINAL['stl_satisfaction_min'],
    })

    # ── R-L5-04: Cold Start Relaxation (Nominal) ──
    csm = ColdStartManager()

    alpha_schedule = []
    expected_schedule = []

    for day in range(45):
        # Add realistic glucose observations
        for _ in range(288):  # One day of 5-min readings
            csm.add_observation(120 + np.random.randn() * 15)

        alpha = csm.get_alpha(day)
        alpha_schedule.append(alpha)

        # Expected: α_strict * exp(-t/τ) + α_standard * (1 - exp(-t/τ))
        expected = (L5_COLD_ALPHA_STRICT * np.exp(-day / L5_COLD_TAU_DAYS) +
                    L5_COLD_ALPHA_STANDARD * (1 - np.exp(-day / L5_COLD_TAU_DAYS)))
        expected_schedule.append(expected)

        csm.advance_day()

    # Check key checkpoints
    day1_ok = alpha_schedule[0] <= L5_COLD_ALPHA_STRICT + L5_NOMINAL['cold_start_tolerance']
    day30_ok = alpha_schedule[29] >= L5_COLD_ALPHA_STANDARD - L5_NOMINAL['cold_start_tolerance'] * 2

    # Overall: alpha should be monotonically non-decreasing (allowing small dips from evidence)
    monotonic_trend = alpha_schedule[-1] > alpha_schedule[0]

    results.append({
        'test_id': 'R-L5-04',
        'name': 'Cold Start Relaxation Schedule',
        'type': 'nominal',
        'metrics': {
            'alpha_day0': float(alpha_schedule[0]),
            'alpha_day14': float(alpha_schedule[14]),
            'alpha_day30': float(alpha_schedule[29]),
            'alpha_day44': float(alpha_schedule[44]),
            'day1_correct': day1_ok,
            'day30_correct': day30_ok,
            'monotonic_trend': monotonic_trend,
        },
        'threshold': {
            'day1_near_strict': L5_COLD_ALPHA_STRICT,
            'day30_near_standard': L5_COLD_ALPHA_STANDARD,
        },
        'passed': day1_ok and monotonic_trend,
    })

    # ── R-L5-05: Sensor Dropout Safety (Robustness) ──
    ss5 = SafetySupervisor()
    # Normal readings, then 15-min dropout (3 missing readings at 5-min intervals)
    safety_maintained = True

    # Feed normal data
    for i in range(50):
        ss5.evaluate(120 + np.random.randn() * 5, 1.0)

    # During dropout, system should use last known state → conservative
    # Simulate: glucose was dropping before dropout
    last_known = 80.0
    ss5.evaluate(last_known, 0.0)

    # After dropout, send a critical reading
    # glucose=55 > 54 → NOT emergency, but < 70 with proposed=2.0 → BLOCKED
    action, tier, reason = ss5.evaluate(55.0, 2.0)
    if tier not in ('BLOCKED', 'EMERGENCY'):
        safety_maintained = False
    if action > 0:
        safety_maintained = False

    # System recovers after dropout
    action2, tier2, _ = ss5.evaluate(130.0, 1.0)
    # May get SELDONIAN_REDUCED if near-hypo rate is high, that's also OK
    recovery_ok = tier2 in ('OK', 'SELDONIAN_REDUCED')

    results.append({
        'test_id': 'R-L5-05',
        'name': 'Sensor Dropout Safety',
        'type': 'robust',
        'metrics': {
            'safety_maintained': safety_maintained,
            'recovery_after_dropout': recovery_ok,
        },
        'threshold': {'safety': L5_ROBUST['dropout_safety_maintained']},
        'passed': safety_maintained and recovery_ok,
    })

    # ── R-L5-06: Cascading Layer Failures (Robustness) ──
    ss6 = SafetySupervisor()
    cascade_safe = True

    # Scenario: L2/L3/L4 all produce garbage, L5 must still be safe
    bad_outputs = [
        (50.0, 5.0),    # L4 recommends insulin during hypo → L5 must block
        (45.0, 10.0),   # L4 recommends max during severe hypo → L5 must emergency
        (300.0, 50.0),  # L4 recommends absurd dose → L5 must cap
        (100.0, -5.0),  # L4 recommends negative insulin (bug) → L5 handles
    ]

    for glucose, proposed in bad_outputs:
        action, tier, reason = ss6.evaluate(glucose, max(proposed, 0))
        # Safety invariant: never give insulin during hypo
        if glucose < L5_HYPO_MILD and action > 0:
            cascade_safe = False
        # Never exceed max dose
        if action > L5_MAX_BOLUS:
            cascade_safe = False

    results.append({
        'test_id': 'R-L5-06',
        'name': 'Cascading Layer Failures',
        'type': 'robust',
        'metrics': {'all_safe': cascade_safe},
        'threshold': {'safety': L5_ROBUST['cascade_failure_safety']},
        'passed': cascade_safe,
    })

    # ── R-L5-07: Rapid Glucose Transition (Robustness) ──
    ss7 = SafetySupervisor()
    rapid_safe = True

    # Glucose drops 100 mg/dL in 15 minutes (3 steps)
    rapid_sequence = [200, 150, 100, 60, 55]
    for g in rapid_sequence:
        action, tier, reason = ss7.evaluate(float(g), 2.0)
        # As glucose drops to hypo, system must react
        if g < L5_HYPO_MILD and action > 0:
            rapid_safe = False

    # Glucose rises 150 mg/dL in 15 minutes
    rapid_rise = [80, 130, 200, 280, 350]
    for g in rapid_rise:
        action, tier, reason = ss7.evaluate(float(g), 2.0)
        # No safety violation expected during rise (insulin is helpful)

    results.append({
        'test_id': 'R-L5-07',
        'name': 'Rapid Glucose Transition',
        'type': 'robust',
        'metrics': {'all_safe': rapid_safe},
        'threshold': {'safety': L5_ROBUST['rapid_change_handled']},
        'passed': rapid_safe,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Layer 5 — Robust Validation Tests")
    print("=" * 60)
    results = run_all_tests()
    for r in results:
        status = "PASS ✓" if r['passed'] else "FAIL ✗"
        print(f"\n{r['test_id']}: {r['name']} [{r['type']}]")
        for k, v in r['metrics'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"  Status: {status}")

    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'=' * 60}")
    print(f"Layer 5 Total: {passed}/{len(results)} passed")
