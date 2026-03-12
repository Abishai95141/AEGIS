"""
AEGIS 3.0 Robust Validation — Layer 4 (Decision Engine) Tests

6 tests: 3 nominal + 3 robustness
Tests real ACB + CTS with conjugate Bayesian updates.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.layer4_decision import (
    ActionCenteredBandit, CounterfactualThompsonSampling, DecisionEngine,
)
from config import L4_NOMINAL, L4_ROBUST, N_MONTE_CARLO


def run_all_tests():
    results = []
    rng_global = np.random.RandomState(42)

    # ── R-L4-01: ACB Variance Reduction (Nominal) ──
    acb_ratios = []
    for mc in range(min(N_MONTE_CARLO, 50)):
        np.random.seed(mc)
        n_arms = 4
        true_effects = [0.0, 0.3, 0.6, 0.9]
        T = 1000

        # ACB with baseline
        acb = ActionCenteredBandit(n_arms)
        acb_rewards = {a: [] for a in range(n_arms)}

        # Q-learning (no centering) as control
        ql_counts = np.zeros(n_arms)
        ql_sums = np.zeros(n_arms)
        ql_rewards = {a: [] for a in range(n_arms)}

        for t in range(T):
            baseline = np.random.randn() * 5  # Large state-dependent noise

            arm_acb = acb.select_arm()
            reward_acb = true_effects[arm_acb] + baseline + np.random.randn() * 0.5
            acb.update(arm_acb, reward_acb, baseline=baseline)
            acb_rewards[arm_acb].append(reward_acb - baseline)

            arm_ql = int(np.argmax(ql_sums / np.maximum(ql_counts, 1)))
            if np.random.random() < 0.1 or min(ql_counts) == 0:
                arm_ql = np.random.randint(n_arms)
            reward_ql = true_effects[arm_ql] + baseline + np.random.randn() * 0.5
            ql_counts[arm_ql] += 1
            ql_sums[arm_ql] += reward_ql
            ql_rewards[arm_ql].append(reward_ql)

        # Compare variance of mean estimates
        acb_var = np.mean([np.var(v) for v in acb_rewards.values() if len(v) > 5])
        ql_var = np.mean([np.var(v) for v in ql_rewards.values() if len(v) > 5])
        acb_ratios.append(acb_var / max(ql_var, 1e-6))

    mean_ratio = np.mean(acb_ratios)

    results.append({
        'test_id': 'R-L4-01',
        'name': 'ACB Variance Reduction',
        'type': 'nominal',
        'metrics': {'mean_variance_ratio': float(mean_ratio)},
        'threshold': {'ratio_max': L4_NOMINAL['acb_variance_ratio_max']},
        'passed': mean_ratio <= L4_NOMINAL['acb_variance_ratio_max'],
    })

    # ── R-L4-02: Regret Scaling (Nominal) ──
    np.random.seed(42)
    n_arms = 4
    true_effects = [0.0, 0.3, 0.6, 0.9]
    best_arm = 3
    T = 5000

    cumulative_regret = []
    acb = ActionCenteredBandit(n_arms)
    total_regret = 0

    for t in range(T):
        arm = acb.select_arm()
        reward = true_effects[arm] + np.random.randn() * 0.5
        acb.update(arm, reward)
        regret = true_effects[best_arm] - true_effects[arm]
        total_regret += regret
        cumulative_regret.append(total_regret)

    # Log-log slope of regret
    log_t = np.log(np.arange(100, T + 1))
    log_r = np.log(np.maximum(cumulative_regret[99:], 1))
    slope = np.polyfit(log_t, log_r, 1)[0]

    results.append({
        'test_id': 'R-L4-02',
        'name': 'Regret Scaling',
        'type': 'nominal',
        'metrics': {'log_log_slope': float(slope), 'final_regret': float(total_regret)},
        'threshold': {
            'slope_min': L4_NOMINAL['regret_slope_min'],
            'slope_max': L4_NOMINAL['regret_slope_max'],
        },
        'passed': L4_NOMINAL['regret_slope_min'] <= slope <= L4_NOMINAL['regret_slope_max'],
    })

    # ── R-L4-03: Posterior Collapse Prevention (Nominal) ──
    np.random.seed(42)
    cts = CounterfactualThompsonSampling(4)

    # Arm 2 is always blocked (never observed directly)
    blocked_arm = 2
    initial_var = cts.post_vars[blocked_arm]

    # Observe other arms
    for t in range(200):
        for a in [0, 1, 3]:
            cts.update(a, np.random.randn() * 0.5 + [0.0, 0.3, 0.6, 0.9][a])

    # Apply counterfactual updates for blocked arm
    for t in range(100):
        cts.counterfactual_update(blocked_arm)

    final_var = cts.post_vars[blocked_arm]
    var_ratio = final_var / initial_var

    results.append({
        'test_id': 'R-L4-03',
        'name': 'Posterior Collapse Prevention (CTS)',
        'type': 'nominal',
        'metrics': {
            'initial_var': float(initial_var),
            'final_var': float(final_var),
            'var_ratio': float(var_ratio),
        },
        'threshold': {'ratio_max': L4_NOMINAL['cts_posterior_ratio_max']},
        'passed': var_ratio < L4_NOMINAL['cts_posterior_ratio_max'],
    })

    # ── R-L4-04: Biased Digital Twin (Robustness) ──
    np.random.seed(42)
    cts_biased = CounterfactualThompsonSampling(4)
    blocked_arm = 2
    true_mean = 0.6

    # Observe other arms normally
    for t in range(300):
        for a in [0, 1, 3]:
            cts_biased.update(a, np.random.randn() * 0.5 + [0, 0.3, 0.6, 0.9][a])

    # Bias the global mean (simulating a biased Digital Twin)
    cts_biased.global_mean = 0.0  # Biased low (true is 0.6)

    # Apply many counterfactual updates with biased twin
    for t in range(200):
        cts_biased.counterfactual_update(blocked_arm)

    # CTS should NOT diverge — mean should stay reasonable
    est_mean = cts_biased.post_means[blocked_arm]
    est_var = cts_biased.post_vars[blocked_arm]
    # With biased twin, mean might be pulled toward 0, but error must be bounded
    mean_error = abs(est_mean - true_mean)
    # Stricter check: error must be < 2x the true mean (not just < 10)
    error_bounded = mean_error < 2.0 * true_mean

    results.append({
        'test_id': 'R-L4-04',
        'name': 'Biased Digital Twin',
        'type': 'robust',
        'metrics': {
            'estimated_mean': float(est_mean),
            'true_mean': float(true_mean),
            'mean_error': float(mean_error),
            'post_var': float(est_var),
        },
        'threshold': {'error_bounded_by_2x_true_mean': True},
        'passed': error_bounded and est_var > 0,
    })

    # ── R-L4-05: Non-Stationary Rewards (Robustness) ──
    np.random.seed(42)
    # Use a higher epsilon floor so the bandit keeps exploring after shift
    acb_ns = ActionCenteredBandit(4, epsilon_decay=0.999, epsilon_min=0.15)
    T = 3000

    # Phase 1: arm 3 is best
    phase1_effects = [0.0, 0.3, 0.6, 0.9]
    # Phase 2: arm 1 is best (shift at T/2)
    phase2_effects = [0.0, 0.9, 0.3, 0.1]

    selections_phase2 = []
    for t in range(T):
        effects = phase1_effects if t < T // 2 else phase2_effects
        arm = acb_ns.select_arm()
        reward = effects[arm] + np.random.randn() * 0.5
        acb_ns.update(arm, reward)
        if t >= T // 2:
            selections_phase2.append(arm)

    # After shift, arm 1 should appear more than random (>25% = 1/4 arms)
    late_selections = selections_phase2[-500:]
    best_arm_rate = np.mean([s == 1 for s in late_selections])

    # ε-greedy will explore arm 1 at least ε/4 of the time
    # Plus exploitation once means update. Threshold: better than random (25%)
    results.append({
        'test_id': 'R-L4-05',
        'name': 'Non-Stationary Rewards',
        'type': 'robust',
        'metrics': {'best_arm_rate_late': float(best_arm_rate)},
        'threshold': {'min_rate': 0.25},
        'passed': best_arm_rate > 0.25,  # Must beat random (1/4 arms)
    })

    # ── R-L4-06: High Blocking Rate (Robustness) ──
    np.random.seed(42)
    de = DecisionEngine()
    T = 1000
    blocked_count = 0
    functional_decisions = 0

    for t in range(T):
        glucose = 60 + np.random.rand() * 200  # Mix of ranges
        arm, info = de.select_action(glucose)

        # Simulate 60% blocking rate
        if np.random.random() < 0.6:
            de.counterfactual_update(arm)
            blocked_count += 1
        else:
            reward = -abs(glucose - 120) / 100
            de.update(arm, reward)
            # Check if arm selection is non-random (functional)
            if info.get('epsilon', 1.0) < 0.5:
                functional_decisions += 1

    blocked_rate = blocked_count / T

    # System should still function — posteriors should have narrowed
    var_changed = np.any(de.cts.post_vars < de.cts.prior_var)

    results.append({
        'test_id': 'R-L4-06',
        'name': 'High Blocking Rate (>50%)',
        'type': 'robust',
        'metrics': {
            'blocking_rate': float(blocked_rate),
            'posteriors_narrowed': bool(var_changed),
        },
        'threshold': {'still_functional': True},
        'passed': var_changed,
    })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("AEGIS 3.0 Layer 4 — Robust Validation Tests")
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
    print(f"Layer 4 Total: {passed}/{len(results)} passed")
