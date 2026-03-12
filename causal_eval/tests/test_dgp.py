"""
Tests for the Data Generating Process — ground truth computation.

These tests MUST pass before any estimator is run.
They validate the four conditions from system_design.md §5:
  1. Range: τ_i varies by ≥ 0.5 mg/dL/U across patients
  2. Stability: finite difference is in the linear regime
  3. Sign: insulin lowers glucose (τ < 0)
  4. State-dependence: τ differs at different glucose levels
"""

import sys
import os
import unittest
import numpy as np

# Ensure causal_eval package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from causal_eval.dgp.hovorka_wrapper import HovorkaWrapper, create_cohort, micro_randomize
from causal_eval.dgp.ground_truth import (
    GroundTruthComputer, validate_ground_truth,
    DELTA_PRIMARY, DELTA_VALIDATION, HORIZON_STEPS,
    MIN_TAU_RANGE, MAX_STABILITY_RELDIFF,
)


class TestMicroRandomization(unittest.TestCase):
    """Verify the micro-randomization mechanism."""

    def test_propensity_in_bounds(self):
        """Propensity must always be in [0.3, 0.7]."""
        rng = np.random.RandomState(42)
        for glucose in [40, 70, 100, 150, 200, 300, 400]:
            _, p = micro_randomize(glucose, rng=rng)
            self.assertGreaterEqual(p, 0.3, f"Propensity {p} < 0.3 at glucose={glucose}")
            self.assertLessEqual(p, 0.7, f"Propensity {p} > 0.7 at glucose={glucose}")

    def test_propensity_increases_with_glucose(self):
        """Higher glucose → higher treatment probability."""
        _, p_low = micro_randomize(80.0)
        _, p_high = micro_randomize(250.0)
        self.assertGreater(p_high, p_low)

    def test_randomization_produces_both_arms(self):
        """Over many draws, both A=0 and A=1 should appear."""
        rng = np.random.RandomState(42)
        actions = [micro_randomize(150.0, rng=rng)[0] for _ in range(100)]
        self.assertIn(0, actions)
        self.assertIn(1, actions)


class TestHovorkaWrapper(unittest.TestCase):
    """Verify wrapper state save/restore and basic simulation."""

    def test_save_restore_determinism(self):
        """Restoring state must produce identical forward trajectories."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        w.init_scenario(n_days=1)

        # Warm up
        for _ in range(10):
            scenario = w.get_scenario()
            w.sim.sim_step(0.05, scenario['carb_input'])

        # Save state
        snapshot = w.save_state()

        # Run forward 20 steps
        glucose_run1 = []
        for _ in range(20):
            scenario = w.get_scenario()
            g = w.sim.sim_step(0.1, scenario['carb_input'])
            glucose_run1.append(g)

        # Restore and run again
        w.restore_state(snapshot)
        glucose_run2 = []
        for _ in range(20):
            scenario = w.get_scenario()
            g = w.sim.sim_step(0.1, scenario['carb_input'])
            glucose_run2.append(g)

        np.testing.assert_array_almost_equal(
            glucose_run1, glucose_run2, decimal=10,
            err_msg="State restore did not produce identical trajectories"
        )

    def test_insulin_lowers_glucose(self):
        """More insulin should result in lower glucose (basic sanity)."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        w.init_scenario(n_days=1)

        # Warm up
        for _ in range(24):
            scenario = w.get_scenario()
            w.sim.sim_step(0.05, scenario['carb_input'])

        snapshot = w.save_state()

        # Run with no extra insulin
        for _ in range(HORIZON_STEPS):
            w.step(0.0)
        g_no_insulin = w.get_glucose()

        # Run with extra insulin
        w.restore_state(snapshot)
        w.step(1.0)  # 1 Unit bolus at first step
        for _ in range(HORIZON_STEPS - 1):
            w.step(0.0)
        g_with_insulin = w.get_glucose()

        self.assertLess(g_with_insulin, g_no_insulin,
                        "Insulin did not lower glucose")


class TestGroundTruthComputation(unittest.TestCase):
    """Verify finite-difference ground truth τ_i computation."""

    def test_tau_is_negative(self):
        """τ should be negative (insulin lowers glucose)."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        w.init_scenario(n_days=1)

        # Warm up to a representative state
        for _ in range(24):
            scenario = w.get_scenario()
            w.sim.sim_step(0.05, scenario['carb_input'])

        gt = GroundTruthComputer(delta=DELTA_PRIMARY)
        tau = gt.compute_tau_at_time(w)
        self.assertLess(tau, 0, f"τ = {tau} is not negative")

    def test_tau_finite_difference_stable(self):
        """τ(δ=0.01) and τ(δ=0.001) should agree within 1%."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        w.init_scenario(n_days=1)

        for _ in range(24):
            scenario = w.get_scenario()
            w.sim.sim_step(0.05, scenario['carb_input'])

        snapshot = w.save_state()

        gt_coarse = GroundTruthComputer(delta=DELTA_PRIMARY)
        tau_coarse = gt_coarse.compute_tau_at_time(w)

        w.restore_state(snapshot)
        gt_fine = GroundTruthComputer(delta=DELTA_VALIDATION)
        tau_fine = gt_fine.compute_tau_at_time(w)

        if abs(tau_fine) > 1e-10:
            reldiff = abs(tau_coarse - tau_fine) / abs(tau_fine)
            self.assertLess(reldiff, MAX_STABILITY_RELDIFF,
                            f"Relative diff {reldiff:.6f} > {MAX_STABILITY_RELDIFF}")

    def test_tau_state_dependent(self):
        """τ at glucose=200 should differ from τ at glucose=100."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        gt = GroundTruthComputer(delta=DELTA_PRIMARY)

        Vg = w.params['Vg']
        BW = w.BW

        # τ at glucose = 200
        w.sim.state[6] = 200.0 * Vg * BW / 18.0
        tau_200 = gt.compute_tau_at_time(w)

        # τ at glucose = 100
        w.sim._init_state()
        w.sim.state[6] = 100.0 * Vg * BW / 18.0
        tau_100 = gt.compute_tau_at_time(w)

        self.assertNotAlmostEqual(tau_200, tau_100, places=2,
                                  msg=f"τ(200)={tau_200:.4f} ≈ τ(100)={tau_100:.4f}")

    def test_tau_varies_across_patient_types(self):
        """τ should differ between child, adolescent, and adult."""
        gt = GroundTruthComputer(delta=DELTA_PRIMARY)
        taus = {}
        for ptype in ['child', 'adolescent', 'adult']:
            w = HovorkaWrapper(patient_id=0, seed=42, patient_type=ptype)
            w.init_scenario(n_days=1)
            for _ in range(24):
                scenario = w.get_scenario()
                w.sim.sim_step(0.05, scenario['carb_input'])
            taus[ptype] = gt.compute_tau_at_time(w)

        # At least two types should have different τ
        tau_list = list(taus.values())
        tau_range = max(tau_list) - min(tau_list)
        self.assertGreater(tau_range, 0.01,
                           f"τ values too similar across types: {taus}")


class TestFullValidation(unittest.TestCase):
    """Run the complete validation suite from system_design.md §5."""

    def test_full_validation(self):
        """All four validation checks must pass."""
        cohort = create_cohort(n_patients=10, base_seed=42)
        results = validate_ground_truth(cohort, verbose=True)

        # Report results regardless of pass/fail
        print("\n=== Ground Truth Validation Summary ===")
        for test_name, result in results.items():
            status = "PASS" if result['pass'] else "FAIL"
            print(f"  {test_name}: {status}")

        # The range test and sign test are the critical ones.
        # Stability and state-dependence are validation checks.
        self.assertTrue(results['sign_test']['pass'],
                        "Sign test failed: some τ values are not negative")
        self.assertTrue(results['state_dependence_test']['pass'],
                        "State-dependence test failed: τ does not vary with glucose level")
        self.assertTrue(results['stability_test']['pass'],
                        "Stability test failed: finite difference is not in linear regime")
        # Range test may fail if patient parameters are too similar —
        # this is a finding, not a bug, but we still assert it
        if not results['range_test']['pass']:
            print(f"\n⚠ RANGE TEST FAILED: τ range = {results['range_test']['tau_range']:.4f} "
                  f"< {MIN_TAU_RANGE}. Patient population may be too homogeneous.")
        # We still assert because the experiment cannot proceed without this
        self.assertTrue(results['range_test']['pass'],
                        f"Range test failed: τ range = {results['range_test']['tau_range']:.4f} "
                        f"< {MIN_TAU_RANGE}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
