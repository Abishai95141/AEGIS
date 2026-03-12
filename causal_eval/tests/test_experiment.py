"""
Tests for the full experiment orchestration.

Runs a small-scale version of the experiment (10 patients, 288 obs each)
to verify end-to-end correctness before the full 50-patient run.

Semi-synthetic DGP: outcomes come from known structural equation
Y = μ(S) + τ_i·A + γ·U + ε, where τ_i is from Hovorka finite differencing.
"""

import sys
import os
import unittest
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from causal_eval.evaluation.experiment import CausalExperiment, generate_patient_data
from causal_eval.evaluation.metrics import compute_metrics, format_results_table
from causal_eval.dgp.hovorka_wrapper import HovorkaWrapper, create_cohort
from causal_eval.dgp.proxy_generator import create_proxy_generator, verify_proxy_independence


# A reference τ for testing data generation (typical adult value)
TEST_TAU = -5.0


class TestDataGeneration(unittest.TestCase):
    """Test the semi-synthetic data generation pipeline."""

    def test_generate_patient_data_shapes(self):
        """Generated data arrays must have correct shapes."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, latent = generate_patient_data(w, tau_i=TEST_TAU, n_steps=100, seed=42)

        self.assertEqual(len(data['Y']), 100)
        self.assertEqual(len(data['A']), 100)
        self.assertEqual(len(data['glucose']), 100)
        self.assertEqual(len(data['carbs']), 100)
        self.assertEqual(len(data['hour']), 100)
        self.assertEqual(len(data['propensity']), 100)
        self.assertEqual(len(latent['stress']), 100)
        self.assertEqual(len(latent['fatigue']), 100)
        self.assertEqual(len(latent['U']), 100)

    def test_treatment_dose_nonnegative(self):
        """Insulin dose A must be non-negative."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, _ = generate_patient_data(w, tau_i=TEST_TAU, n_steps=288, seed=42)
        self.assertTrue(np.all(data['A'] >= 0),
                        f"Negative insulin doses found: min={np.min(data['A']):.4f}")

    def test_propensity_in_bounds(self):
        """All propensities must be in [0.3, 0.7]."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, _ = generate_patient_data(w, tau_i=TEST_TAU, n_steps=288, seed=42)
        self.assertTrue(np.all(data['propensity'] >= 0.29))
        self.assertTrue(np.all(data['propensity'] <= 0.71))

    def test_glucose_physiological(self):
        """Glucose covariate values must be in physiological range [20, 600]."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, _ = generate_patient_data(w, tau_i=TEST_TAU, n_steps=288, seed=42)
        self.assertTrue(np.all(data['glucose'] >= 20))
        self.assertTrue(np.all(data['glucose'] <= 600))


class TestProxyIndependence(unittest.TestCase):
    """Test that proxy independence conditions hold empirically."""

    def test_strong_proxy_correlates_with_confounder(self):
        """Strong proxy Z should correlate with stress (relevance)."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, latent = generate_patient_data(w, tau_i=TEST_TAU, n_steps=288, seed=42)
        pg = create_proxy_generator(condition='strong', patient_id=0)
        Z, W = pg.generate_trajectory(latent['stress'], latent['fatigue'])
        corr = np.corrcoef(Z, latent['stress'])[0, 1]
        self.assertGreater(abs(corr), 0.1,
                           f"Strong proxy Z has weak correlation {corr:.3f} with stress")

    def test_proxy_independence_diagnostic(self):
        """Proxy independence diagnostic should pass."""
        w = HovorkaWrapper(patient_id=0, seed=42, patient_type='adult')
        data, latent = generate_patient_data(w, tau_i=TEST_TAU, n_steps=288, seed=42)
        pg = create_proxy_generator(condition='strong', patient_id=0)
        Z, W = pg.generate_trajectory(latent['stress'], latent['fatigue'])
        result = verify_proxy_independence(
            Z, W, data['glucose'], data['treatment']
        )
        self.assertTrue(result['all_pass'],
                        f"Proxy independence failed: {result}")


class TestMetrics(unittest.TestCase):
    """Test metric computation."""

    def test_perfect_estimation(self):
        """Perfect estimates should give zero MAE and bias."""
        tau_true = {0: -5.0, 1: -10.0, 2: -3.0}
        tau_est = {0: -5.0, 1: -10.0, 2: -3.0}
        m = compute_metrics(tau_true, tau_est)
        self.assertAlmostEqual(m['mae'], 0.0, places=10)
        self.assertAlmostEqual(m['bias'], 0.0, places=10)

    def test_known_error(self):
        """Known errors should produce correct MAE."""
        tau_true = {0: -5.0, 1: -10.0}
        tau_est = {0: -4.0, 1: -8.0}  # errors: 1.0, 2.0
        m = compute_metrics(tau_true, tau_est)
        self.assertAlmostEqual(m['mae'], 1.5, places=10)
        self.assertAlmostEqual(m['bias'], 1.5, places=10)

    def test_coverage_computation(self):
        """Coverage should be correct with known bounds."""
        tau_true = {0: -5.0, 1: -10.0}
        tau_est = {0: -5.0, 1: -10.0}
        bounds = {0: (-6.0, -4.0), 1: (-12.0, -8.0)}
        m = compute_metrics(tau_true, tau_est, confidence_bounds=bounds)
        self.assertAlmostEqual(m['coverage'], 1.0)

        bounds_miss = {0: (-6.0, -4.0), 1: (-9.0, -8.0)}
        m2 = compute_metrics(tau_true, tau_est, confidence_bounds=bounds_miss)
        self.assertAlmostEqual(m2['coverage'], 0.5)


class TestSmallExperiment(unittest.TestCase):
    """Run a small-scale experiment to verify end-to-end correctness."""

    def test_small_experiment_runs(self):
        """10-patient experiment should complete without error."""
        exp = CausalExperiment(n_patients=10, n_obs=288, base_seed=42)
        results, table_str = exp.run()

        # Should have 8 result rows: 4 estimators × 2 proxy conditions
        self.assertEqual(len(results), 8,
                         f"Expected 8 results, got {len(results)}")

        # All results should have required keys
        for r in results:
            for key in ['estimator', 'proxy_condition', 'mae', 'p95_error', 'bias']:
                self.assertIn(key, r, f"Missing key '{key}' in result")

        # MAE should be finite for all estimators
        for r in results:
            self.assertTrue(np.isfinite(r['mae']),
                            f"MAE is not finite for {r['estimator']}")

        # Table string should be non-empty
        self.assertGreater(len(table_str), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
