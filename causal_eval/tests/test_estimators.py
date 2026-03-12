"""
Tests for the four estimators.

Phase 3 validation: each estimator must recover τ≈-5.0 on a simple
linear DGP with no confounding and no individuality.

Treatment is CONTINUOUS (insulin dose in Units), not binary.
τ is in units of mg/dL per Unit of insulin.
"""

import sys
import os
import unittest
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from causal_eval.estimators.naive_regression import NaiveOLSEstimator
from causal_eval.estimators.population_aipw import PopulationAIPWEstimator
from causal_eval.estimators.standard_gestimation import StandardGEstimator
from causal_eval.estimators.proximal_gestimation import ProximalGEstimatorWrapper


# ---------------------------------------------------------------------------
# Simple linear DGP for validation (system_design.md §7)
# τ_true = -5.0 mg/dL per Unit, no confounding, no individuality
# Using continuous treatment (insulin dose) to match experiment units
# ---------------------------------------------------------------------------

SIMPLE_TAU = -5.0         # mg/dL per Unit of insulin (negative: insulin lowers glucose)
SIMPLE_T = 200            # observations
SIMPLE_TOL = 2.0          # maximum acceptable error for T=200, continuous treatment

def _generate_simple_data(seed=42):
    """
    Generate simple linear DGP:
        Y = 150 + 0.3*S1 - 0.2*S2 + τ*A + ε

    where A is continuous insulin dose (Units), τ = -5.0 mg/dL per Unit.
    """
    rng = np.random.RandomState(seed)
    T = SIMPLE_T
    S = rng.randn(T, 2)
    # Continuous treatment: insulin dose ~ Uniform(0, 2) Units
    A = rng.uniform(0.0, 2.0, T)
    Y = 150.0 + 0.3 * S[:, 0] - 0.2 * S[:, 1] + SIMPLE_TAU * A + 2.0 * rng.randn(T)
    # Dummy proxies (not needed without confounding)
    Z = rng.randn(T)
    W = rng.randn(T)
    # Propensity placeholder (not used for continuous treatment)
    e = np.full(T, 0.5)
    return Y, A, S, e, Z, W


class TestNaiveOLS(unittest.TestCase):
    def test_simple_recovery(self):
        """Naive OLS recovers τ≈-5.0 without confounding."""
        Y, A, S, e, Z, W = _generate_simple_data()
        est = NaiveOLSEstimator()
        r = est.estimate(Y, A, S)
        self.assertAlmostEqual(r['tau'], SIMPLE_TAU, delta=SIMPLE_TOL,
                               msg=f"Naive OLS: τ̂={r['tau']:.4f}, expected ≈{SIMPLE_TAU}")

    def test_returns_required_keys(self):
        Y, A, S, e, Z, W = _generate_simple_data()
        r = NaiveOLSEstimator().estimate(Y, A, S)
        for key in ['tau', 'se', 'method']:
            self.assertIn(key, r)

    def test_individual_returns_dict(self):
        Y, A, S, e, Z, W = _generate_simple_data()
        pids = np.array([0]*100 + [1]*100)
        r = NaiveOLSEstimator().estimate_individual(Y, A, S, pids)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 2)


class TestPopulationAIPW(unittest.TestCase):
    def test_simple_recovery(self):
        """Population partially-linear estimator recovers τ≈-5.0."""
        Y, A, S, e, Z, W = _generate_simple_data()
        est = PopulationAIPWEstimator()
        r = est.estimate(Y, A, S, propensity=e)
        self.assertAlmostEqual(r['tau'], SIMPLE_TAU, delta=SIMPLE_TOL,
                               msg=f"Pop AIPW: τ̂={r['tau']:.4f}, expected ≈{SIMPLE_TAU}")

    def test_individual_returns_same_for_all(self):
        """Population estimator returns same τ for all patients."""
        Y, A, S, e, Z, W = _generate_simple_data()
        pids = np.array([0]*100 + [1]*100)
        est = PopulationAIPWEstimator()
        ind = est.estimate_individual(Y, A, S, pids, propensity=e)
        values = list(ind.values())
        self.assertAlmostEqual(values[0], values[1], places=10,
                               msg="Population estimator should return same τ for all patients")


class TestStandardGEstimation(unittest.TestCase):
    def test_simple_recovery(self):
        """Standard G-estimation recovers τ≈-5.0 without confounding."""
        Y, A, S, e, Z, W = _generate_simple_data()
        est = StandardGEstimator()
        r = est.estimate(Y, A, S, propensity=e)
        self.assertAlmostEqual(r['tau'], SIMPLE_TAU, delta=SIMPLE_TOL,
                               msg=f"Std G-est: τ̂={r['tau']:.4f}, expected ≈{SIMPLE_TAU}")


class TestProximalGEstimation(unittest.TestCase):
    def test_simple_recovery(self):
        """Proximal G-estimation recovers τ≈-5.0 without confounding."""
        Y, A, S, e, Z, W = _generate_simple_data()
        est = ProximalGEstimatorWrapper()
        r = est.estimate(Y, A, S, propensity=e, Z=Z, W=W)
        self.assertAlmostEqual(r['tau'], SIMPLE_TAU, delta=SIMPLE_TOL,
                               msg=f"Prox G-est: τ̂={r['tau']:.4f}, expected ≈{SIMPLE_TAU}")

    def test_requires_proxies(self):
        """Proximal G-estimation must raise if Z or W is missing."""
        Y, A, S, e, Z, W = _generate_simple_data()
        est = ProximalGEstimatorWrapper()
        with self.assertRaises(AssertionError):
            est.estimate(Y, A, S, propensity=e, Z=None, W=W)
        with self.assertRaises(AssertionError):
            est.estimate(Y, A, S, propensity=e, Z=Z, W=None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
