"""
Population-level AIPW-style estimator for continuous treatment.

With continuous treatment A (insulin dose in Units), the standard binary
AIPW formula does not apply. Instead, we use a partially linear model
(Robinson 1988) with doubly-robust properties:

    Y = τ·A + g(S) + ε

Estimated via partialling out:
    1. Regress Y on S → residual Ỹ = Y - μ̂_Y(S)
    2. Regress A on S → residual Ã = A - μ̂_A(S)
    3. Regress Ỹ on Ã → coefficient is τ̂

This is doubly-robust in that it is consistent if EITHER the outcome
model μ̂_Y or the treatment model μ̂_A is correctly specified.

However, it estimates a single τ_pop for ALL patients — it cannot
recover individual τ_i. This makes it the "ignores individuality"
comparator in the experiment.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from causal_eval.estimators._utils import partial_out

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum observations for estimation
MIN_OBS_FOR_ESTIMATION = 50


class PopulationAIPWEstimator:
    """
    Population partially-linear estimator — doubly robust but ignores individuality.

    Returns a single population-level τ. When patients are heterogeneous,
    this averages over individual effects and cannot recover τ_i.
    """

    def __init__(self):
        self.tau_hat = None
        self.se_hat = None

    def estimate(self, Y, A, S, propensity=None, **kwargs):
        """
        Estimate population average treatment effect via Robinson's
        partially linear model (partialling-out estimator).

        Args:
            Y: outcome array, shape (T,)
            A: continuous treatment array (insulin Units), shape (T,)
            S: observed covariate matrix, shape (T, d)
            propensity: not used for continuous treatment (kept for interface)
            **kwargs: ignored (Z, W not used — this estimator ignores proxies)

        Returns:
            dict with tau, se, method
        """
        T = len(Y)
        assert T >= MIN_OBS_FOR_ESTIMATION, (
            f"Need at least {MIN_OBS_FOR_ESTIMATION} observations, got {T}"
        )

        if S.ndim == 1:
            S = S.reshape(-1, 1)

        # --- Partial out S from Y and A (SVD-stable) ---
        Y_resid, A_resid = partial_out(S, Y, A)

        # --- Step 3: Regress Y_resid on A_resid ---
        # τ̂ = Σ(Ã·Ỹ) / Σ(Ã²)
        numerator = np.sum(A_resid * Y_resid)
        denominator = np.sum(A_resid ** 2)

        if abs(denominator) < 1e-10:
            self.tau_hat = 0.0
            self.se_hat = float('inf')
        else:
            self.tau_hat = float(numerator / denominator)

            # Heteroskedasticity-robust SE
            residuals = Y_resid - self.tau_hat * A_resid
            score_var = np.mean((A_resid * residuals) ** 2)
            denom_sq = (np.mean(A_resid ** 2)) ** 2
            self.se_hat = float(np.sqrt(score_var / (denom_sq * T)))

        return {
            'tau': self.tau_hat,
            'se': self.se_hat,
            'method': 'population_aipw',
            'n_obs': T,
        }

    def estimate_individual(self, Y, A, S, patient_ids, propensity=None, **kwargs):
        """
        This estimator cannot recover individual effects — it returns the same
        population estimate for all patients.

        This is the methodological limitation that the experiment
        is designed to expose: population methods give τ_pop for everyone,
        which is wrong when τ_i varies across patients.

        Returns:
            dict mapping patient_id → τ_pop (same value for all)
        """
        # Get population estimate
        result = self.estimate(Y, A, S, propensity=propensity, **kwargs)
        tau_pop = result['tau']

        unique_ids = np.unique(patient_ids)
        return {int(pid): tau_pop for pid in unique_ids}
