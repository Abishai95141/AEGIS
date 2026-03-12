"""
Standard G-estimation for continuous treatment — per-patient OLS
without proxy adjustment.

In this DGP, treatment A is continuous and confounded by unmeasured U.
There is no known treatment assignment mechanism (no valid propensity
score): the micro-randomization function computes a binary probability
that does not determine the actual continuous dose. Therefore, propensity-
based identification is not available.

This estimator uses per-patient partialling-out OLS:
    1. Regress Y on S → residual Ỹ
    2. Regress A on S → residual Ã
    3. τ̂ = Σ(Ã·Ỹ) / Σ(Ã²)

It differs from Naive OLS only in that it is fitted per-patient
(allowing heterogeneous τ_i), whereas Population AIPW returns a
single pooled estimate.

Consistent when there are no unmeasured confounders, but biased when U
exists and affects both A and Y — which is the case in our DGP.

NOTE (from forensic audit): The previous version used propensity weights
derived from micro_randomize(), but those propensity scores describe a
binary mechanism that does not determine the actual continuous treatment.
The weights were near-uniform (range [4.08, 5.71] pre-normalization)
and had negligible effect. They have been removed for honesty — this
estimator is now explicitly per-patient OLS with no propensity adjustment.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from causal_eval.estimators._utils import partial_out, stable_residualize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum observations per patient for individual G-estimation.
MIN_OBS_FOR_ESTIMATION = 50


class StandardGEstimator:
    """
    Per-patient OLS estimator — no proxy adjustment, no propensity weighting.

    Fits the partialling-out regression per-patient, allowing heterogeneous
    τ_i recovery. Does NOT use propensity scores because the DGP has no
    known treatment assignment mechanism for continuous dose A.

    Consistent under sequential ignorability (no unmeasured confounders).
    Biased when unmeasured confounders are present.
    """

    def __init__(self):
        self.tau_hat = None
        self.se_hat = None

    def estimate(self, Y, A, S, propensity=None, **kwargs):
        """
        Estimate treatment effect via per-patient partialling-out OLS.

        Propensity argument is accepted for interface compatibility but
        is NOT used. There is no valid known propensity in this DGP.

        Args:
            Y: outcome array, shape (T,)
            A: continuous treatment array (insulin Units), shape (T,)
            S: observed covariate matrix, shape (T, d)
            propensity: IGNORED — no valid propensity in this DGP
            **kwargs: ignored (Z, W available but deliberately not used)

        Returns:
            dict with tau, se, method
        """
        T = len(Y)
        assert T >= MIN_OBS_FOR_ESTIMATION, (
            f"Need at least {MIN_OBS_FOR_ESTIMATION} observations, got {T}"
        )

        if S.ndim == 1:
            S = S.reshape(-1, 1)

        # --- Partial out S (SVD-stable) ---
        Y_resid, A_resid = partial_out(S, Y, A)

        # --- Unweighted OLS on residuals ---
        # No propensity weighting: there is no known treatment assignment
        # mechanism for the continuous dose A in this DGP.
        weights = np.ones(T)

        numerator = np.sum(weights * A_resid * Y_resid)
        denominator = np.sum(weights * A_resid ** 2)

        if abs(denominator) < 1e-10:
            self.tau_hat = 0.0
            self.se_hat = float('inf')
        else:
            self.tau_hat = float(numerator / denominator)

            # Heteroskedasticity-robust SE
            residuals = Y_resid - self.tau_hat * A_resid
            score_var = np.mean((A_resid * residuals) ** 2)
            denom_sq = (np.mean(A_resid ** 2)) ** 2
            self.se_hat = float(np.sqrt(score_var / max(denom_sq * T, 1e-10)))

        return {
            'tau': self.tau_hat,
            'se': self.se_hat,
            'method': 'standard_gestimation',
            'n_obs': T,
        }

    def estimate_individual(self, Y, A, S, patient_ids, propensity=None, **kwargs):
        """
        Estimate patient-specific treatment effects via per-patient OLS.

        Propensity is accepted for interface compatibility but IGNORED.

        Args:
            Y, A, S: as in estimate()
            patient_ids: integer array identifying which patient each obs belongs to
            propensity: IGNORED

        Returns:
            dict mapping patient_id → tau_i estimate
        """
        unique_ids = np.unique(patient_ids)
        tau_individual = {}

        for pid in unique_ids:
            mask = patient_ids == pid
            n_pid = np.sum(mask)

            if n_pid < MIN_OBS_FOR_ESTIMATION:
                # Fall back to population estimate
                result = self.estimate(Y, A, S, **kwargs)
                tau_individual[int(pid)] = result['tau']
            else:
                result = self.estimate(
                    Y[mask], A[mask], S[mask], **kwargs
                )
                tau_individual[int(pid)] = result['tau']

        return tau_individual
