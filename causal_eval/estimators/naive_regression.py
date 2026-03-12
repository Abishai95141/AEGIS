"""
Naive OLS regression estimator — baseline that ignores confounding.

Estimates the treatment effect by regressing outcome on treatment
and observed covariates, without adjusting for unmeasured confounders.

This estimator is EXPECTED to be biased when confounding is present.
It serves as the lower bound in the experiment — any method that
cannot beat naive OLS is not contributing useful causal information.

Model:
    Y_{t+1} = α + τ·A_t + β'·S_t + ε_t

where:
    Y_{t+1} = glucose at next timestep (mg/dL)
    A_t = continuous insulin dose at time t (Units)
    S_t = observed covariates (glucose_t, carbs_t, hour_t)
    τ = estimated treatment effect (mg/dL per Unit, biased by confounding)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from causal_eval.estimators._utils import partial_out, stable_residualize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum observations required for stable OLS estimation.
# With ~5 covariates, need at least 10x for reasonable standard errors.
MIN_OBS_FOR_ESTIMATION = 50


class NaiveOLSEstimator:
    """
    Naive OLS estimator that ignores unmeasured confounding.

    Consistent only if there are no unmeasured confounders affecting
    both treatment and outcome — which is FALSE in our DGP by design.
    """

    def __init__(self):
        self.model = None
        self.tau_hat = None
        self.se_hat = None

    def estimate(self, Y, A, S, **kwargs):
        """
        Estimate average treatment effect via OLS.

        Args:
            Y: outcome array (glucose at t+1), shape (T,)
            A: treatment array (continuous insulin dose, Units), shape (T,)
            S: observed covariate matrix, shape (T, d)
            **kwargs: ignored (accepts Z, W, propensity for interface compatibility)

        Returns:
            dict with:
                tau: estimated treatment effect (mg/dL per Unit)
                se: standard error of tau
                method: 'naive_ols'
        """
        T = len(Y)
        assert T >= MIN_OBS_FOR_ESTIMATION, (
            f"Need at least {MIN_OBS_FOR_ESTIMATION} observations, got {T}"
        )

        if S.ndim == 1:
            S = S.reshape(-1, 1)

        # Design matrix: [1, S, A]
        X = np.column_stack([np.ones(T), S, A])

        # OLS fit via SVD for numerical stability
        from causal_eval.estimators._utils import stable_ols_coef
        beta, Y_hat = stable_ols_coef(X, Y)
        self.tau_hat = beta[-1]  # coefficient on A

        # Standard error via residual-based variance
        residuals = Y - Y_hat
        sigma2 = np.sum(residuals ** 2) / max(T - X.shape[1], 1)
        try:
            U_svd, s_svd, Vt_svd = np.linalg.svd(X, full_matrices=False)
            tol = max(T, X.shape[1]) * np.finfo(float).eps * s_svd[0]
            s_inv2 = np.where(s_svd > tol, 1.0 / (s_svd ** 2), 0.0)
            XtX_inv_diag = np.sum((Vt_svd.T ** 2) * s_inv2[np.newaxis, :], axis=1)
            self.se_hat = np.sqrt(sigma2 * XtX_inv_diag[-1])
        except np.linalg.LinAlgError:
            self.se_hat = np.std(residuals) / np.sqrt(T)

        return {
            'tau': float(self.tau_hat),
            'se': float(self.se_hat),
            'method': 'naive_ols',
            'n_obs': T,
        }

    def estimate_individual(self, Y, A, S, patient_ids, **kwargs):
        """
        Estimate patient-specific treatment effects via per-patient OLS.

        Args:
            Y, A, S: as in estimate()
            patient_ids: integer array identifying which patient each obs belongs to

        Returns:
            dict mapping patient_id → tau_i estimate
        """
        unique_ids = np.unique(patient_ids)
        tau_individual = {}

        for pid in unique_ids:
            mask = patient_ids == pid
            if np.sum(mask) < MIN_OBS_FOR_ESTIMATION:
                # Not enough data for this patient — use population estimate
                result = self.estimate(Y, A, S, **kwargs)
                tau_individual[int(pid)] = result['tau']
            else:
                result = self.estimate(Y[mask], A[mask], S[mask], **kwargs)
                tau_individual[int(pid)] = result['tau']

        return tau_individual
