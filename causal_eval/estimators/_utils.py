"""
Shared numerical utilities for causal estimators.

Provides numerically stable OLS partialling-out via truncated SVD,
avoiding overflow/divide-by-zero when covariate matrices are
ill-conditioned (e.g. glucose ≈ 120 creates near-collinearity
with the intercept column).
"""

import numpy as np


def stable_residualize(X, y):
    """
    Compute OLS residuals y - X @ β̂ using SVD for numerical stability.

    Handles ill-conditioned X'X by using truncated SVD instead of
    normal equations. Computes ŷ = U diag(d) U'y directly, avoiding
    the overflow-prone X @ β̂ path.

    Args:
        X: design matrix, shape (n, p). Should include intercept column.
        y: target vector, shape (n,)

    Returns:
        residuals: y - ŷ, shape (n,)
    """
    n, p = X.shape

    # SVD-based OLS: ŷ = U U' y (projection), with truncation for rank deficiency
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Truncate singular values below machine precision × max(s) × dim
        tol = max(n, p) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-12
        mask = s > tol
        # Compute ŷ directly: ŷ = U_trunc @ U_trunc' @ y
        # This avoids forming β̂ and the overflow-prone X @ β̂
        U_trunc = U[:, mask]
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            y_hat = U_trunc @ (U_trunc.T @ y)
        # Safety: replace any non-finite values with the mean
        if not np.all(np.isfinite(y_hat)):
            y_hat = np.where(np.isfinite(y_hat), y_hat, np.mean(y))
    except np.linalg.LinAlgError:
        # Fallback: use mean as prediction
        y_hat = np.full(n, np.mean(y))

    return y - y_hat


def stable_ols_coef(X, y):
    """
    Compute OLS coefficient vector β̂ using SVD for numerical stability.

    Args:
        X: design matrix, shape (n, p)
        y: target vector, shape (n,)

    Returns:
        beta: coefficient vector, shape (p,)
        y_hat: fitted values X @ beta, shape (n,) — computed stably via SVD
    """
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        n, p = X.shape
        tol = max(n, p) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-12
        mask = s > tol
        s_inv = np.where(mask, 1.0 / s, 0.0)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Uty = U.T @ y
            beta = Vt.T @ (s_inv * Uty)
            # Compute y_hat stably: ŷ = U diag(mask) U' y
            d = np.where(mask, 1.0, 0.0)
            y_hat = U @ (d * Uty)
        # Safety: replace any non-finite values
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(y_hat)):
            y_hat = np.where(np.isfinite(y_hat), y_hat, np.mean(y))
        return beta, y_hat
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1]), np.full(len(y), np.mean(y))


def partial_out(S, *arrays):
    """
    Partial out covariates S from one or more arrays.

    Builds design matrix [1, S] and returns residuals for each input array.

    Args:
        S: covariate matrix, shape (n, d)
        *arrays: one or more arrays of shape (n,) to residualize

    Returns:
        tuple of residualized arrays (same length as *arrays)
    """
    n = S.shape[0]
    X_s = np.column_stack([np.ones(n), S])
    results = tuple(stable_residualize(X_s, arr) for arr in arrays)
    return results if len(results) > 1 else results[0]
