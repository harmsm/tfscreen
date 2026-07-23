
import numpy as np


def get_cov(y, residuals, params, J):
    """
    Estimate the parameter covariance matrix and standard errors from a fit.

    The covariance is built from the singular value decomposition of the
    Jacobian ``J`` rather than by inverting ``JᵀJ`` directly. Forming ``JᵀJ``
    squares the condition number, so a merely ill-conditioned-but-identifiable
    Jacobian can tip into numerical singularity and yield a spurious all-NaN
    covariance. Working from the SVD of ``J`` avoids that -- this mirrors how
    ``scipy.optimize.curve_fit`` builds ``pcov``.

    For ``J = U S Vᵀ`` we have ``(JᵀJ)⁻¹ = V S⁻² Vᵀ`` and
    ``cov = chi2_red · V S⁻² Vᵀ``, where ``chi2_red`` is the reduced chi-square
    of the (already weighted) residuals.

    A genuinely rank-deficient Jacobian (a truly unidentified parameter or
    parameter combination -- e.g. the width of a bell whose amplitude has gone
    to zero) is detected by a singular value at or below
    ``eps · max(J.shape) · s_max``. The covariance is then not defined, and an
    all-NaN matrix / NaN standard errors are returned. This is the same contract
    as before, so callers that test the covariance for NaN (``cat_fit``'s
    selection guard, ``predict_with_error``) keep working unchanged; a converged
    fit with an unusable curvature stays out of any weighted comparison.

    Parameters
    ----------
    y : np.ndarray
        Observed data. Used only to count valid (non-NaN) observations for the
        degrees of freedom.
    residuals : np.ndarray
        The (weighted) residuals at the solution.
    params : np.ndarray
        Best-fit parameter values.
    J : np.ndarray
        The Jacobian of the residuals with respect to the parameters, shape
        ``(num_obs, num_params)``.

    Returns
    -------
    cov_matrix : np.ndarray
        The ``(num_params, num_params)`` covariance matrix, or an all-NaN matrix
        if ``J`` is rank-deficient / non-finite.
    std_errors : np.ndarray
        The per-parameter standard errors (sqrt of the covariance diagonal), or
        all-NaN if ``J`` is rank-deficient / non-finite.
    """
    num_params = len(params)
    num_obs = np.sum(~np.isnan(y))  # Count only valid observations
    dof = num_obs - num_params
    if dof < 1:
        dof = 1  # Avoid division by zero for poorly constrained fits

    chi2_red = np.sum(residuals ** 2) / dof

    nan_cov = np.full((num_params, num_params), np.nan)
    nan_std = np.full(num_params, np.nan)

    J = np.asarray(J, dtype=float)
    if not np.all(np.isfinite(J)):
        return nan_cov, nan_std

    try:
        # Economy SVD of the Jacobian: J = U @ diag(s) @ Vt.
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return nan_cov, nan_std

    # Rank-deficiency test (the same threshold scipy.optimize.curve_fit uses).
    # Fewer singular values than parameters (an underdetermined fit) or a tiny
    # singular value both mean a parameter direction is unidentified, so the
    # covariance is not defined.
    if s.size < num_params or s[0] == 0.0:
        return nan_cov, nan_std
    threshold = np.finfo(float).eps * max(J.shape) * s[0]
    if np.any(s <= threshold):
        return nan_cov, nan_std

    # cov = chi2_red * V S^-2 Vt, formed from the SVD without ever building JtJ.
    # Scaling the columns of V (= rows of Vt) by 1/s and taking the Gram product
    # gives V S^-2 Vt, which is symmetric positive semidefinite by construction.
    Vs = Vt.T / s
    cov_matrix = chi2_red * (Vs @ Vs.T)
    std_errors = np.sqrt(np.diagonal(cov_matrix))

    return cov_matrix, std_errors
