
import numpy as np
import pytest
from tfscreen.mle.fitters._util import get_cov

def test_get_cov_basic():
    # Linear problem: y = X @ params
    # J = X
    # Covariance = sigma^2 * (X^T X)^-1
    # sigma^2 = sum(residuals^2) / dof
    
    X = np.eye(2)
    y = np.array([1.0, 2.0])
    params = np.array([1.0, 2.0]) # Perfect fit
    residuals = y - X @ params # [0, 0]
    
    # Perfect fit -> residuals=0 -> sigma^2=0 -> cov=0
    cov, std = get_cov(y, residuals, params, X)
    assert np.allclose(cov, 0.0)
    assert np.allclose(std, 0.0)

def test_get_cov_with_error():
    X = np.array([[1.0], [1.0]])
    y = np.array([1.0, 1.2]) # mean 1.1
    params = np.array([1.1])
    residuals = y - X @ params # [-0.1, 0.1]
    
    # DOF = 2 - 1 = 1
    # Chi2_red = (0.01 + 0.01) / 1 = 0.02
    # J = [[1], [1]] -> J.T @ J = [2] -> inv = [0.5]
    # Cov = 0.02 * 0.5 = 0.01
    # Std = sqrt(0.01) = 0.1
    
    cov, std = get_cov(y, residuals, params, X)
    
    assert np.allclose(cov, [[0.01]])
    assert np.allclose(std, [0.1])

def test_get_cov_singular():
    # If J is singular (e.g. column of zeros)
    X = np.array([[0.0], [0.0]])
    y = np.array([1.0, 1.0])
    params = np.array([0.0])
    residuals = y # [1, 1]
    
    # Should catch LinAlgError and return NaNs
    cov, std = get_cov(y, residuals, params, X)
    
    assert np.all(np.isnan(cov))
    assert np.all(np.isnan(std))

def test_get_cov_dof_limit():
    # num_obs <= num_params -> dof < 1 -> dof clamped to 1
    X = np.array([[1.0]])
    y = np.array([1.0])
    params = np.array([1.0])
    residuals = np.array([0.1])
    
    # dof = 1 - 1 = 0 -> clamped to 1
    # chi2_red = 0.01 / 1 = 0.01
    # J.T @ J = 1 -> inv = 1
    # Cov = 0.01
    
    cov, std = get_cov(y, residuals, params, X)
    assert np.allclose(cov, [[0.01]])

def test_get_cov_nans_in_y():
    # Nans in y should be excluded from dof count
    y = np.array([1.0, np.nan, 2.0])
    X = np.ones((3, 1))
    params = np.array([1.0])
    residuals = np.array([0.0, 0.0, 1.0])
    
    # num_obs = 2 (valid)
    # params = 1
    # dof = 2 - 1 = 1
    # chi2_red = (0 + 0 + 1) / 1 = 1.0
    # J.T @ J = 3 (Wait. J is passed in. Does get_cov filter J rows for NaNs?
    # No, it uses J as is. It assumes J corresponds to the residuals provided.
    # The residuals passed usually should correspond to valid data or be handled by caller.
    # The implementation:
    # num_obs = np.sum(~np.isnan(y))
    # JTJ = J.T @ J
    # So it uses FULL J even if y has NaNs?
    # Yes, it seems so.
    
    # J = ones(3,1) -> JTJ = 3
    # inv = 1/3
    # cov = 1.0 * 1/3 = 0.333
    
    cov, std = get_cov(y, residuals, params, X)
    assert np.allclose(cov, [[1.0/3.0]])


def _jacobian_from_svd(singular_values, m, seed=0):
    """Build a dense J = U diag(s) Vt with prescribed singular values."""
    rng = np.random.default_rng(seed)
    n = len(singular_values)
    # Random orthonormal columns for U (m x n) and a random orthogonal Vt (n x n).
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Vt, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return U @ np.diag(singular_values) @ Vt


def test_get_cov_matches_jtj_on_well_conditioned():
    """SVD covariance equals the classic chi2_red * inv(JtJ) when well posed."""
    J = _jacobian_from_svd([2.0, 0.5, 1.0], m=8, seed=1)
    y = np.zeros(8)
    params = np.zeros(3)
    residuals = np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05])

    cov, std = get_cov(y, residuals, params, J)

    dof = 8 - 3
    chi2_red = np.sum(residuals ** 2) / dof
    cov_ref = chi2_red * np.linalg.inv(J.T @ J)
    assert np.allclose(cov, cov_ref)
    assert np.allclose(std, np.sqrt(np.diagonal(cov_ref)))


def test_get_cov_ill_conditioned_but_identifiable_is_finite():
    """A full-rank but ill-conditioned J (kappa ~ 1e8) stays finite.

    Forming JtJ would square the condition number (~1e16, ~float64 epsilon) and
    make the direct inverse unreliable/NaN; the SVD path recovers a correct,
    finite covariance.
    """
    s = [1.0, 1e-8]
    J = _jacobian_from_svd(s, m=6, seed=2)
    y = np.zeros(6)
    params = np.zeros(2)
    residuals = np.full(6, 0.1)

    cov, std = get_cov(y, residuals, params, J)
    assert np.all(np.isfinite(cov))
    assert np.all(np.isfinite(std))

    # Matches the analytic V S^-2 Vt scaling (huge but finite variance along the
    # weakly-constrained direction).
    dof = 6 - 2
    chi2_red = np.sum(residuals ** 2) / dof
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    cov_ref = chi2_red * (Vt.T / sv) @ (Vt.T / sv).T
    assert np.allclose(cov, cov_ref)


def test_get_cov_genuinely_rank_deficient_is_nan():
    """A singular value below the rank threshold -> unidentified -> all-NaN."""
    J = _jacobian_from_svd([1.0, 1e-18], m=6, seed=3)
    y = np.zeros(6)
    params = np.zeros(2)
    residuals = np.full(6, 0.1)

    cov, std = get_cov(y, residuals, params, J)
    assert np.all(np.isnan(cov))
    assert np.all(np.isnan(std))


def test_get_cov_underdetermined_is_nan():
    """Fewer observations than parameters -> not identifiable -> all-NaN."""
    J = np.array([[1.0, 2.0]])   # 1 obs, 2 params
    y = np.array([1.0])
    params = np.array([0.0, 0.0])
    residuals = np.array([0.1])

    cov, std = get_cov(y, residuals, params, J)
    assert np.all(np.isnan(cov))
    assert np.all(np.isnan(std))


def test_get_cov_nonfinite_jacobian_is_nan():
    """A non-finite Jacobian -> all-NaN (no spurious numbers)."""
    J = np.array([[1.0], [np.nan]])
    y = np.array([1.0, 2.0])
    params = np.array([1.0])
    residuals = np.array([0.0, 0.0])

    cov, std = get_cov(y, residuals, params, J)
    assert np.all(np.isnan(cov))
    assert np.all(np.isnan(std))
