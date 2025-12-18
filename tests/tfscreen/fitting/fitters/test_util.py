
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from tfscreen.fitting.fitters._util import get_cov

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

def test_get_cov_sparse():
    # Test with sparse Jacobian
    X = csr_matrix([[1.0], [1.0]])
    y = np.array([1.0, 1.2])
    params = np.array([1.1])
    residuals = y - 1.1
    
    cov, std = get_cov(y, residuals, params, X)
    
    assert np.allclose(cov, [[0.01]])

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

