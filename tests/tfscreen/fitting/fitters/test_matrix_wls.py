
import numpy as np
import pytest
from tfscreen.fitting.fitters.matrix_wls import run_matrix_wls

def test_run_matrix_wls_basic():
    # Simple linear regression: y = 2x + 1
    # X needs a column of ones for intercept if we want one
    
    x = np.array([0, 1, 2, 3])
    X = np.column_stack([x, np.ones_like(x)])
    y = 2*x + 1
    
    weights = np.ones_like(y)
    
    params, std_errors, cov_matrix, _ = run_matrix_wls(X, y, weights)
    
    assert np.allclose(params, [2.0, 1.0])

def test_run_matrix_wls_weighted():
    # Test that weights affect the fit
    # Point at x=0 is y=0, weight=1000 (Strong pull)
    # Point at x=1 is y=10, weight=1 (Weak pull)
    # Model y = c (intercept only)
    
    X = np.ones((2, 1))
    y = np.array([0.0, 10.0])
    weights = np.array([1000.0, 1.0])
    
    params, _, _, _ = run_matrix_wls(X, y, weights)
    
    # The result should be very close to 0
    assert np.isclose(params[0], 0.0, atol=0.1)

def test_matrix_wls_singular():
    # Test behavior with singular matrix (not enough data or redundant cols)
    # This might raise LinAlgError or return garbage depending on implementation
    # The implementation uses np.linalg.solve, which raises LinAlgError for singular matrix
    
    X = np.array([[1.0, 1.0], [1.0, 1.0]]) # Singular
    y = np.array([1.0, 2.0])
    weights = np.ones(2)
    
    with pytest.raises(np.linalg.LinAlgError):
        run_matrix_wls(X, y, weights)
