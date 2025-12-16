
import numpy as np
import pytest
from tfscreen.fitting.fitters.matrix_nls import run_matrix_nls

def test_matrix_nls_linear():
    # Matrix NLS is essentially solving y = X @ params via least_squares
    # Let's test a simple linear case
    
    # y = 2*x1 + 3*x2
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    true_params = np.array([2.0, 3.0])
    y = X @ true_params
    
    weights = np.ones_like(y)
    guesses = np.array([0.0, 0.0])
    
    params, std_errors, cov_matrix, fit = run_matrix_nls(
        X=X,
        y=y,
        weights=weights,
        guesses=guesses
    )
    
    assert np.allclose(params, true_params)
    assert fit.success

def test_matrix_nls_bounds():
    # Test with bounds
    X = np.eye(2)
    # Target is [10, 10]
    y = np.array([10.0, 10.0])
    weights = np.array([1.0, 1.0])
    guesses = np.array([0.0, 0.0])
    
    # Upper bound of 5
    upper_bounds = np.array([5.0, 5.0])
    
    params, _, _, _ = run_matrix_nls(
        X=X,
        y=y,
        weights=weights,
        guesses=guesses,
        upper_bounds=upper_bounds
    )
    
    assert np.allclose(params, [5.0, 5.0])

def test_matrix_nls_default_bounds():
    # Verify default bounds (-inf, inf) behave as expected (unbounded)
    X = np.array([[1.0]])
    y = np.array([-100.0])
    weights = np.array([1.0])
    guesses = np.array([0.0])
    
    params, _, _, _ = run_matrix_nls(X, y, weights, guesses)
    assert np.allclose(params, [-100.0])
