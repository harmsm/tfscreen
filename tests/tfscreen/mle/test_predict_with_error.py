
import pytest
import numpy as np
from tfscreen.mle.predict_with_error import predict_with_error

def linear_model(params, x):
    # y = m*x + b
    m, b = params
    return m * x + b

def test_predict_with_error_basic():
    # Model: y = m*x + b
    # x = [1, 2, 3]
    # params = [2.0, 1.0] (m=2, b=1)
    # y_pred = [3, 5, 7]
    
    x = np.array([1, 2, 3])
    params = np.array([2.0, 1.0])
    
    # Covariance matrix:
    # Var(m) = 0.1, Var(b) = 0.2, Cov(m,b) = 0
    cov = np.array([[0.1, 0.0],
                    [0.0, 0.2]])
    
    val, se = predict_with_error(linear_model, params, cov, args=(x,))
    
    assert np.allclose(val, np.array([3.0, 5.0, 7.0]))
    
    # Variance propagation:
    # y = m*x + b
    # dy/dm = x
    # dy/db = 1
    # Var(y) = (dy/dm)^2 * Var(m) + (dy/db)^2 * Var(b) + 2*...Cov...
    # Var(y) = x^2 * 0.1 + 1^2 * 0.2
    
    expected_var = (x**2) * 0.1 + 0.2
    expected_se = np.sqrt(expected_var)
    
    assert np.allclose(se, expected_se)

def test_predict_with_error_nan_cov():
    x = np.array([1])
    params = np.array([2.0, 1.0])
    cov = np.array([[np.nan, 0], [0, 0]])
    
    val, se = predict_with_error(linear_model, params, cov, args=(x,))
    
    assert np.allclose(val, [3.0])
    assert np.all(np.isnan(se))

def test_predict_with_error_no_args():
    def constant_model(params):
        return np.array([params[0]])
        
    params = np.array([5.0])
    cov = np.array([[0.5]])
    
    val, se = predict_with_error(constant_model, params, cov)
    
    assert val[0] == 5.0
    # J = 1. Var = 1*0.5*1 = 0.5. SE = sqrt(0.5)
    assert np.allclose(se, np.sqrt(0.5))

def test_numerical_derivative_epsilon():
    # Test that epsilon is used
    # Use a non-linear model where epsilon matters slightly compared to analytical?
    # actually, with linear model, numerical deriv is exact.
    # y = x^2. dy/dx = 2x.
    pass


def test_full_cov_diagonal_matches_se():
    """full_cov returns the (M, M) prediction covariance; its diagonal is se^2."""
    x = np.array([1.0, 2.0, 3.0])
    params = np.array([2.0, 1.0])
    cov = np.array([[0.1, 0.0],
                    [0.0, 0.2]])

    val, se, pred_cov = predict_with_error(linear_model, params, cov,
                                           args=(x,), full_cov=True)

    assert pred_cov.shape == (3, 3)
    assert np.allclose(np.diag(pred_cov), se ** 2)

    # Off-diagonal: Cov(y_i, y_j) = x_i*x_j*Var(m) + Var(b).
    expected = np.outer(x, x) * 0.1 + 0.2
    assert np.allclose(pred_cov, expected)


def test_full_cov_nan_cov():
    """Invalid param covariance -> NaN-filled (M, M) prediction covariance."""
    x = np.array([1.0, 2.0])
    params = np.array([2.0, 1.0])
    cov = np.array([[np.nan, 0.0], [0.0, 0.2]])

    val, se, pred_cov = predict_with_error(linear_model, params, cov,
                                           args=(x,), full_cov=True)

    assert pred_cov.shape == (2, 2)
    assert np.all(np.isnan(pred_cov))
    assert np.all(np.isnan(se))


def test_full_cov_default_off():
    """Without full_cov, the return is still a 2-tuple (back-compat)."""
    x = np.array([1.0, 2.0])
    out = predict_with_error(linear_model, np.array([2.0, 1.0]),
                             np.eye(2), args=(x,))
    assert len(out) == 2
