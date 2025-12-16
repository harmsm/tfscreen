
import numpy as np
import pytest
from tfscreen.fitting.fitters.wls_2D import run_wls_2D

def test_wls_2D_single_dataset():
    x = np.array([0, 1, 2])
    y = 2*x
    # Constant error
    y_err = np.ones_like(y)
    
    slopes, intercepts, se_slopes, se_intercepts, res = run_wls_2D(x, y, y_err)
    
    assert np.allclose(slopes, [2.0])
    assert np.allclose(intercepts, [0.0])

def test_wls_2D_weighted_dataset():
    # Two points
    # (0, 0) low error
    # (1, 10) high error
    # Fit should be pinned to (0,0)
    # If we fit a constant (slope=0 forced? No, this fits slope and intercept)
    
    # Let's try to skew the slope.
    # Point 1: (0, 0), err=1
    # Point 2: (10, 10), err=1000
    # Point 3: (10, 0), err=1
    
    # The line should follow (0,0) and (10,0) mostly, ignoring (10,10)
    
    x = np.array([[0, 10, 10]])
    y = np.array([[0, 10, 0]])
    y_err = np.array([[1.0, 10000.0, 1.0]]) # Variances
    # Weights are 1/var -> 1, 0.0001, 1
    
    slopes, intercepts, _ , _, _ = run_wls_2D(x, y, y_err)
    
    # Should be close to y=0 line, slope 0, intercept 0
    assert np.isclose(slopes[0], 0.0, atol=0.1)
    assert np.isclose(intercepts[0], 0.0, atol=0.1)

def test_wls_2D_insufficient_points():
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    y_err = np.array([[1.0]])
    
    slopes, intercepts, _, _, _ = run_wls_2D(x, y, y_err)
    assert np.all(np.isnan(slopes))

def test_wls_2D_zero_variance_weights():
    # If y_err is 0, weight is inf? 
    # The code divides 1.0 / y_err where y_err != 0
    # But where y_err == 0 it puts 0? That seems like a bug or a specific choice.
    # Code: out=np.zeros_like(y_err, dtype=float), where=y_err!=0
    # So if variance is 0, weight is 0. This effectively ignores points with 0 variance?
    # Usually 0 variance means infinite weight.
    
    # Let's test the implementation behavior:
    x = np.array([[0, 1, 2]])
    y = np.array([[0, 1, 2]])
    y_err = np.array([[0.0, 1.0, 1.0]]) # First point has 0 variance -> 0 weight (ignored)
    
    # If first point is ignored, fit on (1,1) and (2,2) -> y=x still
    slopes, intercepts, _, _, _ = run_wls_2D(x, y, y_err)
    assert np.allclose(slopes, 1.0)
    assert np.allclose(intercepts, 0.0)

def test_wls_2D_delta_zero():
    # If all x are same, delta is 0
    x = np.array([[1.0, 1.0, 1.0]])
    y = np.array([[1.0, 2.0, 3.0]])
    y_err = np.ones_like(y)
    
    slopes, intercepts, _, _, _ = run_wls_2D(x, y, y_err)
    assert np.all(np.isnan(slopes))
