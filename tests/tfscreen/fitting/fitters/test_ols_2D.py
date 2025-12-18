
import numpy as np
import pytest
from tfscreen.fitting.fitters.ols_2D import run_ols_2D

def test_ols_2D_single_dataset():
    # Test with 1D input (single dataset)
    x = np.array([0, 1, 2, 3])
    y = 2*x + 1
    
    slopes, intercepts, se_slopes, se_intercepts, residuals = run_ols_2D(x, y)
    
    assert np.allclose(slopes, [2.0])
    assert np.allclose(intercepts, [1.0])
    assert slopes.shape == (1,)

def test_ols_2D_multiple_datasets():
    # Dataset 1: y = x
    # Dataset 2: y = -x + 5
    
    x = np.array([
        [0, 1, 2],
        [0, 1, 2]
    ])
    y = np.array([
        [0, 1, 2],
        [5, 4, 3]
    ])
    
    slopes, intercepts, se_slopes, se_intercepts, residuals = run_ols_2D(x, y)
    
    assert np.allclose(slopes, [1.0, -1.0])
    assert np.allclose(intercepts, [0.0, 5.0])

def test_ols_2D_insufficient_points():
    # Only 1 point
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    
    slopes, intercepts, se_slopes, se_intercepts, residuals = run_ols_2D(x, y)
    
    assert np.all(np.isnan(slopes))
    assert np.all(np.isnan(intercepts))

def test_ols_2D_constant_x():
    # Division by zero variance in X
    x = np.array([[1.0, 1.0, 1.0]])
    y = np.array([[1.0, 2.0, 3.0]])
    
    slopes, intercepts, _, _, _ = run_ols_2D(x, y)
    
    assert np.isnan(slopes[0])

def test_ols_2D_exact_fit_se():
    # For perfect fit, SE should be 0 (or very close)
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 4])
    
    _, _, se_slopes, se_intercepts, _ = run_ols_2D(x, y)
    
    assert np.allclose(se_slopes, 0.0)
    assert np.allclose(se_intercepts, 0.0)

def test_ols_2D_less_than_two_df():
    # 2 points = 0 dof (n-2)
    # The code handles df > 0 specifically
    x = np.array([0, 1])
    y = np.array([0, 1])
    
    _, _, se_slopes, se_intercepts, _ = run_ols_2D(x, y)
    # Should get NaNs for SE because df=0
    assert np.all(np.isnan(se_slopes))
    assert np.all(np.isnan(se_intercepts))
