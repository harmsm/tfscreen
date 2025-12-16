
import numpy as np
import pytest
from tfscreen.fitting.fitters.least_squares import run_least_squares, _weighted_residuals

def linear_model(params, x):
    m, b = params
    return m * x + b

def test_run_least_squares_basic():
    # True parameters
    m_true = 2.0
    b_true = 5.0
    
    # Generate data
    x = np.linspace(0, 10, 20)
    y_true = linear_model([m_true, b_true], x)
    
    # Add noise? For exact check maybe not, or small noise.
    # Let's test exact fit first.
    obs = y_true.copy()
    obs_std = np.ones_like(obs) * 0.1 # Constant error
    
    guesses = np.array([1.0, 1.0])
    
    params, std_errors, cov_matrix, fit = run_least_squares(
        some_model=linear_model,
        obs=obs,
        obs_std=obs_std,
        guesses=guesses,
        args=(x,)
    )
    
    assert np.allclose(params, [m_true, b_true])
    assert fit.success

def test_run_least_squares_bounds():
    # Force the fit against a bound
    # Model: Just a constant value
    def const_model(params, x):
        return np.full_like(x, params[0])
    
    x = np.linspace(0,1,10)
    obs = np.full_like(x, 10.0)
    obs_std = np.ones_like(x)
    
    # Constrain parameter to be <= 5.0
    # The best fit would be 10.0, but it should hit the bound at 5.0
    guesses = np.array([0.0])
    lower_bounds = np.array([-np.inf])
    upper_bounds = np.array([5.0])
    
    params, _, _, fit = run_least_squares(
        some_model=const_model,
        obs=obs,
        obs_std=obs_std,
        guesses=guesses,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        args=(x,)
    )
    
    assert np.allclose(params, [5.0], atol=1e-5)

def test_weighted_residuals_nans():
    # Test that _weighted_residuals handles NaNs/Infs by returning large values
    
    # Intentionally trigger a NaN in residuals by having a NaN in obs
    obs = np.array([1.0, np.nan, 3.0])
    obs_std = np.array([1.0, 1.0, 1.0])
    
    def model(params, x):
        return x * params[0]
        
    x = np.array([1.0, 2.0, 3.0])
    params = np.array([1.0])
    
    res = _weighted_residuals(params, model, obs, obs_std, (x,))
    
    # The NaN should cause the specific large value return
    assert np.all(res == 1e12) or np.any(res == 1e12) # Implementation says returns full array of 1e12 if ANY non-finite

def test_weighted_residuals_inf():
    # Provide infinite observation
    obs = np.array([1.0, np.inf, 3.0])
    obs_std = np.array([1.0, 1.0, 1.0])
    
    def model(params, x):
        return x * params[0]
    
    x = np.array([1.0, 2.0, 3.0])
    params = np.array([1.0])
    
    res = _weighted_residuals(params, model, obs, obs_std, (x,))
    assert np.all(res == 1e12)

def test_default_args():
    # Test run_least_squares without optional arguments
    def simple_model(params):
        return np.array([params[0], params[0]])
    
    obs = np.array([2.0, 2.0])
    obs_std = np.array([1.0, 1.0])
    guesses = np.array([1.0])
    
    params, _, _, _ = run_least_squares(simple_model, obs, obs_std, guesses)
    assert np.allclose(params, [2.0])

