
import pytest
import numpy as np
from unittest.mock import patch
from tfscreen.fitting.fitters.least_squares import run_least_squares

def test_run_least_squares_svd_error():
    """
    Test that run_least_squares handles SVD convergence error correctly.
    """
    
    def model(params, x):
        return params[0] * x
    
    obs = np.array([1.0, 2.0, 3.0])
    obs_std = np.array([0.1, 0.1, 0.1])
    guesses = np.array([1.0])
    args = (np.array([1.0, 2.0, 3.0]),)
    
    # Mock least_squares to raise LinAlgError("SVD did not converge")
    with patch("tfscreen.fitting.fitters.least_squares.least_squares") as mock_lsq:
        mock_lsq.side_effect = np.linalg.LinAlgError("SVD did not converge")
        
        with pytest.warns(UserWarning, match="Fit failed because SVD did not converge"):
            params, std_errors, cov_matrix, fit = run_least_squares(
                model, obs, obs_std, guesses, args=args
            )
            
    # Check that it returns nan 
    assert np.all(np.isnan(params))
    assert np.all(np.isnan(std_errors))
    assert np.all(np.isnan(cov_matrix))
    
    # Check shapes
    assert params.shape == guesses.shape
    assert std_errors.shape == guesses.shape
    assert cov_matrix.shape == (len(guesses), len(guesses))
    
    # Check that fit is the exception
    assert isinstance(fit, np.linalg.LinAlgError)
    assert "SVD did not converge" in str(fit)

def test_run_least_squares_other_linalg_error():
    """
    Test that run_least_squares re-raises other LinAlgErrors.
    """
    
    def model(params, x):
        return params[0] * x
    
    obs = np.array([1.0, 2.0, 3.0])
    obs_std = np.array([0.1, 0.1, 0.1])
    guesses = np.array([1.0])
    args = (np.array([1.0, 2.0, 3.0]),)
    
    # Mock least_squares to raise a different LinAlgError
    with patch("tfscreen.fitting.fitters.least_squares.least_squares") as mock_lsq:
        mock_lsq.side_effect = np.linalg.LinAlgError("Some other error")
        
        with pytest.raises(np.linalg.LinAlgError, match="Some other error"):
            run_least_squares(model, obs, obs_std, guesses, args=args)
