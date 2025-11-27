import numpy as np
from scipy.optimize import least_squares

from ._util import get_cov

def _weighted_residuals(params, some_model, obs, obs_std, args):
    """
    Calculate standard-deviation-weighted residuals for a given model.

    This function serves as a generic cost function for `scipy.optimize.least_squares`.
    It computes the difference between observed data and a model's prediction,
    weights it by the standard deviation of the observations, and handles
    missing data.

    Parameters
    ----------
    params : np.ndarray
        A 1D array of parameters to be optimized, passed to `some_model`.
    some_model : callable
        The model function to be fitted. It must have the signature
        `some_model(params, *args)`.
    obs : np.ndarray
        An array of the observed data points.
    obs_std : np.ndarray
        An array of the standard deviations for each observed data point.
    args : tuple
        A tuple of additional fixed arguments (e.g., x-values) required by
        `some_model`.

    Returns
    -------
    np.ndarray
        A 1D array of flattened, weighted residuals. NaN or infinite values
        resulting from missing data are replaced with 0.0.
    """
    # Calculate the model prediction
    calc = some_model(params, *args)

    # Calculate weighted residuals
    residuals = ((obs - calc) / obs_std).flatten()
    
    
    # nan/inf return high value to guide optimizer away from this region of 
    # parameter space
    if np.any(~np.isfinite(residuals)):
        return np.full(len(residuals),1e12)
    
    return residuals


def run_least_squares(some_model,
                      obs,
                      obs_std,
                      guesses,
                      lower_bounds=None,
                      upper_bounds=None,
                      args=None):
    """
    Perform a nonlinear least-squares fit for a generic model.

    This function provides a wrapper around `scipy.optimize.least_squares`
    to simplify the fitting process and includes robust calculation of
    parameter standard errors and the covariance matrix.

    Parameters
    ----------
    some_model : callable
        The model function to fit, with a signature `some_model(params, *args)`.
    obs : np.ndarray
        The array of observed data.
    obs_std : np.ndarray
        The array of standard deviations of the observations.
    guesses : np.ndarray
        A 1D array of initial guesses for the model parameters.
    lower_bounds : np.ndarray, optional
        A 1D array of lower bounds for each parameter. Defaults to -inf.
    upper_bounds : np.ndarray, optional
        A 1D array of upper bounds for each parameter. Defaults to +inf.
    args : tuple, optional
        A tuple of additional arguments to pass to `some_model`.

    Returns
    -------
    params : np.ndarray
        The array of best-fit parameter values.
    std_errors : np.ndarray
        The array of standard errors for each fitted parameter.
    cov_matrix : np.ndarray
        The full covariance matrix of the fitted parameters.
    fit : scipy.optimize.OptimizeResult
        result from least_squares call
    """
    if args is None:
        args = ()

    residuals_args = (some_model, obs, obs_std, args)

    # Set default unbounded limits if none are provided
    if lower_bounds is None:
        lower_bounds = np.full_like(guesses, -np.inf)
    if upper_bounds is None:
        upper_bounds = np.full_like(guesses, np.inf)

    # Run the regression
    fit = least_squares(
        _weighted_residuals,
        x0=guesses,
        bounds=(lower_bounds, upper_bounds),
        args=residuals_args,
        method='trf'
    )

    # Extract results
    params = fit.x
    cov_matrix, std_errors = get_cov(y=obs,
                                     residuals=fit.fun,
                                     params=params,
                                     J=fit.jac)

    return params, std_errors, cov_matrix, fit
