import numpy as np
from scipy.optimize import least_squares

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
    residuals = (obs - calc) / obs_std

    # Replace any NaNs/Infs from missing data or math errors with 0.0,
    # so they do not contribute to the cost function.
    return np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

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

    # Process results
    params = fit.x
    J = fit.jac

    # Build covariance matrix and estimate standard errors
    num_params = len(params)
    num_obs = np.sum(~np.isnan(obs))  # Count only valid observations
    dof = num_obs - num_params
    if dof < 1:
        dof = 1  # Avoid division by zero for poorly constrained fits

    chi2_red = np.sum(fit.fun**2) / dof

    try:
        JTJ = J.T @ J
        cov_matrix = chi2_red * np.linalg.inv(JTJ)
        with np.errstate(invalid='ignore'): 
            std_errors = np.sqrt(np.diagonal(cov_matrix))
    except (np.linalg.LinAlgError, ValueError):
        cov_matrix = np.full((num_params, num_params), np.nan)
        std_errors = np.full(num_params, np.nan)

    return params, std_errors, cov_matrix
