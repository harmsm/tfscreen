import numpy as np
from scipy.optimize import least_squares

from ._util import get_cov

def run_matrix_nls(X,
                   y,
                   weights,
                   guesses,
                   lower_bounds=None,
                   upper_bounds=None):
    """
    Estimate parameters by weighted non-linear least squares given a design 
    matrix.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix.
    y : numpy.ndarray
        The target vector.
    weights : numpy.ndarray
        The weights for each data point.
    guesses : numpy.ndarray
        Initial guesses for the parameters.
    lower_bounds : numpy.ndarray, optional
        Lower bounds for the parameters. Default is None.
    upper_bounds : numpy.ndarray, optional
        Upper bounds for the parameters. Default is None.

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

    def _weighted_residuals(params,X,y,sqrt_weights):
        residuals = y - X @ params
        return sqrt_weights * residuals
    
    sqrt_weights = np.sqrt(weights)

    if lower_bounds is None:
        lower_bounds = np.repeat(-np.inf*(len(guesses)))
    if upper_bounds is None:
        upper_bounds = np.repeat( np.inf*(len(guesses)))

    fit_result = least_squares(fun=_weighted_residuals,
                               x0=guesses,
                               bounds=[lower_bounds,upper_bounds],
                               args=(X, y, sqrt_weights),
                               method='trf')
    
    # Extract results
    params = fit_result.x
    cov_matrix, std_errors = get_cov(y=y,
                                     residuals=fit_result.fun,
                                     params=params,
                                     J=fit_result.J)
    
    return params, std_errors, cov_matrix, fit_result