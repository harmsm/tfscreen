import numpy as np
from scipy.optimize import least_squares

def nls_estimate(X,
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
    parameters : numpy.ndarray
        The estimated parameters.
    cov_matrix : numpy.ndarray
        The covariance matrix of the parameters.
    """

    def _weighted_residuals(params,X,y,sqrt_weights):
        residuals = y - X @ params
        return sqrt_weights * residuals
    
    sqrt_weights = np.sqrt(weights)

    if lower_bounds is None:
        lower_bounds = -np.inf*np.ones(len(guesses))
    if upper_bounds is None:
        upper_bounds = np.inf*np.ones(len(guesses))

    fit_result = least_squares(fun=_weighted_residuals,
                               x0=guesses,
                               bounds=[lower_bounds,upper_bounds],
                               args=(X, y, sqrt_weights),
                               method='trf')
    
    parameters = fit_result.x

    J = fit_result.jac
    num_obs, num_param = J.shape
    dof = num_obs - num_param
    chi2_red = np.sum(fit_result.fun**2) / dof
    
    try:
        cov_matrix = chi2_red * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        print("Could not invert Jacobian matrix to estimate parameter uncertainty",flush=True)
        cov_matrix = np.ones((num_param,num_param))*np.nan
    
    return parameters, cov_matrix