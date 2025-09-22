import numpy as np

def run_matrix_wls(X,
                   y,
                   weights):
    """
    Estimate parameters by weighted linear least squares given a design matrix.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix.
    y : numpy.ndarray
        The target vector.
    weights : numpy.ndarray
        The weights for each data point.

    Returns
    -------
    params : np.ndarray
        The array of best-fit parameter values.
    std_errors : np.ndarray
        The array of standard errors for each fitted parameter.
    cov_matrix : np.ndarray
        The full covariance matrix of the fitted parameters.
    fit : None
        for api consistency, return None where the least squares fit results are 
        sent by other functions
    """
    
    # Calculate X^T * W * X and X^T * W * y
    # (w[:, np.newaxis] * X) is an efficient way to apply weights row-wise
    XTWX = X.T @ (weights[:, np.newaxis] * X)
    XTWy = X.T @ (weights * y)

    # Solve for parameters: beta = (X^T * W * X)^-1 * (X^T * W * y)
    # np.linalg.solve is more stable and faster than computing the inverse
    parameters = np.linalg.solve(XTWX, XTWy)

    # Get total residuals
    residuals = y - X @ parameters
    
    # Degrees of freedom (num observations - num parameters)
    dof = X.shape[0] - len(parameters)
    
    # Calculate reduced chi-squared (variance of unit weight)
    chi2_red = np.sum(weights * residuals**2) / dof
    
    # Parameter covariance matrix is chi2_red * (X^T * W * X)^-1
    cov_matrix = chi2_red * np.linalg.inv(XTWX)

    with np.errstate(invalid='ignore'): 
        std_errors = np.sqrt(np.diagonal(cov_matrix))
    
    return parameters, std_errors, cov_matrix, None