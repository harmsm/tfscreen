
import numpy as np
from scipy.sparse import issparse

def get_cov(y,residuals,params,J):
    
    # Build covariance matrix and estimate standard errors
    num_params = len(params)
    num_obs = np.sum(~np.isnan(y))  # Count only valid observations
    dof = num_obs - num_params
    if dof < 1:
        dof = 1  # Avoid division by zero for poorly constrained fits

    chi2_red = np.sum(residuals**2) / dof

    try:
        JTJ = J.T @ J
        if issparse(J):
            JTJ =JTJ.toarray()
        cov_matrix = chi2_red * np.linalg.inv(JTJ)
        with np.errstate(invalid='ignore'): 
            std_errors = np.sqrt(np.diagonal(cov_matrix))
    except (np.linalg.LinAlgError, ValueError):
        cov_matrix = np.full((num_params, num_params), np.nan)
        std_errors = np.full(num_params, np.nan)

    return cov_matrix, std_errors