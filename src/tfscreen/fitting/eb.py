import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm

def map_cost_function(params, obs, obs_std, model_args, priors):
    """
    Calculates the MAP cost, including a hybrid prior penalty.

    priors: A list of 10 fitted scipy.stats distribution objects.
    """
    # 1. Likelihood part (Sum of Squares)
    # Assumes you have a function to calculate this
    sum_of_squares = calculate_sum_of_squares(params, obs, obs_std, model_args)

    # 2. Hybrid Prior Penalty part
    # We sum the negative log-probabilities from each prior distribution
    total_log_prior_penalty = 0
    for i in range(len(params)):
        
        # Add a small epsilon to prevent log(0) for out-of-bounds values
        param_value = params[i]
        prior_dist = priors[i]
        pdf_val = prior_dist.pdf(param_value)
        
        # Guard against pdf_val being zero
        if pdf_val > 1e-9:
            total_log_prior_penalty -= np.log(pdf_val)
        else:
            # Assign a large penalty for very improbable values
            total_log_prior_penalty += 1e9 

    return sum_of_squares + total_log_prior_penalty

# --- Inside your main loop ---
# priors = [skew_param_dist, norm_param1_dist, norm_param2_dist, ...]
# result = minimize(map_cost_function, x0=guesses, args=(..., priors))


# Get hessian as fake cov matrix
from scipy.optimize import minimize

# result = minimize(map_cost_function, ...)

# The optimizer estimates the inverse of the Hessian
# This is a good approximation of the covariance matrix
cov_matrix_approx = result.hess_inv

# If you are using an algorithm that doesn't provide it, you
# can compute it numerically, but it's slower.

with np.errstate(invalid='ignore'):
    std_errors = np.sqrt(np.diagonal(cov_matrix_approx))