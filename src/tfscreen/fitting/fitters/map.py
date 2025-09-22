import numpy as np
import scipy.optimize
import scipy.stats as st

# =============================================================================
# Prior Negative Log-PDFs (Private Helper Functions)
# These functions calculate the negative log of the prior probability density.
# They are the building blocks for the penalty term in the objective function.
# =============================================================================

def _nll_normal(p, mu, sigma):
    """Negative log-likelihood of a normal prior."""
    if sigma <= 0:
        return np.inf  # Sigma must be positive
    return -st.norm.logpdf(p, loc=mu, scale=sigma)

def _nll_lognormal(p, log_mu, log_sigma):
    """Negative log-likelihood of a lognormal prior."""
    if p <= 0 or log_sigma <= 0:
        return np.inf  # Lognormal variable must be positive
    # Note: scipy's lognormal takes s=sigma and scale=exp(mu)
    return -st.lognorm.logpdf(p, s=log_sigma, scale=np.exp(log_mu))

def _nll_laplace(p, mu, b):
    """Negative log-likelihood of a Laplace (double exponential) prior."""
    if b <= 0:
        return np.inf # Scale parameter b must be positive
    return -st.laplace.logpdf(p, loc=mu, scale=b)

def _nll_uniform(p, lower, upper):
    """Negative log-likelihood of a uniform prior (hard bounds)."""
    return -st.uniform.logpdf(p, loc=lower, scale=upper-lower)
    
def _nll_none(p, *args):
    """A null prior that applies no penalty."""
    return 0.0

# Dispatcher dictionary to map prior type strings to functions
_PRIOR_FUNCTIONS = {
    'normal': _nll_normal,
    'lognormal': _nll_lognormal,
    'laplace': _nll_laplace,
    'uniform': _nll_uniform,
    'none': _nll_none,
}

# =============================================================================
# Objective Function Components (Private Helper Functions)
# =============================================================================

def _negative_log_prior(params, prior_types, prior_params):
    """
    Calculates the total negative log-prior for all parameters.
    """
    total_log_prior = 0.0
    for i, p in enumerate(params):
        prior_type = prior_types[i]
        p_params = prior_params[i]
        
        prior_func = _PRIOR_FUNCTIONS.get(prior_type)
        if prior_func is None:
            raise ValueError(f"Unknown prior type: {prior_type}")
            
        total_log_prior += prior_func(p, *p_params)
        
    return total_log_prior

def _negative_log_likelihood(params, model, obs, obs_std, args):
    """
    Calculates the negative log-likelihood, assuming Gaussian noise.
    This is equivalent to a weighted sum of squared residuals.
    """

    prediction = model(params, *args)
    residuals = obs - prediction
    return 0.5 * np.sum((residuals / obs_std)**2)


def _objective_function(params, model, obs, obs_std, prior_types, prior_params, args):
    """
    The full objective function (negative log-posterior) to be minimized.
    This is the sum of the negative log-likelihood and negative log-prior.
    """
    
    nll = _negative_log_likelihood(params, model, obs, obs_std, args)
    nlp = _negative_log_prior(params, prior_types, prior_params)
    
    # Return total negative log-posterior
    # If the value is non-finite, return a very large number to guide the
    # optimizer away from this region of parameter space.
    result = nll + nlp
    if not np.isfinite(result):
        return 1e12
    return result

def run_map(some_model,
            obs,
            obs_std,
            guesses,
            prior_types,
            prior_params,
            args=None):
    """
    Estimates parameters for a model using Maximum a Posteriori (MAP). 
    This minimizes the negative log-posterior (the sum of the -ln(likelihood) 
    and -log(priors). It is flexible, allowing for different prior distributions
    on different parameters.

    Parameters
    ----------
    some_model : callable
        The model function with signature `some_model(parameters, *args)`. It
        should return a NumPy array of predictions of the same shape as `obs`.
    obs : np.ndarray
        Array of observed data.
    obs_std : np.ndarray
        Array of standard deviations for each observation.
    guesses : np.ndarray or list
        Initial guess for the parameters.
    prior_types : list of str
        A list of strings specifying the prior for each parameter. Must be the
        same length as `guesses`. Supported types: 'normal', 'lognormal',
        'laplace', 'uniform', 'none'.
    prior_params : list of tuples
        A list of tuples, where each tuple contains the parameters for the
        corresponding prior in `prior_types`.
        - 'normal': (mu, sigma)
        - 'lognormal': (log_mu, log_sigma)
        - 'laplace': (mu, b)
        - 'uniform': (lower, upper)
        - 'none': (None,) or any single-element tuple.
    args : tuple, optional
        Additional arguments to be passed to `some_model`.

    Returns
    -------
    params : np.ndarray
        The array of best-fit parameter values.
    std_errors : np.ndarray
        The array of standard errors for each fitted parameter.
    cov_matrix : np.ndarray or None
        An approximation of the covariance matrix derived from the inverse of
        the Hessian of the objective function at the minimum. Returns None if
        the optimization fails or the matrix is not finite.
    scipy.optimize.minimize :
        object returned by minimizer
    
    """
    
    if len(guesses) != len(prior_types) or len(guesses) != len(prior_params):
        raise ValueError(
            "Length of 'guesses', 'prior_types', and 'prior_params' must be equal."
        )

    # Arguments to pass to the objective function
    obj_args = (some_model, obs, obs_std, prior_types, prior_params, args)

    # Perform the minimization
    result = scipy.optimize.minimize(
        fun=_objective_function,
        x0=np.array(guesses),
        args=obj_args,
        method='BFGS'
    )

    # --- Extract and validate results ---
    params = result.x

    cov_matrix = None
    if result.success:
        
        # The BFGS algorithm conveniently provides the inverse of the Hessian,
        # which is an excellent approximation of the covariance matrix.
        hess_inv = result.hess_inv
        
        # Final check for numerical stability before returning
        if np.all(np.isfinite(hess_inv)):
            cov_matrix = hess_inv
            with np.errstate(invalid='ignore'): 
                std_errors = np.sqrt(np.diagonal(cov_matrix))
        else:
            num_params = len(params)
            cov_matrix = np.full((num_params, num_params), np.nan)
            std_errors = np.full(num_params, np.nan)
    else:
        params = np.full(num_params, np.nan)
        cov_matrix = np.full((num_params, num_params), np.nan)
        std_errors = np.full(num_params, np.nan)

    return params, std_errors, cov_matrix, result

