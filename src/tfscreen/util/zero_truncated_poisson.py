import numpy as np
from scipy.stats import poisson
from typing import Optional

def zero_truncated_poisson(
    num_samples: int,
    poisson_lambda: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample from a zero-truncated Poisson distribution.

    This function generates random samples from a Poisson distribution that
    are guaranteed to be greater than zero. It uses the inverse transform
    sampling method for efficiency.

    Parameters
    ----------
    num_samples : int
        The number of samples to generate. Must be an integer > 0.
    poisson_lambda : float
        The lambda parameter (rate) of the Poisson distribution. Must be > 0.
    rng : numpy.random.Generator, optional
        A pre-initialized NumPy random number generator. If None, a new
        generator is created. By default None.

    Returns
    -------
    np.ndarray
        An array of shape (`num_samples`,) containing integer samples from
        the zero-truncated Poisson distribution.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> samples = zero_truncated_poisson(5, 1.5, rng=rng)
    >>> samples
    array([1, 1, 2, 1, 2])
    """

    # ------------------ Input Validation ------------------
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be an integer > 0")
    
    try:
        # check for vector-ness
        if not np.isscalar(poisson_lambda):
            raise ValueError
        
        # Check for float (or float coercibility)
        poisson_lambda = float(poisson_lambda)

        # Check for positive
        if poisson_lambda <= 0:
            raise ValueError
    except (TypeError,ValueError):
        raise ValueError("poisson_lambda must be a scalar > 0")

    # initialize the random number generator if this was not already done.
    if rng is None:
        rng = np.random.default_rng()

    # Calculate the cumulative probability of the zero outcome given lambda
    p_zero = np.exp(-poisson_lambda)
    
    # Generate uniform random numbers in the valid range [p_zero, 1]
    u = rng.uniform(low=p_zero, high=1.0, size=num_samples)
    
    # Use the Poisson Percent Point Function (ppf) to find the integer 
    # outcomes from these values. 
    ztp_samples = poisson.ppf(u, mu=poisson_lambda).astype(int)
    
    return ztp_samples