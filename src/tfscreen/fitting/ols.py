
from tfscreen.fitting.linear_regression import fast_linear_regression

def get_growth_rates_ols(times,ln_cfu):
    """
    Estimate growth rates using ordinary least squares regression on
    log-transformed CFU/mL data. 

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).

    Returns
    -------
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : np.ndarray
        1D array of standard deviations on estimated growth rates, shape (num_genotypes,).
    """

    _results = fast_linear_regression(x_arrays=times,
                                      y_arrays=ln_cfu)

    growth_rate_est = _results[0]
    growth_rate_std = _results[2]

    return growth_rate_est, growth_rate_std