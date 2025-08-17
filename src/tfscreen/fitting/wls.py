from tfscreen.fitting.linear_regression import fast_weighted_linear_regression

def get_growth_rates_wls(times,
                         ln_cfu,
                         ln_cfu_var):
    """
    Estimate growth rates using weighted least squares regression (WLS) on
    log-transformed CFU/mL. The weights are based on the variance of the
    log-transformed data.

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).
    ln_cfu_var : np.ndarray
        2D array of variance of the estimate of ln_cfu each genotype, 
        shape (num_genotypes, num_times).

    Returns
    -------
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : np.ndarray
        1D array of standard deviations on estimated growth rates, shape (num_genotypes,).
    """

    _results = fast_weighted_linear_regression(x_arrays=times,
                                               y_arrays=ln_cfu,
                                               y_err_arrays=ln_cfu_var)

    growth_rate_est = _results[0]
    growth_rate_std = _results[2]

    return growth_rate_est, growth_rate_std