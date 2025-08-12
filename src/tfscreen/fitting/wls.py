from tfscreen.fitting.linear_regression import fast_weighted_linear_regression
from tfscreen.util import process_counts

def get_growth_rates_wls(times,
                         sequence_counts,
                         total_counts,
                         total_cfu_ml,
                         pseudocount=1.0):
    """
    Estimate growth rates using weighted least squares regression (WLS) on
    log-transformed CFU/mL. The weights are based on the variance of the
    log-transformed data.

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    sequence_counts : np.ndarray
        2D array of sequence counts for each genotype, shape (num_genotypes, num_times).
    total_counts : np.ndarray
        2D array of total sequence counts for each time point, shape (num_genotypes, num_times).
    total_cfu_ml : np.ndarray
        2D array of total CFU/mL measurements, shape (num_genotypes, num_times).
    pseudocount : float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default: 1.0.

    Returns
    -------
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : np.ndarray
        1D array of standard deviations on estimated growth rates, shape (num_genotypes,).
    """

    _counted = process_counts(sequence_counts,
                              total_counts,
                              total_cfu_ml,
                              pseudocount=pseudocount)
    
    ln_cfu = _counted["ln_cfu"]
    ln_cfu_var = _counted["ln_cfu_var"]

    _results = fast_weighted_linear_regression(x_arrays=times,
                                                y_arrays=ln_cfu,
                                                y_err_arrays=ln_cfu_var)

    growth_rate_est = _results[0]
    growth_rate_std = _results[2]

    return growth_rate_est, growth_rate_std