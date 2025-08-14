from tfscreen.util import process_counts
from tfscreen.fitting.linear_regression import (
    fast_linear_regression,
)

import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm


def _estimate_delta(times, cfu):
    """
    Estimates the variance power parameter 'delta' for the GLM.
    This parameter describes the relationship Var(cfu) ~ E[cfu]**delta.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    cfu : numpy.ndarray
        2D array of CFU/mL measurements, shape (num_genotypes, num_times).

    Returns
    -------
    delta : float
        Estimated variance power parameter.
    """

    print("Estimating delta ...", end="", flush=True)

    # 1. Run a quick OLS on the log-data to get a reasonable starting
    #    point for the growth trend.
    ln_cfu = np.log(cfu)
    _ols_results = fast_linear_regression(x_arrays=times, y_arrays=ln_cfu)
    slopes, intercepts = _ols_results[0], _ols_results[1]

    # 2. Calculate fitted values and residuals on the ORIGINAL scale.
    fitted_values_log_scale = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * times
    fitted_cfu = np.exp(fitted_values_log_scale)
    residuals_original_scale = cfu - fitted_cfu

    # 3. Perform a single log-log regression to find the variance relationship.
    #    The plot is log(|residual_original|) vs. log(|fitted_original|).
    #    Flatten the arrays to pool all data for a global estimate.
    ln_abs_resid = np.log(np.abs(residuals_original_scale.flatten()))
    ln_abs_fitted = np.log(np.abs(fitted_cfu.flatten()))
    
    # Filter out -inf values that arise from zero residuals/fits
    finite_mask = np.isfinite(ln_abs_resid) & np.isfinite(ln_abs_fitted)
    
    delta_model = sm.OLS(ln_abs_resid[finite_mask], 
                         sm.add_constant(ln_abs_fitted[finite_mask])).fit()
    
    # 4. The slope of the log-log plot is delta/2.
    delta = 2 * delta_model.params[1]
    
    print(f" Done. Delta = {delta:.4f}", flush=True)
    
    return delta

def _do_glm(times,cfu,delta):
    """
    Performs a GLM regression for each genotype.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    cfu : numpy.ndarray
        2D array of CFU/mL measurements, shape (num_genotypes, num_times).
    delta : float
        Variance power parameter.

    Returns
    -------
    growth_rate_est : numpy.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : numpy.ndarray
        1D array of standard errors of estimated growth rates, shape (num_genotypes,).
    """

    growth_rate_est = np.nan*np.ones(times.shape[0],dtype=float)
    growth_rate_std = np.nan*np.ones(times.shape[0],dtype=float)

    with tqdm(total=times.shape[0]) as pbar:

        for i in range(times.shape[0]):


            y = cfu[i, :]
            X = sm.add_constant(times[i,:])
                    
            # Call GLM
            glm_model = sm.GLM(y, X,
                               family=sm.families.Tweedie(link=sm.families.links.Log(),
                                                          var_power=delta))
    
            glm_results = glm_model.fit()
    
            # Store the results
            growth_rate_est[i] = glm_results.params[1]
            growth_rate_std[i] = glm_results.bse[1]

            if i > 0 and i % 1000 == 0:
                pbar.update(1000)

    
        pbar.n = pbar.total - 1
        pbar.refresh()
    
    return growth_rate_est, growth_rate_std
    

def get_growth_rates_glm(times,
                         sequence_counts,
                         total_counts,
                         total_cfu_ml,
                         pseudocount=1):
    """
    Estimate growth rates using a Generalized Linear Model (GLM).

    This function estimates growth rates by fitting a GLM to the CFU/mL data
    for each genotype. It uses a Tweedie family with a log link function to
    model the relationship between time and CFU/mL. The variance power
    parameter 'delta' is estimated from the data.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    sequence_counts : numpy.ndarray
        2D array of sequence counts for each genotype, shape (num_genotypes, num_times).
    total_counts : numpy.ndarray
        2D array of total sequence counts for each time point, shape (num_genotypes, num_times).
    total_cfu_ml : numpy.ndarray
        2D array of total CFU/mL measurements, shape (num_genotypes, num_times).
    pseudocount : float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default: 1.

    Returns
    -------
    growth_rate_est : numpy.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : numpy.ndarray
        1D array of standard errors of estimated growth rates, shape (num_genotypes,).
    """

    # Get ln_cfu for fitting
    _count = process_counts(sequence_counts=sequence_counts,
                            total_counts=total_counts,
                            total_cfu_ml=total_cfu_ml,
                            pseudocount=pseudocount)

    cfu = _count["cfu"]

    delta = _estimate_delta(times,cfu)

    growth_rate_est, growth_rate_std = _do_glm(times=times,
                                               cfu=cfu,
                                               delta=delta)

    return growth_rate_est, growth_rate_std
