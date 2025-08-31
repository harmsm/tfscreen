from tfscreen.fitting.linear_regression import (
    fast_linear_regression,
)

import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm
import pandas as pd


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
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std)
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    A0_est = np.repeat(np.nan,times.shape[0])
    A0_std = np.repeat(np.nan,times.shape[0])
    k_est = np.repeat(np.nan,times.shape[0])
    k_std = np.repeat(np.nan,times.shape[0])

    obs = np.ones(times.shape,dtype=float)*np.nan
    pred = np.ones(times.shape,dtype=float)*np.nan

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
            A0_est[i] = glm_results.params[0]
            A0_std[i] = glm_results.params[1]
            k_est[i] = glm_results.params[1]
            k_std[i] = glm_results.bse[1]

            obs[i,:] = y
            pred[i,:] = glm_results.fittedvalues

            if i > 0 and i % 1000 == 0:
                pbar.update(1000)

        pbar.n = pbar.total - 1
        pbar.refresh()
    
    obs = obs.flatten()
    pred = pred.flatten()

    param_df = pd.DataFrame({"A0_est":A0_est,
                             "A0_std":A0_std,
                             "k_est":k_est,
                             "k_std":k_std})

    pred_df = pd.DataFrame({"obs":obs,
                            "pred":pred})
    

    return param_df, pred_df
    

def get_growth_rates_glm(times,cfu):
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
    cfu : np.ndarray
        2D array of cfu each genotype, shape (num_genotypes, num_times).

    Returns
    -------
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std)
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    delta = _estimate_delta(times,cfu)

    param_df, pred_df = _do_glm(times=times,
                                cfu=cfu,
                                delta=delta)

    return param_df, pred_df
