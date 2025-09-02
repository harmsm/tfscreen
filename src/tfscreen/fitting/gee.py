from tfscreen.fitting.linear_regression import (
    fast_linear_regression,
)

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf # Use the formula API
import pandas as pd

def _estimate_delta(times, cfu):
    """
    Estimates the variance power parameter 'delta' for the GEE.
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

def _do_gee(times,
            cfu,
            delta,
            cov_model=None):
    """
    Performs a GEE regression for each genotype.

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
    A0_est : np.ndarray
        1D array of estimated initial populations, shape (num_genotypes,)
    A0_std : np.ndarray
        1D array of standard errors on estimated initial populations, shape (num_genotypes,)
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,)
    growth_rate_std : np.ndarray
        1D array of standard errors on estimated growth rates, shape (num_genotypes,)
    """

    # Flatten the (N, t) cfu array into a single (N * t) vector for 'y'
    y_long = cfu.flatten()

    # Flatten the (N, t) times array and add a constant for 'X'
    X_long = sm.add_constant(times.flatten())

    # Create an array [0, 1, 2, ..., N-1]
    subject_indices = np.arange(cfu.shape[0],dtype=int)
    
    # Repeat each index 't' times to create the group array
    # Result: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    group_ids = np.repeat(subject_indices, cfu.shape[1])

    # Create a pandas DataFrame in a 'long' format 
    df = pd.DataFrame({
        'cfu': y_long,
        'time': X_long[:, 1], # The time column without the constant
        'group_id': group_ids
    })
    
    #  --- Define the interaction model formula ---
    # 'C(group_id)' tells patsy to treat group_id as a categorical variable
    # 'C(group_id):time' creates the interaction term (a unique slope for each group)
    # '- 1' removes the global intercept, so we get a unique intercept for each group
    formula = "cfu ~ C(group_id) + C(group_id):time - 1"
    
    #  --- Run the GEE model using the formula API ---
    # Assume 'delta' is already estimated

     #sm.cov_struct.Autoregressive())
    cov_model = sm.cov_struct.Exchangeable()
    
    # The formula API automatically creates the correct design matrix
    gee_model = sm.GEE.from_formula(formula,
                                     groups="group_id",
                                     data=df,
                                     family=sm.families.Tweedie(var_power=delta),
                                     cov_struct=cov_model)
    
    gee_results = gee_model.fit()

    A0_est = np.asarray(gee_results.param[:cfu.shape[0]])
    A0_std = np.asarray(gee_results.base[:cfu.shape[0]])
    k_est = np.asarray(gee_results.param[cfu.shape[0]:])
    k_std = np.asarray(gee_results.base[cfu.shape[0]:])
    
    pred = gee_results.fittedvalues
    obs = y_long

    param_df = pd.DataFrame({"A0_est":A0_est,
                             "A0_std":A0_std,
                             "k_est":k_est,
                             "k_std":k_std})

    pred_df = pd.DataFrame({"obs":obs,
                            "pred":pred})
    
    return param_df, pred_df


def get_growth_rates_gee(times,cfu):
    """
    Estimate growth rates using Generalized Estimating Equations (GEE).

    This allows us to fit untransformed data with heteroscedastic errors and 
    autocorrelation. 
    
    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    cfu : np.ndarray
        2D array of cfu each genotype, shape (num_genotypes, num_times).

    Returns
    -------
    A0_est : np.ndarray
        1D array of estimated initial populations, shape (num_genotypes,)
    A0_std : np.ndarray
        1D array of standard errors on estimated initial populations, shape (num_genotypes,)
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,)
    growth_rate_std : np.ndarray
        1D array of standard errors on estimated growth rates, shape (num_genotypes,)
    """

    delta = _estimate_delta(times,cfu)

    param_df, pred_df = _do_gee(times=times,
                                cfu=cfu,
                                delta=delta)

    return param_df, pred_df