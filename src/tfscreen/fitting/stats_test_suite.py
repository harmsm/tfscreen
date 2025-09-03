import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

def stats_test_suite(param_est,param_std,param_real):
    """
    Run a test suite comparing parameter estimates against true values.

    This function takes arrays of estimated parameters, their standard errors,
    and the corresponding true (known) values from a simulation. It computes
    a dictionary of key metrics to evaluate the performance, accuracy, and
    robustness of the estimation method.

    Parameters
    ----------
    param_est : np.ndarray
        A 1D array of the parameter estimates from a model. May contain NaNs
        for failed fits.
    param_std : np.ndarray
        A 1D array of the standard errors associated with each parameter
        estimate.
    param_real : np.ndarray
        A 1D array of the true, known parameter values used to generate the
        simulated data.

    Returns
    -------
    dict
        A dictionary containing the following performance metrics:

        pct_success : float
            The fraction of fits that returned a valid number (i.e., not
            NaN). This is a measure of the model's numerical robustness.
        rmse : float
            Root Mean Squared Error. This is a measure of the overall
            accuracy and precision of the model, combining both bias and
            variance into a single number. Lower is better.
        normalized_rmse : float
            The RMSE divided by the 95% percentile range of the true
            parameter values. This measures the model's resolving power; a
            low value indicates the error is small compared to the signal.
        pearson_r : float
            Pearson correlation coefficient between the estimated and real 
            parameter values. A value of 1 indicates perfect correlation; 0
            no correlation.
        r_squared : float
            Squared Pearson correlation coefficient. Measures the fraction of
            the variation in the real parameters captured by the estimated 
            parameters.
        mean_error : float
            The average difference between estimated and real values (bias).
            A value close to zero indicates the model is unbiased on
            average.
        coverage_prob : float
            The fraction of times the true parameter value falls within the
            estimated 95% confidence interval. For a perfectly calibrated
            model, this value should be close to 0.95. If this is higher than
            0.95, the model *overestimates* error; if this value is lower than
            0.95, the model *underestimates* error. 
        residual_corr : float
            The Pearson correlation coefficient between the estimation errors
            and the true parameter values. A value near zero is ideal,
            indicating the model's errors are not dependent on the magnitude
            of the true value.
        residual_corr_p_value : float
            The p-value for the residual correlation. A low p-value
            (e.g., < 0.05) suggests the correlation is statistically
            significant.
        bp_p_value : float
            The p-value from the Breusch-Pagan test for heteroscedasticity.
            A low p-value (e.g., < 0.05) suggests the variance of the
            errors is not constant and changes with the true parameter value.
    """
    
    # Get percent of estimates succeded
    not_nan_mask = np.logical_not(np.isnan(param_est))
    pct_success = np.sum(not_nan_mask)/param_est.shape[0]

    # Filter out bad fits for this analysis
    param_est = param_est[not_nan_mask]
    param_std = param_std[not_nan_mask]
    param_real = param_real[not_nan_mask]
    
    # Get RMSE and mean error
    diff = param_est - param_real
    rmse = np.sqrt(np.mean(diff**2))

    # --- New Calculation ---
    # Calculate the 95% percentile range of the true signal
    signal_range = np.percentile(param_real, 97.5) - np.percentile(param_real, 2.5)

    # Avoid division by zero if all real values are the same
    if signal_range > 0:
        normalized_rmse = rmse / signal_range
    else:
        normalized_rmse = np.inf 

    # R and R^2
    r_val, _ = pearsonr(param_real, param_est)
    r_squared = r_val**2

    mean_error = np.mean(diff)

    # Get coverage probability (probability real values fall in the 95% CI). 
    # This will be 95% for perfectly calibrated error estimator. 
    lower_ci = param_est - param_std*1.96
    upper_ci = param_est + param_std*1.96

    in_ci = np.logical_and(param_real >= lower_ci,
                           param_real <= upper_ci)
    coverage_prob = np.sum(in_ci)/param_est.shape[0]

    # Look for correlation in residuals
    residual_corr, residual_corr_p_value = pearsonr(diff, param_real)
    
    # Look for heteroscedasticity in the residuals
    bp_test = het_breuschpagan(diff, sm.add_constant(param_real))
    bp_p_value = bp_test[1]


    out = {
        "pct_success":pct_success,
        "rmse":rmse,
        "normalized_rmse":normalized_rmse,
        "pearson_r": r_val,
        "r_squared": r_squared,
        "mean_error":mean_error,
        "coverage_prob":coverage_prob,
        "residual_corr":residual_corr,
        "residual_corr_p_value":residual_corr_p_value,
        "bp_p_value":bp_p_value
    }
    
    return out

