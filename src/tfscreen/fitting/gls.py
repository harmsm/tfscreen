from tfscreen.fitting.linear_regression import (
    fast_linear_regression,
    fast_weighted_linear_regression
)

import numpy as np
from tqdm.auto import tqdm
import statsmodels.api as sm

def _estimate_delta(times,
                    ln_cfu,
                    convergence_criterion=1e-4,
                    max_iteration=50):
    """
    Estimates the variance power parameter 'delta' for the GLS model.

    This parameter describes the relationship between the variance of the
    CFU/mL measurements and the expected CFU/mL, i.e.,
    Var(CFU/mL) ~ E[CFU/mL]**delta.  The estimation is done iteratively
    using weighted least squares until convergence.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : numpy.ndarray
        2D array of log-transformed CFU/mL measurements,
        shape (num_genotypes, num_times).
    convergence_criterion : float, optional
        Criterion for convergence of the iterative estimation of delta.
        The estimation stops when the absolute change in delta is less than
        this value. Default is 1e-4.
    max_iteration : int, optional
        Maximum number of iterations for the estimation of delta.
        Default is 50.

    Returns
    -------
    delta : float
        Estimated variance power parameter.
    weighted_residuals : numpy.ndarray
        2D array of weighted residuals from the final WLS regression,
        shape (num_genotypes, num_times). Returns np.nan if estimation did not converge.
    """

    all_deltas = []
    
    _ols_results = fast_linear_regression(x_arrays=times,
                                          y_arrays=ln_cfu)
    slopes = _ols_results[0]
    intercepts = _ols_results[1]
    fitted_values = intercepts[:, np.newaxis] + slopes[:, np.newaxis]*times
    residuals = fitted_values - ln_cfu
    
    ln_abs_fitted = np.log(np.abs(fitted_values.reshape(fitted_values.shape[0]*fitted_values.shape[1])))
    ln_abs_resid = np.log(np.abs(residuals.reshape(residuals.shape[0]*residuals.shape[1])))
    
    # Fit a line to find delta
    delta_model = sm.OLS(ln_abs_resid, sm.add_constant(ln_abs_fitted)).fit()
    delta = 2*delta_model.params[1]
    all_deltas.append(delta)

    print("Estimating delta ...",end=None,flush=True)
    with tqdm(total=max_iteration) as pbar:
        
        success = False
        for _ in range(max_iteration):
            fitted_values_original_scale = np.exp(fitted_values)
            std = (fitted_values_original_scale)**(delta / 2)

            _wls_results = fast_weighted_linear_regression(x_arrays=times,
                                                           y_arrays=ln_cfu,
                                                           y_err_arrays=std)
            slopes = _wls_results[0]
            intercepts = _wls_results[1]
            fitted_values = intercepts[:, np.newaxis] + slopes[:, np.newaxis]*times
            residuals = (fitted_values - ln_cfu)
            
            ln_abs_fitted = np.log(np.abs(fitted_values.reshape(fitted_values.shape[0]*fitted_values.shape[1])))
            ln_abs_resid = np.log(np.abs(residuals.reshape(residuals.shape[0]*residuals.shape[1])))
            
            # Fit a line to find delta
            delta_model = sm.OLS(ln_abs_resid, sm.add_constant(ln_abs_fitted)).fit()
            delta = 2*delta_model.params[1]
        
            if np.abs(delta - all_deltas[-1]) < convergence_criterion:
                pbar.n = pbar.total
                pbar.refresh()
                success = True
                break

            all_deltas.append(delta)
            pbar.update(1)

        if not success:
            print("Estimate did not converge!")
            return np.nan, np.nan*np.ones(ln_cfu.shape)

    print(f"Done. Delta = {delta:.4f}",flush=True)

    weighted_residuals = _wls_results[4]
    
    return delta, weighted_residuals

def _estimate_phi(weighted_residuals):
    """
    Estimates the autocorrelation parameter 'phi' for the GLS model.

    This parameter describes the autocorrelation between residuals at
    consecutive time points. It is estimated by performing a linear
    regression of the residuals at time t on the residuals at time t-1.

    Parameters
    ----------
    weighted_residuals : numpy.ndarray
        2D array of weighted residuals from the WLS regression,
        shape (num_genotypes, num_times).

    Returns
    -------
    phi : float
        Estimated autocorrelation parameter.
    """

    print("Estimating phi ...",end=None,flush=True)
    
    # Create the y-variable: residuals from time point 1 to the end
    y = weighted_residuals[:, 1:]
    
    # Create the x-variable: residuals from time point 0 to t-1
    x = weighted_residuals[:, :-1]
    
    # Reshape the arrays into long 1D vectors to pool all N conditions
    y_flat = y.flatten()

    # Add a constant (for the intercept) to the x-variable
    x_flat_with_const = np.vstack([x.flatten(), np.ones(len(x.flatten()))]).T
    
    # Perform the linear regression of y on x
    # lstsq returns the coefficients, residuals, rank, and singular values
    # The first element of the coefficients is the slope (phi)
    phi, intercept = np.linalg.lstsq(x_flat_with_const, y_flat, rcond=None)[0]
    
    print(f"Done. phi = {phi:.4f}")

    return phi

def _do_gls(times,
            ln_cfu,
            delta,
            phi):
    """
    Performs a GLS regression for each genotype.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : numpy.ndarray
        2D array of log-transformed CFU/mL measurements,
        shape (num_genotypes, num_times).
    delta : float
        Variance power parameter.
    phi : float
        Autocorrelation parameter.

    Returns
    -------
    growth_rate_est : numpy.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : numpy.ndarray
        1D array of standard errors of estimated growth rates,
        shape (num_genotypes,).
    """

    growth_rate_est = np.nan*np.ones(times.shape[0],dtype=float)
    growth_rate_std = np.nan*np.ones(times.shape[0],dtype=float)

    with tqdm(total=times.shape[0]) as pbar:
        
        for i in range(times.shape[0]):
        
            y = ln_cfu[i,:]
            X = sm.add_constant(times[i,:])
                    
            # 1. Get initial estimates for the fitted values to construct omega
            # Using OLS for this is a standard and reasonable approach.
            ols_fit = sm.OLS(y, X).fit()
            
            # Back-transform fitted values to the original data scale
            mean_vals_i = np.exp(ols_fit.fittedvalues)
            
            # 2. Construct the omega matrix on the LOG-SCALE
            
            # The variance of the LOGGED errors is proportional to (mean_value)**(delta - 2)
            variances_log_scale = (mean_vals_i)**(delta - 2)
            D = np.diag(np.sqrt(variances_log_scale))
            
            # The correlation matrix R is correct
            R = np.array([[1,     phi,   phi**2],
                          [phi,   1,     phi],
                          [phi**2, phi,   1]])
            
            # The final covariance matrix for the log-scale errors
            omega = D @ R @ D
            
            # 3. Run the GLS model on the log-transformed data
            gls_model = sm.GLS(y, X, sigma=omega).fit()
            
            # 4. Extract results
            growth_rate_est[i] = gls_model.params[1]
            growth_rate_std[i] = gls_model.bse[1]

            if i > 0 and i % 1000 == 0:
                pbar.update(1000)
    
        pbar.n = pbar.total - 1
        pbar.refresh()
    
    return growth_rate_est, growth_rate_std
    

def get_growth_rates_gls(times,ln_cfu):
    """
    Estimate growth rates using a Generalized Least Squares (GLS) model.

    This function estimates growth rates by fitting a GLS model to the
    CFU/mL data for each genotype. It estimates the variance power
    parameter 'delta' and the autocorrelation parameter 'phi' from the data
    and uses these parameters to perform the GLS regression.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).

    Returns
    -------
    growth_rate_est : numpy.ndarray
        1D array of estimated growth rates, shape (num_genotypes,).
    growth_rate_std : numpy.ndarray
        1D array of standard errors of estimated growth rates,
        shape (num_genotypes,).
    """

    delta, weighted_residuals = _estimate_delta(times,ln_cfu)
    phi = _estimate_phi(weighted_residuals)

    growth_rate_est, growth_rate_std = _do_gls(times=times,
                                               ln_cfu=ln_cfu,
                                               delta=delta,
                                               phi=phi)

    return growth_rate_est, growth_rate_std