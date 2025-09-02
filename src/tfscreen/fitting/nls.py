
from tfscreen.fitting import get_growth_rates_wls

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import issparse
from tqdm.auto import tqdm
import pandas as pd



def _model(A0, k, t):
    """
    Calculates exponential growth for multiple curves simultaneously.

    Parameters
    ----------
    A0 : numpy.ndarray
        1D array of initial population sizes (at t=0).
    k : numpy.ndarray
        1D array of growth rates.
    t : numpy.ndarray
        1D array of time points.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_curves, n_times) representing the
        calculated population size at each time point for each curve.
    """

    # Broadcasting: (n, 1) * (1, t) -> (n, t)
    return A0[:, np.newaxis] * np.exp(k[:, np.newaxis] * t)


def _model_residuals(params, cfu, cfu_std, t):
    """
    Calculates weighted residuals for a block of exponential growth curves.

    This function is designed to be called by scipy.optimize.least_squares.
    It unpacks the parameter array into initial populations (A0) and growth
    rates (k), calculates the model predictions, and returns the
    standard-deviation-weighted residuals.

    Parameters
    ----------
    params : numpy.ndarray
        A 1D array containing the fitting parameters. The first half of the
        array contains the initial population (A0) guesses, and the second
        half contains the growth rate (k) guesses.
    cfu : numpy.ndarray
        A 2D array (n_curves, n_times) of observed CFU/mL values.
    cfu_std : numpy.ndarray
        A 2D array (n_curves, n_times) of the standard deviations
        associated with each CFU/mL measurement.
    t : numpy.ndarray
        A 2D array of time points at which measurements were taken.

    Returns
    -------
    numpy.ndarray
        A 1D array of the flattened, weighted residuals, suitable for
        minimization.
    """

    n = cfu.shape[0]
    A0_params = params[:n]
    k_params = params[n:]

    calc = _model(A0_params, k_params, t)
    
    # Return the flattened array of weighted residuals
    return ((cfu - calc) / cfu_std).flatten()


def get_growth_rates_nls(times,
                         cfu,
                         cfu_var,
                         block_size=100):
    """
    Performs block-wise non-linear least squares fitting.

    This function iterates through all genotypes in blocks,
    performing an independent exponential fit for each one simultaneously
    within the block. It leverages a sparse Jacobian to accelerate the
    computation.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points.
    cfu : numpy.ndarray
        2D array of observed CFU/mL values for all genotypes.
    cfu_std : numpy.ndarray
        2D array of standard deviations for the CFU/mL values.
    block_size : int
        The number of genotypes to fit simultaneously in each block. Default 
        is 100.

    Returns
    -------
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std)
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    param_out = {"A0_est":[],
                 "A0_std":[],
                 "k_est":[],
                 "k_std":[]}

    pred_out = {"obs":[],
                "pred":[]}

    num_genotypes = cfu.shape[0]
    num_times = cfu.shape[1]
    block_size = int(np.round(block_size,0))

    ln_cfu = np.log(cfu)
    ln_cfu_var = cfu_var/(cfu**2)

    wls_param_df, _ = get_growth_rates_wls(times=times,
                                            ln_cfu=ln_cfu,
                                            ln_cfu_var=ln_cfu_var)
    
    k_guess = np.array(wls_param_df["k_est"])
    A0_guess = np.exp(np.array(wls_param_df["A0_est"]))
        
    for i in tqdm(range(0, num_genotypes, block_size), desc="Fitting Growth Rates"):

        # Grab a block of data for fitting
        indices = slice(i, i + block_size)
        times_block = times[indices,:]
        cfu_block = cfu[indices, :]
        cfu_std_block = np.sqrt(cfu_var[indices, :])
        rate_guess_block = k_guess[indices]
        pop_guess_block = A0_guess[indices]

        n = cfu_block.shape[0]
        if n == 0:
            continue

        # Create array of parameter guesses: [A0_1, A0_2..., k_1, k_2,...]
        params = np.concatenate([pop_guess_block, rate_guess_block])

        # Set parameter bounds: A0 >= 0, -1 <= k <= 1
        lower_bounds = np.concatenate([np.zeros(n), -1.0 * np.ones(n)])
        upper_bounds = np.concatenate([np.inf * np.ones(n), np.ones(n)])
        bounds = (lower_bounds, upper_bounds)

        # Create a sparse jacobian because the genotypes are independent
        num_residuals = n * num_times
        num_params = 2 * n
        jac_sparsity = np.zeros((num_residuals, num_params), dtype=int)

        for j in range(n):

            # Rows for genotype j
            row_start = num_times * j
            row_end = num_times * (j + 1)
            
            # Non-zero derivatives are only for A0_j and k_j
            # Column for A0_j parameter
            jac_sparsity[row_start:row_end, j] = 1

            # Column for k_j parameter
            jac_sparsity[row_start:row_end, n + j] = 1

        # Do nonlinear regression
        fit_result = least_squares(_model_residuals,
                                   x0=params,
                                   bounds=bounds,
                                   jac_sparsity=jac_sparsity,
                                   args=(cfu_block, cfu_std_block, times_block))
        
        # Get results
        parameters = fit_result.x

        J = fit_result.jac

        num_obs, num_param = J.shape
        dof = num_obs - num_param
        chi2_red = np.sum(fit_result.fun**2) / dof
        
        JTJ = J.T @ J
        if issparse(J):
            JTJ =JTJ.toarray()
    
        try:
            cov_matrix = chi2_red * np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            cov_matrix = np.ones((num_param,num_param))*np.nan

        with np.errstate(invalid='ignore'): # Ignore sqrt(negative) for bad fits
            std = np.sqrt(np.diagonal(cov_matrix))

        # Record fit parameters
        param_out["A0_est"].extend(parameters[:n])
        param_out["A0_std"].extend(std[:n])
        param_out["k_est"].extend(parameters[n:])
        param_out["k_std"].extend(std[n:])

        # Record prediction obs and prediction
        calc = _model(parameters[:n], parameters[n:], times_block)
        pred_out["obs"].extend(cfu_block.flatten())
        pred_out["pred"].extend(calc.flatten())

    
    param_df = pd.DataFrame(param_out)
    pred_df = pd.DataFrame(pred_out)


    return param_df, pred_df
