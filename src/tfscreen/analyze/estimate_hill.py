from tfscreen.fitting import (
    run_least_squares,
    predict_with_error
)

from tfscreen.util import read_dataframe

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def hill_model(params, x):
    """
    A four-parameter Hill model.

    The model describes a sigmoidal curve with defined left (low x) and
    right (high x) baselines.

    Parameters
    ----------
    params : np.ndarray
        1D numpy float array of four parameters:
        - logK: Natural log of the Hill equilibrium constant (midpoint).
        - n: The Hill coefficient (steepness).
        - left_base: The model's value as x -> 0.
        - right_base: The model's value as x -> +inf.
    x : numpy.ndarray
        1D array of concentrations (independent variable).

    Returns
    -------
    np.ndarray
        A 1D array with the model calculated at each concentration in `x`.
    """
    logK, n, left_base, right_base = params

    with np.errstate(invalid='ignore'):
        K = np.exp(logK)
        K_to_n = np.power(K, n)
        x_to_n = np.power(x, n)
        fx = x_to_n / (x_to_n + K_to_n)

    shift = right_base - left_base
    return fx * shift + left_base


def fit_hill_model(theta_est, theta_std, conc, hill_guess=2.0):
    """
    Generate guesses and fit a single Hill curve.

    This function creates intelligent initial guesses for the Hill model
    parameters based on the data and then calls the generic least-squares
    solver.

    Parameters
    ----------
    theta_est : np.ndarray
        1D array of observed y-values for a single curve.
    theta_std : np.ndarray
        1D array of standard deviations for `theta_est`.
    conc : np.ndarray
        1D array of concentration x-values, assumed to be sorted descending.
    hill_guess : float, optional
        An initial guess for the Hill coefficient `n`.

    Returns
    -------
    params : np.ndarray
        The array of best-fit parameter values.
    std_errors : np.ndarray
        The array of standard errors for each fitted parameter.
    cov_matrix : np.ndarray
        The full covariance matrix of the fitted parameters.
    """
    # Start with default guesses
    guesses = np.array([np.log(2e-5), hill_guess, 1.0, 0.0])

    # Try to infer better guesses from the data if possible
    valid_idx = ~np.isnan(theta_est)
    if np.any(valid_idx):

        # Get guesses for baseline from lowest and highest values
        right_base, left_base = theta_est[valid_idx][[0, -1]]

        # Estimate K as the concentration where theta is halfway
        half_point = left_base + (right_base - left_base) / 2
        try:
            idx = np.nanargmin(np.abs(theta_est - half_point))
            K_guess = conc[idx]
        except ValueError:
            K_guess = np.nanmedian(conc)

        if K_guess <= 0: K_guess = 1e-9 # Prevent log(0)

        guesses = np.array([np.log(K_guess), hill_guess, left_base, right_base])

    # Define parameter bounds for [logK, n, left_base, right_base]
    lower_bounds = np.array([-np.inf, 0.1, 0, 0])
    upper_bounds = np.array([np.inf, 10.0, 1, 1])

    return run_least_squares(
        hill_model,
        theta_est,
        theta_std,
        guesses,
        lower_bounds,
        upper_bounds,
        args=(conc,)
    )


def estimate_hill(df,
                  left_log_extend=5,
                  right_log_extend=1,
                  num_smooth_steps=100):
    """
    Fit the Hill model to every genotype in a DataFrame.

    This function orchestrates the entire analysis pipeline:
    1. Reshapes the input DataFrame into matrices.
    2. Loops through each genotype (row).
    3. Fits the Hill model to the data for that genotype.
    4. Generates predictions over a smooth concentration range.
    5. Compiles and returns the results in two clean DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        A long-format DataFrame containing columns 'genotype', 'iptg',
        'theta_est', and 'theta_std'.
    left_log_extend : float, optional   
        how much to extend the smooth prediction curve to the left of the 
        lowest iptg concentration (in np.log(mM) units). The default value of 5
        works well assuming the lowest iptg concentration is 1E-4 mM and that 
        we assign the iptg = 0 point a fake concentration of 1E-5 mM. 
    right_log_extend : float, optional   
        how much to extend the smooth prediction curve to the right of the 
        highest iptg concentration (in np.log(mM) units).
    num_smooth_steps : int, optional
        predict the value of the hill model over this many iptg concentrations 
        interpolated between the lowest and highest iptg. 

    Returns
    -------
    param_df : pd.DataFrame
        A DataFrame containing the fitted parameters and their standard errors
        for each genotype.
    pred_df : pd.DataFrame
        A long-format DataFrame containing the model predictions and their
        standard errors over a smooth range of concentrations for plotting.
    """

    df = read_dataframe(df)

    # Get all genotypes and iptg values
    genotypes = pd.unique(df['genotype'])
    iptg = np.sort(df['iptg'].unique())[::-1]

    # Pivot dataframe to genotype x iptg
    pivoted_df = df.pivot_table(
        index='genotype', columns='iptg', values=['theta_est', 'theta_std']
    ).reindex(index=genotypes).reindex(columns=iptg, level=1)

    # New arrays with shape genotype x iptg
    theta_est = pivoted_df['theta_est'].values
    theta_std = pivoted_df['theta_std'].values

    # Create a smooth concentration vector for predictions
    smooth_iptg = iptg[iptg > 0]
    smooth_iptg = np.log(smooth_iptg)
    smooth_iptg = np.linspace(np.min(smooth_iptg) - left_log_extend,
                              np.max(smooth_iptg) + right_log_extend,
                              num_smooth_steps)
    smooth_iptg = np.exp(smooth_iptg)

    # Prep outputs
    param_out = {"genotype":[],
                 "K_est":[],
                 "K_std":[],
                 "n_est":[],
                 "n_std":[],
                 "left_base_est":[],
                 "left_base_std":[],
                 "right_base_est":[],
                 "right_base_std":[]}
    pred_out = {"genotype":[],
                "iptg":[],
                "hill_est":[],
                "hill_std":[]}
    
    for i in tqdm(range(theta_est.shape[0]), desc="Fitting Genotypes"):

        # Fit hill model to data
        param, std_err, cov_matrix = fit_hill_model(theta_est[i,:],
                                                    theta_std[i,:],
                                                    iptg)

        lnK_est = param[0]
        lnK_std = param[1]
    
        # Propagate error: std(K) ≈ K * std(logK)
        K_est = np.exp(lnK_est)
        K_std = K_est * lnK_std

        # Record fit results
        param_out["genotype"].append(genotypes[i])
        param_out["K_est"].append(K_est)
        param_out["n_est"].append(param[1])
        param_out["left_base_est"].append(param[2])
        param_out["right_base_est"].append(param[3])
        param_out["K_std"].append(K_std)
        param_out["n_std"].append(std_err[1])
        param_out["left_base_std"].append(std_err[2])
        param_out["right_base_std"].append(std_err[3])

        # Predict hill model (with error) for all iptg in smooth_iptg
        hill_est, hill_std = predict_with_error(hill_model,
                                                param,
                                                cov_matrix,
                                                args=[smooth_iptg])


        # Record results
        pred_out["genotype"].extend(np.repeat(genotypes[i],smooth_iptg.size))
        pred_out["iptg"].extend(smooth_iptg)
        pred_out["hill_est"].extend(hill_est)
        pred_out["hill_std"].extend(hill_std)

    # Construct final dataframes
    param_df = pd.DataFrame(param_out)
    pred_df = pd.DataFrame(pred_out)

    return param_df, pred_df
