from tfscreen.util import read_dataframe

from tfscreen.calibration import (
    read_calibration,
    get_wt_k,
    get_wt_theta
)

from tfscreen.fitting import (
    matrix_wls,
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error
)


import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from tqdm.auto import tqdm


def theta_model(params, X):
    """
    Calculate predicted values from a linear model.

    This model is defined by `y = X @ params`, where `X` is the design matrix
    and `params` is the vector of model parameters.

    Parameters
    ----------
    params : np.ndarray
        1D array of model parameters.
    X : np.ndarray
        2D design matrix, where rows are observations and columns are parameters.

    Returns
    -------
    np.ndarray
        1D array of predicted `y` values.
    """
    return X @ params

def fit_theta_model_nls(y, y_std, X, guesses):
    """
    Fit a linear model using bounded Non-Linear Least Squares (NLS).

    This function serves as a wrapper to fit the `theta_model` while enforcing
    bounds on the parameters. The first parameter is treated as an unbounded
    global offset, while all others are constrained to be between 0 and 1.

    Parameters
    ----------
    y : np.ndarray
        1D array of observed dependent variable values.
    y_std : np.ndarray
        1D array of standard deviations for each observation in `y`.
    X : np.ndarray
        2D design matrix for the linear model.
    guesses : np.ndarray
        1D array of initial guesses for the parameters.

    Returns
    -------
    param : np.ndarray
        1D array of the best-fit parameters.
    std_error : np.ndarray
        1D array of the standard errors for each parameter.
    pred_est : np.ndarray
        1D array of the predicted `y` values at the best-fit parameters.
    pred_std : np.ndarray
        1D array of the standard errors for each predicted value.
    """
    
    # Set bounds: first param is unbounded, rest are [0, 1].
    lower_bounds = np.zeros(len(guesses), dtype=float)
    upper_bounds = np.ones(len(guesses), dtype=float)
    lower_bounds[0] = -np.inf
    upper_bounds[0] = np.inf

    param, std_error, cov_matrix = run_least_squares(
        theta_model,
        obs=y,
        obs_std=y_std,
        guesses=guesses,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        args=(X,)
    )

    pred_est, pred_std = predict_with_error(theta_model,
                                            param,
                                            cov_matrix,
                                            args=(X,))

    return param, std_error, pred_est, pred_std

def fit_theta_model_wls(y, y_std, X):
    """
    Fit a linear model using Weighted Linear Least Squares (WLS).

    This is an analytical solution for the linear model that does not support
    bounds but is generally faster and more stable than NLS.

    Parameters
    ----------
    y : np.ndarray
        1D array of observed dependent variable values.
    y_std : np.ndarray
        1D array of standard deviations for each observation in `y`.
    X : np.ndarray
        2D design matrix for the linear model.

    Returns
    -------
    param : np.ndarray
        1D array of the best-fit parameters.
    std_error : np.ndarray
        1D array of the standard errors for each parameter.
    pred_est : np.ndarray
        1D array of the predicted `y` values at the best-fit parameters.
    pred_std : np.ndarray
        1D array of the standard errors for each predicted value.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = (1 / y_std)**2
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    param, cov_matrix = matrix_wls(X, y, weights)

    with np.errstate(invalid="ignore"):
        std_error = np.sqrt(np.diag(cov_matrix))

    pred_est, pred_std = predict_with_error(theta_model, param, cov_matrix,args=(X,))

    return param, std_error, pred_est, pred_std


def estimate_theta(df,
                   calibration_file,
                   block_size=100,
                   method="nls"):
    """
    Estimate mutation effects and fractional occupancy (theta) for genotypes.

    For each genotype, it fits a linear model to estimate a global growth effect
    of the mutation and a `theta` parameter for each IPTG concentration, which
    represents the fractional occupancy of a binding site.

    Parameters
    ----------
    df : pd.DataFrame or str
        A long-format DataFrame (or path to one) with columns: "genotype",
        "k_est", "k_std", "iptg", "marker", and "select".
    calibration_file : str or dict
        Path to the calibration file or a pre-loaded calibration dictionary.
    method : {"nls", "wls"}, default="nls"
        The fitting method to use. "nls" for bounded Non-Linear Least Squares,
        or "wls" for unbounded Weighted Linear Least Squares.

    Returns
    -------
    growth_df : pd.DataFrame
        DataFrame with columns "genotype", "mut_effect_est", and
        "mut_effect_std".
    theta_df : pd.DataFrame
        DataFrame with columns "genotype", "iptg", "theta_est", and "theta_std".
    pred_df : pd.DataFrame
        DataFrame of model predictions for each observed condition, with columns
        "genotype", "marker", "select", "iptg", "k_pred_est", "k_pred_std".
    """

    df = read_dataframe(df)

    # Get canonical order of all experimental conditions
    order_cols = ['marker', 'select', 'iptg']
    canonical_order = df[order_cols].drop_duplicates().sort_values(by=order_cols)
    canonical_multi_index = pd.MultiIndex.from_frame(canonical_order)

    # Pivot k_est and k_std into (num_genotypes, num_conditions) matrices
    k_est = df.pivot_table(
        index='genotype', columns=order_cols, values="k_est"
    ).reindex(columns=canonical_multi_index).values

    k_std = df.pivot_table(
        index='genotype', columns=order_cols, values="k_std"
    ).reindex(columns=canonical_multi_index).values

    # Extract aligned arrays for genotypes and conditions
    genotypes = pd.unique(df['genotype'])
    marker = np.asarray(canonical_multi_index.get_level_values("marker"))
    select = np.asarray(canonical_multi_index.get_level_values("select"))
    iptg = np.asarray(canonical_multi_index.get_level_values("iptg"))
    
    # Load calibration dictionary
    calibration_dict = read_calibration(calibration_file)
    k_wt, _ = get_wt_k(marker=["none" for _ in range(len(iptg))],
                            select=["none" for _ in range(len(iptg))],
                            iptg=iptg,
                            calibration_data=calibration_dict,
                            theta=None,
                            calc_err=False)

    # Get baselines and slopes relating theta to growth for these markers +
    # selection. 

    param_values = calibration_dict["param_values"]
    param_idx_dict = {p: i for i, p in enumerate(calibration_dict["param_names"])}

    intercepts = []
    slopes = []
    for m, s in zip(marker, select):
        intercepts.append(param_values[param_idx_dict[f"{m}|{s}|b"]])
        slopes.append(param_values[param_idx_dict[f"{m}|{s}|m"]])
    intercepts = np.array(intercepts)
    slopes = np.array(slopes)

    # Calculate delta_k: the observed growth rate shift from wildtype with 
    # no theta
    delta_k = k_est - (k_wt + intercepts)

    # map each iptg concentration to a unique theta parameter
    unique_iptg = np.sort(np.unique(iptg))
    iptg_to_idx = {val: i + 1 for i, val in enumerate(unique_iptg)}
    iptg_mapper = np.array([iptg_to_idx[val] for val in iptg])
    
    # Build a design matrix X
    num_conditions = len(canonical_order)
    num_params = len(unique_iptg) + 1

    X = np.zeros((num_conditions, num_params))
    X[:, 0] = 1.0                                      # global mutational effect
    X[np.arange(num_conditions), iptg_mapper] = slopes # theta effects

    # Get initial guesses from wildtype theta values
    wt_theta = get_wt_theta(unique_iptg, calibration_dict)
    base_guesses = np.zeros(num_params)
    base_guesses[1:] = wt_theta
    
    growth_out = {#"genotype":[],
                  "mut_effect_est":[],
                  "mut_effect_std":[]}
    
    theta_out = {#"genotype":[],
                 #"iptg":[],
                 "theta_est":[],
                 "theta_std":[]}
    
    pred_out = {#"genotype":[],
                "marker":[],
                "select":[],
                "iptg":[],
                "k_pred_est":[],
                "k_pred_std":[]}

    for i in tqdm(range(0,k_est.shape[0],block_size), desc="Fitting Genotypes"):

        row = np.arange(i,i+block_size,dtype=int)
        
        block_delta_k = delta_k[row,:]
        block_k_std = k_std[row,:]


        block_X = block_diag(*([X] * actual_block_size))
        block_guesses = np.tile(base_guesses,actual_block_size)
        block_marker = np.tile(marker,actual_block_size)
        block_select = np.tile(select,actual_block_size)
        block_iptg = np.tile(iptg,actual_block_size)
        block_unique_iptg = np.tile(unique_iptg,actual_block_size)

        block_genotypes = np.full(X.shape[0]*actual_block_size,object)
        for j in range(actual_block_size):
            j_span = np.arange(j*X.shape[0],(j+1)*X.shape[0],dtype=int)
            #block_genotypes[j_span] = genotypes[i+j-1]

        valid_mask = ~np.any(np.isnan(block_delta_k) | np.isnan(block_k_std),axis=0)
        if ~np.all(valid_mask):
           continue

        clean_delta_k = block_delta_k[:, valid_mask]
        clean_k_std = block_k_std[:, valid_mask]
        clean_X = block_X[valid_mask, :].copy()
        
        # Remove any parameters that have no data after cleaning
        param_to_estimate_mask = (clean_X != 0).any(axis=0)
        final_X = clean_X[:, param_to_estimate_mask]
        if final_X.shape[1] == 0:
            continue
        
        clean_guesses = block_guesses[param_to_estimate_mask].copy()
        #clean_iptg = unique_iptg[param_to_estimate_mask[1:]]
        
        # Improve the guess for the mutational effect if possible
        # Note: Assumes select values are numeric (e.g., 0/1) not str ('On'/'Off')
        no_select_mask = (select[valid_mask] == 0)
        if np.any(no_select_mask):
            clean_guesses[0] = np.nanmean(clean_delta_k[no_select_mask])
        
        # Fit the model
        if method == "nls":
            params, stds, pred_est, pred_std = fit_theta_model_nls(
                clean_delta_k, clean_k_std, final_X, clean_guesses
            )
        elif method == "wls":
            params, stds, pred_est, pred_std = fit_theta_model_wls(
                clean_delta_k, clean_k_std, final_X
            )
        else:
            raise ValueError(f"Method '{method}' not recognized.")
            
        #growth_out["genotype"].append(genotypes)
        growth_out["mut_effect_est"].append(params[0])
        growth_out["mut_effect_std"].append(stds[0])

        #theta_out["genotype"].extend(np.repeat(genotypes[i],len(clean_iptg)))
        #theta_out["iptg"].extend(clean_iptg)
        theta_out["theta_est"].extend(params[1:])
        theta_out["theta_std"].extend(stds[1:])

        #pred_out["genotype"].extend(np.repeat(genotypes[i],clean_X.shape[0]))
        pred_out["marker"].extend(block_marker[valid_mask])
        pred_out["select"].extend(block_select[valid_mask])
        pred_out["iptg"].extend(block_iptg[valid_mask])
        pred_out["k_pred_est"].extend(pred_est)
        pred_out["k_pred_std"].extend(pred_std)

    # Create final DataFrames from the lists of results
    growth_df = pd.DataFrame(growth_out)
    theta_df = pd.DataFrame(theta_out)
    pred_df = pd.DataFrame(pred_out)

    return growth_df, theta_df, pred_df


