import numpy as np

def _build_design_matrix(marker,
                         select,
                         iptg,
                         K,
                         n,
                         log_iptg_offset):
    """
    Build a design matrix in a stereotyped way. 

    Parameters
    ----------
    marker : np.ndarray
        1D array of condition markers (the special value 'none' is ignored). 
    select : np.ndarray
        1D array of selection state for each condition 
    iptg : np.ndarray
        1D array of iptg concentration for each condition
    K : float
        binding coefficient for the operator occupancy Hill model
    n : float
        Hill coefficient for the operator occupancy Hill model
    log_iptg_offset : float
        add this to value to all iptg concentrations before taking the log 
        to fit the linear model

    Methods
    -------
    param_names : list
        list of parameter names
    X : np.ndarray
        design matrix (num_conditions by num parameters array)
    """

    # -------------------------------------------------------------------------
    # Build a list of all parameters needed to describe the data passed in.

    # Always have a base growth intercept and slope
    param_names = ["base_b",
                   "base_m"]

    # Get unique marker and selection values and sort. Ignore "none" markers.
    unique_marker = np.unique(marker)
    unique_marker.sort()
    unique_marker = unique_marker[unique_marker != "none"]

    unique_select = np.unique(select)
    unique_select.sort()

    # Create a parameter intercept/slope pair for every marker,selector 
    # combo seen. 
    combos_seen = set(list(zip(marker,select)))
    for m in unique_marker:
        for s in unique_select:
            if (m,s) in combos_seen:
                param_names.append(f"{m}-{s}_b")
                param_names.append(f"{m}-{s}_m")

    # Build the empty design matrix
    num_params = len(param_names)    
    num_conditions = len(iptg)
    X = np.zeros((num_conditions,num_params),dtype=float)

    # This indexer can be masked to select specific rows
    row_indexer = np.arange(num_conditions,dtype=int)

    # Calculate log_iptg (for base) and theta (for all others) as our 
    # x-axis
    K = float(K)
    n = float(n)
    log_iptg = np.log(iptg + float(log_iptg_offset))
    theta = (iptg**n)/(K**n + iptg**n)

    # base growth intercept (@ log(iptg + log_iptg_offset) == 0)
    X[:,0] = 1

    # base growth slope. The independent variable is log_iptg. 
    X[:,1] = log_iptg

    # Now go through remaining combinations and add slope and intercept 
    # parameters. The independent variable is theta. 
    param_counter = 2
    for m in unique_marker:
        for s in unique_select:
            if (m,s) in combos_seen:
                mask = np.logical_and(marker == m,select == s)
                X[row_indexer[mask],param_counter] = 1
                X[row_indexer[mask],param_counter + 1] = theta[mask]
                param_counter += 2

    return param_names, X
                

def _wls_estimate(X,
                  y,
                  weights):
    """
    Estimate parameters by weighted linear least squares.

    Parameters
    ----------
    X : np.ndarray
        design matrix
    y : np.ndarray
        observed values
    weights : np.ndarray
        weights for regression (typically 1/k_std^2)

    Returns
    -------
    parameters : np.ndarray
        1D float array of parameter estimates
    cov_matrix : np.ndarray
        2D float array of parameter estimate covariance
    """
    
    # Calculate X^T * W * X and X^T * W * y
    # (w[:, np.newaxis] * X) is an efficient way to apply weights row-wise
    XTWX = X.T @ (weights[:, np.newaxis] * X)
    XTWy = X.T @ (weights * y)

    # Solve for parameters: beta = (X^T * W * X)^-1 * (X^T * W * y)
    # np.linalg.solve is more stable and faster than computing the inverse
    parameters = np.linalg.solve(XTWX, XTWy)

    # Get total residuals
    residuals = y - X @ parameters
    
    # Degrees of freedom (num observations - num parameters)
    dof = X.shape[0] - len(parameters)
    
    # Calculate reduced chi-squared (variance of unit weight)
    chi2_red = np.sum(weights * residuals**2) / dof
    
    # Parameter covariance matrix is chi2_red * (X^T * W * X)^-1
    cov_matrix = chi2_red * np.linalg.inv(XTWX)
    
    return parameters, cov_matrix

                
def perform_calibration(df,K=0.015854,n=2,log_iptg_offset=1e-6):
    """
    Build a calibration dictionary given a dataframe of observed growth rates
    under specific conditions. 

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with experimental data. The function expects the dataframe
        will have columns 'marker', 'select', 'iptg', 'k_est', and 'k_std'.
    K : float, default = 0.015854
        binding coefficient for the operator occupancy Hill model
    n : float, default = 2
        Hill coefficient for the operator occupancy Hill model
    log_iptg_offset : float, default = 1e-6
        add this to value to all iptg concentrations before taking the log 
        to fit the linear model

    Returns
    -------
    calibration : dict
        a dictionary holding calibration values. it has the following keys:
        + "param_names" : list of parameter names
        + "param_values": np.ndarray of float parameter values
        + "cov_matrix": np.ndarray of the covariance matrix of the parameter
          estimates
        + "K": binding coefficient for the operator occupancy Hill model used
          in the calibration
        + "n": Hill coefficient for the operator occupancy Hill model used in
          the calibration
        + "log_iptg_offset": add this to value to all iptg concentrations
          before taking the log to fit the linear model
    """
        
    # Get relevant values from input dataframe
    marker = np.array(df["marker"])
    select = np.array(df["select"])
    iptg = np.array(df["iptg"])

    param_names, X = _build_design_matrix(marker=marker,
                                          select=select,
                                          iptg=iptg,
                                          K=K,
                                          n=n,
                                          log_iptg_offset=log_iptg_offset)
    
    k_est = np.array(df["k_est"]) 
    k_std = np.array(df["k_std"]) 
    
    values, cov = _wls_estimate(X,k_est,(1/k_std**2))
    
    return {"param_names":param_names,
            "param_values":values,
            "cov_matrix":cov,
            "K":K,
            "n":n,
            "log_iptg_offset":log_iptg_offset}

def predict_growth_rate(marker,
                        select,
                        iptg,
                        calibration):
    """
    Predict the growth rate of a wildtype clone under specific conditions. 

    Parameters
    ----------
    marker : np.ndarray
        1D array of condition markers (the special value 'none' is ignored). 
    select : np.ndarray
        1D array of selection state for each condition 
    iptg : np.ndarray
        1D array of iptg concentration for each condition
    calibration : dict
        a dictionary holding calibration values

    Returns
    -------
    y_est : np.ndarray
        predicted growth rates
    y_std : np.ndarray
        standard error on the predicted growth rates
    """

    param_names, X_pred = _build_design_matrix(marker=marker,
                                               select=select,
                                               iptg=iptg,
                                               K=calibration["K"],
                                               n=calibration["n"],
                                               log_iptg_offset=calibration["log_iptg_offset"])
    
    if tuple(param_names) != tuple(calibration["param_names"]):
        print("Warning. param name mismatch between inputs and calibration.")
        print("Inferred param_names",param_names)
        print("Calibration param_names",calibration["param_names"])
        print("This could mean the model does not describe your data.",flush=True)

    y_est = X_pred @ calibration["param_values"]
    y_var_matrix = X_pred @ calibration["cov_matrix"] @ X_pred.T
    y_std = np.sqrt(np.diag(y_var_matrix))
    
    return y_est, y_std 