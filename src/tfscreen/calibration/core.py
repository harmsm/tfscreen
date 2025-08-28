from tfscreen.fitting import matrix_wls

import numpy as np

def _build_design_matrix(marker,
                         select,
                         iptg,
                         theta=None,
                         K=None,
                         n=None,
                         log_iptg_offset=1e-6,
                         param_names=None):
    """
    Build a design matrix in a stereotyped way. 

    Parameters
    ----------
    marker : np.ndarray
        1D array of sample markers (the special value 'none' is ignored). 
    select : np.ndarray
        1D array of selection state for each sample 
    iptg : np.ndarray
        1D array of iptg concentration for each sample
    K : float
        binding coefficient for the operator occupancy Hill model
    n : float
        Hill coefficient for the operator occupancy Hill model
    log_iptg_offset : float
        add this to value to all iptg concentrations before taking the log 
        to fit the linear model
    param_names : np.ndarray or None, default=None
        if param_names is passed in, this is treated as the names of each 
        column in the design matrix
        
    Methods
    -------
    param_names : list
        list of parameter names
    X : np.ndarray
        design matrix (num_samples by num parameters array)
    """

    # Get unique marker and selection values and sort. Ignore "none" markers.
    unique_marker = np.unique(marker)
    unique_marker.sort()
    unique_marker = unique_marker[unique_marker != "none"]

    unique_select = np.unique(select)
    unique_select.sort()

    # Get all marker/select combos seen in the data
    combos_seen = set(list(zip(marker,select)))

    if param_names is None:

          # Always have a base growth intercept and slope
        param_names = ["base|b",
                       "base|m"]

        # Create a parameter intercept/slope pair for every marker,selector 
        # combo seen. 
        for m in unique_marker:
            for s in unique_select:
                if (m,s) in combos_seen:
                    param_names.append(f"{m}|{s}|b")
                    param_names.append(f"{m}|{s}|m")


    param_name_dict = dict([(p,i) for i, p in enumerate(param_names)])

    # Build the empty design matrix
    num_params = len(param_names) 
    num_samples = len(iptg)
    X = np.zeros((num_samples,num_params),dtype=float)

    # This indexer can be masked to select specific rows
    row_indexer = np.arange(num_samples,dtype=int)

    # If theta was not specified, calculate it from K and n
    if theta is None:

        if K is None or n is None:
            err = "K and n must be specified if theta is not\n"
            raise ValueError(err)
        
        K = float(K)
        n = float(n)
        theta = (iptg**n)/(K**n + iptg**n)

    
    # base growth intercept (@ log(iptg + log_iptg_offset) == 0)
    X[:,0] = 1

    # base growth slope. The independent variable is log_iptg. 
    X[:,1] = np.log(iptg + float(log_iptg_offset))

    # Now go through remaining combinations and add slope and intercept 
    # parameters. The independent variable is theta. 
    for m in unique_marker:
        for s in unique_select:
            if (m,s) in combos_seen:

                b_idx = param_name_dict[f"{m}|{s}|b"]
                m_idx = param_name_dict[f"{m}|{s}|m"]

                mask = np.logical_and(marker == m,select == s)
                X[row_indexer[mask],b_idx] = 1
                X[row_indexer[mask],m_idx] = theta[mask]
                
    return param_names, X
                
       
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
    calibration_dict : dict
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
    k_est = np.array(df["k_est"]) 
    k_std = np.array(df["k_std"]) 

    # Build the design matrix relating each data point (defined by marker, 
    # selection, and iptg) to its parameters
    param_names, X = _build_design_matrix(marker=marker,
                                          select=select,
                                          iptg=iptg,
                                          K=K,
                                          n=n,
                                          log_iptg_offset=log_iptg_offset)
    
    # Fit the model parameters to the experimental data
    values, cov = matrix_wls(X,k_est,(1/k_std**2))
    
    # Create the output dictionary
    calibration_dict = {"param_names":param_names,
                        "param_values":values,
                        "cov_matrix":cov,
                        "K":K,
                        "n":n,
                        "log_iptg_offset":log_iptg_offset}
    
    return calibration_dict

def predict_growth_rate(marker,
                        select,
                        iptg,
                        calibration_dict,
                        theta=None,
                        calc_err=True):
    """
    Predict the growth rate of a wildtype clone under the conditions specified 
    using the model stored in calibration_dict.

    Parameters
    ----------
    marker : np.ndarray
        1D array of condition markers (the special value 'none' is ignored). 
    select : np.ndarray
        1D array of selection state for each condition 
    iptg : np.ndarray
        1D array of iptg concentration for each condition
    calibration_dict : dict
        a dictionary holding calibration values
    theta : np.ndarray, default=None
        1D array of theta values over the conditions. if None, calculate theta
        from the K and n values in the calibration dictionary

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
                                               theta=theta,
                                               K=calibration_dict["K"],
                                               n=calibration_dict["n"],
                                               log_iptg_offset=calibration_dict["log_iptg_offset"],
                                               param_names=calibration_dict["param_names"])
    
    if tuple(param_names) != tuple(calibration_dict["param_names"]):
        print("Warning. param name mismatch between inputs and calibration.")
        print("Inferred param_names",param_names)
        print("Calibration param_names",calibration_dict["param_names"])
        print("This could mean the model does not describe your data.",flush=True)
        print(X_pred.shape)

    y_est = X_pred @ calibration_dict["param_values"]

    if calc_err: 
        y_var_matrix = X_pred @ calibration_dict["cov_matrix"] @ X_pred.T
        y_std = np.sqrt(np.diag(y_var_matrix))
    else:
        y_std = np.repeat(np.nan,len(y_est))
    
    return y_est, y_std 