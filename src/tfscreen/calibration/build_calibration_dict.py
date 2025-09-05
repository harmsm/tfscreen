
from tfscreen.calibration import build_design_matrix
from tfscreen.fitting import matrix_wls

import numpy as np

def build_calibration_dict(df,K=0.015854,n=2,log_iptg_offset=1e-6):
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
    param_names, X = build_design_matrix(marker=marker,
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