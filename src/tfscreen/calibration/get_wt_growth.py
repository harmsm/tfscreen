from tfscreen.calibration import build_design_matrix

import numpy as np


def get_wt_growth(marker,
                  select,
                  iptg,
                  calibration_dict,
                  theta=None,
                  calc_err=True):
    """
    Get the growth rate of a wildtype clone under the conditions specified 
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

    param_names, X_pred = build_design_matrix(marker=marker,
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