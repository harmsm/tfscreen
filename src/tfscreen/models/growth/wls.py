from tfscreen.fitting import (
    run_wls_2D
)

import numpy as np
import pandas as pd

def wls(t_sel,
        ln_cfu,
        ln_cfu_var):
    """
    Estimate growth rates using weighted least squares regression (WLS) on
    log-transformed CFU/mL. The weights are based on the variance of the
    log-transformed data.

    Parameters
    ----------
    t_sel : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).
    ln_cfu_var : np.ndarray
        2D array of variance of the estimate of ln_cfu each genotype, 
        shape (num_genotypes, num_times).

    Returns
    -------
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std)
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    _results = run_wls_2D(x_arrays=t_sel,
                          y_arrays=ln_cfu,
                          y_err_arrays=ln_cfu_var)

    k_est = _results[0]
    A0_est = _results[1]
    k_std = _results[2]
    A0_std = _results[3]

    param_out = {"lnA0_est":A0_est,
                 "lnA0_std":A0_std,
                 "k_est":k_est,
                 "k_std":k_std}
    param_df = pd.DataFrame(param_out)

    pred = t_sel*k_est[:,np.newaxis] + A0_est[:,np.newaxis]
    pred_out = {"obs":ln_cfu.flatten(),
                "pred":pred.flatten()}
    pred_df = pd.DataFrame(pred_out)

    return param_df, pred_df

    
