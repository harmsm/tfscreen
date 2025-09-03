
from tfscreen.fitting.linear_regression import linear_regression

import pandas as pd
import numpy as np

def get_growth_rates_ols(times,ln_cfu):
    """
    Estimate growth rates using ordinary least squares regression on
    log-transformed CFU/mL data. 

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).

    Returns
    -------
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std)
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    _results = linear_regression(x_arrays=times,
                                      y_arrays=ln_cfu)

    k_est = _results[0]
    A0_est = _results[1]
    k_std = _results[2]
    A0_std = _results[3]

    param_out = {"A0_est":A0_est,
                 "A0_std":A0_std,
                 "k_est":k_est,
                 "k_std":k_std}
    param_df = pd.DataFrame(param_out)

    pred = times*k_est[:,np.newaxis] + A0_est[:,np.newaxis]
    pred_out = {"obs":ln_cfu.flatten(),
                "pred":pred.flatten()}
    pred_df = pd.DataFrame(pred_out)

    return param_df, pred_df