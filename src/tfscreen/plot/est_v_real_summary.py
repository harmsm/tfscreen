
from tfscreen.plot.helper import (
    subsample_index,
    clean_arrays,
)

from tfscreen.plot import (
    xy_corr,
    uncertainty_calibration,
    err_zscore
)


import numpy as np
from matplotlib import pyplot as plt

def est_v_real_summary(
        param_est,
        param_std,
        param_real,
        axis_prefix=None,
        suptitle=None,
        subsample=10000,
        scatter_kwargs=None
    ):
    """
    Generate a summary plot of estimated vs real values.

    Parameters
    ----------
    param_est : array_like
        The estimated parameters.
    param_std : array_like
        The standard error on the estimated parameters.
    param_real : array_like
        The true values.
    axis_prefix : str, optional
        append a prefix to the axis titles
    suptitle : str, optional
        A title for the whole figure.
    subsample : int, optional
        The number of points to subsample for plotting.
    scatter_kwargs : dict, optional
        Keyword arguments to pass to the scatter plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : array_like of matplotlib.axes._axes.Axes
        An array of axes objects.
    """


    param_est, param_std, param_real = clean_arrays(param_est,
                                                    param_std,
                                                    param_real)
    idx = subsample_index(param_est,subsample)
    param_est = param_est[idx]
    param_std = param_std[idx]
    param_real = param_real[idx]

    if axis_prefix is not None:
        est = f"{axis_prefix}_est"
        real = f"{axis_prefix}_real"
        std = f"{axis_prefix}_std"
    else:
        est = "est"
        real = "real"
        std = "std"
         

    fig, ax = plt.subplots(1,3,figsize=(14,6))
    
    _ = xy_corr(param_real,
                param_est,
                scatter_kwargs=scatter_kwargs,
                ax=ax[0])

    ax[0].set_xlabel(real)
    ax[0].set_ylabel(est)
    
    _ = uncertainty_calibration(
            param_est,
            param_std,
            param_real,
            scatter_kwargs=scatter_kwargs,
            ax=ax[1]
        )

    ax[1].set_xlabel(f"{est} - {real}")
    ax[1].set_ylabel(std)

    _ = err_zscore(param_est,
                   param_std,
                   param_real,
                   ax=ax[2])

    ax[2].set_xlabel(f"Z-score ({est} - {real})")
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    fig.tight_layout()
                     
    return fig, ax

