from tfscreen.plot.default_styles import DEFAULT_SCATTER_KWARGS

from tfscreen.plot.helper import (
    get_ax_limits,
    clean_arrays,
    subsample_index
)

from matplotlib import pyplot as plt
import numpy as np

import copy

def uncertainty_calibration(
        param_est,
        param_std,
        param_real,
        percentile=0.005,
        pad_by=0.0,
        subsample=10000,
        scatter_kwargs=None,
        ax=None
    ):
    """
    Create an uncertainty calibration plot. This plots parameter standard error
    against the (estimated - real) values of those parameters. 

    A well-calibrated model will produce a plot that looks like a funnel
    centered on zero. The points should be symmetrically distributed around the
    vertical line at x=0. If the center of the cloud of points is shifted to the
    left or right, it indicates your fitting procedure has a bias (it
    consistently over- or underestimates the parameter). When the standard error
    is small (bottom of the plot), the actual estimation error
    (param_est - param_real) should also be very small, tightly clustered around
    zero. As the standard error gets larger (moving up the plot), the spread of
    the actual estimation error should increase proportionally. The model is
    correctly reporting that it is less certain about these estimates, and the
    wider spread of errors confirms this.

    Parameters
    ----------
    param_est : array_like
        The estimated values.
    param_std : array_like
        The standard error on the estimated parameters.
    param_real : array_like
        The true values.
    percentile : float, optional
        set axis limits to cover only the middle percentile of the data (removes
        wild outliers)
    pad_by : float, optional
        pad axis limits by this amount
    scatter_kwargs : dict, optional
        Keyword arguments to pass to the scatter plot.
    ax : matplotlib.axes._axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object with the plot.
    """

    param_est, param_std, param_real = clean_arrays(param_est,
                                                    param_real,
                                                    param_std)
    idx = subsample_index(param_est,subsample)
    param_est = param_est[idx]
    param_std = param_std[idx]
    param_real = param_real[idx]

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]

    diff = (param_est - param_real)

    #abs_diffs = np.abs(diff)

    ax_max, _ = get_ax_limits(diff,
                              percentile=percentile,
                              pad_by=pad_by)

    ax.scatter(diff,param_std,**final_scatter_kwargs)
    ax.set_xlabel("est_value - real_value")
    ax.set_ylabel("est_err")
    ax.plot([0,0],[0,2*ax_max],'--',color='gray',zorder=-20)

    for i in range(1,4):
        ax.plot([0,ax_max],[0,ax_max/i],'--',color='gray',zorder=-20)
        ax.plot([-ax_max,0],[ax_max/i,0],'--',color='gray',zorder=-20)
    
    ax.set_xlim(-ax_max,ax_max)
    ax.set_ylim(0,ax_max*2)
    ax.set_aspect('equal', adjustable='box')

    return ax