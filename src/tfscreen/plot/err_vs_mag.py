
from tfscreen.plot.default_styles import DEFAULT_SCATTER_KWARGS
from tfscreen.plot.helper import (
    get_ax_limits,
    clean_arrays,
    subsample_index
)

from matplotlib import pyplot as plt

import copy

def err_vs_mag(
        obs,
        pred,
        axis_name="",
        subsample=10000,
        scatter_kwargs=None,
        ax=None
    ):
    """
    Plot the error vs real value. 

    Parameters
    ----------
    obs : array_like
        Observed values.
    pred : array_like
        Predicted values.
    subsample : int, optional
        Number of points to subsample for plotting.
    axis_name : str, optional
        Name to prepend to the axis labels.
    scatter_kwargs : dict, optional
        Keyword arguments to pass to the scatter plot. 
    ax : matplotlib.axes._axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object with the plot.
    """

    obs, pred = clean_arrays(obs,pred)
    idx = subsample_index(obs,subsample)
    obs = obs[idx]
    pred = pred[idx]

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]

    

    x_min, x_max = get_ax_limits(obs)
    span = x_max - x_min
    
    ax.scatter(obs,
               pred - obs,
               **final_scatter_kwargs)
    
    ax.plot([x_min,x_max],[0,0],'--',color='gray',zorder=-5)
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(-span/2,span/2)
    ax.set_aspect('equal')
    
    ax.set_xlabel(f"{axis_name} obs")
    ax.set_ylabel(f"{axis_name} predicted - obs")

    return ax