
from tfscreen.plot.default_styles import DEFAULT_SCATTER_KWARGS
from tfscreen.plot.helper import (
    get_ax_limits,
    clean_arrays,
    subsample_index
)

from matplotlib import pyplot as plt


import copy

def xy_corr(
        x_values,
        y_values,
        percentile=0.005,
        pad_by=0.01,
        subsample=10000,
        scatter_kwargs=None,
        ax=None
    ):
    """
    Plot the correlation between two sets of values.

    Parameters
    ----------
    x_values : array_like
        Values for the x-axis.
    y_values : array_like
        Values for the y-axis.
    percentile : float, optional
        set axis limits to cover only the middle percentile of the data (removes
        wild outliers)
    pad_by : float, optional
        pad axis limits by this amount
    subsample : int, optional
        limit the number of points plotted to subsample
    scatter_kwargs : dict, optional
        Keyword arguments to pass to the scatter plot.
    ax : matplotlib.axes._axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object with the plot.
    """

    # Clean up and subsample if needed
    x_values, y_values = clean_arrays(x_values,y_values)
    idx = subsample_index(x_values,subsample)
    x_values = x_values[idx]
    y_values = y_values[idx]

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]
    
    ax_min, ax_max = get_ax_limits(x_values=x_values,
                                   y_values=y_values,
                                   percentile=percentile,
                                   pad_by=pad_by)

    ax.scatter(x_values,
               y_values,
               **final_scatter_kwargs)
    ax.plot([ax_min,ax_max],[ax_min,ax_max],'--',color='gray',lw=2,zorder=5)
    
    ax.set_xlim(ax_min,ax_max)
    ax.set_ylim(ax_min,ax_max)
    ax.set_aspect('equal', adjustable='box')

    return ax