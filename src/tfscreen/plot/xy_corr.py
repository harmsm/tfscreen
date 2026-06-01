
from tfscreen.plot.default_styles import DEFAULT_SCATTER_KWARGS, DEFAULT_HEXBIN_KWARGS
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
        as_hexbin=False,
        percentile=0.005,
        pad_by=0.01,
        subsample=10000,
        scatter_kwargs=None,
        hexbin_kwargs=None,
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
    as_hexbin : bool, optional
        If True, render as a hexbin density plot instead of a scatter plot.
        A colorbar is automatically added to the figure.  Defaults to False.
    percentile : float, optional
        Set axis limits to cover only the middle percentile of the data
        (removes wild outliers).
    pad_by : float, optional
        Pad axis limits by this fraction.
    subsample : int, optional
        Limit the number of points plotted to this count.
    scatter_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.scatter``.  These override the
        defaults in ``default_styles.DEFAULT_SCATTER_KWARGS``.  Only used
        when ``as_hexbin=False``.
    hexbin_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.hexbin``.  These override the
        defaults in ``default_styles.DEFAULT_HEXBIN_KWARGS``.  Only used
        when ``as_hexbin=True``.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.  If ``None``, a new 6×6 figure and axes
        are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """

    # Clean up and subsample if needed
    x_values, y_values = clean_arrays(x_values,y_values)
    idx = subsample_index(x_values,subsample)
    x_values = x_values[idx]
    y_values = y_values[idx]

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    ax_min, ax_max = get_ax_limits(x_values=x_values,
                                   y_values=y_values,
                                   percentile=percentile,
                                   pad_by=pad_by)

    if as_hexbin:
        final_hexbin_kwargs = DEFAULT_HEXBIN_KWARGS | (hexbin_kwargs or {})
        hb = ax.hexbin(x_values, y_values,
                       extent=[ax_min, ax_max, ax_min, ax_max],
                       **final_hexbin_kwargs)
        ax.get_figure().colorbar(hb, ax=ax)
    else:
        final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
        if scatter_kwargs is not None:
            for k in scatter_kwargs:
                final_scatter_kwargs[k] = copy.deepcopy(scatter_kwargs[k])
        ax.scatter(x_values,
                y_values,
                **final_scatter_kwargs)

    ax.plot([ax_min,ax_max],[ax_min,ax_max],'--',color='gray',lw=2,zorder=5)

    ax.set_xlim(ax_min,ax_max)
    ax.set_ylim(ax_min,ax_max)
    ax.set_aspect('equal', adjustable='box')

    return ax