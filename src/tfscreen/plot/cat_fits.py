
from tfscreen.plot.helper import clean_arrays
from tfscreen.plot.default_styles import (
    DEFAULT_EXPT_SCATTER_KWARGS,
    DEFAULT_EXPT_ERROR_KWARGS,
    DEFAULT_FIT_LINE_KWARGS,
)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import copy

def cat_fits(x,y,y_std,
             pred_df,
             title=None,
             data_color="black",
             best_model_color="firebrick",
             other_model_color="darkgray",
             err_area_color="lightgray",
             best_model_lw=3,
             other_model_lw=1,
             scatter_kwargs=None,
             error_kwargs=None,
             fit_line_kwargs=None,
             ax=None):
    """
    Plot experimental data and model fits for a given genotype.

    This function generates a plot showing experimental data points with error bars,
    along with model predictions. It distinguishes the best-fit model from other models
    using different colors and line widths.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x-coordinates (e.g., IPTG concentrations).
    y : numpy.ndarray
        Array of y-coordinates (e.g., operator occupancy).
    y_std : numpy.ndarray
        Array of standard deviations for the y-coordinates.
    pred_df : pandas.DataFrame
        DataFrame containing model predictions. Must include columns 'x', 'y', 'y_std',
        'model', and 'is_best_model'.
    title : str, optional
        Put this title on the plot
    data_color : str, optional
        Color for the experimental data points and error bars. Default is "black".
    best_model_color : str, optional
        Color for the line representing the best-fit model. Default is "firebrick".
    other_model_color : str, optional
        Color for the lines representing other models. Default is "darkgray".
    err_area_color : str, optional
        Color for the shaded area representing the uncertainty in the best-fit model.
        Default is "lightgray".
    best_model_lw : int, optional
        Line width for the best-fit model. Default is 3.
    other_model_lw : int, optional
        Line width for the other models. Default is 1.
    scatter_kwargs : dict, optional
        Keyword arguments to pass to the `ax.scatter` function for plotting the data points.
        Default is None (uses default plotting styles).
    error_kwargs : dict, optional
        Keyword arguments to pass to the `ax.errorbar` function for plotting the error bars.
        Default is None (uses default plotting styles).
    fit_line_kwargs : dict, optional
        Keyword arguments to pass to the `ax.plot` function for plotting the model fit lines.
        Default is None (uses default plotting styles).
    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes object to plot on. If None, a new figure and axes
        are created. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.
    """

    # Get rid of nans
    x, y, y_std = clean_arrays(x,y,y_std)
    
    # Add zero point
    x_plot = x.copy()
    x_plot[x_plot == 0] = np.min(x[x_plot > 0])/50
    
    # Load in various plot kwargs
    final_scatter_kwargs = copy.deepcopy(DEFAULT_EXPT_SCATTER_KWARGS)
    for key in ["color","edgecolor","facecolor"]:
        if key in final_scatter_kwargs:
            final_scatter_kwargs.pop(key)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]

    final_error_kwargs = copy.deepcopy(DEFAULT_EXPT_ERROR_KWARGS)
    for key in ["color","edgecolor","facecolor"]:
        if key in final_error_kwargs:
            final_error_kwargs.pop(key)
            
    if error_kwargs is not None:
        for k in error_kwargs:
            final_error_kwargs[k] = error_kwargs[k]
    
    final_fit_line_kwargs = copy.deepcopy(DEFAULT_FIT_LINE_KWARGS)
    for key in ["color","lw"]:
        if key in final_fit_line_kwargs:
            final_fit_line_kwargs.pop(key)
    if fit_line_kwargs is not None:
        for k in fit_line_kwargs:
            final_fit_line_kwargs[k] = fit_line_kwargs[k]
    
    # Build in ax
    if ax is None:
        fig, ax = fig, ax = plt.subplots(1,figsize=(6,6))
    else:
        fig = ax.get_figures()

    # plot data
    ax.scatter(x_plot,y,
               edgecolor=data_color,
               facecolor="none",
               **final_scatter_kwargs,
               zorder=10)
    ax.errorbar(x_plot,y,y_std,
                color=data_color,
                **final_error_kwargs,zorder=9)

    # Go through the model list
    model_list = pd.unique(pred_df["model"])
    for i, m in enumerate(model_list):

        model_df = pred_df.loc[pred_df["model"] == m,:]

        if model_df["is_best_model"].iloc[0]:
            ax.plot(model_df['x'],
                    model_df['y'],
                    '-',
                    color=best_model_color,
                    lw=best_model_lw,
                    label=m,
                    zorder=8)
            ax.fill_between(model_df['x'],
                             model_df['y'] - model_df['y_std'],
                             model_df['y'] + model_df['y_std'],
                             color=err_area_color,
                             zorder=0)
        else:
            ax.plot(model_df['x'],
                    model_df['y'],
                    '-',
                    color=other_model_color,
                    lw=other_model_lw,
                    label=m,
                    zorder=7)

    # Add legend
    ax.legend(frameon=True,
              loc='lower left',
              prop={'size': 10})

    # Clean up plot
    ax.set_xlabel("iptg (mM)")
    ax.set_ylabel("operator occupancy")
    ax.set_xscale("log")
    ax.set_ylim((-0.1,1.1))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # clean up full fig
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, ax
    