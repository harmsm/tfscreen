
from tfscreen.plot.helper import clean_arrays
from tfscreen.plot.default_styles import (
    DEFAULT_EXPT_SCATTER_KWARGS,
    DEFAULT_EXPT_ERROR_KWARGS,
    DEFAULT_FIT_LINE_KWARGS,
)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def cat_fits(x,y,y_std,
             pred_df,
             title=None,
             data_color="black",
             best_model_color="firebrick",
             other_model_color="darkgray",
             err_area_color="lightgray",
             best_model_lw=3,
             other_model_lw=1,
             xlabel=None,
             ylabel=None,
             xlim=None,
             ylim=None,
             xlog=False,
             ylog=False,
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
        Array of x-coordinates (e.g., titrant concentrations).
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
    xlabel : str, optional
        string to use to label the x-axis
    ylabel : str, optional
        string to use to label the y-axis
    xlim : tuple, optional
        min and max for the x-axis
    ylim : tuple, optional
        min and max for the y-axis
    xlog : bool, optional
        if True, use a log scale for the x-axis. Default: False
    ylog : bool, optional
        if True, use a log scale for the y-axis. Default: False
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
    
    # Load in various plot kwargs, overwriting defaults with user dict
    default_fit_line_kwargs = DEFAULT_FIT_LINE_KWARGS.copy()
    default_fit_line_kwargs["zorder"] = 8
    final_fit_line_kwargs = default_fit_line_kwargs | (fit_line_kwargs or {})

    default_expt_scatter_kwargs = DEFAULT_EXPT_SCATTER_KWARGS.copy()
    default_expt_scatter_kwargs["zorder"] = 9
    default_expt_scatter_kwargs["edgecolor"] = data_color
    final_scatter_kwargs = default_expt_scatter_kwargs | (scatter_kwargs or {}) 
    
    default_expt_error_kwargs = DEFAULT_EXPT_ERROR_KWARGS.copy()
    default_expt_error_kwargs["zorder"] = 7
    default_expt_error_kwargs["color"] = data_color
    final_error_kwargs = default_expt_error_kwargs | (error_kwargs or {})
    
    # Build in ax. If ax is already around, grab the fig for a consistent 
    # return. 
    if ax is None:
        fig, ax = fig, ax = plt.subplots(1,figsize=(6,6))
    else:
        fig = ax.get_figures()

    # plot data
    ax.scatter(x_plot,y,**final_scatter_kwargs)

    # Plot error bars
    ax.errorbar(x_plot,y,y_std,**final_error_kwargs)

    # Go through the model list
    model_list = pd.unique(pred_df["model"])
    for i, m in enumerate(model_list):

        model_df = pred_df.loc[pred_df["model"] == m,:]

        if model_df["is_best_model"].iloc[0]:

            final_fit_line_kwargs["color"] = best_model_color
            final_fit_line_kwargs["lw"] = best_model_lw
            final_fit_line_kwargs["label"] = m

            ax.plot(model_df['x'],model_df['y'],'-',
                    **final_fit_line_kwargs)
            
            ax.fill_between(model_df['x'],
                            model_df['y'] - model_df['y_std'],
                            model_df['y'] + model_df['y_std'],
                            color=err_area_color,
                            zorder=0)
        else:

            final_fit_line_kwargs["color"] = other_model_color
            final_fit_line_kwargs["lw"] = other_model_lw
            final_fit_line_kwargs["label"] = m
            
            ax.plot(model_df['x'],model_df['y'],'-',
                    **final_fit_line_kwargs)

    # Add legend
    ax.legend(frameon=True,
              prop={'size': 10})

    # Clean up plot
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # clean up full fig
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, ax
    