
from tfscreen.calibration import (
    read_calibration,
    manual_fit,
    get_background,
    get_wt_k
)
from tfscreen.plot.helper import get_ax_limits

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def growth_rate_fit(obs,
                    obs_std,
                    calc,
                    calc_std,
                    ax=None):
    """
    Plot the fit results (obs vs. calc)
    """

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    min_value, max_value = get_ax_limits(obs,calc,pad_by=0.05,percentile=0)
    
    # Build clean ticks
    min_ceil = np.ceil(min_value)
    max_floor = np.floor(max_value)
    ticks = np.arange(min_ceil,max_floor + 1,1,dtype=int)

    # Plot points and error bars
    ax.scatter(calc,
               obs,
               s=30,
               facecolor='none',
               edgecolor="black",
               zorder=20)
    ax.errorbar(x=calc,
                xerr=calc_std,
                y=obs,
                yerr=obs_std,
                lw=0,
                elinewidth=1,
                capsize=3,
                color='gray')
    ax.plot((min_value,max_value),
            (min_value,max_value),
            '--',
            color='gray',
            zorder=-5)

    # Label axes
    ax.set_xlabel("calculated ln(cfu/mL)")
    ax.set_ylabel("observed ln(cfu/mL)")

    # Make clean axes
    ax.set_xlim(min_value,max_value)
    ax.set_ylim(min_value,max_value)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)    
    ax.set_aspect("equal")

    return ax

def A0_hist(A0,
            ax):
    """
    Plot a histogram of A0
    """
    
    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

    min_edge = np.min(A0)
    if min_edge > 10:
        min_edge = 10

    max_edge = np.max(A0)
    if max_edge < 22:
        max_edge = 22

    # Create histogram
    counts, edges = np.histogram(A0,
                                 bins=np.arange(min_edge,
                                                max_edge,
                                                0.5))

    # Plot histogram
    for i in range(len(counts)):
        ax.fill(np.array([edges[i],
                        edges[i],
                        edges[i+1],
                        edges[i+1]]),
                np.array([0,
                        counts[i],
                        counts[i],
                        0]),
                edgecolor="black",
                facecolor="gray")
        
    # labels
    ax.set_xlabel("ln(A0)")
    ax.set_ylabel("counts")

    return ax

def k_vs_titrant(df,
                 manual_df,
                 calibration_dict,
                 plot_color_dict,
                 ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))

 
    min_titr = np.log(np.min(df.loc[df["titrant_conc"] > 0,"titrant_conc"])/10)
    max_titr = np.log(np.max(df["titrant_conc"])*10)
    titr_span = np.exp(np.linspace(min_titr,max_titr,100))

    if plot_color_dict is None:
        plot_color_dict = {}

    for key, sub_df in manual_df.groupby(["pre_condition","condition","titrant_name"]):

        condition = key[1]
        
        k_bg = get_background(key[2],titrant_conc=titr_span,calibration_data=calibration_dict)
        if condition == "background":
            k_total = k_bg
        else:
            k_total = get_wt_k(key[1],key[2],titrant_conc=titr_span,calibration_data=calibration_dict)

        x = sub_df["titrant_conc"].to_numpy()
        x[x == 0] = min_titr
        y = sub_df["k_est"].to_numpy()
        y_err = sub_df["k_std"].to_numpy()
        
        if condition not in plot_color_dict:
            plot_color_dict[condition] = "gray"

        ax.scatter(x,y,facecolor="none",edgecolor=plot_color_dict[condition])
        ax.errorbar(x,y,y_err,lw=0,capsize=5,elinewidth=1,color=plot_color_dict[condition])
        ax.plot(titr_span,k_total,'-',lw=2,color=plot_color_dict[condition])

    ax.set_xscale('log')
    ax.set_ylim(0,0.03)
    ax.set_xlabel("titrant ln(mM)")
    ax.set_ylabel("growth rate (cfu/mL/min)")

    return ax


def k_pred_corr(manual_df,
                calibration_dict,
                ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))
    
    # Get manual fit values for k_est
    k_man_est = np.array(manual_df["k_est"])
    k_man_std = np.array(manual_df["k_std"])

    # calculate matched k using model
    k_cal_est = get_wt_k(manual_df["condition"],
                         manual_df["titrant_name"],
                         manual_df["titrant_conc"],
                         calibration_data=calibration_dict)
    
    ax.scatter(k_cal_est,
               k_man_est,
               s=30,
               zorder=5,
               facecolor="none",
               edgecolor="gray")
    ax.errorbar(x=k_cal_est,
                y=k_man_est,
                yerr=k_man_std,
                lw=0,capsize=5,
                elinewidth=1,
                ecolor="gray",
                zorder=0)
    

    ax_min, ax_max = get_ax_limits(k_cal_est,k_man_est,pad_by=0.05,percentile=0)

    ax.plot((ax_min,ax_max),(ax_min,ax_max),'--',color='gray',zorder=-5)
    ax.set_xlim(ax_min,ax_max)
    ax.set_ylim(ax_min,ax_max)
    
    ax.set_aspect("equal")
    ax.set_xlabel("k from global model")
    ax.set_ylabel("k from individual fit")

    return ax

def fit_summary(pred_df,
                A0_df,
                calibration_data,
                plot_color_dict=None):
    
    calibration_dict = read_calibration(calibration_data)

    fig, ax = plt.subplots(2,2,figsize=(12,12))
    growth_rate_fit(pred_df["obs_est"],
                    pred_df["obs_std"],
                    pred_df["calc_est"],
                    pred_df["calc_std"],
                    ax=ax[0,0])
    
    A0_hist(A0_df["A0_est"],
            ax=ax[0,1])

    # Do a manual fit of all data used for the calibration
    manual_df = manual_fit(pred_df,calibration_dict)

    k_vs_titrant(pred_df,
                 manual_df,
                 calibration_dict,
                 plot_color_dict,
                 ax[1,0])

    k_pred_corr(manual_df,
                calibration_dict,
                ax[1,1])

    fig.tight_layout()

    return fig, ax


def indiv_replicates(pred_df,rgb_map=None):
    """
    Plot observed and predicted ln(A0) vs. time for all replicates used in the
    calibration.
    
    Parameters
    ----------
    pred_df : pandas.DataFrame
        prediction dataframe returned by calibrate. The function expects the 
        dataframe has columns: "time", "replicate", "titrant_conc", "obs_est",
        "obs_std", and "calc_est". 
    rgb_map : list or None, optional
        rgb_map defines how the series colors should change as a function of 
        log(titrant). Should be a list of three values indicating how the 
        red, green, and blue channels should change vs. log(titrant). Allowed
        values are 'up' (meaning this channel should increase with titrant),
        'down' (meaning this channel should decrease with titrant), or a float
        value between 0 and 1 (meaning this channel should have this fixed
        value).

    Returns
    -------
    fig, ax : 
        matplotlib figures and axes from the plot
    """
    
    # -------------------------------------------------------------------------
    # Define color mapping. 
    
    class PassThrough:
        """
        Class that stores a single value and returns it when called. 
        """
        def __init__(self,v):
            if v < 0 or v > 1:
                raise ValueError(f"{v} must be between 0 and 1\n")
            self._v = v
        def __call__(self,*args):
            return self._v

    # dictionary defines how keys 'up' and down translate to returns
    rgb_mapper = {"up":lambda v: v,
                  "down":lambda v: 1 - v}
    
    # Populate default
    if rgb_map is None:
        rgb_map = ["down",0.5,"up"]

    # Construct the rgb_fcns list which we'll use to build colors for series on
    # the fly. 
    try:
        if hasattr(rgb_map,"__iter__") and len(rgb_map) == 3:
            rgb_fcns = []
            for v in rgb_map:
                if v in rgb_mapper:
                    rgb_fcns.append(rgb_mapper[v])
                else:
                    rgb_fcns.append(PassThrough(v))  
        else:
            raise ValueError("rgb_map incorrectly defined\n")

    except Exception as e:
        err = "rgb_map should be a list of three values. Allowed values\n"
        err += "are 'up' (meaning this channel should increase with titrant)\n"
        err += "'down' (meaning this channel should decrease with titrant),\n"
        err += "or a float value between 0 and 1 (meaning this channel should\n"
        err += "have this fixed value).\n"
        
        raise ValueError(err) from e

    # -------------------------------------------------------------------------
    # Build a  color gradient on titrant (log scale). Use a dataframe with 
    # titrant concentration as its index for speedy lookup when doing actual 
    # plotting. 
    
    # get unique titrant concs, sorted
    titrants = pd.unique(pred_df["titrant_conc"])
    titrants.sort()
    
    # Get value to assign to zero titrant value. 0.1 times minimum non-zero value
    floor = 0.1*np.min(titrants[titrants > 0])

    # Normalize log(titrant) between 0 and 1. 
    norm_values = titrants.copy()
    norm_values[norm_values == 0] = floor 
    norm_values = np.log(norm_values)
    mx = np.max(norm_values)
    mn = np.min(norm_values)
    norm_values = (norm_values - mn)/(mx - mn)
    
    # Get RGBA values for these norm values.
    rgba = [(rgb_fcns[0](v),
             rgb_fcns[1](v),
             rgb_fcns[2](v),1)
             for v in norm_values]
    
    # Build fast lookup rgba_df
    rgba_df = pd.DataFrame({"titrant":titrants,
                            "color":rgba})
    rgba_df = rgba_df.set_index("titrant")

    # -------------------------------------------------------------------------
    # We're going to plot each replicate on its own subplot. Figure out global 
    # plot information.
    
    # Figure out number of plots to create
    size = int(np.ceil(np.sqrt(len(pred_df.groupby(["replicate"])))))
    
    # Get axis limits
    min_x, max_x = get_ax_limits(pred_df["time"],
                                 pred_df["time"],
                                 pad_by=0.05,percentile=0)
    min_y, max_y = get_ax_limits(pred_df["obs_est"],
                                 pred_df["calc_est"],
                                 pad_by=0.05,percentile=0)
        
    fig, axes = plt.subplots(size,size,figsize=(12,12),sharex=True,sharey=True)

    # -------------------------------------------------------------------------
    # Go through all replicates and generate plots
    
    row_counter = 0
    col_counter = 0
    for key, sub_df in pred_df.groupby(["replicate"]):

        # Axis for plot
        ax = axes[row_counter,col_counter]

        # Go through all conditions...
        condition = sub_df[["condition"]].drop_duplicates()["condition"].iloc[0]
        for _, cond_df in sub_df.groupby(["titrant_conc"]):
    
            # Get values to plot
            x = cond_df["time"].to_numpy()
            y = cond_df["obs_est"].to_numpy()
            y_std = cond_df["obs_std"].to_numpy()
            
            # Get color and titrant label
            t = cond_df["titrant_conc"].iloc[0]
            color = tuple(rgba_df.loc[t,"color"])
            label = f"titrant: {t:.3f}"
            
            # Plot data and fit values
            ax.scatter(x,y,s=30,facecolor="none",edgecolor=color,label=label)
            ax.errorbar(x,y,y_std,lw=0,color=color,capsize=5,elinewidth=1)
            ax.plot(cond_df["time"],cond_df["calc_est"],'-',color=color,lw=2)

            # Put legend on top-left plot
            if row_counter == 0 and col_counter == 0:
                ax.legend(frameon=False,fontsize=8)

        # Define axes
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Label y-axis if left column
        if col_counter == 0:
            ax.set_ylabel("ln(CFU/mL)")
    
        # Label x-axis if bottom row
        if row_counter == size - 1:
            ax.set_xlabel("time (min)")
        
        # Set title
        ax.set_title(f"{condition}, replicate {key[0]}")
        
        # update row/column counters
        col_counter += 1
        if col_counter == size:
            col_counter = 0
            row_counter += 1

    # Fill in empty plots
    while row_counter < size:
        
        # Turn off axes
        axes[row_counter,col_counter].axis("off")

        # If we're on the bottom row, put the x-axis label on the plot above
        if row_counter != 0:
            axes[row_counter-1,col_counter].set_xlabel("time (min)")            

        col_counter += 1
        if col_counter == size:
            col_counter = 0
            row_counter += 1
    
    fig.tight_layout()

    return fig, ax