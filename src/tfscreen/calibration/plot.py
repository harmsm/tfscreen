
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

def fit_summary(df,
                pred_df,
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
    manual_df = manual_fit(df,calibration_dict)

    k_vs_titrant(df,
                 manual_df,
                 calibration_dict,
                 plot_color_dict,
                 ax[1,0])

    k_pred_corr(manual_df,
                calibration_dict,
                ax[1,1])

    fig.tight_layout()

    return fig, ax