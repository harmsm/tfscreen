
from tfscreen.fitting import matrix_wls
from tfscreen.calibration import perform_calibration
from tfscreen.calibration import write_calibration
from tfscreen.calibration import predict_growth_rate

from tfscreen.util import read_dataframe

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COLOR_DICT = {("kanR",0):"wheat",
                ("kanR",1):"orange",
                ("pheS",0):"lightgreen",
                ("pheS",1):"darkgreen",
                ("none",0):"gray"}
PRETTY_SELECT = {"kanR":"kanamycin",
                 "pheS":"4CP"}

def _plot_growth_rate_fit(obs,
                          obs_std,
                          calc,
                          calc_std,
                          ax):
    """Plot the fit results (obs vs. calc)"""

    # Get min and max for plots (assumes min and max are both positive)
    min_value = np.min([np.min(obs),np.min(calc)])*0.95
    max_value = np.max([np.max(obs),np.max(calc)])*1.05

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

def _plot_A0_hist(A0,
                  ax):
    """Plot a histogram of A0"""
    
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

def _plot_k_vs_iptg(k_df,
                    calibration_dict,
                    ax):

    sim_iptg = 10**np.linspace(-5,0,100)
    
    for m in pd.unique(k_df["marker"]):

        m_df = k_df[k_df["marker"] == m]

        for s in pd.unique(m_df["select"]):

            series = (m,s)        
            
            # Figure out label for legend
            if s == 1:
                operator = "+"
            else:
                operator = "-"
    
            if m == "none":
                label = "base"
            else:
                label = f"{m} {operator} {PRETTY_SELECT[m]}"

            ms_df = m_df[m_df["select"] == s].copy()
            ms_df.loc[ms_df["iptg"] == 0,"iptg"] = 1e-5
            
            ax.errorbar(ms_df["iptg"],
                        ms_df["k_est"],
                        ms_df["k_std"],
                        lw=0,
                        elinewidth=1,
                        capsize=5,
                        color=COLOR_DICT[series])
            ax.scatter(ms_df["iptg"],
                       ms_df["k_est"],
                       facecolor="none",
                       edgecolor=COLOR_DICT[series])
            
            sim_marker = np.repeat(m,sim_iptg.shape[0])
            sim_select = np.repeat(s,sim_iptg.shape[0])
            
            pred, _ = predict_growth_rate(sim_marker,
                                          sim_select,
                                          sim_iptg,
                                          calibration_dict)
            ax.plot(sim_iptg,pred,'-',lw=2,color=COLOR_DICT[series],label=label)


    k_est = k_df["k_est"]
    k_std = k_df["k_std"]
    y_min = np.min(k_est-k_std)
    if y_min < 0:
        y_min = y_min*1.05
    else:
        y_min = 0 

    y_max = np.max(k_est+k_std)
    if y_max < 0:
        y_max = y_max*0.95
    else:
        y_max = y_max*1.05

    ax.set_ylim((y_min,y_max))

    # Clean up plot
    ax.set_xscale('log')
    ax.set_xlabel("[iptg] (mM)")
    ax.set_ylabel("growth rate (cfu/mL/min)")
    ax.legend()

def _plot_k_pred_corr(k_df,
                      calibration_dict,
                      ax):
    
    k_est = np.array(k_df["k_est"])
    k_std = np.array(k_df["k_std"])

    k_calc, k_calc_std = predict_growth_rate(k_df["marker"],
                                             k_df["select"],
                                             k_df["iptg"],
                                             calibration_dict)

    edgecolor = [COLOR_DICT[k] for k in zip(k_df["marker"],k_df["select"])]
    ax.scatter(k_calc,
               k_est,
               s=30,
               zorder=5,
               facecolor="none",
               edgecolor=edgecolor)
    ax.errorbar(x=k_calc,
                y=k_est,
                yerr=k_std,
                xerr=k_calc_std,
                lw=0,capsize=5,
                elinewidth=1,
                ecolor="gray",
                zorder=0)
    
    x_min = np.min([np.min(k_est-k_std),np.min(k_calc-k_calc_std)])
    if x_min < 0:
        x_min = x_min*1.05
    else:
        x_min = 0 

    x_max = np.max([np.max(k_est+k_std),np.max(k_calc+k_calc_std)])
    if x_min < 0:
        x_max = x_max*0.95
    else:
        x_max = x_max*1.05

    ax.plot((x_min,x_max),(x_min,x_max),'--',color='gray',zorder=-5)
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(x_min,x_max)
    
    ax.set_aspect("equal")
    ax.set_xlabel("predicted growth rate")
    ax.set_ylabel("observed growth rate (cfu/mL/min)")


def _get_growth_rates(df,
                      offset=0,
                      ax_growth=None,
                      ax_hist=None):
    """
    Get growth rates by nonlinear least squares given cfu_per_mL vs. 
    time data. 

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe with input data. The function expects the columns 
        'day', 'marker', 'select', 'iptg', 'cfu_per_mL', and 'cfu_per_mL_std'.
    offset : float, default = 0
        offset time by this amount (for modeling global lag).
    ax_growth : matplotlib.Axis, default = None
        if specified, plot the fit results on this axis
    ax_hist : matplotlib.Axis, default = None
        if specified, plot a histogram of ln(A0)
        
    Returns
    -------
    A0_df : pandas.DataFrame
        dataframe with fit results for initial cell populations for each
        experiment. It will have day, marker, select, A0_est (estimated 
        initial population), and A0_std (standard errors on the estimated 
        initial population).
    k_df : pandas.DataFrame
        dataframe with fit results for growth rates under specific conditions.
        It will have marker, select, iptg, k_est (estimated growth rates), and
        k_std (standard errors on the estimated growth rates). 
    """

    time = np.array(df["time"],dtype=float) + offset

    cfu = np.array(df["cfu_per_mL"])
    cfu_std = np.array(df["cfu_per_mL_std"])
    cfu_var = cfu_std**2

    obs = np.log(cfu)
    obs_std = cfu_var/(cfu**2)
    weights = (1/obs_std)**2

    # Every day/marker/select combo will have a global A0. Figure out the 
    # unique combinations and create a dictionary mapping day/marker/select
    # to index. 
    A0_keys = list(zip(df["day"],df["marker"],df["select"]))
    unique_A0_keys = list(set(A0_keys))
    unique_A0_keys.sort()
    A0_key_to_idx = dict([(k,i) for i, k in enumerate(unique_A0_keys)])
    A0_idx = np.array([A0_key_to_idx[k] for k in A0_keys])
    
    k_keys = list(zip(df["marker"],df["select"],df["iptg"]))
    unique_k_keys = list(set(k_keys))
    unique_k_keys.sort()
    k_key_to_idx = dict([(k,i + np.max(A0_idx) + 1) for i, k in enumerate(unique_k_keys)])
    k_idx = np.array([k_key_to_idx[k] for k in k_keys])
    
    # Create empty design matrix
    num_obs = obs.shape[0]
    num_param = np.max(k_idx) + 1
    row_indexer = np.arange(num_obs,dtype=int)
    X = np.zeros((num_obs,num_param),dtype=float)
    
    # Associate appropriate A0 and k with appropriate rows
    X[row_indexer,A0_idx] = 1
    X[row_indexer,k_idx] = time

    # Do weighted linear regression 
    parameters, cov = matrix_wls(X,obs,weights)
    std_errors = np.sqrt(np.diag(cov))

    # Calculate predicted ln(cfu) and their standard errors
    calc = X @ parameters
    calc_var_matrix = X @ cov @ X.T
    calc_std = np.sqrt(np.diag(calc_var_matrix))

    # Extract A0 per experiment
    A0_out = {"day":[],
              "marker":[],
              "select":[],
              "lnA0_est":[],
              "lnA0_std":[]}
    for k in A0_key_to_idx:
        A0_out["day"].append(k[0])
        A0_out["marker"].append(k[1])
        A0_out["select"].append(k[2])
        A0_out["lnA0_est"].append(parameters[A0_key_to_idx[k]])
        A0_out["lnA0_std"].append(std_errors[A0_key_to_idx[k]])

    # Extract k per condition
    k_out = {"marker":[],
             "select":[],
             "iptg":[],
             "k_est":[],
             "k_std":[]}
    for k in k_key_to_idx:
        k_out["marker"].append(k[0])
        k_out["select"].append(k[1])
        k_out["iptg"].append(float(k[2]))
        k_out["k_est"].append(parameters[k_key_to_idx[k]])
        k_out["k_std"].append(std_errors[k_key_to_idx[k]])

    # Plot if requested
    if ax_growth is not None:

        _plot_growth_rate_fit(obs,
                              obs_std,
                              calc,
                              calc_std,
                              ax_growth)
    
    # Plot if requested
    if ax_hist is not None:
        _plot_A0_hist(np.array(A0_out["lnA0_est"]),ax_hist)

    # Create dataframes
    A0_df = pd.DataFrame(A0_out)
    k_df = pd.DataFrame(k_out)

    return A0_df, k_df

def calibrate(calibration_file,
              output_root,
              K,
              n,
              log_iptg_offset=1e-6,
              offset=0,
              plot_fit=True):
    """
    Parameters
    ----------
    calibration_file : pd.DataFrame or str
        dataframe or path to spreadsheet with 
    output_root : str
        prefix to use for writing out calibration files
    K : float
        binding coefficient for the operator occupancy Hill model
    n : float
        Hill coefficient for the operator occupancy Hill model
    log_iptg_offset : float, default=1e-6
        add this to value to all iptg concentrations before taking the log 
        to fit the linear model

    Returns
    -------
    calibration : dict
        calibration dictionary
    fig : matplotlib.Figure
        figure object
    ax : matplotlib.Axis
        axis object
    """

    if plot_fit:
        fig, axes = plt.subplots(2,2,figsize=(12,12))
        ax_growth = axes[0,0]
        ax_hist = axes[0,1]
        ax_k_vs_iptg = axes[1,0]
        ax_k_pred_corr = axes[1,1]
    else:
        ax_growth = None
        ax_hist = None
        ax_k_vs_iptg = None
        ax_k_pred_corr = None

    # read in growth file
    df = read_dataframe(calibration_file)

    # Extract growth rates
    A0_df, k_df = _get_growth_rates(df,
                                    offset=offset,
                                    ax_growth=ax_growth,
                                    ax_hist=ax_hist)

    # Do calibration
    calibration_dict = perform_calibration(k_df,
                                           K=K,
                                           n=n,
                                           log_iptg_offset=log_iptg_offset)

    # Write out calibration file
    calibration_file = f"{output_root}.json"
    write_calibration(calibration_dict=calibration_dict,
                      json_file=calibration_file)

    # Recording out
    out = {"A0_df":A0_df,
           "k_df":k_df,
           "calibration":calibration_dict,
           "calibration_file":calibration_file}

    # Plot the results if requested and finalize
    if plot_fit:
        _plot_k_vs_iptg(k_df,
                        calibration_dict,
                        ax_k_vs_iptg)
        _plot_k_pred_corr(k_df,
                        calibration_dict,
                        ax_k_pred_corr)
        fig.tight_layout()

        out["fig"] = fig
        out["axes"] = axes

    return out
        


