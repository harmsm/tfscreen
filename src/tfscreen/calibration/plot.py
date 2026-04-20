
from tfscreen.calibration import (
    read_calibration,
    get_background,
    get_wt_k,
    get_wt_theta
)
from tfscreen.models.growth_linkage import get_model
from tfscreen.models.transition_linkage import get_transition_model
from tfscreen.plot.helper import get_ax_limits
from tfscreen.analysis.independent.get_indiv_growth import get_indiv_growth
from tfscreen.models.occupancy_growth_model import OccupancyGrowthModel

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

    positive_titrants = df.loc[df["titrant_conc"] > 1e-12, "titrant_conc"]
    if len(positive_titrants) > 0:
        min_titr_raw = np.min(positive_titrants)
        max_titr_raw = np.max(positive_titrants)
        
        min_titr = np.log(min_titr_raw / 10)
        max_titr = np.log(max_titr_raw * 10)
        
        # Use a small positive value for 0 titrant in log scale
        floor_titr = min_titr_raw / 100 
    else:
        # If only 0 titrant is present, define a reasonable range
        min_titr = np.log(0.01) # arbitrary small log value
        max_titr = np.log(1.0)  # arbitrary larger log value
        floor_titr = 0.01 # arbitrary small value
    
    titr_span = np.exp(np.linspace(min_titr,max_titr,100))

    if plot_color_dict is None:
        plot_color_dict = {}

    for key, sub_df in manual_df.groupby(["condition_sel","titrant_name"]):

        condition = key[0]
        
        k_bg = get_background(key[1],titrant_conc=titr_span,calibration_data=calibration_dict)
        if condition == "background":
            k_total = k_bg
        else:
            k_total = get_wt_k(key[0],key[1],titrant_conc=titr_span,calibration_data=calibration_dict)

        x = sub_df["titrant_conc"].to_numpy()
        x[x == 0] = floor_titr # Replace 0 with a small positive value for log scale
        y = sub_df["k_est"].to_numpy()
        y_err = sub_df["k_std"].to_numpy()
        
        if condition not in plot_color_dict:
            plot_color_dict[condition] = "gray"

        ax.scatter(x,y,facecolor="none",edgecolor=plot_color_dict[condition])
        ax.errorbar(x,y,y_err,lw=0,capsize=5,elinewidth=1,color=plot_color_dict[condition])
        ax.plot(titr_span,k_total,'-',lw=2,color=plot_color_dict[condition])

    ax.set_xscale('log')
    ax.set_xlim(np.exp(min_titr), np.exp(max_titr))
    
    # Set dynamic y-axis limits based on data
    ax_min, ax_max = get_ax_limits(manual_df["k_est"], center_on_zero=True, pad_by=0.1)
    ax.set_ylim(ax_min, ax_max)

    ax.set_xlabel("titrant conc (mM)")
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

    titrant_conc = manual_df["titrant_conc"].to_numpy(dtype=float)

    # calculate matched k using model
    k_cal_est = get_wt_k(manual_df["condition_sel"].values,
                         manual_df["titrant_name"].values,
                         titrant_conc,
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

def _plot_calibrated_param_vs_titrant(calibration_dict,
                                      param_name="tau",
                                      plot_color_dict=None,
                                      ax=None):
    """
    Plot calibrated parameters vs titrant concentration.
    """
    if ax is None:
        _, ax = plt.subplots(1, figsize=(6, 6))

    if plot_color_dict is None:
        plot_color_dict = {}

    data = calibration_dict.get("dk_cond", {}).get(param_name, {})
    
    # Extract granular items: "cond:titrant:conc" -> value
    items = []
    for k, v in data.items():
        if ":" in k:
            parts = k.split(":")
            if len(parts) == 3:
                try:
                    items.append({"condition": parts[0], 
                                  "titrant": parts[1], 
                                  "conc": float(parts[2]), 
                                  "value": v})
                except ValueError:
                    continue
    
    if not items:
        ax.text(0.5, 0.5, f"No granular {param_name} data", 
                ha='center', va='center', transform=ax.transAxes)
        return ax

    df = pd.DataFrame(items).sort_values("conc")
    
    # Replace 0 conc with small floor for log scale
    concs = df["conc"].unique()
    positive = concs[concs > 0]
    floor = 0.1 * np.min(positive) if len(positive) > 0 else 1.0
    
    # Color map handling
    for key, sub_df in df.groupby(["condition", "titrant"]):
        condition = key[0]
        color = plot_color_dict.get(condition, "gray")
        
        sub_df = sub_df.sort_values("conc")
        x = sub_df["conc"].to_numpy()
        x[x == 0] = floor
        
        # Plot points and lines
        ax.scatter(x, sub_df["value"], facecolor="none", edgecolor=color)
        ax.plot(x, sub_df["value"], '--', lw=1, color=color, label=f"{condition}")

    ax.set_xscale('log')
    ax.set_xlabel("titrant conc (mM)")
    ax.set_ylabel(f"calibrated {param_name}")
    
    return ax

def fit_summary(pred_df,
                param_df,
                calibration_data,
                plot_color_dict=None,
                no_selection_conditions=None):

    if no_selection_conditions is None:
        no_selection_conditions = ["pheS-4CP","kanR-kan"]

    calibration_dict = read_calibration(calibration_data)

    pred_df = pred_df.copy()
    if "genotype" not in pred_df:
        pred_df["genotype"] = "wt"
    
    pred_df["dk_geno_mask"] = pred_df["condition_sel"].isin(no_selection_conditions)

    trans_model_name = calibration_dict.get("transition_model_name", "constant")
    trans_model = get_transition_model(trans_model_name)
    trans_param_defs = trans_model.get_param_defs()
    num_trans_params = len(trans_param_defs)

    per_titrant_tau = calibration_dict.get("per_titrant_tau", False)
    if per_titrant_tau and num_trans_params > 0:
        # Determine grid size
        num_rows = 2 + int(np.ceil(num_trans_params / 2))
        fig, ax = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Filter rows that have observations (drop -t_pre rows)
    mask = ~np.isnan(pred_df["y_obs"])
    clean_pred_df = pred_df.loc[mask,:].copy()

    growth_rate_fit(clean_pred_df["y_obs"],
                    clean_pred_df["y_std"],
                    clean_pred_df["calc_est"],
                    clean_pred_df["calc_std"],
                    ax=ax[0,0])
    
    A0_est = param_df.loc[param_df["class"] == "ln_cfu_0","est"]

    A0_hist(A0_est,ax=ax[0,1])

    manual_param_df, manual_pre_df = get_indiv_growth(
        clean_pred_df,
        series_selector=["replicate","condition_sel","titrant_name","titrant_conc"],
        calibration_data=calibration_dict,
        dk_geno_selector=["genotype"],
        dk_geno_mask_col="dk_geno_mask",
        lnA0_selector=["replicate"],
        num_iterations=3
    )

    k_vs_titrant(pred_df,
                 manual_param_df,
                 calibration_dict,
                 plot_color_dict,
                 ax[1,0])

    k_pred_corr(manual_param_df,
                calibration_dict,
                ax[1,1])
    
    if per_titrant_tau and num_trans_params > 0:
        for i, (suffix, _, _, _) in enumerate(trans_param_defs):
            row = 2 + i // 2
            col = i % 2
            _plot_calibrated_param_vs_titrant(calibration_dict,
                                              param_name=suffix,
                                              plot_color_dict=plot_color_dict,
                                              ax=ax[row, col])
        
        # Turn off last axis if odd number of parameters
        if num_trans_params % 2 != 0:
            ax[-1, -1].axis("off")

    fig.tight_layout()

    return fig, ax


def indiv_replicates(pred_df,calibration_dict=None,rgb_map=None):
    """
    Plot observed and predicted ln(A0) vs. time for all replicates used in the
    calibration.
    
    Parameters
    ----------
    pred_df : pandas.DataFrame
        prediction dataframe returned by calibrate. The function expects the 
        dataframe has columns: "time", "replicate", "titrant_conc", "y_obs",
        "y_std", and "calc_est". 
    calibration_dict : dict, optional
        calibration dictionary used to calculate pre-growth rates.
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
    positive_titrants = titrants[titrants > 0]
    if len(positive_titrants) > 0:
        floor = 0.1*np.min(positive_titrants)
    else:
        floor = 1.0 # arbitrary if no positive titrants
    
    # Normalize log(titrant) between 0 and 1. 
    norm_values = titrants.copy()
    norm_values[norm_values == 0] = floor 
    norm_values = np.log(norm_values)
    mx = np.max(norm_values)
    mn = np.min(norm_values)
    if mx == mn:
        norm_values = np.ones(len(norm_values), dtype=float) * 0.5
    else:
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
    
    # Figure out number of plots to create. Subplots are defined by 
    # (genotype, replicate, condition_pre). 
    subplot_cols = ["genotype","replicate","condition_pre"]
    size = int(np.ceil(np.sqrt(len(pred_df.groupby(subplot_cols)))))
    
    # Get axis limits. Use quantile-based limits for robustness to outliers.
    min_x, max_x = get_ax_limits(pred_df["t_sel"], pad_by=0.05, percentile=0.005)
    
    min_y, max_y = get_ax_limits(pred_df["y_obs"],
                                 pred_df["calc_est"],
                                 pad_by=0.05,percentile=0.005)
        
    fig, axes = plt.subplots(size,
                             size,
                             figsize=(12,12),
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    # -------------------------------------------------------------------------
    # Go through all replicates and generate plots
    
    row_counter = 0
    col_counter = 0
    for key, sub_df in pred_df.groupby(subplot_cols):
        
        # Unpack key for identification
        genotype = key[0]
        replicate = key[1]
        condition_pre = key[2]

        # Axis for plot
        ax = axes[row_counter,col_counter]

        # Go through all conditions in this series. Series are defined by
        # (titrant_name, titrant_conc, condition_sel).
        series_cols = ["titrant_name","titrant_conc","condition_sel"]
        for _, cond_df in sub_df.groupby(series_cols):
    
            # Get values to plot. We use .to_numpy() and then ensure they are 1D
            # in case columns were somehow duplicated or multidimensional.
            def to_1d(data):
                v = np.asarray(data)
                if v.ndim > 1:
                    return v[:, 0]
                return v.ravel()

            # Helper to get a scalar value from potentially redundant columns
            def to_scalar(data):
                return to_1d(data)[0]

            x = to_1d(cond_df["t_sel"])
            y = to_1d(cond_df["y_obs"])
            y_std = to_1d(cond_df["y_std"])
            
            # Get color and titrant label
            t = to_scalar(cond_df["titrant_conc"])
            color = tuple(rgba_df.loc[t,"color"])
            label = f"titrant: {t:.3f}"

            # Plot data 
            mask = ~np.isnan(y)
            ax.scatter(x[mask],y[mask],s=30,facecolor="none",edgecolor=color,label=label)
            ax.errorbar(x[mask],y[mask],y_std[mask],lw=0,color=color,capsize=5,elinewidth=1)
            
            # --- FULL SHIFT GROWTH MODEL CALCULATION ---
            
            # Helper to draw a dashed connecting line if complex fails
            def draw_fallback():
                order = np.argsort(x)
                sort_x = x[order]
                sort_y = to_1d(cond_df["calc_est"])[order]
                valid = np.isfinite(sort_x) & np.isfinite(sort_y)
                ax.plot(sort_x[valid], sort_y[valid], color=color, lw=2)

            # If we have a calibration dict and a -t_pre point, we can draw a 
            # continuous curve based on the shift model.
            if calibration_dict is not None and np.any(to_1d(cond_df["t_sel"]) < 0):
                
                try:
                    # Sort for extracting points
                    order = np.argsort(x)
                    sort_x = x[order]
                    # sort_y = to_1d(cond_df["calc_est"])[order]
                    
                    # Identify lnA0 (t = -t_pre)
                    neg_mask = sort_x < 0
                    if not np.any(neg_mask):
                        raise ValueError("No negative time points found")
                        
                    lnA0 = to_scalar(to_1d(cond_df["calc_est"])[order][neg_mask])
                    t_pre = abs(to_scalar(sort_x[neg_mask]))
                    
                    # Get parameters
                    dilution = calibration_dict.get("dilution", 1.0)
                    cond_pre = to_scalar(cond_df["condition_pre"])
                    cond_sel = to_scalar(cond_df["condition_sel"])
                    titr_name = to_scalar(cond_df["titrant_name"])
                    titr_conc = to_scalar(cond_df["titrant_conc"])
                    
                    def as_float(val):
                        return float(np.asarray(val).ravel()[0])

                    # Growth rates
                    mu1 = as_float(get_wt_k(condition=cond_pre,
                                            titrant_name=titr_name,
                                            titrant_conc=0.0,
                                            calibration_data=calibration_dict))
                    
                    # Get background components for mu2
                    k_bg_val = as_float(get_background(titr_name, titr_conc, calibration_dict))
                    
                    # Get growth perturbation
                    mu_wt_val = as_float(get_wt_k(condition=cond_sel,
                                                  titrant_name=titr_name,
                                                  titrant_conc=titr_conc,
                                                  calibration_data=calibration_dict))
                    
                    dk_val = mu_wt_val - k_bg_val
                    
                    # Add genotype effect if available
                    dk_geno_val = 0.0
                    if "dk_geno" in calibration_dict:
                        dk_geno_val = calibration_dict["dk_geno"].get(genotype, 0.0)
                    
                    mu1 = mu1 + dk_geno_val
                    mu2 = mu_wt_val + dk_geno_val
                    
                    # Model selection
                    growth_model_name = calibration_dict.get("model_name", "linear")
                    trans_model_name = calibration_dict.get("transition_model_name", "constant")
                    
                    growth_model = get_model(growth_model_name)
                    trans_model = get_transition_model(trans_model_name)

                    # Extract theta for transition model prediction (if needed)
                    # For indiv_replicates, we don't have individual theta fit yet usually,
                    # but we can use wt_theta or 1.0/0.0 as appropriate.
                    # Calibration usually assumes wt response.
                    theta_wt = as_float(get_wt_theta(titr_name, titr_conc, calibration_dict))
                    
                    # Shift parameters from transition model
                    dk_cond_df = calibration_dict.get("dk_cond_df")
                    per_titrant_tau = calibration_dict.get("per_titrant_tau", False)
                    
                    # DEBUG: Print dk_cond_df to see what we're working with
                    # print("DEBUG: dk_cond_df contents:")
                    # print(dk_cond_df)
                    
                    trans_param_list = []
                    for suffix, _, _, _ in trans_model.get_param_defs():
                        if per_titrant_tau:
                            granular_key = f"{cond_sel}:{titr_name}:{titr_conc}"
                            val = dk_cond_df.loc[granular_key, suffix] if granular_key in dk_cond_df.index else dk_cond_df.loc[cond_sel, suffix]
                        else:
                            val = dk_cond_df.loc[cond_sel, suffix]
                        trans_param_list.append(val)
                    
                    trans_params = np.array(trans_param_list)
                    tau = trans_model.predict_tau(theta_wt, trans_params)
                    k_sharp = trans_model.predict_k_sharp(theta_wt, trans_params)
                    
                    # Generate dense time points
                    # Pre-growth: [min_x, 0]
                    # Post-growth: [0, max_x]
                    
                    if max_x > 0:
                        # Post-growth: [0, max_x]
                        t_model_post = np.linspace(0, max_x, 100)
                        
                        gm = OccupancyGrowthModel()
                        
                        # Calculate growth component (without C offset)
                        growth_sel_comp = gm.predict_trajectory(
                            t_pre=0, t_sel=t_model_post, 
                            ln_cfu0=0, mu1=mu1, mu2=mu2, 
                            dilution=1.0, tau=tau, k_sharp=k_sharp
                        )
                        
                        # Find anchor: average(calc_est - growth_sel_comp) for observed points
                        obs_mask = x >= 0
                        if np.any(obs_mask):
                            obs_x = x[obs_mask]
                            obs_y = to_1d(cond_df["calc_est"])[obs_mask]
                            
                            obs_growth_comp = gm.predict_trajectory(
                                t_pre=0, t_sel=obs_x, 
                                ln_cfu0=0, mu1=mu1, mu2=mu2, 
                                dilution=1.0, tau=tau, k_sharp=k_sharp
                            )
                            C_post = np.mean(obs_y - obs_growth_comp)
                            ax.plot(t_model_post, growth_sel_comp + C_post, color=color, lw=2)
                        else:
                            # Fallback if no selection points, though unlikely in this loop
                            draw_fallback()

                    if min_x < 0:
                        # Pre-growth segment: [min_x, 0]
                        t_model_pre = np.linspace(min_x, -1e-6, 50)
                        
                        # Linear growth: y = mu1 * t + C
                        # Find anchor from pre-growth points
                        obs_mask = x < 0
                        if np.any(obs_mask):
                            obs_x = x[obs_mask]
                            obs_y = to_1d(cond_df["calc_est"])[obs_mask]
                            C_pre = np.mean(obs_y - mu1 * obs_x)
                            
                            ax.plot(t_model_pre, mu1 * t_model_pre + C_pre, color=color, lw=2)
                        else:
                            # If no pre-growth points observed but we're in this block, 
                            # we could anchor to the selection start? 
                            # But better to just skip or fallback.
                            pass

                except Exception:
                    import traceback
                    traceback.print_exc()
                    draw_fallback()
                
            else:
                draw_fallback()

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
        ax.set_title(f"{genotype}, {condition_pre}, rep {replicate}")
        
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

    return fig, axes