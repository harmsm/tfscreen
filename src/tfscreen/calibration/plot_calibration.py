import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tfscreen.calibration import predict_growth_rate

def plot_calibration(expt_df,
                     calibration):
                            
    iptg = 10**np.linspace(-6,0,100)
    color_dict = {("pheS",1):"darkgreen",
                  ("kanR",1):"orange",
                  ("pheS",0):"lightgreen",
                  ("kanR",0):"wheat",
                  ("none",0):"gray"}
    pretty_select = {"kanR":"kanamycin",
                     "pheS":"4CP"}
    
    fig, ax = plt.subplots(1,2,figsize=(12,6),sharey=True)
    
    # Plot experimental data
    for k in pd.unique(expt_df["key"]):
        
        this_df = expt_df[expt_df["key"] == k].copy()
        this_df.loc[this_df["iptg"] == 0,"iptg"] = 1e-5
    
        this_key = (k[1],k[2])
        
        ax[0].scatter(this_df["iptg"],
                      this_df["k_est"],
                      s=30,
                      edgecolor=color_dict[this_key],
                      facecolor="none")
        ax[0].errorbar(this_df["iptg"],
                       this_df["k_est"],
                       this_df["k_std"],
                       color=color_dict[this_key],
                       lw=0,elinewidth=1,capsize=5)
    
    # Build dataframe for smooth simulation of data
    out = {"key":[],
           "marker":[],
           "select":[],
           "iptg":[]}
    for marker in ["none","kanR","pheS"]:
        for select in [0,1]:
            if marker == "none" and select == 1:
                continue
    
            out["key"].extend([(marker,select) for _ in range(len(iptg))])
            out["marker"].extend([marker for _ in range(len(iptg))])
            out["select"].extend([select for _ in range(len(iptg))])
            out["iptg"].extend(iptg)
            
    sim_df = pd.DataFrame(out)
    
    # Add predicted and predicted error to the smooth dataset
    pred, pred_err = predict_growth_rate(sim_df["marker"],
                                         sim_df["select"],
                                         sim_df["iptg"],
                                         calibration)
    sim_df["predicted"] = pred
    sim_df["predicted_err"] = pred_err
    
    # Plot smooth data 
    for k in pd.unique(sim_df["key"]):
    
        # Create label for legend
        this_key = (k[0],k[1])
        if k[1] == 1:
            operator = "+"
        else:
            operator = "-"
    
        if k[0] == "none":
            label = "base"
        else:
            label = f"{k[0]} {operator} {pretty_select[k[0]]}"
    
        # Plot smooth data
        this_sim_df = sim_df[sim_df["key"] == k]
        ax[0].plot(this_sim_df["iptg"],
                   this_sim_df["predicted"],lw=3,
                   color=color_dict[k],
                   label=label)
    
    # Clean up plot
    ax[0].set_xscale('log')
    ax[0].set_xlabel("[iptg] (mM)")
    ax[0].set_ylabel("growth rate (cfu/mL/min)")
    ax[0].legend()
    
    
    # Now do correlation plot between input and output k
    pred, pred_err = predict_growth_rate(expt_df["marker"],
                                         expt_df["select"],
                                         expt_df["iptg"],
                                         calibration)
    
    edgecolor = [color_dict[(v[1],v[2])] for v in expt_df["key"]]
    ax[1].scatter(pred,expt_df["k_est"],s=40,zorder=5,facecolor="none",edgecolor=edgecolor)
    ax[1].errorbar(x=pred,
                   y=expt_df["k_est"],
                   yerr=expt_df["k_std"],
                   xerr=pred_err,
                   lw=0,capsize=5,
                   elinewidth=1,
                   ecolor="gray",
                   zorder=0)
    
    x_min = 0.03
    x_max = 0.13
    ax[1].plot((x_min,x_max),(x_min,x_max),'--',color='gray',zorder=-5)
    ax[1].set_xlim(x_min,x_max)
    ax[1].set_ylim(x_min,x_max)
    
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("calibration model prediction")
    
    fig.tight_layout()

    return fig, ax