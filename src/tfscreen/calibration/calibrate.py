from tfscreen.fitting import nls
from tfscreen.calibration import perform_calibration
from tfscreen.calibration import plot_calibration
from tfscreen.calibration import write_calibration

from tfscreen.util import read_dataframe

import numpy as np
import pandas as pd

def _get_growth_rates(df,default_err=0.01):
    """
    Get growth rates by nonlinear least squares given cfu_per_mL vs. 
    time data. 

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe with input data. The function expects a column 
        "key" that uniquely identifies each experiment to be fit. (For 
        example, (day-marker-select)). It also expects time and
        cfu_per_mL. It also looks for cfu_per_mL_var (variance on the 
        cfu_per_mL). 
    default_err : pandas.DataFrame
        if no cfu_per_mL_var column is found, assign a variance of 
        default_err*cfu_per_mL.     

    Returns
    -------
    out_df : pandas.DataFrame
        dataframe with fit results. it will add k_est and k_std (estimated
        growth rates and standard errors on the estimated growth rates). 
    """

    # Populate this dictionary for output
    columns = ["key","day","marker","select","iptg","k_est","k_std"]
    out = dict([(c,[]) for c in columns])

    # Go through each unique experiment (key)
    for key in pd.unique(df["key"]):

        times = []
        cfu = []
        cfu_var = []

        # Convert to arrays with shape (num_times,num_iptg)
        this_df = df.loc[df["key"] == key]
        unique_iptg = pd.unique(this_df["iptg"])
        for u in unique_iptg:
            iptg_df = this_df.loc[this_df["iptg"] == u,:]
            times.append(iptg_df["time"])
            cfu.append(iptg_df["cfu_per_mL"])
            cfu_var.append((default_err*iptg_df["cfu_per_mL"])) 

        # Finalize arrays
        times = np.array(times)
        cfu = np.array(cfu)
        cfu_var = np.array(cfu_var)

        # Fit by non linear least squares
        k_est, k_std = nls.get_growth_rates_nls(times,cfu,cfu_var)

        # Record in dictionary
        N = len(k_est)
        out["key"].extend([key for _ in range(N)])
        out["day"].extend([key[0] for _ in range(N)])
        out["marker"].extend([key[1] for _ in range(N)])
        out["select"].extend(np.ones(N,dtype=int)*key[2])
        out["iptg"].extend(unique_iptg)
        out["k_est"].extend(k_est)
        out["k_std"].extend(k_std)
        
    # Build output dataframe
    out_df = pd.DataFrame(out)
    
    return out_df

def calibrate(calibration_file,output_root,K,n):
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

    Returns
    -------
    calibration : dict
        calibration dictionary
    fig : matplotlib.Figure
        figure object
    ax : matplotlib.Axis
        axis object
    """

    # read in growth file
    df = read_dataframe(calibration_file)

    # Extract growth rates
    expt_df = _get_growth_rates(df)

    # Do calibration
    calibration = perform_calibration(expt_df,K=K,n=n)

    # Write out calibration figure
    fig, ax = plot_calibration(expt_df=expt_df,
                               calibration=calibration)
    fig.savefig(f"{output_root}.pdf")

    # Write out calibration file
    write_calibration(calibration,
                      f"{output_root}.json")
    
    return calibration, fig, ax
