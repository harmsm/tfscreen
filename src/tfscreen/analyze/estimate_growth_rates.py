
from tfscreen.util import (
    read_dataframe,
    process_for_fit
)

from tfscreen.calibration import read_calibration

from tfscreen.fitting import (
    get_growth_rates_ols,
    get_growth_rates_wls,
    get_growth_rates_gls,
    get_growth_rates_glm,
    get_growth_rates_gee,
    get_growth_rates_kf,
    get_growth_rates_ukf,
    get_growth_rates_ukf_lin,
    get_growth_rates_nls
)

import numpy as np
import pandas as pd

# k fitters with list of positional arguments to pass. The names of these 
# positional arguments match the keys in the output dictionary from 
# process_for_fit.

_ALLOWED_K_FITTERS = {
    "ols":{
        "fcn":get_growth_rates_ols,
        "args":["times",
                "ln_cfu"],
    },
    "wls":{
        "fcn":get_growth_rates_wls,
        "args":["times",
                "ln_cfu",
                "ln_cfu_var"],
    },
    "gee":{
        "fcn":get_growth_rates_gee,
        "args":["times",
                "cfu"],
    },
    "gls":{
        "fcn":get_growth_rates_gls,
        "args":["times",
                "ln_cfu"],
    },
    "glm":{
        "fcn":get_growth_rates_glm,
        "args":["times",
                "cfu"],
    },
    "kf":{
        "fcn":get_growth_rates_kf,
        "args":["times",
                "ln_cfu",
                "ln_cfu_var",
                "growth_rate_wt"],
    },
    "ukf":{
        "fcn":get_growth_rates_ukf,
        "args":["times",
                "cfu",
                "cfu_var",
                "growth_rate_wls",
                "growth_rate_err_wls"],
    },
    "ukf_lin":{
        "fcn":get_growth_rates_ukf_lin,
        "args":["times",
                "ln_cfu",
                "ln_cfu_var",
                "growth_rate_wls",
                "growth_rate_err_wls"],
    },
    "nls":{
        "fcn":get_growth_rates_nls,
        "args":["times",
                "cfu",
                "cfu_var"],
    }
}

def estimate_growth_rates(combined_df,
                          sample_df,
                          calibration_file,
                          k_fit_method="wls",
                          use_inferred_zero_point=True,
                          pseudocount=1,
                          num_required=2,
                          iptg_out_growth_time=30,
                          k_fitter_kwargs=None):
    """
    Estimate growth rates for each genotype in each sample.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        A dataframe that minimally has columns "genotype", "sample",
        "time", "counts", "total_counts_at_time", and "total_cfu_at_time".  The
        values in the "sample" column should be indexes in the sample_df 
        dataframe. The dataframe must be sorted by genotype, then sample. 
        The combined_df should be exhaustive, having all genotypes in all 
        samples. Genotypes not seen in a particular sample should still be 
        present, just given counts of zero. 
    sample_df : pandas.DataFrame
        Dataframe containing information about the samples. This function assumes
        it is indexed by the values seen in the "sample" column of the 
        combined_df. 
    calibration_file : str or dict
        Path to the calibration file or loaded calibration dictionary.
    k_fit_method : str, optional
        Method to use for fitting growth rates. Must be one of 'ols', 'wls', 
        'gls', 'glm', 'kf', 'ukf', 'ukf_lin', or 'nls'. Default is 'wls'.
    use_inferred_zero_point : bool, optional
        Whether to use the inferred zero time point in the fit. Default is True.
    pseudocount : int, optional
        Pseudocount to add to CFU values before taking the logarithm. 
        Default is 1.
    num_required : int, optional
        Minimum number of time points with non-zero counts required for a 
        successful fit. Default is 2.
    iptg_out_growth_time : int, optional
        Time (in minutes) that the samples grow at their IPTG concentrations 
        before selection is added. Default is 30.
    k_fitter_kwargs : dict, optional
        Keyword arguments to pass to the growth rate fitting function. 
        Default is None.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the estimated growth rates and initial population
        sizes for each genotype in each sample. Includes columns for the 
        estimated A0 (initial population size), A0 standard error, 
        estimated growth rate (k), growth rate standard error, and a 
        boolean indicating whether a fit was attempted or whether there were
        insufficient counts to proceed. If a genotype was not fit (or the fit
        failed), it will have fit values of np.nan.
    
    Raises
    ------
    ValueError
        If `k_fit_method` is not one of the allowed methods.
    """

    # Set up function and arguments to fit growth rates
    if k_fit_method not in _ALLOWED_K_FITTERS:
        err = f"k_fit_method '{k_fit_method}' not recognized. Should be one of:\n"
        for k in _ALLOWED_K_FITTERS:
            err += f"    '{k}'\n"
        err += "\n"
        raise ValueError(err)

    k_fit_fcn = _ALLOWED_K_FITTERS[k_fit_method]["fcn"]
    k_fit_args = _ALLOWED_K_FITTERS[k_fit_method]["args"]

    if k_fitter_kwargs is None:
        k_fitter_kwargs = {}
    
    # Load the calibration dictionary
    calibration_dict = read_calibration(calibration_file)

    # Load the combined and sample data frames
    combined_df = read_dataframe(combined_df)
    sample_df = read_dataframe(sample_df)

    if sample_df.index.name != "sample":

        # Set the index of the sample dataframe to be the sample
        if sample_df.columns[0] == "Unnamed: 0":
            sample_df = sample_df.rename(columns={"Unnamed: 0":"sample"})

        sample_df.index = sample_df["sample"]
        sample_df = sample_df.drop(columns=["sample"])

    # Create arrays with sample/genotype as their primary axis and (if 
    # relevant) time as their secondary axis. These 1D and 2D arrays include 
    # times (2D), cfu (2D), cfu_var (2D), ln_cfu (2D), ln_cfu_var (2D), 
    # genotypes (1D), etc.
    processed = process_for_fit(combined_df=combined_df,
                                sample_df=sample_df,
                                calibration_dict=calibration_dict,
                                pseudocount=pseudocount,
                                iptg_out_growth_time=iptg_out_growth_time)

    genotypes = processed["genotypes"]
    samples = processed["samples"]

    # Get rid of data with too few counts across time points to fit
    sequence_counts = processed["sequence_counts"]
    less_than_required = np.sum(sequence_counts > 0,axis=1) < num_required
    keep_mask = np.logical_not(less_than_required)

    # Get arguments for growth rate fitter from the processed data 
    k_fit_args_filled = []
    for a in k_fit_args:
        
        v = processed[a]

        # Only keep entries with enough counts
        v = v[keep_mask]

        # Drop first (inferred zero) point if requested
        if not use_inferred_zero_point:
            if len(v.shape) == 2:
                v = v[:,1:]
        
        k_fit_args_filled.append(v)

    # Estimate A0 and k for all genotypes/timepoints
    param_df, pred_df = k_fit_fcn(*k_fit_args_filled,
                                  **k_fitter_kwargs)

    # build the output dataframe
    out_param_df = sample_df.loc[samples,:]
    out_param_df["genotype"] = genotypes
    out_param_df["A0_est"] = np.nan
    out_param_df["A0_std"] = np.nan
    out_param_df["k_est"] = np.nan
    out_param_df["k_std"] = np.nan
    out_param_df["fit_attempted"] = False

    # Record the fit values
    out_param_df.loc[keep_mask,"A0_est"] = np.array(param_df["A0_est"])
    out_param_df.loc[keep_mask,"A0_std"] = np.array(param_df["A0_std"])
    out_param_df.loc[keep_mask,"k_est"] = np.array(param_df["k_est"])
    out_param_df.loc[keep_mask,"k_std"] = np.array(param_df["k_std"])
    out_param_df.loc[keep_mask,"fit_attempted"] = True

    # Move sample from index to its own column and reset index
    out_param_df["sample"] = out_param_df.index
    out_param_df.index = np.arange(len(out_param_df.index),dtype=int)
    
    # Clean up column order
    columns = list(out_param_df.columns[:-1])
    columns.insert(0,out_param_df.columns[-1])
    out_param_df = out_param_df.loc[:,columns]

    # Create a dataframe with comparing the observed and predicted values for 
    # each sample/genotype/time. 
    num_t = processed["times"].shape[1]
    out_pred_df = pd.DataFrame({"sample":np.repeat(processed["samples"],num_t),
                                "genotype":np.repeat(processed["genotypes"],num_t),
                                "time":processed["times"].flatten()})
    
    out_pred_df["obs"] = np.nan
    out_pred_df.loc[np.repeat(keep_mask,num_t),"obs"] = pred_df["obs"]
    out_pred_df["pred"] = np.nan
    out_pred_df.loc[np.repeat(keep_mask,num_t),"pred"] = pred_df["pred"]

    # Drop the fake zero points from the prediction. These are part of the 
    # fit. 
    if use_inferred_zero_point:
        out_pred_df = out_pred_df[out_pred_df["time"] != 0]


    return out_param_df, out_pred_df