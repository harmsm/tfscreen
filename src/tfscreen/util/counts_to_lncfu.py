
from tfscreen.util import (
    df_to_arrays,
    argsort_genotypes,
    read_dataframe
)
from tfscreen.calibration import read_calibration
from tfscreen.analyze import estimate_time0

import numpy as np
import pandas as pd

def _load_replicate(combined_df,
                    sample_df,
                    calibration_data,
                    pseudocount,
                    num_required,
                    pre_select_time):
    """
    Load and process data for a single replicate.

    This function processes data for a single replicate, performing steps such as
    converting dataframes to arrays, filtering data based on the number of required
    time points, and estimating initial ln_cfu values.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing combined sequence and sample data.
    sample_df : pandas.DataFrame
        DataFrame containing sample information, including replicate, marker,
        select, and iptg.
    calibration_data : dict
        Dictionary containing calibration data.
    pseudocount : float
        Pseudocount added to sequence counts to avoid division by zero.
    num_required : int
        Minimum number of time points with more than zero reads required for a
        genotype.
    pre_select_time : float
        length of time before selection is applied (used to estimate initial
        ln_cfu values). 

    Returns
    -------
    to_regress_df : pandas.DataFrame
        DataFrame holding row info of the regression arrays.
    times : numpy.ndarray
        2D (num_rows,num_times) array of time points.
    ln_cfu : numpy.ndarray
        2D (num_rows,num_times) array of log-transformed CFU/mL values.
    ln_cfu_var : numpy.ndarray
        2D (num_rows,num_times) array of variances of log-transformed CFU/mL
        values.
    """

    if len(np.unique(sample_df["replicate"])) > 1:
        err = "this should only be called by dataframes with single replicates\n"
        raise ValueError(err)

    # Convert the dataframes into a collection of arrays. 
    _arrays = df_to_arrays(combined_df=combined_df,
                           sample_df=sample_df,
                           pseudocount=pseudocount)

    # Get 1D arrays corresponding to rows of the regression arrays for this
    # replicate
    genotype = _arrays["genotypes"]    
    replicate = sample_df.loc[_arrays["samples"],["replicate"]].values.flatten()
    marker = sample_df.loc[_arrays["samples"],["marker"]].values.flatten()
    select = sample_df.loc[_arrays["samples"],["select"]].values.flatten()
    iptg = sample_df.loc[_arrays["samples"],["iptg"]].values.flatten()
    sample = _arrays["samples"]

    # assemble dataframe holding row info of the regression arrays. 
    to_regress_df = pd.DataFrame({"genotype":genotype,
                                  "replicate":replicate,
                                  "marker":marker,
                                  "select":select,
                                  "iptg":iptg,
                                  "sample":sample})

    # Extract main arrays for regression
    times = _arrays["times"]    
    ln_cfu = _arrays["ln_cfu"]
    ln_cfu_var = _arrays["ln_cfu_var"]
    sequence_counts = _arrays["sequence_counts"]
        
    # Estimate the starting ln_cfu and global effect of each genotype on
    # growth rate. 
    time0_df, _, _, _ = estimate_time0(
        times,
        ln_cfu,
        ln_cfu_var,
        sample_df,
        calibration_data,
        pre_select_time=pre_select_time
    )

    # Extract starting ln_cfu and genotype rate shift from time0_df. 
    time0_df.index = pd.unique(genotype)
    to_regress_df["lnA_pre0_guess"] = time0_df.loc[genotype,"lnA_pre0_est"].values
    to_regress_df["k_shift_guess"] = time0_df.loc[genotype,"k_shift"].values

    # Find seq/condition rows that had at least num_required time points with
    # more than zero reads. We do this here rather than before the estimate_time0
    # call because estimate_time0 assumes observations for all conditions. 
    less_than_required = np.sum(sequence_counts > 0,axis=1) < num_required
    keep_mask = np.logical_not(less_than_required)

    # Drop rows with too few time points that have enough reads
    times = times[keep_mask,:]
    ln_cfu = ln_cfu[keep_mask,:]
    ln_cfu_var = ln_cfu_var[keep_mask,:]
    to_regress_df = to_regress_df.loc[keep_mask,:]
    to_regress_df.index = np.arange(len(to_regress_df),dtype=int)


    return to_regress_df, times, ln_cfu, ln_cfu_var

    
def counts_to_lncfu(combined_df,
                    sample_df,
                    calibration_data,
                    pseudocount,
                    num_required,
                    pre_select_time):
    """
    Use combined and sample dataframe to calculate ln(cfu/mL) array for
    regression analysis.

    This function processes the combined and sample dataframes, inferring 
    
    converting them to arrays, filtering data, and estimating initial values.
    It handles multiple replicates and sorts the data for efficient regression.

    Parameters
    ----------
    combined_df : str or pandas.DataFrame
        Path to or DataFrame containing combined sequence and sample data.
    sample_df : str or pandas.DataFrame
        Path to or DataFrame containing sample information, including replicate,
        marker, select, and iptg.
    calibration_data : str or dict
        Path to or dictionary containing calibration data.
    pseudocount : float
        Pseudocount added to sequence counts to avoid division by zero.
    num_required : int
        Minimum number of time points with more than zero reads required for a
        genotype.
    pre_select_time : float
        length of time before selection is applied (used to estimate initial
        ln_cfu values). Units must match the units of the "time" column in the
        combined_df. 

    Returns
    -------
    to_regress_df : pandas.DataFrame
        DataFrame holding row info of the regression arrays, sorted by genotype,
        replicate, marker, select, and iptg.
    times : numpy.ndarray
        2D (num_rows,num_times) array of time points, sorted according to
        to_regress_df.
    ln_cfu : numpy.ndarray
        2D (num_rows,num_times) array of log-transformed CFU/mL values, sorted
        according to to_regress_df.
    ln_cfu_var : numpy.ndarray
        2D (num_rows,num_times) array of variances of log-transformed CFU/mL
        values, sorted according to to_regress_df.
    """

    # Read dataframes
    combined_df = read_dataframe(combined_df)
    sample_df = read_dataframe(sample_df)
    
    # Make sure sample_df is indexed by sample if it isn't already
    if sample_df.index.name != "sample":

        # Set the index of the sample dataframe to be the sample
        if "sample" not in sample_df.columns:
            if sample_df.columns[0] == "Unnamed: 0":
                sample_df = sample_df.rename(columns={"Unnamed: 0":"sample"})

        if "sample" not in sample_df.columns:
            err = "sample_df must have a column or index named 'sample'\n"
            raise ValueError(err)   

        sample_df.index = sample_df["sample"]
        sample_df = sample_df.drop(columns=["sample"])
    
    calibration_data = read_calibration(calibration_data)

    # These will hold all replicate dataframes and arrays
    to_regress_dfs = []
    times = []
    ln_cfu = []
    ln_cfu_var = []

    # Go through all unique replicates seen in the sample dataframe
    replicates = pd.unique(sample_df["replicate"])
    for rep in replicates:

        # Pull subsets of input dataframes for this replicate
        rep_sample_df = sample_df[sample_df["replicate"] == rep]
        rep_combined_df = combined_df[combined_df["sample"].isin(rep_sample_df.index)]

        # Build replicate dataframes and arrays
        rep_df, rep_times, rep_ln_cfu, rep_ln_cfu_var = _load_replicate(
            rep_combined_df,
            rep_sample_df,
            calibration_data,
            pseudocount,
            num_required,
            pre_select_time
        )

        # record replicate dataframes and arrays
        to_regress_dfs.append(rep_df)
        times.append(rep_times)
        ln_cfu.append(rep_ln_cfu)
        ln_cfu_var.append(rep_ln_cfu_var)

    # Assemble replicate dataframes and arrays
    to_regress_df = pd.concat(to_regress_dfs,ignore_index=True)
    times = np.concat(times)
    ln_cfu = np.concat(ln_cfu)
    ln_cfu_var = np.concat(ln_cfu_var)
    
    # Get all unique genotypes in canonical sorted order
    genotypes = pd.unique(to_regress_df["genotype"])
    idx = argsort_genotypes(genotypes)
    genotype_order = genotypes[idx]

    # Convert genotype to categorical to allow sort
    to_regress_df["genotype"] = pd.Categorical(to_regress_df["genotype"],
                                               categories=genotype_order,
                                               ordered=True)

    # Sort dataframe 
    to_regress_df = to_regress_df.sort_values(["genotype",
                                               "replicate",
                                               "marker",
                                               "select",
                                               "iptg"])
    # Get resulting order of sort
    sort_idx = to_regress_df.index.values.copy()

    # Sort times, ln_cfu, and ln_cfu_var according to sort
    times = times[sort_idx]
    ln_cfu = ln_cfu[sort_idx]
    ln_cfu_var = ln_cfu_var[sort_idx]

    # Reset to_regress_df index to sequential
    to_regress_df.index = np.arange(len(to_regress_df),dtype=int)

    return to_regress_df, times, ln_cfu, ln_cfu_var



