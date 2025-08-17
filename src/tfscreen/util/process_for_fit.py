
from tfscreen.util import get_growth_guesses
from tfscreen.fitting import fast_weighted_linear_regression

import pandas as pd
import numpy as np

def _count_df_to_arrays(df):
    """
    Take a dataframe with columns "genotype", "base_condition", "time",
    "counts", "total_counts_at_time", and "total_cfu_per_time" return numpy
    arrays with shape (num_samples,num_times) where num_samples iterates 
    over all combinations of genotype and and base condition. There must be
    an identical number of time points for each genotype/base_condition pair,
    though the time point values can all differ from one another. 

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to process

    Returns
    -------
    times : np.ndarray
        2D float array of times with shape (num_samples,num_times)
    sequence_counts : np.ndarray 
        2D float array of counts of particular genotype with shape
        (num_samples,num_times)
    total_counts : np.ndarray
        2D float array of total counts in sample with shape
        (num_samples,num_times)
    total_cfu_ml : np.ndarray
        2D float array of total cfu/mL in sample with shape
        (num_samples,num_times)
    genotypes : np.ndarray
        1D object array of genotypes with shape (num_samples,)
    base_conditions : np.ndarray
        1D object array of base_condition strings with shape
        (num_samples,)
    """

    # Check for all required columns
    look_for = ["time","counts","total_counts_at_time","total_cfu_mL_at_time"]
    diff = list(set(look_for) - set(df.columns))
    diff.sort()
    if len(diff) > 0:
        err = "not all columns found in dataframe. missing columns:\n"
        for d in diff:
            err += f"    {d}\n"
        raise ValueError(err + "\n")
        
    # Get number of times per experiment
    number_of_times_per_expt = df.groupby(['genotype', 'base_condition']).size().unique()
    if len(number_of_times_per_expt) != 1:
        err = "each genotype/base_condition must have the same number of time entries\n"
        raise ValueError(err)

    num_l = len(df["time"])
    num_t = number_of_times_per_expt[0]

    # Extract relevant series as arrays
    times = np.array(df["time"])
    sequence_counts = np.array(df["counts"])
    total_counts = np.array(df["total_counts_at_time"])
    total_cfu_ml = np.array(df["total_cfu_mL_at_time"])

    # Reshape into sample x time arrays
    times = times.reshape(num_l//num_t,num_t)
    sequence_counts = sequence_counts.reshape(num_l//num_t,num_t)
    total_counts = total_counts.reshape(num_l//num_t,num_t)
    total_cfu_ml = total_cfu_ml.reshape(num_l//num_t,num_t)

    # Get genotypes and base conditions for masking arrays
    genotypes = np.array(df["genotype"])
    base_conditions = np.array(df["base_condition"])
    genotypes = genotypes[np.arange(0,num_l,num_t,dtype=int)]
    base_conditions = base_conditions[np.arange(0,num_l,num_t,dtype=int)]
    
    return times, sequence_counts, total_counts, total_cfu_ml, genotypes, base_conditions

def _process_counts(sequence_counts,
                    total_counts,
                    total_cfu_ml,
                    pseudocount=1):
    """
    Convert sequence counts to frequencies and variances. 

    Parameters
    ----------
    sequence_counts : numpy.ndarray
        Array of sequence counts for a specific genotype/condition.
    total_counts : numpy.ndarray
        Array of total sequence counts for each sample.
    total_cfu_ml : numpy.ndarray
        Array of total CFU/mL measurements for each sample.
    pseudocount : int or float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default is 1.

    Returns
    -------
    out : dict
        Dictionary containing the following keys:
        - "adj_seq_counts" : numpy.ndarray
            Adjusted sequence counts (sequence_counts + pseudocount).
        - "adj_total_counts" : numpy.ndarray
            Adjusted total counts (total_counts + n*pseudocount).
        - "f" : numpy.ndarray
            Frequency of the genotype (adj_sequence_counts / adj_total_counts).
        - "f_var" : numpy.ndarray
            Variance of the frequency.
        - "cfu" : numpy.ndarray
            CFU/mL of the genotype (f * total_cfu_ml).
        - "cfu_var" : numpy.ndarray
            Variance of the CFU/mL.
        - "ln_f" : numpy.ndarray
            Natural logarithm of the frequency.
        - "ln_f_var" : numpy.ndarray
            Variance of the natural logarithm of the frequency.
        - "ln_cfu" : numpy.ndarray
            Natural logarithm of the CFU/mL.
        - "ln_cfu_var" : numpy.ndarray
            Variance of the natural logarithm of the CFU/mL.    
    """

    n = len(sequence_counts)

    adj_sequence_counts = sequence_counts + pseudocount
    adj_total_counts = total_counts + n*pseudocount
    
    f = (adj_sequence_counts)/(adj_total_counts)
    f_var = f*(1 - f)/(adj_total_counts)
    
    cfu = f*total_cfu_ml
    cfu_var = f_var*(total_cfu_ml**2)
        
    ln_cfu = np.log(cfu)
    ln_cfu_var = cfu_var/(cfu**2)

    return ln_cfu, ln_cfu_var

def process_for_fit(combined_df,
                    condition_df,
                    pseudocount=1,
                    iptg_out_growth_time=30):


    df = pd.read_csv(combined_df)

    _results = _count_df_to_arrays(df)
    
    times = _results[0]
    sequence_counts = _results[1]
    total_counts = _results[2]
    total_cfu_ml = _results[3]
    genotypes = _results[4]
    base_conditions = _results[5]
    
    condition_df = pd.read_csv(condition_df)
    iptg = np.array(condition_df["iptg"])
    marker = np.array(condition_df["marker"])
    select = np.array(condition_df["select"])
    
    num_conditions = len(pd.unique(base_conditions))
    num_genotypes = len(pd.unique(df["genotype"]))
    
    # Get guesses for growth rates from wildtype data
    growth_rate_guess = get_growth_guesses(iptg=iptg,
                                                         select=select,
                                                         marker=marker)
    growth_rate_guess = np.tile(growth_rate_guess,num_genotypes)
    
    _counted = _process_counts(sequence_counts,
                               total_counts,
                               total_cfu_ml,
                               pseudocount=pseudocount)
    
    # Pull out ln_cfu and ln_cfu_var
    ln_cfu = _counted["ln_cfu"]
    ln_cfu_var = _counted["ln_cfu_var"]
    
    # Get initial fit with from data that do not have a real t = 0 point. Fit 
    # Returns estimates for lnA0 (t = 0). 
    _, lnA0, _, lnA0_err, _ = fast_weighted_linear_regression(x_arrays=times,
                                                              y_arrays=ln_cfu,
                                                              y_err_arrays=ln_cfu_var)
    
    # We estimated lnA0 and lnA0_err for multiple conditions. Get a global estimate
    # assuming all conditions share same lnA0. Weight the mean and variance estimates
    # based on 1/lnA0_err. 
    
    # Reshape by condition and extract weights
    lnA0_reshaped = lnA0.reshape((lnA0.shape[0]//num_conditions,num_conditions))
    lnA0_err_reshaped = lnA0_err.reshape((lnA0_err.shape[0]//num_conditions,num_conditions))
    
    # err from fit is standard error, convert to variance then normalize
    lnA0_weight = 1/(lnA0_err_reshaped**2) 
    lnA0_weight = lnA0_weight/np.sum(lnA0_weight,axis=1,keepdims=True)
    
    # This is the pre-growth done in the presence of IPTG and marker, but no selection.
    condition_pre_growth = get_growth_guesses(iptg=np.array(condition_df["iptg"]),
                                              marker=np.array(condition_df["marker"]),
                                              select=np.zeros(len(condition_df["select"])))
    pre_growth = condition_pre_growth*iptg_out_growth_time
    
    # Get a per-genotype mean and variance for the true starting ln_cfu (from 
    # before the pre-growth). 
    ln_A_pre_zero = lnA0_reshaped - pre_growth[np.newaxis,:]
    ln_A_pre_zero_mean = np.average(ln_A_pre_zero,weights=lnA0_weight,axis=1)
    ln_A_pre_zero_var = (np.std(ln_A_pre_zero,axis=1)*np.sqrt(np.sum(lnA0_weight**2,axis=1)))**2
    
    # Grow out pre_zero by pre_growth for each condition and append as a new time zero.
    repeated_pre_zero = np.repeat(ln_A_pre_zero_mean,num_conditions)
    repeated_pre_growth = np.tile(pre_growth,num_genotypes)
    repeated_ln_A0 = repeated_pre_zero + repeated_pre_growth
    ln_cfu_expanded = np.hstack([repeated_ln_A0[:,np.newaxis],ln_cfu])
    
    # Append ln_cfu_variance as new time zero
    repeated = np.repeat(ln_A_pre_zero_var,num_conditions)
    ln_cfu_var_expanded = np.hstack([repeated[:,np.newaxis],ln_cfu_var])
    
    # Add time zero
    time_block = np.zeros(times.shape[0])
    times_expanded = np.hstack([time_block[:,np.newaxis],times])

    cfu_expanded = np.exp(ln_cfu_expanded)
    cfu_var_expanded = (cfu_expanded**2)*ln_cfu_var_expanded

    out = {"conditions":base_conditions[:num_conditions],
           "genotypes":genotypes[:num_genotypes],
           "times":times_expanded,
           "cfu":cfu_expanded,
           "cfu_var":cfu_var_expanded,
           "ln_cfu":ln_cfu_expanded,
           "ln_cfu_var":ln_cfu_var_expanded}

    return out