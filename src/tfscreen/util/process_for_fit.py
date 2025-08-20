
from tfscreen.util import get_growth_guesses
from tfscreen.fitting import fast_weighted_linear_regression

import pandas as pd
import numpy as np

def _count_df_to_arrays(df):
    """
    Take a dataframe with columns "genotype", "base_condition", "time",
    "counts", "total_counts_at_time", and "total_cfu_at_time" return numpy
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

def _get_ln_cfu(sequence_counts,
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
    ln_cfu : numpy.ndarray
        Natural logarithm of the CFU/mL for each genotype
    ln_cfu_var : numpy.ndarray
        Variance of the natural logarithm of the CFU/mL for each genotype.   
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

def _build_time_zero(times,
                     ln_cfu,
                     ln_cfu_var,
                     condition_df,
                     num_genotypes,
                     num_conditions,
                     iptg_out_growth_time):
    """
    Estimate t = 0 points from all conditions and create a new pseudo datapoint
    with ln_cfu and ln_cfu_var at t = 0. This is done by:

    1. Infer lnA0 and lnA0 for each genotype/condition. This is lnA0 at the 
       start of selection, after a pre-selection outgrowth in the relevant IPTG
       concentration.
    2. Calculate the expected growth of each genotype/condition over the pre-
       selection interval. 
    3. Subtract the pre-selection growth from each lnA0. This gives us an 
       independent estimate of the initial ln(CFU) for the genotype in the 
       initial culture. 
    4. Calculate the mean and standard deviation of this initial ln(CFU) from
       all conditions. Weight mean and stdev by 1/parameter_std_err^2.
    5. For each condition, add the pre-selection growth back to the estimate of
       ln(CFU) averaged over all conditions.
    6. Add these new lnA0 values as t = 0 to ln_cfu and ln_cfu_var arrays. 

    Parameters
    ----------
    times : numpy.ndarray
        2D array. Time points corresponding to each genotype/condition. 
        Shape (num_time_points,num_genotypes*num_conditions)
    ln_cfu : numpy.ndarray
        2D array. Natural logarithm of the CFU/mL for each genotype/condition
        Shape (num_time_points,num_genotypes*num_conditions)
    ln_cfu_var : numpy.ndarray
        2D array. Variance of the natural logarithm of the CFU/mL for each
        genotype/condition. Shape (num_time_points,num_genotypes*num_conditions)  
    condition_df : pandas.DataFrame
        Dataframe with conditions. Should have columns "iptg", "marker", and
        "select".
    num_genotypes : int
        number of genotypes
    num_conditions : int
        number of conditions
    iptg_out_growth_time : float
        how long the cultures grew in iptg before being put under selection. 
        Units should match other units.
    
    Returns
    -------
    times_expanded : numpy.ndarray
        2D array of time points, including the added time zero.
    ln_cfu_expanded : numpy.ndarray
        2D array of natural log of CFU values, expanded to include the time zero
        point.
    ln_cfu_var_expanded : numpy.ndarray
        2D array of variances of natural log of CFU values, expanded to
        include the time zero point.
    """

    # Get initial fit with from data that do not have a real t = 0 point. Fit 
    # returns estimates for lnA0 (t = 0). 
    _, lnA0, _, lnA0_err, _ = fast_weighted_linear_regression(x_arrays=times,
                                                              y_arrays=ln_cfu,
                                                              y_err_arrays=ln_cfu_var)
        
    # Reshape by condition and extract weights
    lnA0_reshaped = lnA0.reshape((lnA0.shape[0]//num_conditions,
                                  num_conditions))
    lnA0_err_reshaped = lnA0_err.reshape((lnA0_err.shape[0]//num_conditions,
                                          num_conditions))
        
    # Calculate how much the genotype grew during the IPTG outgrowth prior 
    # to selection.
    pre_selection_k = get_growth_guesses(iptg=np.array(condition_df["iptg"]),
                                         marker=np.array(condition_df["marker"]),
                                         select=np.zeros(len(condition_df["select"])))
    pre_growth = pre_selection_k*iptg_out_growth_time
    
    # Get a per-genotype mean and variance for the true starting ln_cfu (from 
    # before the pre-growth). 
    ln_A_pre_zero = lnA0_reshaped - pre_growth[np.newaxis,:]

    # err from fit is parameter standard error, convert to variance then
    # normalize so we can do weighted mean and standard deviation calculation
    lnA0_weight = 1/(lnA0_err_reshaped**2) 
    lnA0_weight = lnA0_weight/np.sum(lnA0_weight,axis=1,keepdims=True)

    # Calculate the ln_A mean and variance for before the pre-selection growth
    ln_A_pre_zero_mean = np.average(ln_A_pre_zero,weights=lnA0_weight,axis=1)
    ln_A_pre_zero_var = (np.std(ln_A_pre_zero,axis=1)*np.sqrt(np.sum(lnA0_weight**2,axis=1)))**2
    
    # Add back the pre-selection growth to each condition and append as a new
    # time zero.
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

    return times_expanded, ln_cfu_expanded, ln_cfu_var_expanded

def _get_k_guess_from_wt(condition_df,
                         num_genotypes):
    
    iptg = np.array(condition_df["iptg"])
    marker = np.array(condition_df["marker"])
    select = np.array(condition_df["select"])
    
    # Get guesses for growth rates from wildtype data
    growth_rate_wt = get_growth_guesses(iptg=iptg,
                                        select=select,
                                        marker=marker)
    growth_rate_wt = np.tile(growth_rate_wt,num_genotypes)

    return growth_rate_wt


def process_for_fit(combined_df,
                    condition_df,
                    pseudocount=1,
                    iptg_out_growth_time=30):
    """
    Process dataframes for fitting growth parameters.

    This function takes two dataframes, a combined dataframe containing
    sequence counts and a condition dataframe specifying experimental
    conditions, and performs several processing steps to prepare the data
    for fitting. These steps include:

        1. Combining the dataframes to yield numpy arrays of times, cfu, and 
           ln_cfu. 
        2. Estimating and adding a t = 0 datapoint for each genotype/condition.
        3. Generating plausible initial guesses of growth rate for further
           regression.

    This analysis assumes: 1) The combined_df is organized by genotype, then
    condition. 2) The condition_df has all conditions in combined_df. 3) The 
    combined_df has **all** genotypes and **all** conditions--it is exhaustive. 
    Genotype/condition pairs that were not seen in sequencing should still be 
    present, just given counts of zero. 

    Parameters
    ----------
    combined_df : str
        Path to a CSV file containing the combined dataframe. This dataframe
        should have columns "genotype", "base_condition", "time", "counts",
        "total_counts_at_time", and "total_cfu_mL_at_time".
    condition_df : str
        Path to a CSV file containing the condition dataframe. This dataframe
        should have columns "iptg", "marker", and "select".
    pseudocount : int or float, optional
        Pseudocount added to sequence counts to avoid division by zero when
        calculating frequencies. Default is 1.
    iptg_out_growth_time : int, optional
        Growth time outside of IPTG induction. Default is 30.

    Returns
    -------
    out : dict
        A dictionary containing the processed data. The dictionary has the
        following keys:

        *   "genotypes" : numpy.ndarray
            1D array of genotypes. Shape (num_genotypes*num_conditions)
        *   "conditions" : numpy.ndarray
            1D array of base conditions. Shape (num_genotypes*num_conditions)
        *   "times" : numpy.ndarray
            2D array of time points, including the added time zero.
            Shape: (num_times,num_genotypes*num_conditions)
        *   "cfu" : numpy.ndarray
            2D array of CFU values, expanded to include the time zero point.
            Shape: (num_times,num_genotypes*num_conditions)
        *   "cfu_var" : numpy.ndarray
            2D array of CFU variances, expanded to include the time zero point.
            Shape: (num_times,num_genotypes*num_conditions)
        *   "ln_cfu" : numpy.ndarray
            2D array of natural log of CFU values, expanded to include the
            time zero point.
            Shape: (num_times,num_genotypes*num_conditions)
        *   "ln_cfu_var" : numpy.ndarray
            2D array of variances of natural log of CFU values, expanded to
            include the time zero point. 
            Shape: (num_times,num_genotypes*num_conditions)
        *   "growth_rate_wt" : numpy.ndarray
            1D array of wildtype growth rate under the specified conditions.
            Shape (num_genotypes*num_conditions)
        *   "growth_rate_wls" : numpy.ndarray
            1D array of genotype/condition growth rates estimated by weighted 
            linear regression on ln_cfu. Shape (num_genotypes*num_conditions)
        *   "growth_rate_err_wls" : numpy.ndarray
            1D array of growth rate variance for each genotype/condition
            estimated by weighted linear regression on ln_cfu.
            Shape (num_genotypes*num_conditions)
    """

    combined_df = pd.read_csv(combined_df)
    condition_df = pd.read_csv(condition_df)

    # Convert the dataframe into a collection of numpy arrays
    _results = _count_df_to_arrays(combined_df)
    
    # 2D arrays (num_times,num_conditions*num_genotypes)
    times = _results[0]
    sequence_counts = _results[1]
    total_counts = _results[2]
    total_cfu_ml = _results[3]

    # 1D arrays (num_genotypes*num_conditions). 
    genotypes = _results[4]
    base_conditions = _results[5]
    
    num_conditions = len(pd.unique(base_conditions))
    num_genotypes = len(pd.unique(genotypes))
    
    # Extract ln_cfu and ln_cfu_variance from the count data
    ln_cfu, ln_cfu_var = _get_ln_cfu(sequence_counts=sequence_counts,
                                     total_counts=total_counts,
                                     total_cfu_ml=total_cfu_ml,
                                     pseudocount=pseudocount)
    
    # Estimate ln_cfu at t = 0 and extend the times, ln_cfu, and ln_cfu_var 
    # arrays with these data
    _results = _build_time_zero(times=times,
                                ln_cfu=ln_cfu,
                                ln_cfu_var=ln_cfu_var,
                                condition_df=condition_df,
                                num_genotypes=num_genotypes,
                                num_conditions=num_conditions,
                                iptg_out_growth_time=iptg_out_growth_time)
    
    times_expanded, ln_cfu_expanded, ln_cfu_var_expanded = _results

    # Calculate CFU and variance from the ln_cfu results
    cfu_expanded = np.exp(ln_cfu_expanded)
    cfu_var_expanded = (cfu_expanded**2)*ln_cfu_var_expanded

    growth_rate_wt = _get_k_guess_from_wt(condition_df=condition_df,
                                          num_genotypes=num_genotypes)

    # Do a final 
    k_est, _, k_err, _, _ = fast_weighted_linear_regression(x_arrays=times_expanded,
                                                            y_arrays=ln_cfu_expanded,
                                                            y_err_arrays=ln_cfu_var_expanded)

    out = {"genotypes":genotypes,
           "conditions":base_conditions,
           "times":times_expanded,
           "cfu":cfu_expanded,
           "cfu_var":cfu_var_expanded,
           "ln_cfu":ln_cfu_expanded,
           "ln_cfu_var":ln_cfu_var_expanded,
           "growth_rate_wt":growth_rate_wt,
           "growth_rate_wls":k_est,
           "growth_rate_err_wls":k_err}

    return out