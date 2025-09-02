from tfscreen.util import read_dataframe
from tfscreen.fitting import fast_weighted_linear_regression
from tfscreen.calibration import predict_growth_rate

import pandas as pd
import numpy as np

def _count_df_to_arrays(df):
    """
    Take a dataframe with columns "genotype", "sample", "time",
    "counts", "total_counts_at_time", and "total_cfu_at_time" return numpy
    arrays with shape (num_obs,num_times) where num_obs iterates 
    over all combinations of genotype and and samples. There must be
    an identical number of time points for each genotype/sample pair,
    though the time point values can all differ from one another. 

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to process

    Returns
    -------
    times : np.ndarray
        2D float array of times with shape (num_obs,num_times)
    sequence_counts : np.ndarray 
        2D float array of counts of particular genotype with shape
        (num_obs,num_times)
    total_counts : np.ndarray
        2D float array of total counts in sample with shape
        (num_obs,num_times)
    total_cfu_ml : np.ndarray
        2D float array of total cfu/mL in sample with shape
        (num_obs,num_times)
    genotypes : np.ndarray
        1D object array of genotypes with shape (num_obs,)
    samples : np.ndarray
        1D object array of sample strings with shape
        (num_obs,)
    """

    # Check for all required columns
    look_for = [
        "genotype",
        "sample",
        "time",
        "counts",
        "total_counts_at_time",
        "total_cfu_mL_at_time"
        ]
    diff = list(set(look_for) - set(df.columns))
    diff.sort()
    if len(diff) > 0:
        err = "not all columns found in dataframe. missing columns:\n"
        for d in diff:
            err += f"    {d}\n"
        raise ValueError(err + "\n")
        
    # Get number of times per experiment
    number_of_times_per_expt = df.groupby(['genotype', 'sample']).size().unique()
    if len(number_of_times_per_expt) != 1:
        err = "each genotype/sample must have the same number of time entries\n"
        raise ValueError(err)

    num_total_obs = len(df["time"])
    num_t = number_of_times_per_expt[0]

    # Extract relevant series as arrays
    times = np.array(df["time"])
    sequence_counts = np.array(df["counts"])
    total_counts = np.array(df["total_counts_at_time"])
    total_cfu_ml = np.array(df["total_cfu_mL_at_time"])

    # Reshape into num_obs x time arrays
    times = times.reshape(num_total_obs//num_t,num_t)
    sequence_counts = sequence_counts.reshape(num_total_obs//num_t,num_t)
    total_counts = total_counts.reshape(num_total_obs//num_t,num_t)
    total_cfu_ml = total_cfu_ml.reshape(num_total_obs//num_t,num_t)

    # Get genotypes and samples for masking arrays
    genotypes = np.array(df["genotype"])
    genotypes = genotypes[np.arange(0,num_total_obs,num_t,dtype=int)]
    
    samples = np.array(df["sample"])
    samples = samples[np.arange(0,num_total_obs,num_t,dtype=int)]
    
    return times, sequence_counts, total_counts, total_cfu_ml, genotypes, samples

def _get_ln_cfu(sequence_counts,
                total_counts,
                total_cfu_ml,
                pseudocount=1):
    """
    Convert sequence counts to frequencies and variances. 

    Parameters
    ----------
    sequence_counts : numpy.ndarray
        Array of sequence counts for a specific genotype/sample.
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
                     sample_df,
                     num_genotypes,
                     num_samples,
                     iptg_out_growth_time,
                     calibration_dict):
    """
    Estimate t = 0 points from all samples with a specific genotype and create a
    new pseudo datapoint with ln_cfu and ln_cfu_var at t = 0. This is done by:

    1. Infer lnA0 and lnA0 for all genotypes in each sample. This is lnA0 at the 
       start of selection, after a pre-selection outgrowth in the relevant IPTG
       concentration.
    2. Calculate the expected growth of each genotype in each sample over the
       pre-selection interval. 
    3. Subtract the pre-selection growth from each lnA0. This gives us an 
       independent estimate of the initial ln(CFU) for the genotype in the 
       initial culture. 
    4. Calculate the mean and standard deviation of this initial ln(CFU) from
       all samples. Weight mean and stdev by 1/parameter_std_err^2.
    5. For each sample, add the pre-selection growth back to the estimate of
       ln(CFU) averaged over all samples.
    6. Add these new lnA0 values as t = 0 to ln_cfu and ln_cfu_var arrays. 

    Parameters
    ----------
    times : numpy.ndarray
        2D array. Time points corresponding to each genotype/sample. 
        Shape (num_time_points,num_genotypes*num_samples)
    ln_cfu : numpy.ndarray
        2D array. Natural logarithm of the CFU/mL for each genotype/sample
        Shape (num_time_points,num_genotypes*num_samples)
    ln_cfu_var : numpy.ndarray
        2D array. Variance of the natural logarithm of the CFU/mL for each
        genotype/sample. Shape (num_time_points,num_genotypes*num_samples)  
    sample_df : pandas.DataFrame
        Dataframe with samples. Should have columns "marker", "select" and 
        "iptg".
    num_genotypes : int
        number of genotypes
    num_samples : int
        number of samples
    iptg_out_growth_time : float
        how long the cultures grew in iptg before being put under selection. 
        Units should match other units.
    calibration_dict : dict
        calibration_dict dictionary
    
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
        
    # Reshape by sample and extract weights
    lnA0_reshaped = lnA0.reshape((lnA0.shape[0]//num_samples,
                                  num_samples))
    lnA0_err_reshaped = lnA0_err.reshape((lnA0_err.shape[0]//num_samples,
                                          num_samples))
        
    # Calculate how much the genotype grew during the IPTG outgrowth prior 
    # to selection.
    pre_selection_k, _ = predict_growth_rate(marker=np.array(sample_df["marker"]),
                                             select=np.zeros(len(sample_df["select"]),dtype=int),
                                             iptg=np.array(sample_df["iptg"]),
                                             calibration_dict=calibration_dict)

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
    
    # Add back the pre-selection growth to each sample and append as a new
    # time zero.
    repeated_pre_zero = np.repeat(ln_A_pre_zero_mean,num_samples)
    repeated_pre_growth = np.tile(pre_growth,num_genotypes)
    repeated_ln_A0 = repeated_pre_zero + repeated_pre_growth
    ln_cfu_expanded = np.hstack([repeated_ln_A0[:,np.newaxis],ln_cfu])
    
    # Append ln_cfu_variance as new time zero
    repeated = np.repeat(ln_A_pre_zero_var,num_samples)
    ln_cfu_var_expanded = np.hstack([repeated[:,np.newaxis],ln_cfu_var])
    
    # Add time zero
    time_block = np.zeros(times.shape[0])
    times_expanded = np.hstack([time_block[:,np.newaxis],times])

    return times_expanded, ln_cfu_expanded, ln_cfu_var_expanded

def _get_k_guess_from_wt(sample_df,
                         num_genotypes,
                         calibration_dict):
    """
    Get guesses for rates for wildtype in each fo the samples in sample_df.

    Parameters
    ----------
    sample_df : pandas.DataFrame
        pandas dataframe with samples
    num_genotypes : int
        number of genotypes
    calibration_dict : dict
        calibration_dict dictionary

    Returns
    -------
    growth_rate_wt : numpy.ndarray
        1D numpy array of growth rates. Shape: (num_genotypes*num_samples,)
    """
    
    iptg = np.array(sample_df["iptg"])
    marker = np.array(sample_df["marker"])
    select = np.array(sample_df["select"])
    
    # Get guesses for growth rates from wildtype data
    growth_rate_wt, _ = predict_growth_rate(marker=marker,
                                            select=select,
                                            iptg=iptg,
                                            calibration_dict=calibration_dict)
    growth_rate_wt = np.tile(growth_rate_wt,num_genotypes)

    return growth_rate_wt


def process_for_fit(combined_df,
                    sample_df,
                    calibration_dict,
                    pseudocount=1,
                    iptg_out_growth_time=30):
    """
    Process dataframes for fitting growth parameters.

    This function takes two dataframes, a combined dataframe containing
    sequence counts, cfu_mL, and total reads, as well as a sample dataframe
    specifying the conditions for each sample. The function performs several
    processing steps to prepare the data for fitting. These steps include:

        1. Combining the dataframes to yield numpy arrays of times, cfu, and 
           ln_cfu. 
        2. Estimating and adding a t = 0 datapoint for each genotype in each 
           sample.
        3. Generating plausible initial guesses of growth rate for further
           regression.

    The analysis assumes that combined_df is sorted by genotype, then sample. 
    The combined_df should be exhaustive, having all genotypes in all 
    samples. Genotypes not seen in a particular sample should still be 
    present, just given counts of zero. 

    Parameters
    ----------
    combined_df : str
        combined dataframe. This dataframe should have columns "genotype",
        "sample", "time", "counts", "total_counts_at_time", and
        "total_cfu_mL_at_time".
    sample_df : str
        sample dataframe. This dataframe should have columns "marker", "select"
        and "iptg" and be indexed by 'sample' (replicate|marker|select|iptg)
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
            1D array of genotypes. Shape (num_genotypes*num_samples)
        *   "samples" : numpy.ndarray
            1D array of samples. Shape (num_genotypes*num_samples)
        *   "times" : numpy.ndarray
            2D array of time points, including the added time zero.
            Shape: (num_times,num_genotypes*num_samples)
        *   "cfu" : numpy.ndarray
            2D array of CFU values, expanded to include the time zero point.
            Shape: (num_times,num_genotypes*num_samples)
        *   "cfu_var" : numpy.ndarray
            2D array of CFU variances, expanded to include the time zero point.
            Shape: (num_times,num_genotypes*num_samples)
        *   "ln_cfu" : numpy.ndarray
            2D array of natural log of CFU values, expanded to include the
            time zero point.
            Shape: (num_times,num_genotypes*num_samples)
        *   "ln_cfu_var" : numpy.ndarray
            2D array of variances of natural log of CFU values, expanded to
            include the time zero point. 
            Shape: (num_times,num_genotypes*num_samples)
        *   "growth_rate_wt" : numpy.ndarray
            1D array of wildtype growth rate in each sample
            Shape (num_genotypes*num_samples)
        *   "growth_rate_wls" : numpy.ndarray
            1D array of genotype/sample growth rates estimated by weighted 
            linear regression on ln_cfu. Shape (num_genotypes*num_samples)
        *   "growth_rate_err_wls" : numpy.ndarray
            1D array of growth rate variance for each genotype/sample
            estimated by weighted linear regression on ln_cfu.
            Shape (num_genotypes*num_samples)
    """

    combined_df = read_dataframe(combined_df)
    sample_df = read_dataframe(sample_df)

    # Convert the dataframe into a collection of numpy arrays
    _results = _count_df_to_arrays(combined_df)
    
    # 2D arrays (num_times,num_genotypes*num_samples)
    times = _results[0]
    sequence_counts = _results[1]
    total_counts = _results[2]
    total_cfu_ml = _results[3]

    # 1D arrays (num_genotypes*num_samples). 
    genotypes = _results[4]
    samples = _results[5]

    num_samples = len(pd.unique(samples))
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
                                sample_df=sample_df,
                                num_genotypes=num_genotypes,
                                num_samples=num_samples,
                                iptg_out_growth_time=iptg_out_growth_time,
                                calibration_dict=calibration_dict)
    
    times_expanded, ln_cfu_expanded, ln_cfu_var_expanded = _results

    # Calculate CFU and variance from the ln_cfu results
    cfu_expanded = np.exp(ln_cfu_expanded)
    cfu_var_expanded = (cfu_expanded**2)*ln_cfu_var_expanded

    growth_rate_wt = _get_k_guess_from_wt(sample_df=sample_df,
                                          num_genotypes=num_genotypes,
                                          calibration_dict=calibration_dict)

    # Do a final 
    k_est, _, k_err, _, _ = fast_weighted_linear_regression(x_arrays=times_expanded,
                                                            y_arrays=ln_cfu_expanded,
                                                            y_err_arrays=ln_cfu_var_expanded)

    out = {"genotypes":genotypes,
           "samples":samples,
           "sequence_counts":sequence_counts,
           "times":times_expanded,
           "cfu":cfu_expanded,
           "cfu_var":cfu_var_expanded,
           "ln_cfu":ln_cfu_expanded,
           "ln_cfu_var":ln_cfu_var_expanded,
           "growth_rate_wt":growth_rate_wt,
           "growth_rate_wls":k_est,
           "growth_rate_err_wls":k_err}

    return out