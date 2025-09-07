from tfscreen.calibration import get_wt_k
from tfscreen.analyze.get_growth_rates.wls import get_growth_rates_wls

import numpy as np
import pandas as pd

def _estimate_lnA0(times,
                   ln_cfu,
                   ln_cfu_var,
                   no_select_mask,
                   k_wt0,
                   iptg_out_growth_time):
    """
    Estimate t = 0 points from all samples with a specific genotype and create a
    new pseudo datapoint with ln_cfu and ln_cfu_var at t = 0. 

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
    no_select_mask : numpy.ndarray
        1D bool numpy array that is num_samples long and selects sample 
        conditions that did not have selection in them. 
    wt_k0 : float
        1D numpy array holding the wildtype growth rate in the absence of
        selection for each sample condition. Shape (num_genotype*num_samples,). 
    iptg_out_growth_time : float
        how long the cultures grew in iptg before being put under selection. 
        Units should match units in times array. 
        
    Returns
    -------
    lnA0_est : numpy.ndarray
        1D array with shape (num_genotypes*num_samples,) holding an estimate of
        lnA0 for each genotype in each sample
    lnA0_var : numpy.ndarray
        1D array with shape (num_genotypes*num_samples,) holding the variance on
        the estimate of lnA0 for each genotype in each sample
    param_df : numpy.ndarray
        dataframe with columns "lnA_pre0_est", "lnA_pre0_var", and "k_shift"
        for each genotype.
    """

    # Number of samples and genotypes. This works because times is sorted by
    # genotypes then samples. It is genotypes*num_samples long, so dividing by
    # num_samples gives num genotypes. 
    num_samples = len(no_select_mask)
    num_genotypes = times.shape[0]//num_samples

    # Do a fit, extracting growth rates and estimates of A0 for each row in 
    # times/ln_cfu/ln_cfu_var. Each row corresponds to a genotype growing in a
    # particular sample. 
    param_df, _ = get_growth_rates_wls(times,ln_cfu,ln_cfu_var)
    k_est = param_df["k_est"].values
    lnA0 = param_df["A0_est"].values
    lnA0_var = (param_df["A0_std"].values)**2

    # Shift in growth rate in each sample relative to wildtype no-selection
    # expectation
    k_shift = k_est - k_wt0

    # Get mean of k for each genotype in each sample that has no selection
    # applied. (This assumes growth with/without marker and with/without iptg
    # is similar). Grab k_est for rows not under selection using no_select_mask.
    # Reshape into samples-as-columns and take the mean of each row. This 
    # yields an array k_before_select that has the average growth rate of each 
    # genotype under non-selective conditions. 
    to_mean = k_shift[np.tile(no_select_mask,num_genotypes)]
    to_mean = to_mean.reshape((num_genotypes, np.sum(no_select_mask)))
    k_shift = np.nanmean(to_mean,axis=1)
    
    # Any bad estimates -- replace with 0.0
    k_shift[np.isnan(k_shift)] = 0.0
    
    # k_before_select is the growth rate of each genotype in the absence of
    # selection. This is the wildtype growth rate in that sample condition plus
    # the genotype-specific shift.
    k_before_select = k_wt0 + np.repeat(k_shift,num_samples)

    # Calculate how much each genotype in each sample would grow in the
    # pre-selection interval
    pre_growth = k_before_select*iptg_out_growth_time

    # Re-shape lnA0, lnA0_var, and pre_growth. In this new shape, genotypes are 
    # rows and samples are columns. 
    new_dim = (lnA0.shape[0]//num_samples,num_samples)
    lnA0_reshaped = lnA0.reshape(new_dim)
    lnA0_var_reshaped = lnA0_var.reshape(new_dim)
    pre_growth = pre_growth.reshape(new_dim)

    # Grow backwards in time by pre_growth, finding the lnA the mother sample 
    # was split into different tubes with different iptg. **Because the samples 
    # are split from the same starter culture, lnA_pre0 for a given genotype
    # should be the same for all samples.**
    lnA_pre0 = lnA0_reshaped - pre_growth
    
    # Because of uncertainty in our measurements, fits, and growth rates, 
    # lnA_pre0 will be different in each sample. Take the weighted average
    # of the estimate from each sample. This is our estimate of the shared 
    # lnA_pre0 for each genotype across samples. 
    lnA0_weight = 1/lnA0_var_reshaped
    lnA0_weight = lnA0_weight/np.sum(lnA0_weight,axis=1,keepdims=True)
    lnA_pre0_mean = np.average(lnA_pre0,
                                   weights=lnA0_weight,
                                   axis=1)
    
    # Get the weighted variance of our estimates of lnA_pre0. We're keeping
    # full variance rather than doing variance on the mean because we do not
    # want a tiny variance on our fake zero point to dominate the fit.
    lnA_pre0_var = np.sum(lnA0_weight*(lnA_pre0 - lnA_pre0_mean[:,np.newaxis])**2,
                              axis=1)
    
    # lnA0 for each genotype/sample will be lnA_pre0 of each genotype plus
    # the out-growth prior to adding selection. The out-growth will depend 
    # slightly on the conditions, but possibly a lot based on the genotype. 
    lnA0_est = np.repeat(lnA_pre0_mean,num_samples) + pre_growth.flatten()
    
    # Get lnA0 variance on a pre-sample basis
    lnA0_var = np.repeat(lnA_pre0_var,num_samples)

    # Per-genotype dataframe with lnA0 and k_shift estimates
    param_df = pd.DataFrame({"lnA_pre0_est":lnA_pre0_mean,
                             "lnA_pre0_var":lnA_pre0_var,
                             "k_shift":k_shift})    

    return lnA0_est, lnA0_var, param_df


def estimate_time0(times,
                   ln_cfu,
                   ln_cfu_var,
                   sample_df,
                   calibration_data,
                   iptg_out_growth_time=30,
                   num_iterations=3):
    """
    Independently estimate the growth rates of each genotype in each sample, 
    then use these fits to estimate the shared initial ln(CFU) for each
    genotype in the initial culture before any outgrowth. This is done by:

    1. Independently fit lnA(t) ~ lnA0 + kt for each genotypes in each
       sample. This gives us lnA0 at the start of selection, after a
       pre-selection outgrowth.
    2. Calculate the expected growth of each genotype in each sample over the
       pre-selection interval using the wildtype growth rate in the absence of 
       selection plus the genotype-specific shift in growth rate estimated from
       our fits.  
    3. Subtract the pre-selection growth from each lnA0. This gives us an 
       estimate, from each sample, of the initial ln(CFU) for genotypes in the
       initial culture (lnA_pre0).
    4. Calculate the mean lnA_pre0 from all samples, weighting by standard error
       on the lnA0_pre0 estimate.
    5. For each sample, add the pre-selection growth back to the estimate of
       ln(CFU) averaged over all samples. This gives us sample-specific lnA0 
       for each genotype after the pre-selection outgrowth.

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
        DataFrame with information about each sample, including columns for
        "select", "marker", and "iptg".
    calibration_data : dict or str
        Calibration data to use for estimating wildtype growth rates. This can
        either be a path to a calibration data file or a dictionary containing
        calibration data.
    iptg_out_growth_time : float, optional
        how long the cultures grew in iptg before being put under selection.
        Units should match units in times array. Default is 30.
    num_iterations : int, optional
        Number of iterations to perform when estimating time0. Default is 3.
        This means that we will estimate time0 three times, each time using the
        previous estimate to refine the next estimate.

    Returns
    -------
    param_df : pandas.DataFrame
        DataFrame with columns "lnA_pre0_est", "lnA_pre0_var", and "k_shift"
        for each genotype. lnA_pre0 is the estimated ln(CFU) at the start of the
        pre-selection outgrowth, and k_shift is the estimated global growth rate
        shift of that genotype relative to wildtype.
    times : numpy.ndarray
        times array with a new zero point added to the front of it. Shape
        (num_time_points+1,num_genotypes*num_samples)
    ln_cfu : numpy.ndarray
        ln_cfu array with a new zero point added to the front of it. Shape
        (num_time_points+1,num_genotypes*num_samples)
    ln_cfu_var : numpy.ndarray
        ln_cfu_var array with a new zero point added to the front of it. Shape
        (num_time_points+1,num_genotypes*num_samples)
    """

    # Get every sample row that is not under selection
    no_select_mask = np.array(sample_df["select"] == 0,dtype=bool)
    num_samples = len(sample_df["select"])
    num_genotypes = times.shape[0]//num_samples
    
    # Get the wildtype growth rate without selection for model outgrowth
    k_wt0, _ = get_wt_k(marker=sample_df["marker"],
                        select=np.zeros(len(sample_df["select"]),dtype=int),
                        iptg=sample_df["iptg"],
                        calibration_data=calibration_data,
                        calc_err=False)
    k_wt0 = np.tile(k_wt0,num_genotypes)

    # Useful templates for later
    time_block = np.zeros(times.shape[0])
    times_template = times.copy()
    ln_cfu_template = ln_cfu.copy()
    ln_cfu_var_template = ln_cfu_var.copy()

    # Iteratively update our estimates of lnA0_est and lnA0_var.
    # The first iteration has no zero point. The second iteration has a zero 
    # point, but it was estimated (maybe poorly) using the first iteration that
    # had no zero point. By the third iteration, we are now using zero points 
    # estimated using previously estimated zero points. 
    for _ in range(num_iterations):

        # Estimate lnA0 and the k_before_select for each genotype
        lnA0_est, lnA0_var, param_df = _estimate_lnA0(
            times=times,
            ln_cfu=ln_cfu,
            ln_cfu_var=ln_cfu_var,
            no_select_mask=no_select_mask,
            k_wt0=k_wt0,
            iptg_out_growth_time=iptg_out_growth_time
        )
    
        # Build new times, ln_cfu, and ln_cfu_var arrays that have a new fake 
        # zero point on them.
        times = np.hstack([time_block[:,np.newaxis],times_template])
        ln_cfu = np.hstack([lnA0_est[:,np.newaxis],ln_cfu_template])
        ln_cfu_var = np.hstack([lnA0_var[:,np.newaxis],ln_cfu_var_template])
                             
    return param_df, times, ln_cfu, ln_cfu_var




    