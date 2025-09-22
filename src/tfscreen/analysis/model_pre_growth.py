from tfscreen.util import (
    get_group_mean_std
)

import numpy as np

def _validate_group_array(arr, target_shape, name):
    """Helper function to validate optional group arrays."""
    if arr is None:
        return
    
    is_invalid = (
        np.isscalar(arr) or
        arr.ndim > 1 or
        arr.shape[0] != target_shape[0] or
        not np.all(arr >= 0) or
        not np.all(arr == np.floor(arr))
    )
    
    if is_invalid:
        err = f"If specified, {name} must be a 1D array of integers >= 0\n"
        err += f"the same length as the other input arrays."
        raise ValueError(err)

def _process_dk_geno(k_est, k_wt, dk_geno_groups, dk_geno_mask):
    """
    Calculates the change in growth rate (dk_geno) by averaging over
    a masked subset of each group and broadcasting the result to all
    group members.
    """
    # Calculate initial delta-k for all samples
    dk = k_est - k_wt

    # Filter for the values and groups to be used in the averaging
    values_to_average = dk[dk_geno_mask]
    groups_for_average = dk_geno_groups[dk_geno_mask]
    
    # Calculate the per-group summary mean directly, avoiding the double broadcast.
    # This mimics the internal logic of get_group_mean_std.
    sums = np.bincount(groups_for_average, weights=values_to_average)
    counts = np.bincount(groups_for_average)

    # Ensure the group_means array is large enough for all original groups
    max_group_id = np.max(dk_geno_groups)
    group_means = np.zeros(max_group_id + 1, dtype=float)

    # Use safe division to calculate the mean only for groups present in the subset
    valid_indices = counts > 0
    group_means[valid_indices] = sums[valid_indices] / counts[valid_indices]

    # Broadcast the calculated group mean to ALL members of the original groups
    dk_geno = group_means[dk_geno_groups]

    return dk_geno


def _process_lnA0(lnA0_est, lnA0_std, kt_pre, lnA0_groups):
    """
    Calculates the pre-growth lnA0, optionally averaging over groups and
    propagating the associated error.
    """
    # Calculate the initial pre-growth population size for each sample
    lnA0_pre = lnA0_est - kt_pre

    if lnA0_groups is not None:
        
        # Get per-group summary statistics (mean, std, count)
        counts = np.bincount(lnA0_groups)
        sums = np.bincount(lnA0_groups, weights=lnA0_pre)
        sum_x2 = np.bincount(lnA0_groups, weights=lnA0_pre**2)

        group_means = np.divide(sums, counts,
                                out=np.zeros_like(sums, dtype=float),
                                where=(counts != 0))

        mean_of_squares = np.divide(sum_x2, counts,
                                    out=np.zeros_like(sum_x2, dtype=float),
                                    where=(counts != 0))
        pop_variance = mean_of_squares - group_means**2
        correction_factor = np.divide(counts, counts - 1,
                                      out=np.ones_like(counts, dtype=float),
                                      where=(counts > 1))
        group_stds = np.sqrt(pop_variance * correction_factor)

        # Broadcast the shared group mean back to the per-sample arrays
        lnA0_pre = group_means[lnA0_groups]
        lnA0_est = lnA0_pre + kt_pre

        # Calculate the Standard Error of the Mean (SEM) for each group.
        # This now works because group_stds and counts have compatible shapes.
        group_sem = np.divide(group_stds, np.sqrt(counts),
                              out=np.zeros_like(group_stds),
                              where=(counts > 1))

        # Map the group SEM to each individual sample.
        sem_per_sample = group_sem[lnA0_groups]

        # Add the variances in quadrature and take the square root.
        lnA0_std = np.sqrt(lnA0_std**2 + sem_per_sample**2)

    return lnA0_pre, lnA0_est, lnA0_std

def model_pre_growth(k_est,
                     lnA0_est,
                     lnA0_std,
                     k_wt,
                     t_pre,
                     dk_geno_groups=None,
                     dk_geno_mask=None,
                     lnA0_groups=None):
    """
    Model the effect of pre-selection growth on initial population size.

    This function models the effect of a pre-growth phase on the estimated
    growth rate and initial population size (lnA0) of a set of genotypes. It
    estimates the global effect of each genotype on growth rate and adjusts
    the lnA0 values to account for the pre-growth phase. It also allows for
    grouping observations to estimate a shared initial population size. 

    Parameters
    ----------
    k_est : numpy.ndarray
        Array of estimated growth rates for each sample.
    lnA0_est : numpy.ndarray
        Array of estimated log-transformed initial population sizes (lnA0) for
        each sample.
    lnA0_std : numpy.ndarray
        Array of standard errors on the estimated lnA0 values.
    k_wt : float or numpy.ndarray
        Wild-type growth rate. If a scalar, it is applied to all samples. If an
        array, it must have the same shape as `k_est`.
    t_pre : float or numpy.ndarray
        Pre-growth time. If a scalar, it is applied to all samples. If an array,
        it must have the same shape as `k_est`.
    dk_geno_groups : numpy.ndarray, optional
        Array of group assignments for values in k_est, used to estimate a shared
        growth rate effect (i.e., the effect of a genotype). If specified, it
        must be a 1D array of integers > 0 the same length as `k_est`.
    dk_geno_mask : numpy.ndarray, optional
        Array of samples that are best for estimating dk_geno. If specified, it
        must be a 1D array of bool the same length as `k_est`. If a given 
        dk_geno_group has all dk_geno_mask == False, all members of the group 
        will be used for the calculation. 
    lnA0_groups : numpy.ndarray, optional
        Array of group assignments for samples, used to estimate a shared
        initial population size (lnA0_pre). If specified, it must be a 1D array
        of integers > 0 the same length as `k_est`.

    Returns
    -------
    dk_geno : numpy.ndarray
        Change in growth rate in each sample relative to wildtype.
    lnA0_pre : numpy.ndarray
        Estimated log-transformed initial population size (lnA0) before the
        pre-growth phase.
    lnA0_est : numpy.ndarray
        Updated array of estimated log-transformed initial population sizes (lnA0)
        for each sample, accounting for the pre-growth phase.
    lnA0_std : numpy.ndarray
        Updated array of standard deviations for the estimated lnA0 values,
        incorporating the uncertainty in the shared initial population size
        (lnA0_pre), where applicable.

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes or if the group assignment
        arrays are not valid.
    """

    # --- Parameter Sanity Checking ---
    # (Validation code remains here, with a new check for dk_geno_mask)
    if not (k_est.ndim == 1 and k_est.shape == lnA0_est.shape and k_est.shape == lnA0_std.shape):
        raise ValueError("k_est, lnA0_est, and lnA0_std must be 1D arrays of the same length.")

    _validate_group_array(dk_geno_groups, k_est.shape, "dk_geno_groups")
    _validate_group_array(lnA0_groups, k_est.shape, "lnA0_groups")
    
    if dk_geno_mask is not None:
        if not (dk_geno_mask.ndim == 1 and dk_geno_mask.shape == k_est.shape and dk_geno_mask.dtype == bool):
            raise ValueError("dk_geno_mask must be a 1D boolean array of the same length as k_est.")

    # Check that mask and groups are specified together
    if (dk_geno_groups is None) != (dk_geno_mask is None):
        raise ValueError("dk_geno_groups and dk_geno_mask must be specified together.")


    # --- Broadcast scalars to arrays ---
    if np.isscalar(k_wt):
        k_wt = np.full(k_est.shape, k_wt)
    if np.isscalar(t_pre):
        t_pre = np.full(k_est.shape, t_pre)
    
    # --- Estimate growth rate effect (dk_geno) ---
    if dk_geno_groups is not None and dk_geno_mask is not None:
        dk_geno = _process_dk_geno(k_est, k_wt, dk_geno_groups, dk_geno_mask)
    else:
        dk_geno = k_est - k_wt

    # --- Calculate total growth during pre-incubation ---
    if dk_geno_groups is not None:
        k_pre = k_wt + dk_geno
    else:
        k_pre = k_wt
    kt_pre = k_pre * t_pre

    # --- Estimate initial population size (lnA0) and propagate error ---
    lnA0_pre, lnA0_est, lnA0_std = _process_lnA0(
        lnA0_est, lnA0_std, kt_pre, lnA0_groups
    )

    return dk_geno, lnA0_pre, lnA0_est, lnA0_std


