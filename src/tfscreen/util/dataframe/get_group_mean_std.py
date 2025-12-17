import numpy as np

def get_group_mean_std(values, groups):
    """
    Calculate the mean and standard deviation of values for groups in an array.

    This function calculates the mean and sample standard deviation of a set of
    1D values, grouped according to a corresponding array of group assignments.
    It uses `np.bincount` for efficient summation and counting.

    Parameters
    ----------
    values : array_like
        A 1D array of numerical values.
    groups : array_like
        A 1D array of group assignments, with the same shape as `values`.
        Each element in `groups` specifies the group to which the
        corresponding element in `values` belongs. Group numbers must be
        non-negative integers.

    Returns
    -------
    values_means : np.ndarray
        An array of the same shape as `values`, where each element
        contains the mean value for its group.
    values_stds : np.ndarray
        An array of the same shape as `values`, where each element contains the
        sample standard deviation for its group. Standard deviations for
        groups with 1 or 0 members are set to 0.

    Raises
    ------
    ValueError
        If `values` and `groups` do not have the same shape or if `groups`
        contains negative values.

    Examples
    --------
    >>> values = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([0, 0, 1, 1, 2, 2])
    >>> means, stds = get_group_mean_std(values, groups)
    >>> print(np.round(means, 2))
    [1.5 1.5 3.5 3.5 5.5 5.5]
    >>> print(np.round(stds, 2))
    [0.71 0.71 0.71 0.71 0.71 0.71]
    """

    # Convert to NumPy arrays and perform basic shape check
    values = np.asarray(values)
    groups = np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError("values and groups must be arrays of the same shape.")

    # Ensure groups are non-negative integers
    if not np.issubdtype(groups.dtype, np.integer):
        
        # Check if float array can be safely cast to int
        if not np.all(groups == np.floor(groups)):
            raise ValueError("groups array must contain integer values.")
        groups = groups.astype(int)
    
    if np.any(groups < 0):
        raise ValueError("group numbers must be non-negative.")

    # Get counts and sums for each group
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=values)
    sum_x2 = np.bincount(groups, weights=values**2)

    # Calculate means safely, avoiding division by zero for empty groups
    group_means = np.divide(sums, counts,
                            out=np.zeros_like(sums, dtype=float),
                            where=(counts != 0))

    # --- Calculate sample standard deviation ---
    # Using the formula: std = sqrt( (E[X^2] - E[X]^2) * (n / (n-1)) )
    mean_of_squares = np.divide(sum_x2, counts,
                                out=np.zeros_like(sum_x2, dtype=float),
                                where=(counts != 0))
    
    pop_variance = mean_of_squares - group_means**2

    # Correct for sample variance (Bessel's correction)
    correction_factor = np.divide(counts, counts - 1,
                                  out=np.ones_like(counts, dtype=float),
                                  where=(counts > 1))

    sample_variance = pop_variance * correction_factor
    group_stds = np.sqrt(sample_variance)

    # Map group statistics back to the original array shape
    values_means = group_means[groups]
    values_stds = group_stds[groups]

    return values_means, values_stds