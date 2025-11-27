import numpy as np

def xfill(x,
          num_points=100,
          use_log=None,
          pad_by=0.1):
    """
    Smoothly fill points between minimum and maximum values in x.

    Ensures that all original values in 'x' are present in the output
    (even if this means the spacing is not perfectly smooth). This allows
    one-to-one comparisons between measured values and a model calculated
    using the filled-in x values.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of values.
    num_points : int, optional
        Number of points in the final array, by default 100.
    use_log : bool, optional
        Interpolate on a log scale. If None, the function chooses based
        on the data's dynamic range.
    pad_by : float, optional
        Expand the range beyond the min/max of x by this factor, by default 0.1.
    
    Returns
    -------
    np.ndarray
        An array of expanded values with a length of `num_points`.
    """
    
    x_finite = x[np.isfinite(x)]

    # Handle edge cases: empty or single-point arrays
    if x_finite.size == 0:
        return np.array([])
    if x_finite.size == 1:
        return np.full(num_points, x_finite[0])

    # Determine if a log scale should be used
    if use_log and np.min(x_finite) < 0:
        raise ValueError("x has negative values. use_log cannot be set to True.")

    if use_log is None:
        x_pos = x_finite[x_finite > 0]
        # Use log scale if data is all non-negative and spans 3+ orders of magnitude
        if np.min(x_finite) >= 0 and x_pos.size > 1 and \
           np.max(x_pos) / np.min(x_pos) > 1000:
            use_log = True
        else:
            use_log = False

    if use_log:
        x_pos = x_finite[x_finite > 0]
        # If only zeros were present, fall back to linear
        if x_pos.size < 2:
             use_log = False # Can't make a log scale from one point or less
        else:
            log_span = np.log(np.max(x_pos)) - np.log(np.min(x_pos))
            x_min_log = np.log(np.min(x_pos)) - log_span * pad_by
            x_max_log = np.log(np.max(x_pos)) + log_span * pad_by
            x_filled = np.exp(np.linspace(x_min_log, x_max_log, num_points))

    # Fallback to linear scale if use_log is or became False
    if not use_log:
        span = np.max(x_finite) - np.min(x_finite)
        # If span is zero (all points are the same), avoid padding
        pad = span * pad_by if span > 0 else 0
        x_min = np.min(x_finite) - pad
        x_max = np.max(x_finite) + pad
        x_filled = np.linspace(x_min, x_max, num_points)

    # Re-insert original points into the filled array at the closest positions.
    # This uses broadcasting to create a distance matrix and find the minimums.
    indices = np.argmin(np.abs(x_filled[:, np.newaxis] - x_finite), axis=0)
    x_filled[indices] = x_finite

    return x_filled