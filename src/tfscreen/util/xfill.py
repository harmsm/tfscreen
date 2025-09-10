
import numpy as np

def xfill(x,
          num_points=100,
          use_log=None,
          pad_by=0.1):
    """
    Smoothly fill points between minimum and maximum values in x using linspace.
    Makes sure that all values in 'x' are present in the output (even if this 
    means the spacing is not perfectly smooth) to allow one-to-one comparisons
    between values measured at x and calculated in a model using the filled-in x. 
    
    Parameters
    ----------
    x : np.ndarray
        1D array of values
    num_points : int, optional
        number of points in final array
    use_log : bool, optional
        interpolate on a log scale. if None, the function chooses whether to 
        use the log based on the data
    pad_by : float default=0.1
        expand beyond x by this factor on both sides
    
    Returns
    -------
    np.ndarray
        num_points long array of expanded values
    """

    x = x[np.isfinite(x)]

    if use_log and np.min(x) < 0:
        err = "x has negative values. use_log cannot be set to True.\n"
        raise ValueError(err)

    if use_log is None:
        if np.min(x) >= 0 and np.max(x)/np.min(x[x > 0]) > 1000:
            use_log = True
        else:
            use_log = False

    if use_log:

        # Get rid of zero
        x_pos = x[x > 0]

        # Smoothly fill between min and max with padding
        log_span = np.log(np.max(x_pos)) - np.log(np.min(x_pos))
        x_min = np.log(np.min(x_pos)) - log_span*pad_by
        x_max = np.log(np.max(x_pos)) + log_span*pad_by
        x_filled = np.exp(np.linspace(x_min,x_max,num_points))

        # Stick zero point back in
        if np.min(x) == 0:
            x_filled[0] = 0

    else:

        # Smoothly fill between min and max with padding
        span = np.max(x) - np.min(x)
        x_min = np.min(x) - span*pad_by
        x_max = np.max(x) + span*pad_by
        x_filled = np.linspace(x_min,x_max,num_points)

    # Replace values in x_filled with their closest values in the original 
    # x array. 

    # Find the indices of the closest values in x_filled for each value in x
    # This creates a 2D array of differences, then finds the minimum in each
    # column
    indices = np.argmin(np.abs(x_filled[:, np.newaxis] - x), axis=0)
    x_filled[indices] = x

    return x_filled

        
