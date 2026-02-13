
import numpy as np

def get_ax_limits(x_values,
                  y_values=None,
                  pad_by=0.0,
                  percentile=0.005,
                  center_on_zero=False):
    """
    Calculate symmetrical axis limits for a plot.

    This function determines appropriate axis limits by considering the combined
    range of both x and y data. It uses quantiles to find the core data range,
    making it robust to extreme outliers. It can also add symmetrical padding
    to the limits.

    Parameters
    ----------
    x_values : array_like
        The x-coordinates of the data to be plotted.
    y_values : array_like, optional
        The y-coordinates of the data to be plotted.
    pad_by : float, optional
        A fractional value to expand the axis range for padding. For example,
        a value of 0.1 will add 10% padding to each end of the axis range.
        If None, no padding is added. Defaults to None.
    percentile: float, optional
        The limits will be set at this percentile. Default is 0.005, removing 
        only the most extreme outliers. 
    center_on_zero: bool, optional
        make ax_min and ax_max symmetrical about zero. 

    Returns
    -------
    ax_min : float
        The calculated lower axis limit.
    ax_max : float
        The calculated upper axis limit.
    """

    # Ensure x and y values are converted to 1D numpy arrays of numerical values
    # before combining to avoid including column names if they are DataFrames.
    all_values = [np.asarray(x_values).ravel()]
    if y_values is not None:
        all_values.append(np.asarray(y_values).ravel())
    
    combined = np.concatenate(all_values)
    ax_min, ax_max = np.nanquantile(combined, [percentile, 1-percentile])

    if center_on_zero:
        ax_max = np.max([np.abs(ax_min),np.abs(ax_max)])
        ax_min = -ax_max

    span = ax_max - ax_min
    ax_min = ax_min - pad_by*span
    ax_max = ax_max + pad_by*span

    return ax_min, ax_max
    
def clean_arrays(*arrs):
    """
    Takes an arbitrary number of arrays of the same length and removes 
    entries where any of the arrays have a nan. clean_arrays(x,y) where
    x = [0,1,nan] and y = [4,5,6] would return [0,1], [4,5]. 
    """

    combined = np.array(arrs)

    good_mask = ~np.any(~np.isfinite(combined),axis=0)
    combined = combined[:,good_mask]

    return tuple(combined)

def subsample_index(some_array,
                    subsample=10000):
    """
    Generate an index array that sub-samples an array.

    Parameters
    ----------
    some_array : np.ndarray
        array to be subsampled
    subsample : int, optional
        how many samples to take. if None, no sampling -- just return array 
        covering whole array

    Returns
    -------
    np.ndarray
        numpy array that indexes the original array
    """
    
    index = np.arange(len(some_array),dtype=int)
    if subsample is not None:
        if subsample < len(index):
            index = np.random.choice(index,
                                     size=subsample,
                                     replace=False)
            
    return index