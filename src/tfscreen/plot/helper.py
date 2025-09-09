
import numpy as np

def get_ax_limits(x_values,
                  y_values=None,
                  pad_by=0.0,
                  percentile=0.005):
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

    Returns
    -------
    ax_min : float
        The calculated lower axis limit.
    ax_max : float
        The calculated upper axis limit.
    """

    all_values = list(x_values)
    if y_values is not None:
        all_values.extend(y_values)
    all_values = np.array(all_values)
    ax_min, ax_max = np.nanquantile(all_values, [percentile, 1-percentile])

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