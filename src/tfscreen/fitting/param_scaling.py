
import numpy as np

def scale(value, scaling):
    """
    Scale a value using a mean and standard deviation.

    Parameters
    ----------
    value : float or np.ndarray
        The value(s) to scale.
    scaling : tuple
        A tuple of (mean, std) used for scaling.

    Returns
    -------
    float or np.ndarray
        The scaled value(s).
    """
    mean, std = scaling
    
    return (value - mean) / std

def unscale(scaled_value, scaling):
    """
    Unscale a value that has been scaled using a mean and standard deviation.

    Parameters
    ----------
    scaled_value : float or np.ndarray
        The scaled value(s) to unscale.
    scaling : tuple
        A tuple of (mean, std) used for scaling.

    Returns
    -------
    float or np.ndarray
        The unscaled value(s).
    """

    mean, std = scaling
    return scaled_value * std + mean

def get_scaling(value):
    """
    Get the mean and standard deviation of a value for scaling.

    Parameters
    ----------
    value : float or np.ndarray
        The value(s) to get scaling for.

    Returns
    -------
    tuple
        A tuple of (mean, std) used for scaling.
    """
    mean = np.mean(value)
    std = np.std(value)

    return (mean, std)