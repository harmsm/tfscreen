
from tfscreen.calibration import (
    read_calibration
)

import numpy as np

def get_slopes(marker,
               select,
               calibration_data):
    """
    Get the slope of k vs. theta for combinations of marker and selection.

    Parameters
    ----------
    marker : np.ndarray
        1D array of condition markers. marker values of "none" are ignored
        (slope is zero)
    select : np.ndarray
        1D array of selection state for each condition 
    
    Returns
    """

    calibration_dict = read_calibration(calibration_data)
    param_dict = calibration_dict["param_dict"]

    slopes = []
    for m, s in zip(marker,select):
        
        if m == "none":
            slopes.append(0.0)
        else:
            slopes.append(param_dict[f"{m}|{s}|m"])

    slopes = np.array(slopes)

    return slopes