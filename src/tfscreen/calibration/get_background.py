from ._models import simple_poly

from tfscreen.calibration import (
    read_calibration
)

from tfscreen.util import (
    broadcast_args
)

import numpy as np
import pandas as pd

def get_background(
    titrant_name: np.ndarray,
    titrant_conc: np.ndarray,
    calibration_data: dict or str
) -> np.ndarray:
    """
    Calculate background growth rates from calibration data.

    This function looks up the background growth model parameters for each
    specified titrant and calculates the growth rate at the corresponding
    titrant concentration using a polynomial model.

    Parameters
    ----------
    titrant_name : numpy.ndarray
        A 1D array of titrant name strings.
    titrant_conc : numpy.ndarray
        A 1D array of titrant concentrations, corresponding to each entry in
        `titrant_name`. Must be the same length.
    calibration_data : dict or str
        A pre-loaded calibration dictionary or the file path to the
        calibration JSON file.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the calculated background growth rate for each
        corresponding input titrant name and concentration.
    """

    # Make sure these are both arrays of the same length
    titrant_name, titrant_conc = broadcast_args(titrant_name,titrant_conc)

    # Read calibration data
    calibration_dict = read_calibration(calibration_data)
    bg_param_dict = calibration_dict["bg_model_param"]

    # Create a pandas Series for fast, vectorized lookups
    param_map = pd.Series(bg_param_dict)

    # Use the Series to map each input titrant_name to its parameter array.
    # .tolist() converts the resulting Series of arrays into a list of lists.
    bg_params = np.array(param_map[titrant_name].tolist())
    
    return simple_poly(bg_params.T, titrant_conc)


