from tfscreen.models.generic import MODEL_LIBRARY
from tfscreen.calibration import read_calibration
from tfscreen.util import broadcast_args

import numpy as np
import pandas as pd

def get_wt_theta(
    titrant_name: np.ndarray,
    titrant_conc: np.ndarray,
    calibration_data: dict or str,
    override_K: float or np.ndarray = None,
    override_n: float or np.ndarray = None
) -> np.ndarray:
    """
    Calculate wildtype theta using the calibrated Hill model.

    This function looks up the calibrated Hill model parameters for each
    specified titrant. It allows for optionally overriding the equilibrium
    constant (K) and Hill coefficient (n) before calculating the fractional
    saturation (theta).

    Parameters
    ----------
    titrant_name : numpy.ndarray
        A 1D array of titrant name strings.
    titrant_conc : numpy.ndarray
        A 1D array of corresponding titrant concentrations.
    calibration_data : dict or str
        A pre-loaded calibration dictionary or the file path to the
        calibration JSON file.
    override_K : float or numpy.ndarray, optional
        If provided, this value is used for the dissociation constant (K)
        instead of the value from the calibration data. Must have a shape
        compatible with `titrant_name`. Defaults to None. Note that the 
        calibration stores lnK, but this function expects K. 
    override_n : float or numpy.ndarray, optional
        If provided, this value is used for the Hill coefficient (n)
        instead of the value from the calibration data. Must have a shape
        compatible with `titrant_name`. Defaults to None.

    Returns
    -------
    numpy.ndarray
        A 1D array of the calculated theta values for each corresponding input.
    
    Notes
    -----
    The function assumes the Hill parameters in the calibration data are
    ordered as `[baseline, amplitude, K, n]`.
    """
    
    # Read calibration data
    calibration_dict = read_calibration(calibration_data)

    # Create arrays of identical length from inputs
    titrant_name, titrant_conc = broadcast_args(titrant_name,
                                                titrant_conc)

    # Create series to map between theta and parameter arrays
    theta_param_dict = calibration_dict["theta_param"]
    theta_map = pd.Series(theta_param_dict)

    # param array 
    param_array = np.array(theta_map[titrant_name].tolist())

    # Get positions of named parameters in the model to allow us to override
    param_names = MODEL_LIBRARY["hill_repressor"]["param_names"]
    param_positions = dict([(p,i) for i, p in enumerate(param_names)])

    # Override calibration K value
    if override_K is not None:
        idx = param_positions["lnK"]
        param_array[:, idx] = np.log(override_K)
    
    # Override calibration n value
    if override_n is not None:
        idx = param_positions["n"]
        param_array[:,idx] = override_n

    # Return hill model
    return MODEL_LIBRARY["hill_repressor"]["model_func"](param_array.T,
                                                         titrant_conc)

