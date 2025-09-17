
from tfscreen.calibration import (
    read_calibration,
    get_background,
    get_wt_theta,
    get_k_vs_theta
)

from tfscreen.util import (
    broadcast_args
)

import numpy as np

def get_wt_k(
    condition: np.ndarray,
    titrant_name: np.ndarray,
    titrant_conc: np.ndarray,
    calibration_data: dict or str,
    theta: np.ndarray = None
) -> np.ndarray:
    """
    Calculate the wildtype growth rate using the full calibration model.

    This function combines the background growth rate with a theta-dependent
    perturbation to predict the final growth rate (`k`) for a given set of
    experimental conditions.

    Parameters
    ----------
    condition : numpy.ndarray
        A 1D array of condition strings.
    titrant_name : numpy.ndarray
        A 1D array of titrant name strings.
    titrant_conc : numpy.ndarray
        A 1D array of corresponding titrant concentrations.
    calibration_data : dict or str
        A pre-loaded calibration dictionary or the file path to the
        calibration JSON file.
    theta : numpy.ndarray, optional
        A 1D array of pre-calculated theta values. If None (the default),
        theta will be calculated internally using the calibrated Hill model.

    Returns
    -------
    numpy.ndarray
        A 1D array of the final predicted wildtype growth rate (`k`) for each
        corresponding set of inputs.
    """

    # Read calibration data
    calibration_dict = read_calibration(calibration_data)

    condition, titrant_name, titrant_conc = broadcast_args(condition,
                                                           titrant_name,
                                                           titrant_conc)


    # Get background growth
    background = get_background(titrant_name,
                                titrant_conc,
                                calibration_dict)
    
    # Get slope/intercept vs. theta 
    slopes, intercepts = get_k_vs_theta(condition,
                                        titrant_name,
                                        calibration_dict)
    
    # Get theta if not passed in
    if theta is None:
        theta = get_wt_theta(titrant_name,
                             titrant_conc,
                             calibration_dict)
                
    return background + intercepts + slopes*theta