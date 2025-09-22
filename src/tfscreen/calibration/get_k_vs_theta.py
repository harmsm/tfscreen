
from tfscreen.calibration import (
    read_calibration
)

from tfscreen.util import (
    broadcast_args
)

import numpy as np

def get_k_vs_theta(
    condition: np.ndarray,
    calibration_data: dict or str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the slopes and intercepts of the growth perturbation vs. theta.

    This function looks up the calibrated linear model parameters (slope `m` and
    intercept `b`) for a given set of experimental conditions. These parameters
    describe how the growth rate (`k`) is perturbed by transcription factor
    occupancy (`theta`).

    Parameters
    ----------
    condition : numpy.ndarray
        A 1D array of condition strings (e.g., 'wt', 'mut1').
    calibration_data : dict or str
        A pre-loaded calibration dictionary or the file path to the
        calibration JSON file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing two 1D NumPy arrays:
        - The slopes (`m`) of `k` vs. `theta`.
        - The intercepts (`b`) of `k` vs. `theta`.
    """
    # Read calibration data
    calibration_dict = read_calibration(calibration_data)

    slopes = calibration_dict["dk_cond_df"].loc[condition,"m"].to_numpy().flatten()
    intercepts = calibration_dict["dk_cond_df"].loc[condition,"b"].to_numpy().flatten()

    return slopes, intercepts

