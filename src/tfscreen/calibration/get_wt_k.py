
from tfscreen.calibration import (
    read_calibration,
    get_background,
    get_wt_theta
)

from tfscreen.util.numerical import (
    broadcast_args
)

import numpy as np
import pandas as pd

from tfscreen.models.growth_linkage import get_model

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
    
    # Get model
    model_name = calibration_dict.get("model_name", "linear")
    model = get_model(model_name)
    param_defs = model.get_param_defs()

    condition, titrant_name, titrant_conc = broadcast_args(condition,
                                                           titrant_name,
                                                           titrant_conc)

    # Get background growth
    background = get_background(titrant_name,
                                titrant_conc,
                                calibration_dict)
    
    # Get theta if not passed in
    if theta is None:
        theta = get_wt_theta(titrant_name,
                             titrant_conc,
                             calibration_dict)

    # Get condition-specific parameters
    dk_cond_df = calibration_dict["dk_cond_df"]
    per_titrant_tau = calibration_dict.get("per_titrant_tau", False)
    
    params_list = []
    
    # Build lookup keys
    if per_titrant_tau:
        # Vectorized key construction
        # We need to be careful with float formatting to match calibrate.py
        # Using the same default string conversion as df.apply(lambda r: f"{r[...]}")
        lookup_keys = pd.Series(zip(condition, titrant_name, titrant_conc)).apply(lambda x: f"{x[0]}:{x[1]}:{x[2]}")
    else:
        lookup_keys = pd.Series(condition)

    for suffix, _, _, _ in param_defs:
        # Try granular lookup
        vals = lookup_keys.map(dk_cond_df[suffix])
        
        # Identify missing (NaN)
        mask = vals.isna()
        if mask.any():
            # Fallback to condition name only
            fallback_vals = pd.Series(condition[mask]).map(dk_cond_df[suffix])
            vals.loc[mask] = fallback_vals
            
            # Still missing? Final fallback to 0.0
            mask = vals.isna()
            if mask.any():
                vals.loc[mask] = 0.0
                
        params_list.append(vals.to_numpy())
        
    params = np.array(params_list)
    
    # Predict perturbation
    dk = model.predict(theta, params)
                
    v = background + dk

    return v