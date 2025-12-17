from tfscreen.util.io import read_yaml

import numpy as np
from typing import Tuple

def od600_to_cfu(od600: np.ndarray,
                 constants: str | dict) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Convert plate reader OD600 values to cfu/mL using empirical
    calibration parameters. 

    The parameters MUST be determined empirically for a specific experimental 
    set up. The constants are all defined in the constants dictionary/yaml).
    See the directory tfscreen/notebooks/od600-to-cfu/ for details on the
    calibration. 

    Parameters
    ----------
    od600 : np.ndarray
        array of OD600 measurements
    constants : str or dict
        dictionary holding calibration constants for converting od600 to 
        cfu/mL. if a string, treat as a path to a yaml file with the constants

    Returns
    -------
    cfu_est : np.ndarray
        1D array of estimated cfu/mL from OD600
    cfu_std : np.ndarray
        1D array of standard error on the estimated cfu/mL
    detectable : np.ndarray
        1D boolean array indicating whether the OD600 was above the 
        detection threshold. (True means measurable; False means too low to see). 
    """

    constants = read_yaml(constants)

    required_constants = ["OD600_MEAS_THRESHOLD",
                          "A_CFU","B_CFU","C_CFU",
                          "P_JCJT_CFU","Q_JCJT_CFU","R_JCJT_CFU",
                          "OD600_PCT_STD"]

    required_set = set(required_constants)
    seen_set = set(constants.keys())
    if not required_set.issubset(seen_set):
        missing = required_set - seen_set
        err = "Constants are missing. Missing constants:\n"
        for c in missing:
            err += f"    {c}\n"
        err += "\n"
        raise ValueError(err)

    OD600_MEAS_THRESHOLD = constants["OD600_MEAS_THRESHOLD"]
    A_CFU = constants["A_CFU"]
    B_CFU = constants["B_CFU"]
    C_CFU = constants["C_CFU"]
    P_JCJT_CFU = constants["P_JCJT_CFU"]
    Q_JCJT_CFU = constants["Q_JCJT_CFU"]
    R_JCJT_CFU = constants["R_JCJT_CFU"]
    OD600_PCT_STD = constants["OD600_PCT_STD"]
    
    # see if we are dealing with a single value. Turn into an array for
    # simplicity. 
    single_value = False
    if np.isscalar(od600):
        single_value = True
        od600 = np.array([od600])
    
    # work on a copy
    od600 = od600.copy()
    
    # left-cap values below the measurement threshold
    detectable = np.ones(len(od600),dtype=bool)
    detectable[od600 < OD600_MEAS_THRESHOLD] = False
    od600[od600 < OD600_MEAS_THRESHOLD] = OD600_MEAS_THRESHOLD
        
    # caclulate cfu estimate
    cfu_est = A_CFU + B_CFU*(od600) + C_CFU*(od600**2)
    
    # caclulate J @ Cp @ J^T 
    cfu_est_std_2 = (P_JCJT_CFU + Q_JCJT_CFU*(od600) + R_JCJT_CFU*(od600**2))**2
    
    # caclulate (dy/dx)^2 of the cfu vs od600 curve
    dy_dx_2 = (B_CFU + 2*C_CFU*od600)**2
    
    # caclulate (error in od600 measurement)^2
    od600_std_2 = (OD600_PCT_STD*od600)**2 
    
    # Final cfu_err is square root of combined variances
    cfu_std = np.sqrt(cfu_est_std_2 + dy_dx_2*od600_std_2)
    
    if single_value:
        cfu_est = cfu_est[0]
        cfu_std = cfu_std[0]
        detectable = detectable[0]
    
    return cfu_est, cfu_std, detectable
