
from tfscreen.calibration import read_calibration

def get_wt_theta(iptg,
                 calibration_data,
                 override_K=None,
                 override_n=None):
    """
    Get the slope of k vs. theta for combinations of marker and selection 
    using the model stored in calibration_data.

    Parameters
    ----------
    iptg : np.ndarray
        1D array of iptg concentration for each condition
    calibration_data : dict or str
        a dictionary holding calibration values or the path to the calibration
        json file
    override_K : np.ndarray or float
        use this K value for the hill model, not the K in calibration_data. if
        an array, must match dimensions of iptg
    override_n : np.ndarray or float
        use this n value for the hill model, not the n in calibration_data. if
        an array, must match dimensions of iptg
    
    Returns
    -------
    np.ndarray
        1D array of the slope of k vs. theta
    """
    
    calibration_dict = read_calibration(calibration_data)
    
    K = calibration_dict["K"]
    n = calibration_dict["n"]

    if override_K:
        K = override_K
    if override_n:
        n = override_n

    theta = 1 - (iptg**n)/(K**n + iptg**n)

    return theta
