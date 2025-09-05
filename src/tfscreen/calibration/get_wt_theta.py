from tfscreen.calibration import read_calibration

def get_wt_theta(iptg,
                 calibration_file,
                 override_K=None,
                 override_n=None):
    
    
    calibration_dict = read_calibration(calibration_file)
    K = calibration_dict["K"]
    n = calibration_dict["n"]

    if override_K:
        K = override_K
    if override_n:
        n = override_n

    theta = 1 - (iptg**n)/(K**n + iptg**n)

    return theta
