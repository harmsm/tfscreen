import numpy as np

# Constants are **only** valid when measuring samples of XX uL in luria broth
# in clear-bottom 96 well plates on the SpectraMaxI3 in the Harms lab. Upper
# bound for reliable measurements is an OD600 of 0.600. The lower detection
# threshold is 0.096. Calibration done 2025/09/19. (See notebooks/od600-to-cfu
# for calibration details.)

# od600 measurement threshold
OD600_MEAS_THRESHOLD = 0.09573000000000001

# od600 to cfu, 2nd-order polynomial
A_CFU = -5388603.36438282 
B_CFU = 66683325.953339 
C_CFU = 38292706.79524336 

# cfu polynomial standard error vs. od600, 2nd-order polynomial
# This approximates J @ P @ J^T.
P_JCJT = 801108.73371897 
Q_JCJT = -5133635.54769162 
R_JCJT = 13418613.55590658 

# pct error in od600 measurements
OD600_PCT_STD = 0.02


def od600_to_cfu(od600):
    """
    Convert plate reader OD600 values to cfu/mL using empirical
    calibration parameters. 

    The parameters MUST be determined empirically for a specific experimental 
    set up. The constants are all defined globally in this file. See the 
    directory notebooks/od600-to-cfu/ for details on the calibration. 

    Parameters
    ----------
    od600 : np.ndarray
        array of OD600 measurements

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
    cfu_est_std_2 = (P_JCJT + Q_JCJT*(od600) + R_JCJT*(od600**2))**2
    
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
