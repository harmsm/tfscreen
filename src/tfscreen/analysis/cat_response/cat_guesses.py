"""
A library of empirical mathematical functions used to generate guesses for the
functions in cat_library.py. Each function adheres to the signature: 
`guess_func(x,y)`
- `x`: A NumPy array of independent variable values (e.g., titrant concentration).
- `y`: A NumPy array of dependent variable values (e.g., titrant concentration).
"""

import numpy as np

def guess_flat(x, y):
    """
    Generates a guess for the flat model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays

    Returns
    -------
    list
        A guess for the parameter [baseline].
    """
    # Guess the constant is the mean of the observed y-values.
    baseline = np.mean(y)
    return [baseline]

def guess_linear(x, y):
    """
    Generates guesses for the linear model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [m, b].
    """
    # Guess slope from the first and last points. Handle division by zero.
    delta_x = x[-1] - x[0]
    if np.abs(delta_x) < 1e-9:
        m = 0.0
    else:
        m = (y[-1] - y[0]) / delta_x
    
    # Guess intercept from the first point.
    b = y[0] - m * x[0]
    
    return [m, b]

def guess_repressor(x, y):
    """
    Generates guesses for the repressor model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, lnK]
    """

    baseline = np.max(y)
    amplitude = np.min(y) - baseline

    y_half = baseline + 0.5 * amplitude
    K = x[np.argmin(np.abs(y - y_half))]
    lnK = np.log(max(K,1e-9))

    return [baseline, amplitude, lnK]

def guess_inducer(x, y):
    """
    Generates guesses for the inducer model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, lnK]
    """
    
    baseline = np.min(y)
    amplitude = np.max(y) - baseline

    y_half = baseline + 0.5 * amplitude
    K = x[np.argmin(np.abs(y - y_half))]
    lnK = np.log(max(K,1e-9))

    return [baseline, amplitude, lnK]


def guess_hill_repressor(x, y):
    """
    Generates guesses for the Hill repressor model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, lnK, n]
    """

    guesses = guess_repressor(x,y)
    guesses.append(2.0) # n hard to guess -- set

    return guesses

def guess_hill_inducer(x, y):
    """
    Generates guesses for the Hill inducer model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, lnK, n]
    """
    
    guesses = guess_inducer(x,y)
    guesses.append(2.0) # n hard to guess -- set

    return guesses

def guess_bell_peak(x, y):
    """
    Generates guesses for the symmetric bell-shaped peak model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, ln_x0, ln_width].
    """
    # Baseline is the minimum observed value.
    baseline = np.min(y)
    
    # Find the location and amplitude of the peak.
    peak_idx = np.argmax(y)
    amplitude = y[peak_idx] - baseline
    x0 = x[peak_idx]
    
    # Guess the width is a fraction of the total x-range (in linear space).
    width = max((x[-1] - x[0]) / 4.0, 1e-9)

    # Now take the log of the linear-space parameters.
    ln_x0 = np.log(max(x0, 1e-9))
    ln_width = np.log(width)

    return [baseline, amplitude, ln_x0, ln_width]

def guess_bell_dip(x, y):
    """
    Generates guesses for the symmetric bell-shaped dip model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, ln_x0, ln_width].
    """
    # Baseline is the maximum observed value.
    baseline = np.max(y)
    
    # Find the location and amplitude of the dip.
    dip_idx = np.argmin(y)
    amplitude = y[dip_idx] - baseline
    x0 = x[dip_idx]
    
    # Guess the width is a fraction of the total x-range (in linear space).
    width = max((x[-1] - x[0]) / 4.0, 1e-9)

    # Now take the log of the linear-space parameters.
    ln_x0 = np.log(max(x0, 1e-9))
    ln_width = np.log(width)

    return [baseline, amplitude, ln_x0, ln_width]

def guess_biphasic_peak(x, y):
    """
    Generates guesses for the biphasic peak model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline, amplitude, lnK_a, lnK_i]
    """
    # Baseline C is the first y-value (at low ligand).
    baseline = y[0]
    
    # Find the location and amplitude of the peak.
    peak_idx = np.argmax(y)
    
    amplitude = y[peak_idx] - baseline
    
    # Heuristic: activation constant is before the peak, dissociation is at the peak.
    x_peak = x[peak_idx]
    lnK_a = np.log(max(x_peak / 2.0, 1e-9))
    lnK_i = np.log(max(x_peak, 1e-9))
    
    return [baseline, amplitude, lnK_a, lnK_i]


def guess_biphasic_dip(x, y):
    """
    Generates guesses for the biphasic dip model.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays.

    Returns
    -------
    list
        Guesses for parameters [baseline,amplitude,lnK_dip,lnK_rise]
    """

    # Initial amplitude is the first point, final amplitude is the last.
    baseline = y[0]
    amplitude = y[-1]
    
    # Find the location of the dip (minimum y-value).
    dip_idx = np.argmin(y)
    x_dip = x[dip_idx]
    
    # Heuristic: K_dip is near the dip, K_rise is in the second half of the data.
    lnK_dip = np.log(max(x_dip, 1e-9))
    if dip_idx < len(x) -1:
        lnK_rise = np.log(max(np.median(x[dip_idx:]), 1e-9))
    else: # If dip is the last point, no rise has occurred yet.
        lnK_rise = np.log(max(x[-1], 1e-9))

    return [baseline, amplitude, lnK_dip, lnK_rise]


