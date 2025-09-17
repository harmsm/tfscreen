import numpy as np
from numpy.polynomial import polynomial as P

# At the top of your _models.py or a constants.py file
HILL_PARAM_IDX = {"baseline": 0, "amplitude": 1, "K": 2, "n": 3}

def hill_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculate the Hill equation for cooperative binding.

    This model describes a sigmoidal binding curve, often used for phenomena
    like transcription factor occupancy (`theta`) vs. ligand concentration (`x`).

    Parameters
    ----------
    params : numpy.ndarray
        An array of the four Hill model parameters. The shape can be 1D (4,)
        for a single parameter set, or 2D (4, N) for vectorized calculations
        where N is the number of points in `x`. The parameters must be in the
        order: [baseline, amplitude, K, n].
    x : numpy.ndarray
        A 1D array of the independent variable (e.g., ligand concentration),
        with shape (N,).

    Returns
    -------
    numpy.ndarray
        A 1D array of the calculated dependent variable values.
    """

    baseline = params[HILL_PARAM_IDX["baseline"]]
    amplitude = params[HILL_PARAM_IDX["amplitude"]]
    K = params[HILL_PARAM_IDX["K"]]
    n = params[HILL_PARAM_IDX["n"]]

    c = (x**n)
    
    return baseline + amplitude*(c/(c + K**n))

def simple_poly(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial at specific x values.

    The polynomial is defined by the coefficients in `params`. The function
    assumes the coefficients are ordered from the lowest degree to the highest
    (c_0, c_1, c_2, ...).

    Parameters
    ----------
    params : numpy.ndarray
        An array of polynomial coefficients in order `[c0, c1, c2, ...]`.
        Shape should be 1D. 
    x : numpy.ndarray
        A 1D array of the points at which to evaluate the polynomial(s),
        with shape (N,).

    Returns
    -------
    numpy.ndarray
        A 1D array containing the result of the polynomial evaluation(s).
    """
    
    # Create an exponent for each coefficient row: [0, 1, 2, ...]'
    exponents = np.arange(params.shape[0])

    # Use broadcasting to raise each x to all powers, then sum the terms.
    # For a 2D params array, exponents[:, np.newaxis] ensures correct alignment.
    if params.ndim == 2:
        terms = params * (x ** exponents[:, np.newaxis])
    else:
        terms = params[:, np.newaxis] * (x ** exponents[:, np.newaxis])
    
    return np.sum(terms, axis=0)