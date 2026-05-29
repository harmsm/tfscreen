"""
A library of empirical mathematical functions for fitting 1D curves. Each
function adheres to the signature: `model_func(params, x)`
- `params`: A list or NumPy array of the model's parameters.
- `x`: A NumPy array of independent variable values (e.g., titrant concentration).
"""
import numpy as np

EPSILON = 1e-12
EXP_CLIP = 700
POWER_CLIP = 25

def model_flat(params, x):
    """
    A constant, flat line (null model).

    Parameters
    ----------
    params : array-like
        A single-element array containing the constant value [C].
    x : np.ndarray
        The independent variable values.

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: y = baseline
    - Biological Interpretation: This model represents a non-responsive or
      "dead" mutant. It could be permanently bound to the operator (C ~ 1)
      or permanently unbound/broken (C ~ 0), with its state being
      independent of the ligand concentration. It serves as a baseline or
      null hypothesis for model comparison.
    """
    
    return np.full_like(x,params[0])
    
def model_linear(params, x):
    """
    A simple linear model.

    Parameters
    ----------
    params : array-like
        A two-element array: [m, b].
        - m: The slope of the line.
        - b: The y-intercept.
    x : np.ndarray
        The independent variable values.

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: y = m*x + b
    - Biological Interpretation: Represents a dose-dependent response that
      does not saturate within the tested concentration range.
    """

    return params[0]*x + params[1]

def _hill(params, x):
    """
    Core private hill model used by public models. 
    """
    baseline, amplitude, lnK, n = params

    # For numerical stability, especially if x contains 0
    x_safe = x + EPSILON 

    # Clip the Hill coefficient 'n' to a plausible and safe range
    n_safe = np.clip(n, -POWER_CLIP, POWER_CLIP)

    # Calculate ln(K^n) and ln(x^n) safely
    ln_K_to_n = np.clip(n_safe * lnK, -EXP_CLIP, EXP_CLIP)
    ln_x_to_n = np.clip(n_safe * np.log(x_safe), -EXP_CLIP, EXP_CLIP)

    K_to_n = np.exp(ln_K_to_n)
    x_to_n = np.exp(ln_x_to_n)

    fx = x_to_n / (x_to_n + K_to_n)
    
    return baseline + amplitude * fx

def model_hill_3p(params, x):
    """
    A flexible three-parameter equation for dose-response curves.

    This function can model both increasing (activation/induction) and
    decreasing (repression/inhibition) responses based on the sign of the
    'amplitude' parameter.

    Parameters
    ----------
    params : array-like
        A three-element array: [baseline, amplitude, logK].
        - baseline: The response at zero ligand concentration (y-intercept).
        - amplitude: The total change in response. If positive, the curve
          increases (activation); if negative, it decreases (repression).
          The final asymptote is (baseline + amplitude).
        - logK: The natural logarithm of the concentration constant (EC50/IC50).
          This is the concentration at which the response is halfway between
          the baseline and the final value.
    x : np.ndarray
        The independent variable values (e.g., concentration).

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: $y = \text{baseline} + \text{amplitude} \times \frac{x}{K + x}$
    - This single, unified model replaces the need for separate 'repressor'
      and 'inducer' functions. The optimizer determines the direction of the
      response by fitting the sign of the 'amplitude'.
    - The model fits the logarithm of K ($logK$) for better numerical
      stability, allowing the optimizer to search for K in an unbounded space.
    """

    baseline, amplitude, logK = params

    return _hill([baseline,amplitude,logK,1.0],x)

def model_hill_4p(params, x):
    """
    A flexible four-parameter Hill equation for dose-response curves.

    This function can model both increasing (activation/induction) and
    decreasing (repression/inhibition) responses based on the sign of the
    'amplitude' parameter.

    Parameters
    ----------
    params : array-like
        A four-element array: [baseline, amplitude, logK, n].
        - baseline: The response at zero ligand concentration (y-intercept).
        - amplitude: The total change in response. If positive, the curve
          increases (activation); if negative, it decreases (repression).
          The final asymptote is (baseline + amplitude).
        - logK: The natural logarithm of the concentration constant (EC50/IC50).
          This is the concentration at which the response is halfway between
          the baseline and the final value.
        - n: The Hill coefficient, which describes the steepness
          (cooperativity) of the transition.
    x : np.ndarray
        The independent variable values (e.g., concentration).

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: $y = \text{baseline} + \text{amplitude} \times \frac{x^n}{K^n + x^n}$
    - This single, unified model replaces the need for separate 'repressor'
      and 'inducer' functions. The optimizer determines the direction of the
      response by fitting the sign of the 'amplitude'.
    - The model fits the logarithm of K ($logK$) for better numerical
      stability, allowing the optimizer to search for K in an unbounded space.
    """
    baseline, amplitude, logK, n = params

    return _hill([baseline, amplitude, logK, n], x)


def model_bell(params, x):
    """
    A symmetric, bell-shaped (Gaussian) peak model.

    Parameters
    ----------
    params : array-like
        A four-element array: [baseline, amplitude, ln_x0, ln_width].
        - baseline: Baseline occupancy (asymptote on both sides).
        - amplitude: The height of the peak relative to the baseline.
        - ln_x0: The log of the ligand concentration at the center of the peak.
        - ln_width: The log of the standard deviation, or width, of the peak.
    x : np.ndarray
        The independent variable values.

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: y = baseline + amplitude * exp(-0.5 * ((x - x0) / width)^2)
    - Biological Interpretation: This empirical model captures a response whose
      activity changes symmetrically within a specific concentration range. It
      provides a distinct alternative to the asymmetric biphasic peak and can be
      useful for classifying mutants that show this symmetric behavior.
    """
    
    baseline, amplitude, ln_x0, ln_width = params

    ln_x0_safe = np.clip(ln_x0,-EXP_CLIP,EXP_CLIP)
    ln_width_safe = np.clip(ln_width,-EXP_CLIP,EXP_CLIP)

    x0 = np.exp(ln_x0_safe)
    width = max(np.exp(ln_width_safe),EPSILON)
    
    exponent = -0.5 * ((x - x0) / width) ** 2
    exponent_safe = np.clip(exponent,-EXP_CLIP,EXP_CLIP)
    
    return baseline + amplitude * np.exp(exponent_safe)


def model_biphasic_peak(params, x):
    """
    A biphasic model describing an asymmetric peak-like response.

    Parameters
    ----------
    params : array-like
        A four-element array: [baseline, amplitude, lnK_a, lnK_i].
        - baseline: Baseline occupancy at zero ligand (y-intercept).
        - amplitude: The amplitude of the peak relative to the baseline.
        - K_a: Activation constant (controls the rising phase).
        - K_i: Inhibition constant (controls the falling phase).
    x : np.ndarray
        The independent variable values.

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: y = baseline + amplitude * (x / (K_a + x)) * (1 / (1 + x/K_i))
    - Biological Interpretation: This model describes a sequential process of
      activation followed by inhibition. It suggests two ligand binding events
      with different affinities: a high-affinity event that enhances binding,
      and a lower-affinity event that inhibits binding, resulting in a peak
      response at intermediate ligand concentrations. The shape is typically
      asymmetric.
    """

    baseline, amplitude, lnK_a, lnK_i = params

    lnKa_safe = np.clip(lnK_a,-EXP_CLIP,EXP_CLIP)
    Ka = max(np.exp(lnKa_safe),EPSILON)

    lnKi_safe = np.clip(lnK_i,-EXP_CLIP,EXP_CLIP)
    Ki = max(np.exp(lnKi_safe),EPSILON)

    activation_term = x / (Ka + x)
    inhibition_term = 1.0 / (1.0 + (x / Ki))
    
    return baseline + amplitude * activation_term * inhibition_term

def model_biphasic_dip(params, x):
    """
    A biphasic model describing a dip-then-rise response.

    Parameters
    ----------
    params : array-like
        A four-element array: [baseline, amplitude, lnK_dip, lnK_rise].
        - baseline: The initial amplitude at zero ligand (y-intercept).
        - amplitude: The amplitude of the subsequent rising phase.
        - lnK_dip: The log of the constant for the initial dip.
        - lnK_rise: The log of the constant for the subsequent rise.
    x : np.ndarray
        The independent variable values.

    Returns
    -------
    np.ndarray
        The calculated y-values.

    Notes
    -----
    - Mathematical Form: y = baseline/(1+x/K_dip) + amplitude*(x/(K_rise+x))
    - Biological Interpretation: This model describes the sum of two competing
      processes. The first is a high-affinity inhibitory effect that causes
      the initial dip (controlled by 'baseline' and 'K_dip'). The second is a
      lower-affinity activating effect that causes the response to rise at
      high concentrations (controlled by 'amplitude' and 'K_rise').
    """
    baseline, amplitude, lnK_dip, lnK_rise = params

    lnK_dip_safe = np.clip(lnK_dip, -EXP_CLIP, EXP_CLIP)
    K_dip = max(np.exp(lnK_dip_safe), EPSILON)

    lnK_rise_safe = np.clip(lnK_rise, -EXP_CLIP, EXP_CLIP)
    K_rise = max(np.exp(lnK_rise_safe), EPSILON)

    repressor_term = baseline / (1.0 + (x / K_dip))
    activator_term = amplitude * (x / (K_rise + x))
    
    return repressor_term + activator_term

def model_poly(params, x):
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
    terms = params[:, np.newaxis] * (x ** exponents[:, np.newaxis])
    
    return np.sum(terms, axis=0)
