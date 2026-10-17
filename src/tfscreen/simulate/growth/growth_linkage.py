"""
Numpy implementations of the growth model components for simulation.

Each class corresponds to one inference-side growth component in
growth_model/components/growth/ and implements the same mathematical
formula using numpy instead of JAX.

The growth_params dict passed to thermo_to_growth uses the 'model' key to
select one of these classes.  All remaining keys are forwarded as keyword
arguments to predict().

    growth_params = {
        "M9":        {"model": "linear",     "b": 0.025, "m": -0.01},
        "M9+kan":    {"model": "saturation", "kmin": 0.001, "kmax": 0.04},
        "M9+kan_hi": {"model": "power",      "b": 0.001, "a": 0.04, "n": 2.0},
    }
"""

import numpy as np


def warp_theta(theta, a, x0):
    """
    The monotone warp of linear growth's ``m_warp`` (numpy):
    f(theta) = (s(theta) - s(0)) / (s(1) - s(0)), s(t) = 1 / (1 + exp(-a (t - x0))),
    so f(0) = 0 and f(1) = 1; steepness a > 0, threshold x0. Matches
    ``generative/components/growth/linear.warp_theta``.
    """
    theta = np.asarray(theta, dtype=float)

    def s(t):
        return 1.0 / (1.0 + np.exp(-a * (t - x0)))

    return (s(theta) - s(0.0)) / (s(1.0) - s(0.0))


class LinearGrowth:
    """
    k = b + activity * m * f(theta)

    Matches the 'linear' inference component. f is the identity, or with
    ``warp_a`` the monotone warp of the component's ``m_warp`` (steepness
    ``warp_a``, the fit's exp(warp_log_a), and threshold ``warp_x0``), for
    a marker whose growth is a switch-like or graded function of occupancy.

    Parameters
    ----------
    b : float
        Baseline growth rate at theta=0.
    m : float
        Slope: change in growth rate per unit theta.
    activity : float or array-like, default 1.0
        Per-genotype TF activity scaling factor.
    warp_a : float, optional
        Warp steepness (> 0). None: no warp.
    warp_x0 : float, default 0.5
        Warp threshold.
    """
    def predict(self, theta, b, m, activity=1.0, warp_a=None, warp_x0=0.5):
        theta = np.asarray(theta)
        if warp_a is not None:
            if not float(warp_a) > 0:
                raise ValueError(f"warp_a must be > 0, got {warp_a}")
            theta = warp_theta(theta, float(warp_a), float(warp_x0))
        return b + np.asarray(activity) * m * theta


class PowerGrowth:
    """
    k = b + activity * a * theta**n

    Matches the 'power' inference component.

    Parameters
    ----------
    b : float
        Baseline growth rate at theta=0.
    a : float
        Coefficient on the power-law term.
    n : float
        Exponent (n=1 reduces to linear).
    activity : float or array-like, default 1.0
        Per-genotype TF activity scaling factor.
    """
    def predict(self, theta, b, a, n, activity=1.0):
        theta = np.asarray(theta)
        return b + np.asarray(activity) * a * (theta ** n)


class SaturationGrowth:
    """
    k = kmin + activity * (kmax - kmin) * theta / (1 + theta)

    Matches the 'saturation' inference component.

    Parameters
    ----------
    kmin : float
        Growth rate at theta=0.
    kmax : float
        Asymptotic growth rate as theta → ∞.
    activity : float or array-like, default 1.0
        Per-genotype TF activity scaling factor.
    """
    def predict(self, theta, kmin, kmax, activity=1.0):
        theta = np.asarray(theta)
        return kmin + np.asarray(activity) * (kmax - kmin) * theta / (1.0 + theta)


MODEL_REGISTRY = {
    "linear":     LinearGrowth,
    "power":      PowerGrowth,
    "saturation": SaturationGrowth,
}


def get_growth_model(name):
    """Return an instantiated growth model by name.

    Parameters
    ----------
    name : str
        One of the keys in MODEL_REGISTRY.

    Returns
    -------
    LinearGrowth | PowerGrowth | SaturationGrowth

    Raises
    ------
    ValueError
        If name is not a known model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown growth model '{name}'. "
            f"Available models: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()
