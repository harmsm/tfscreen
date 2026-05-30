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


class LinearGrowth:
    """
    k = b + activity * m * theta

    Matches the 'linear' and 'linear_fixed' inference components.

    Parameters
    ----------
    b : float
        Baseline growth rate at theta=0.
    m : float
        Slope: change in growth rate per unit theta.
    activity : float or array-like, default 1.0
        Per-genotype TF activity scaling factor.
    """
    def predict(self, theta, b, m, activity=1.0):
        return b + np.asarray(activity) * m * np.asarray(theta)


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
