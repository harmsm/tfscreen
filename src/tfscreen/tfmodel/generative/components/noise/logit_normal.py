"""
Logit-normal theta noise: additive Normal noise on the logit scale.

Each per-(genotype, condition) occupancy theta_pred receives an independent
noise draw epsilon ~ Normal(0, sigma_logit), giving:

    theta_noisy = sigmoid(logit(theta_pred) + epsilon)

sigma_logit is a single global positive scalar inferred from data.
Noise is naturally heteroscedastic: variance in theta-space is maximized
at theta=0.5 and shrinks to zero at saturation (theta→1) or depletion
(theta→0), which matches the physical expectation that occupancy near
the midpoint is most sensitive to fluctuations in TF copy number or
effector concentration.

Prior
-----
    sigma_logit ~ HalfNormal(sigma_logit_scale)

sigma_logit_scale should be set to a physically meaningful scale:
sigma_logit=0.1 corresponds to ~2.5% absolute noise at theta=0.5;
sigma_logit=1.0 corresponds to ~25% absolute noise at theta=0.5.
A scale of 0.5 is weakly informative, allowing values from near-zero
to ~1.5.

Guide
-----
    sigma_logit ~ LogNormal(loc, scale)   (positive support)
    epsilon     ~ Normal(0, sigma_logit)  (prior-as-guide; KL=0 per-obs)

Using the prior as the guide for the per-observation epsilon sites avoids
introducing O(n_obs) variational parameters while still correctly
registering the sample sites required by SVI.
"""

import jax.numpy as jnp
import jax.nn as jax_nn
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.tfmodel.data_class import GrowthData
from typing import Dict, Any

# Clip theta away from boundaries before logit to avoid ±inf.
_LOGIT_EPS = 1e-6


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the logit-normal theta noise model.

    Attributes
    ----------
    sigma_logit_scale : float
        Scale of the HalfNormal prior on sigma_logit. In logit units.
        sigma_logit=0.1 gives ~2.5% absolute noise at theta=0.5.
    """
    sigma_logit_scale: float


def define_model(name: str,
                 fx_calc: jnp.ndarray,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Apply logit-normal noise to a deterministic occupancy array.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample / deterministic sites.
    fx_calc : jnp.ndarray
        Deterministic theta values in (0, 1).
    priors : ModelPriors
        Pytree containing ``sigma_logit_scale``.

    Returns
    -------
    jnp.ndarray
        Noisy theta values in (0, 1), same shape as ``fx_calc``.
    """
    sigma_logit = pyro.sample(
        f"{name}_sigma_logit",
        dist.HalfNormal(priors.sigma_logit_scale)
    )
    pyro.deterministic(f"{name}_sigma_logit_value", sigma_logit)

    fx_safe = jnp.clip(fx_calc, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    logit_calc = jnp.log(fx_safe / (1.0 - fx_safe))

    # Plate epsilon on titrant_conc (dim=-2) and genotype (dim=-1) so that
    # prediction.py's slicing logic can match these named plates to TensorManager
    # dim names and correctly reduce epsilon to the prediction-time shape.
    # Without plates, prediction.py passes training-time epsilon unchanged into
    # a model expecting prediction-time shapes, causing a broadcast error.
    with pyro.plate(f"{name}_titrant_conc", fx_calc.shape[-2], dim=-2):
        with pyro.plate(f"{name}_genotype", fx_calc.shape[-1], dim=-1):
            epsilon = pyro.sample(
                f"{name}_epsilon",
                dist.Normal(jnp.zeros_like(fx_calc), sigma_logit)
            )

    fx_noisy = jax_nn.sigmoid(logit_calc + epsilon)
    pyro.deterministic(name, fx_noisy)
    return fx_noisy


def guide(name: str,
          fx_calc: jnp.ndarray,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Variational guide for the logit-normal theta noise model.

    sigma_logit uses a LogNormal guide (positive support). epsilon uses
    the prior as its guide distribution (KL = 0 per-obs), sampled inside
    the same titrant_conc / genotype plates as define_model so that
    prediction.py can slice it correctly when predicting at a subset of
    the training concentrations or genotypes.

    Parameters
    ----------
    name : str
        Prefix for Numpyro parameter / sample sites.
    fx_calc : jnp.ndarray
        Deterministic theta values (used for shape and logit computation).
    priors : ModelPriors
        Used for initialisation of the variational loc parameter.

    Returns
    -------
    jnp.ndarray
        Noisy theta values drawn from the variational distribution.
    """
    init_loc = jnp.log(jnp.array(priors.sigma_logit_scale) / 4.0)

    loc = pyro.param(f"{name}_sigma_logit_loc", init_loc)
    scale = pyro.param(
        f"{name}_sigma_logit_scale",
        jnp.array(0.5),
        constraint=dist.constraints.greater_than(1e-4),
    )
    sigma_logit = pyro.sample(f"{name}_sigma_logit", dist.LogNormal(loc, scale))

    with pyro.plate(f"{name}_titrant_conc", fx_calc.shape[-2], dim=-2):
        with pyro.plate(f"{name}_genotype", fx_calc.shape[-1], dim=-1):
            epsilon = pyro.sample(
                f"{name}_epsilon",
                dist.Normal(jnp.zeros_like(fx_calc), sigma_logit)
            )

    fx_safe = jnp.clip(fx_calc, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    logit_calc = jnp.log(fx_safe / (1.0 - fx_safe))
    fx_noisy = jax_nn.sigmoid(logit_calc + epsilon)
    return fx_noisy


def get_hyperparameters() -> Dict[str, Any]:
    """
    Return default hyperparameter values.

    The default ``sigma_logit_scale = 0.5`` is weakly informative, placing
    meaningful prior weight on values from near-zero up to ~1.5 logit units,
    corresponding to absolute theta noise from negligible to ~37% at theta=0.5.

    Returns
    -------
    dict
        ``{"sigma_logit_scale": 0.5}``
    """
    return {"sigma_logit_scale": 0.5}


def get_priors() -> ModelPriors:
    """Return a populated ModelPriors Pytree with default values."""
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    """
    Return initial guess values for the variational parameters.

    Initialises sigma_logit near zero so inference starts from the
    no-noise baseline.

    Parameters
    ----------
    name : str
        Parameter name prefix (e.g. ``"theta_growth_noise"``).
    data : GrowthData
        Unused.

    Returns
    -------
    dict
        ``{"{name}_sigma_logit": small_positive_value}``
    """
    return {f"{name}_sigma_logit": jnp.array(1e-3)}


def get_extract_specs(ctx) -> list:
    """Return extraction specs for posterior summarisation."""
    import pandas as pd
    sigma_df = pd.DataFrame({"parameter": ["sigma_logit"], "map_all": [0]})
    return [dict(
        input_df=sigma_df,
        params_to_get=["sigma_logit"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="theta_growth_noise_",
    )]
