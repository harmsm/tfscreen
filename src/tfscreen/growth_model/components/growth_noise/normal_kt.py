"""
Normal growth noise component: a global additive noise on accumulated log-growth.

The observation model becomes:

    ln_cfu_obs ~ StudentT(nu,  ln_cfu_pred,  sqrt(ln_cfu_std² + sigma_k²))

where sigma_k is a single positive scalar learned from data.  This is
mathematically equivalent to placing an independent Normal(0, sigma_k)
noise on each kt observation and marginalising it out analytically, giving
a combined scale without introducing per-observation latent variables.

sigma_k captures biological growth-rate variability not explained by theta,
dk_geno, or measurement noise — e.g. between-replicate environmental
fluctuations.  It has units of accumulated log-growth (dimensionless kt).

Prior
-----
    sigma_k ~ HalfNormal(sigma_k_scale)

A weakly informative prior: ``sigma_k_scale`` should be set to roughly
5 % of the expected total accumulated log-growth (kt ≈ k_pre*t_pre +
k_sel*t_sel).  For typical experiments with k ~ 0.03 h⁻¹ and t ~ 100 h,
kt ~ 3, so a scale of 0.15 is a reasonable starting point.

Guide
-----
    sigma_k ~ LogNormal(loc, scale)   (LogNormal gives positive support)
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.growth_model.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the normal_kt growth noise model.

    Attributes
    ----------
    sigma_k_scale : float
        Scale of the HalfNormal prior on sigma_k.  In kt units (accumulated
        log-growth).  A value of ~5 % of expected kt is a sensible default.
    """
    sigma_k_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Sample sigma_k from a HalfNormal prior and return it.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample / deterministic sites.
    data : GrowthData
        Unused (kept for interface consistency).
    priors : ModelPriors
        Pytree containing ``sigma_k_scale``.

    Returns
    -------
    jnp.ndarray
        Positive scalar sigma_k.
    """
    sigma_k = pyro.sample(
        f"{name}_sigma_k",
        dist.HalfNormal(priors.sigma_k_scale)
    )
    pyro.deterministic(f"{name}_sigma_k_value", sigma_k)
    return sigma_k


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    LogNormal variational guide for sigma_k.

    Uses a LogNormal guide to ensure positive support.  Initialised with
    ``loc = log(sigma_k_scale / 4)`` so the initial mean is well below the
    prior scale, letting the data push sigma_k up if needed.

    Parameters
    ----------
    name : str
        Prefix for Numpyro parameter / sample sites.
    data : GrowthData
        Unused.
    priors : ModelPriors
        Used for initialisation of the variational loc parameter.

    Returns
    -------
    jnp.ndarray
        Positive scalar sigma_k drawn from the variational distribution.
    """
    init_loc = jnp.log(jnp.array(priors.sigma_k_scale) / 4.0)

    loc = pyro.param(
        f"{name}_sigma_k_loc",
        init_loc,
    )
    scale = pyro.param(
        f"{name}_sigma_k_scale",
        jnp.array(0.5),
        constraint=dist.constraints.greater_than(1e-4),
    )
    sigma_k = pyro.sample(f"{name}_sigma_k", dist.LogNormal(loc, scale))
    return sigma_k


def get_hyperparameters() -> Dict[str, Any]:
    """
    Return default hyperparameter values.

    The default ``sigma_k_scale = 0.15`` is appropriate for experiments
    where kt ~ 3 (k ~ 0.03 h⁻¹, t ~ 100 h) and the expected noise
    fraction is about 5 %.  Adjust to match the expected kt magnitude.

    Returns
    -------
    dict
        ``{"sigma_k_scale": 0.15}``
    """
    return {"sigma_k_scale": 0.15}


def get_priors() -> ModelPriors:
    """Return a populated ModelPriors Pytree with default values."""
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    """
    Return initial guess values for the variational parameters.

    Initialises sigma_k close to zero so inference starts from the
    no-noise baseline.

    Parameters
    ----------
    name : str
        Parameter name prefix (e.g. ``"growth_noise"``).
    data : GrowthData
        Unused.

    Returns
    -------
    dict
        ``{"{name}_sigma_k": small_positive_value}``
    """
    return {f"{name}_sigma_k": jnp.array(1e-3)}


def get_extract_specs(ctx) -> list:
    """Return extraction specs for posterior summarisation."""
    import pandas as pd
    sigma_k_df = pd.DataFrame({"parameter": ["sigma_k"], "map_all": [0]})
    return [dict(
        input_df=sigma_k_df,
        params_to_get=["sigma_k"],
        map_column="map_all",
        get_columns=["parameter"],
        in_run_prefix="growth_noise_",
    )]
