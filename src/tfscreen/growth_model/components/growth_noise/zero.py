"""
Pass-through growth noise component: no growth rate noise (sigma_k = 0).

This is the default. It preserves the original model behaviour exactly —
the observation scale is determined entirely by data.ln_cfu_std.
"""

import jax.numpy as jnp
from flax.struct import dataclass
from tfscreen.growth_model.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    Empty Pytree — no priors needed for zero growth noise.
    """
    pass


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Return zero — no additive noise on growth.

    Parameters
    ----------
    name : str
        Unused (kept for interface consistency).
    data : GrowthData
        Unused.
    priors : ModelPriors
        Unused.

    Returns
    -------
    jnp.ndarray
        Scalar zero.
    """
    return jnp.array(0.0)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for zero growth noise — returns zero.
    """
    return jnp.array(0.0)


def get_hyperparameters() -> Dict[str, Any]:
    return {}


def get_priors() -> ModelPriors:
    return ModelPriors()


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    return {}


def get_extract_specs(ctx):
    return []
