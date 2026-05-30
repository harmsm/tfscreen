"""
Pass-through sample offset: no per-tube noise (delta_sample = 0).

This is the default. All tube-to-tube growth-rate variation is absent;
noise is captured entirely by data.ln_cfu_std and the Student-T nu.
"""

import jax.numpy as jnp
from flax.struct import dataclass
from tfscreen.tfmodel.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    pass


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    return jnp.array(0.0)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    return jnp.array(0.0)


def get_hyperparameters() -> Dict[str, Any]:
    return {}


def get_priors() -> ModelPriors:
    return ModelPriors()


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    return {}


def get_extract_specs(ctx) -> list:
    return []
