"""
Pass-through sample offset: no per-tube noise (delta_sample = 0).

The ModelOrchestrator default (so configs written before the count
likelihood read back unchanged); tfs-configure-model defaults to 'level'.
All tube-to-tube variation is absent; noise is captured by the likelihood
alone.
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
