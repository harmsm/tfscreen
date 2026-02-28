import jax.numpy as jnp
import numpyro as pyro
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the instant growth transition model.
    Currently, there are no priors needed for an instant transition.
    """
    pass

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors,
                 g_pre: jnp.ndarray,
                 g_sel: jnp.ndarray,
                 t_pre: jnp.ndarray,
                 t_sel: jnp.ndarray) -> jnp.ndarray:
    """
    Combines the pre-selection and selection growth phases with an instant 
    transition between them.

    Parameters
    ----------
    name : str
        The prefix for Numpyro sample/deterministic sites in this component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for the priors.
    g_pre : jnp.ndarray
        Pre-selection growth rate tensor.
    g_sel : jnp.ndarray
        Selection growth rate tensor.
    t_pre : jnp.ndarray
        Pre-selection time tensor.
    t_sel : jnp.ndarray
        Selection time tensor.

    Returns
    -------
    total_growth : jnp.ndarray
        The total growth over both phases.
    """

    return g_pre * t_pre + g_sel * t_sel


def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray) -> jnp.ndarray:
    """
    Guide corresponding to the instant growth transition model.
    """

    return g_pre * t_pre + g_sel * t_sel


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {}

def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    return {}

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors()
