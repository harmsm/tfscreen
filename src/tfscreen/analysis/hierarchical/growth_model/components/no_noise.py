import jax.numpy as jnp
from flax.struct import (
    dataclass
)
from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the no-noise model.
    This is an empty placeholder, as this model has no priors.
    """

    pass


def define_model(name: str, 
                 fx_calc: jnp.ndarray, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    A null model that applies no noise to the input.

    This function serves as a placeholder in the model architecture.
    It simply returns the deterministic input array ``fx_calc``
    without adding any stochasticity or sampling any parameters.

    Parameters
    ----------
    name : str
        The prefix for Numpyro sites. (Unused in this model).
    fx_calc : jnp.ndarray
        The deterministically calculated input array (e.g., fractional
        occupancy).
    priors : ModelPriors
        A Pytree of hyperparameters. (Unused in this model).

    Returns
    -------
    jnp.ndarray
        The original, unmodified ``fx_calc`` array.
    """

    return fx_calc

def guide(name: str, 
          fx_calc: jnp.ndarray, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for define_model
    """

    return fx_calc


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no hyperparameters.
    """

    parameters = {}

    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Get guess values for the model's latent parameters.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites. (Unused).
    data : DataClass
        A Pytree containing data metadata. (Unused).

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """
    
    guesses = {}

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        An empty, populated Pytree (Flax dataclass).
    """
    return ModelPriors(**get_hyperparameters())