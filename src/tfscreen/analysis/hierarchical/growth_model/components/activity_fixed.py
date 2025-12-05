import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the fixed activity model.
    This is an empty placeholder, as this model has no priors.
    """
    pass

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the fixed model for genotype-specific activity.

    This model provides a fixed, non-learned value for activity. It sets
    the activity for all genotypes to 1.0, effectively making the
    growth rate dependent only on `k` and `m*theta` without any
    additional scaling.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro deterministic sites.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_genotype`` : (int) The total number of genotypes.
        - ``data.map_genotype`` : (jnp.ndarray) Index array to map
          per-genotype parameters to the full set of observations.
    priors : ModelPriors
        A Pytree of hyperparameters. (Unused in this model).

    Returns
    -------
    jnp.ndarray
        An array of 1.0s, expanded to match the shape of
        the observations via ``data.map_genotype``.
    """

    # Set activity for all genotypes to 1.0
    activity_dists = jnp.ones(data.batch_size)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity_dists)  

    # Broadcast to full-sized tensor
    activity = activity_dists[None,None,None,None,None,None,:]

    return activity


def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    """

    # Set activity for all genotypes to 1.0
    activity_dists = jnp.ones(data.batch_size)*1.0

    # Broadcast to full-sized tensor
    activity = activity_dists[None,None,None,None,None,None,:]

    return activity

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no hyperparameters.
    """
    return {}


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    """
    Get guess values for the model's latent parameters.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites. (Unused).
    data : GrowthData
        A Pytree containing data metadata. (Unused).

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """
    return {}

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        An empty, populated Pytree (Flax dataclass).
    """
    return ModelPriors(**get_hyperparameters())