import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """
    pass

def define_model(name,data,priors):
    """
    The activity (scale of transcriptional output given a fractional occupancy).
    Depends on genotype. Sets all to 1.0. Returns a full tensor. 

    Priors
    ------

    Data
    ----
    data.num_genotype
    data.map_genotype
    """

    # Set all to 1.0
    activity_dists = jnp.ones(data.num_genotype)

    # Register dists
    pyro.deterministic(name, activity_dists)  

    # Expand to full-sized tensor
    activity = activity_dists[data.map_genotype]

    return activity

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """
    return {}


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """
    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())