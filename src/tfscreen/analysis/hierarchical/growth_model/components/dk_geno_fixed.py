import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """
    pass

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    The pleiotropic effect of a genotype on growth rate independent of
    transcription factor occupancy. Fixed to zero. Returns a full tensor. 

    Priors
    ------

    Data
    ----
    data.num_genotype
    data.map_genotype
    """

    # Create fixed dk_geno (0)
    dk_geno_per_genotype = jnp.zeros(data.num_genotype,dtype=float)

    # Register dists
    pyro.deterministic(name, dk_geno_per_genotype)  

    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    """

    # Create fixed dk_geno (0)
    dk_geno_per_genotype = jnp.zeros(data.num_genotype,dtype=float)

    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

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