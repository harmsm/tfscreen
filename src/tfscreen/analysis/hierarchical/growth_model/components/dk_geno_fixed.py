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
        A tensor of zeros, expanded to match the shape of
        the observations via ``data.map_genotype``.
    """

    # Create fixed dk_geno (0)
    dk_geno_per_genotype = jnp.zeros(data.batch_size,dtype=float)

    # Register dists
    pyro.deterministic(name, dk_geno_per_genotype)  

    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for the fixed dk_geno model.

    Since all parameters are fixed and deterministic (to 0.0), this guide
    simply returns zeros and does not register any learnable parameters.
    """

    # Create fixed dk_geno (0)
    dk_geno_per_genotype = jnp.zeros(data.batch_size,dtype=float)

    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no hyperparameters.
    """
    return {}


def get_guesses(name,data):
    """
    Get guess values for the model parameters. 

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """
    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())