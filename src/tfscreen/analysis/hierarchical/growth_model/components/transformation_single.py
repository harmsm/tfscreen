import jax.numpy as jnp
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    
    No priors needed for single transformation (dummy).
    """
    pass

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors,
                 anchors: tuple = None) -> jnp.ndarray:
    """
    Dummy model definition for single transformation.
    """
    return 1.0, 1.0, 1.0 # Dummy lambda, a, b

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors,
          anchors: tuple = None) -> jnp.ndarray:
    """
    Dummy guide for single transformation.
    """
    return 1.0, 1.0, 1.0

def update_thetas(theta, params, mask=None):
    """
    Pass-through update_thetas for single transformation.
    
    Returns theta unchanged.
    """
    return theta

def get_hyperparameters():
    """
    No hyperparameters for single transformation.
    """
    return {}

def get_guesses(name, data):
    """
    No guesses needed.
    """
    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())
