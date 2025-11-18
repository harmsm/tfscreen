import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the Horseshoe activity model.

    Attributes
    ----------
    global_scale_tau_scale : float
        The scale of the HalfNormal prior on the global scale parameter `tau`.
        This controls the overall "slab" width for non-zero effects.
    """

    global_scale_tau_scale: float

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the Horseshoe-regularized model for genotype-specific activity.

    Activity is the scale of transcriptional output (range 0 to infinity).
    This model assumes the wild-type genotype has a fixed activity of 1.0.
    The activities of all other (mutant) genotypes are modeled using a
    Horseshoe prior. This is a sparse prior that strongly shrinks
    most mutant activities towards 1.0 (i.e., `log(activity) = 0.0`)
    unless there is strong evidence for a non-zero effect.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites in this
        component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_not_wt`` : (int) The number of non-wild-type genotypes.
        - ``data.num_genotype`` : (int) The total number of genotypes.
        - ``data.not_wt_mask`` : (jnp.ndarray) A boolean mask that is
          `True` for non-wild-type genotypes.
        - ``data.map_genotype`` : (jnp.ndarray) Index array to map
          per-genotype parameters to the full set of observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for
        the Horseshoe prior.

    Returns
    -------
    jnp.ndarray
        The sampled `activity` values, expanded to match the shape of
        the observations via ``data.map_genotype``.
    """

    # Global scale: How big can the "slab" (real effects) be?
    # This is `tau` in the Horseshoe prior.
    global_scale_tau = pyro.sample(f"{name}_global_scale",
                                   dist.HalfNormal(priors.global_scale_tau_scale)) 
    
    # Sample local scales and offsets for mutant genotypes only
    with pyro.plate(f"{name}_parameters", data.num_not_wt):
        
        # Local scale `lambda`. HalfNormal(1) is the standard Horseshoe.
        local_scale_lambda = pyro.sample(f"{name}_local_scale",
                                         dist.HalfNormal(1.0))

        # Non-centered offset `z` (always Normal(0,1))
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0, 1))

    # Combine scales: `beta = z * (tau * lambda)`
    # The mean `log(activity)` is 0.0 (i.e., activity = 1.0)
    # The effective scale allows for deviations from 0.0
    effective_scale = global_scale_tau * local_scale_lambda
    log_activity_mutant_dists = activity_offset * effective_scale
    
    activity_mutant_dists = jnp.clip(jnp.exp(log_activity_mutant_dists), max=1e30)

    # Build array with wildtype set to 1.0 by default
    activity_dists = jnp.ones(data.num_genotype)
    # Set the mutant values in the array
    activity_dists = activity_dists.at[data.not_wt_mask].set(activity_mutant_dists)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity_dists)  

    # Expand to full-sized tensor
    activity = activity_dists[data.map_genotype]

    return activity


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their
        default values.
    """
    
    parameters = {}
    parameters["global_scale_tau_scale"] = 0.1

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `numpyro.handlers.substitute` for testing
    or initializing inference. The local scales and offsets are set to
    zero, forcing all mutant `log(activity)` values to 0.0 and thus
    all mutant activities to 1.0 (same as wild-type).

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A Pytree containing data metadata, used to determine the
        shape of the guess arrays. Requires:
        - ``data.num_not_wt``

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names to JAX arrays of
        guess values.
    """

    guesses = {}
    guesses[f"{name}_global_scale"] = 0.1
    guesses[f"{name}_local_scale"] = jnp.zeros(data.num_not_wt)
    guesses[f"{name}_offset"] = jnp.zeros(data.num_not_wt)

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    return ModelPriors(**get_hyperparameters())