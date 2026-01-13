import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from .genotype_utils import (sample_genotype_parameter, 
                             sample_genotype_parameter_guide,
                             get_genotype_parameter_guesses)

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
        - ``data.num_genotype`` : (int) The total number of genotypes.
        - ``data.wt_indexes`` : (jnp.ndarray) A boolean mask that is
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
    
    def sample_activity_per_unit(site_name, size):
        # Local scale `lambda`. HalfNormal(1) is the standard Horseshoe.
        local_scale_lambda = pyro.sample(f"{site_name}_local_scale",
                                        dist.HalfNormal(1.0))

        # Non-centered offset `z` (always Normal(0,1))
        activity_offset = pyro.sample(f"{site_name}_offset", dist.Normal(0.0, 1.0))

        # Guard against full-sized array substitution during initialization or re-runs 
        # with full-sized initial values
        if data.epistasis_mode == "genotype":
            if local_scale_lambda.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
                local_scale_lambda = local_scale_lambda[..., data.batch_idx]
            if activity_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
                activity_offset = activity_offset[..., data.batch_idx]

        # Combine scales: `beta = z * (tau * lambda)`
        return activity_offset * (global_scale_tau * local_scale_lambda)

    log_activity_per_genotype = sample_genotype_parameter(name, data, sample_activity_per_unit)
    activity = jnp.clip(jnp.exp(log_activity_per_genotype), max=1e30)

    # Set wildtype activity to 1.0
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    activity = jnp.where(is_wt_mask, 1.0, activity)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity)  

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:]

    return activity


def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the Horseshoe-regularized activity model.

    This guide implements the variational family for the Horseshoe prior,
    using:
    - LogNormal distributions for the global scale `tau` and local scales `lambda`.
    - Normal distributions for the non-centered offsets `z`.
    """

    t_scale_loc = pyro.param(f"{name}_t_hyper_scale_loc", jnp.array(-5.0))
    t_scale_scale = pyro.param(f"{name}_t_hyper_scale_scale",jnp.array(0.1),
                               constraint=dist.constraints.positive)
    global_scale_tau = pyro.sample(f"{name}_global_scale",
                                   dist.LogNormal(t_scale_loc, t_scale_scale)) 
    
    def guide_activity_per_unit(site_name, size):
        actual_size = data.num_genotype if data.epistasis_mode == "genotype" else data.num_mutation
        
        l_locs = pyro.param(f"{site_name}_lambda_offset_locs", jnp.zeros(actual_size))
        l_scales = pyro.param(f"{site_name}_lambda_offset_scales", jnp.ones(actual_size),
                               constraint=dist.constraints.positive)
        
        z_locs = pyro.param(f"{site_name}_activity_offset_locs", jnp.zeros(actual_size))
        z_scales = pyro.param(f"{site_name}_activity_offset_scales", jnp.ones(actual_size),
                                constraint=dist.constraints.positive)

        if data.epistasis_mode == "genotype":
            l_batch_locs = l_locs[data.batch_idx]
            l_batch_scales = l_scales[data.batch_idx]
            z_batch_locs = z_locs[data.batch_idx]
            z_batch_scales = z_scales[data.batch_idx]
        else:
            l_batch_locs = l_locs
            l_batch_scales = l_scales
            z_batch_locs = z_locs
            z_batch_scales = z_scales

        # Local scale `lambda`
        local_scale_lambda = pyro.sample(f"{site_name}_local_scale",
                                        dist.LogNormal(l_batch_locs, l_batch_scales))

        if local_scale_lambda.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
             local_scale_lambda = local_scale_lambda[..., data.batch_idx]

        # Non-centered offset `z`
        activity_offset = pyro.sample(f"{site_name}_offset", dist.Normal(z_batch_locs, z_batch_scales))

        if activity_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
             activity_offset = activity_offset[..., data.batch_idx]

        # Combine
        return activity_offset * (global_scale_tau * local_scale_lambda)

    log_activity_per_genotype = sample_genotype_parameter_guide(name, data, guide_activity_per_unit)
    activity = jnp.clip(jnp.exp(log_activity_per_genotype), max=1e30)

    # Set wildtype activity to 1.0
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    activity = jnp.where(is_wt_mask, 1.0, activity)

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:]

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
        - ``data.num_genotype`

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names to JAX arrays of
        guess values.
    """

    def guess_activity_per_unit(site_name, size):
        return {
            f"{site_name}_local_scale": jnp.ones(size) * 0.1,
            f"{site_name}_offset": jnp.zeros(size)
        }

    guesses = {}
    guesses[f"{name}_global_scale"] = 0.1
    
    guesses.update(get_genotype_parameter_guesses(name, data, guess_activity_per_unit))

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