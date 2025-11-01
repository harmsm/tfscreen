import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass 
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    global_scale_tau_scale: float

def define_model(name,data,priors):
    """
    The activity (scale of transcriptional output given a fractional occupancy).
    Depends on genotype. Uses a horseshoe prior that enforces sparsity: keep the
    value 1.0 **unless** there's strong evidence to the contrary for a specific
    genotype. 

    Priors
    ------
    priors.global_scale_tau_scale

    Data
    ----
    data.num_not_wt
    data.num_genotype
    data.not_wt_mask
    data.map_genotype
    """

    # Global scale: How big can the "slab" (real effects) be?
    global_scale_tau = pyro.sample(f"{name}_global_scale",
                                   dist.HalfNormal(priors.global_scale_tau_scale)) 
    
    with pyro.plate(f"{name}_parameters", data.num_not_wt):
        
        # HalfNormal(1) is the standard Horseshoe.
        local_scale_lambda = pyro.sample(f"{name}_local_scale",
                                         dist.HalfNormal(1.0))

        # Non-centered offset (always Normal(0,1))
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0, 1))

    # Combine them loc is 0.0 (tight prior on activity = exp(0)). scale is
    # (global_scale_tau * local_scale_lambda). Only becomes meaningfully 
    # different from 0 if global_scale_tau gets big (meaning there it is
    # possible for at least some large values of activity to exist) and we have
    # a sample with a high value for the activity for this genotype.
    effective_scale = 0.0 + global_scale_tau * local_scale_lambda
    log_activity_mutant_dists = activity_offset * effective_scale
    
    activity_mutant_dists = jnp.clip(jnp.exp(log_activity_mutant_dists),1e30)

    # Build array with wildtype set to 1.0
    activity_dists = jnp.ones(data.num_genotype)
    activity_dists = activity_dists.at[data.not_wt_mask].set(activity_mutant_dists)

    # Register dists
    pyro.deterministic(name, activity_dists)  

    # Expand to full-sized tensor
    activity = activity_dists[data.map_genotype]

    return activity


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """
    
    parameters = {}
    parameters["global_scale_tau_scale"] = 0.1

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    guesses = {}
    guesses[f"{name}_global_scale"] = 0.1
    guesses[f"{name}_local_scale"] = jnp.zeros(data.num_not_wt)
    guesses[f"{name}_offset"] = jnp.zeros(data.num_not_wt)

    return guesses

