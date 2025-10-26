import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass 
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    activity_hyper_loc_loc: float
    activity_hyper_loc_scale: float
    activity_hyper_scale_loc: float

def define_model(name,data,priors):
    """
    The activity (scale of transcriptional output given a fractional occupancy).
    Depends on genotype. Uses a pooled normal prior across all genotypes. 
    wildtype is set to 1.0. Returns a full activity tensor. 

    Priors
    ------
    priors.activity_hyper_loc_loc
    priors.activity_hyper_loc_scale
    priors.activity_hyper_scale_loc

    Data
    ----
    data.num_not_wt
    data.num_genotype
    data.not_wt_mask
    data.map_genotype
    """

    # Priors are on log(activity), so their mean is log(1.0) = 0.0
    log_activity_hyper_loc = pyro.sample(
        f"{name}_log_hyper_loc",
        dist.Normal(priors.activity_hyper_loc_loc, # This prior should be ~Normal(0.0, ...)
                    priors.activity_hyper_loc_scale)
    )
    log_activity_hyper_scale = pyro.sample(
        f"{name}_log_hyper_scale",
        dist.HalfNormal(priors.activity_hyper_scale_loc) # Using HalfNormal
    )

    with pyro.plate(f"{name}_parameters", data.num_not_wt):
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0, 1))
    
    # Calculate in log-space, then exponentiate
    log_activity_mutant_dists = log_activity_hyper_loc + activity_offset * log_activity_hyper_scale
    activity_mutant_dists = jnp.exp(log_activity_mutant_dists)

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
    parameters["activity_hyper_loc_loc"] = 0.0
    parameters["activity_hyper_loc_scale"] = 0.01
    parameters["activity_hyper_scale_loc"] = 0.1

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    guesses = {}
    guesses[f"{name}_log_hyper_loc"] = 0.0
    guesses[f"{name}_log_hyper_scale"] = 0.1
    guesses[f"{name}_offset"] = jnp.zeros(data.num_not_wt)

    return guesses

