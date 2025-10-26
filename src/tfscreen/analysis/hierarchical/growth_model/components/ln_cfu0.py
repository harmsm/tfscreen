import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass 
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float

def define_model(name,data,priors):
    """
    ln_cfu0 for each genotype/replicate combination. This uses a pooled, learned
    normal distribution across all combinations as a prior. Returns a full tensor. 

    Priors
    ------
    priors.ln_cfu0_hyper_loc_loc
    priors.ln_cfu0_hyper_loc_scale
    priors.ln_cfu0_hyper_scale_loc

    Data
    ----
    data.num_ln_cfu0
    data.map_ln_cfu0
    """
    
    ln_cfu0_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.ln_cfu0_hyper_loc_loc,
                    priors.ln_cfu0_hyper_loc_scale)
    )
    ln_cfu0_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.ln_cfu0_hyper_scale_loc)
    )
    with pyro.plate(f"{name}_parameters", data.num_ln_cfu0):
        ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0, 1))

    ln_cfu0_dists = ln_cfu0_hyper_loc + ln_cfu0_offsets*ln_cfu0_hyper_scale

    # Register dists
    pyro.deterministic(name,ln_cfu0_dists)

    # Expand tensor
    ln_cfu0 = ln_cfu0_dists[data.map_ln_cfu0]

    return ln_cfu0

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_loc"] = -2.5
    parameters["ln_cfu0_hyper_loc_scale"] = 3.0
    parameters["ln_cfu0_hyper_scale_loc"] = 2.0
               
    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """
    
    guesses = {}
    guesses[f"{name}_hyper_loc"] = -2.5
    guesses[f"{name}_hyper_scale"] = 3.0
    guesses[f"{name}_offset"] = jnp.zeros(data.num_ln_cfu0)

    return guesses

