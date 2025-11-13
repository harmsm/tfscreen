import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    dk_geno_hyper_loc_loc: float
    dk_geno_hyper_loc_scale: float
    dk_geno_hyper_scale_loc: float
    dk_geno_hyper_shift_loc: float
    dk_geno_hyper_shift_scale: float
    

def define_model(name,data,priors):
    """
    The pleiotropic effect of a genotype on growth rate independent of 
    transcription factor occupancy. This applies a pooled, left-skewed prior 
    shifted to allow for negative values. (Implemented as shift - log_normal).
    This describes the expectation that most mutations will be neutral, a few
    will be favorable, and a long tail will be deleterious. wildtype is assigned
    dk_geno = 0.  Returns a full dk_geno tensor. 

    Priors
    ------
    priors.dk_geno_hyper_shift_loc
    priors.dk_geno_hyper_shift_scale
    priors.dk_geno_hyper_loc_loc
    priors.dk_geno_hyper_loc_scale
    priors.dk_geno_hyper_scale_loc

    Data
    ----
    data.num_not_wt
    data.num_genotype
    data.not_wt_mask
    data.map_genotype
    """

    dk_geno_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.dk_geno_hyper_loc_loc,
                    priors.dk_geno_hyper_loc_scale)
    )
    dk_geno_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.dk_geno_hyper_scale_loc)
    )

    dk_geno_hyper_shift = pyro.sample(
        f"{name}_shift",
        dist.Normal(priors.dk_geno_hyper_shift_loc,
                    priors.dk_geno_hyper_shift_scale)
    )

    with pyro.plate(f"{name}_parameters", data.num_not_wt):
        dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(0, 1))
    
    dk_geno_lognormal = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale),a_max=1e30)
    dk_geno_mutant_dists = dk_geno_hyper_shift - dk_geno_lognormal

    # Set to zero by default (wt) then set rest to dk_geno_mutants
    dk_geno_dists = jnp.zeros(data.num_genotype)
    dk_geno_dists = dk_geno_dists.at[data.not_wt_mask].set(dk_geno_mutant_dists)
    
    # Register dists
    pyro.deterministic(name, dk_geno_dists)    

    # Expand to full-sized tensor
    dk_geno = dk_geno_dists[data.map_genotype]

    return dk_geno

def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}
    parameters["dk_geno_hyper_loc_loc"] = -3.5
    parameters["dk_geno_hyper_loc_scale"] = 1
    parameters["dk_geno_hyper_scale_loc"] = 1.0
    parameters["dk_geno_hyper_shift_loc"] = 0.02
    parameters["dk_geno_hyper_shift_scale"] = 0.2
               
    return parameters
    

def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """

    # these values give a distribution that looks right by eye, clustering 
    # slightly below zero with a few above zero and a tail reaching down to 
    # to -0.02. 
    # dk_geno_hyper_loc = -3.5
    # dk_geno_hyper_scale = 0.5
    # dk_geno_hyper_shift = 0.02
    # The offset of -0.-0.8240460108562919 is dk_geno = 0 on this distribution

    guesses = {}
    guesses[f"{name}_hyper_loc"] = -3.5
    guesses[f"{name}_hyper_scale"] = 0.5
    guesses[f"{name}_shift"] = 0.02
    guesses[f"{name}_offset"] = -0.8240460108562919*jnp.ones(data.num_not_wt)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
    