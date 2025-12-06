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

    dk_geno_hyper_loc_loc: float
    dk_geno_hyper_loc_scale: float
    dk_geno_hyper_scale_loc: float
    dk_geno_hyper_shift_loc: float
    dk_geno_hyper_shift_scale: float
    

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    The pleiotropic effect of a genotype on growth rate independent of 
    transcription factor occupancy. 
    
    This applies a pooled, left-skewed prior shifted to allow for negative
    values. (Implemented as shift - log_normal). This describes the expectation
    that most mutations will be neutral, a few will be favorable, and a long
    tail will be deleterious. wildtype is assigned dk_geno = 0.  Returns a full
    dk_geno tensor. 

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - data.num_genotype
        - data.wt_indexes
        - data.map_genotype
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model, including:
        - priors.dk_geno_hyper_shift_loc
        - priors.dk_geno_hyper_shift_scale
        - priors.dk_geno_hyper_loc_loc
        - priors.dk_geno_hyper_loc_scale
        - priors.dk_geno_hyper_scale_loc

    Returns
    -------
    jnp.ndarray
        full tensor with shape (num_replicate,num_time,num_treatment,num_genotype)
        with dk_geno values for each position in the tensor.
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

    with pyro.plate("shared_genotype_plate", size=data.num_genotype,subsample_size=data.batch_size,dim=-1):
        dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))
    
    dk_geno_lognormal_values = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale),max=1e30)
    dk_geno_per_genotype = dk_geno_hyper_shift - dk_geno_lognormal_values

    # Force wildtype to be zero. 
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    dk_geno_per_genotype = jnp.where(is_wt_mask, 0.0, dk_geno_per_genotype)
    
    # Register dists
    pyro.deterministic(name, dk_geno_per_genotype)    

    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the define_model function for dk_geno.
    """

    # --- Global Parameters ---

    # Hyper Loc (Normal guide for Normal prior)
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.dk_geno_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.dk_geno_hyper_loc_scale), 
                             constraint=dist.constraints.positive)
    dk_geno_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal guide for HalfNormal prior)
    # Initialized to -1.0 (approx 0.37) to start with reasonable spread
    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0))
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1), 
                               constraint=dist.constraints.positive)
    dk_geno_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))

    # Shift (Normal guide for Normal prior)
    shift_loc = pyro.param(f"{name}_shift_loc", jnp.array(priors.dk_geno_hyper_shift_loc))
    shift_scale = pyro.param(f"{name}_shift_scale", jnp.array(priors.dk_geno_hyper_shift_scale), 
                             constraint=dist.constraints.positive)
    dk_geno_hyper_shift = pyro.sample(f"{name}_shift", dist.Normal(shift_loc, shift_scale))

    # --- Local Parameters (Per Genotype) ---
    
    offset_locs = pyro.param(f"{name}_offset_locs", jnp.zeros(data.num_genotype,dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales", jnp.ones(data.num_genotype,dtype=float), 
                               constraint=dist.constraints.positive)

    # --- Batching ---
    with pyro.plate("shared_genotype_plate", size=data.num_genotype, subsample_size=data.batch_size, dim=-1) as idx:
        
        batch_locs = offset_locs[idx]
        batch_scales = offset_scales[idx]

        dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(batch_locs, batch_scales))

    # --- Deterministic Calculation ---
    
    # Replicate the Shift - LogNormal logic
    dk_geno_lognormal_values = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale), max=1e30)
    dk_geno_per_genotype = dk_geno_hyper_shift - dk_geno_lognormal_values

    # Force wildtype to be zero
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    dk_geno_per_genotype = jnp.where(is_wt_mask, 0.0, dk_geno_per_genotype)
    
    # Expand to full-sized tensor
    dk_geno = dk_geno_per_genotype[None,None,None,None,None,None,:]

    return dk_geno

def get_hyperparameters():
    """
    Gets default values for the model hyperparameters.

    Returns
    -------
    dict
        A dictionary of hyperparameter names and their default values.
    """

    parameters = {}
    parameters["dk_geno_hyper_loc_loc"] = -3.5
    parameters["dk_geno_hyper_loc_scale"] = 1.0
    parameters["dk_geno_hyper_scale_loc"] = 1.0
    parameters["dk_geno_hyper_shift_loc"] = 0.02
    parameters["dk_geno_hyper_shift_scale"] = 0.2
               
    return parameters
    

def get_guesses(name,data):
    """
    Gets initial guess values for model parameters.

    These are used to initialize the MCMC sampler (e.g., via
    ``numpyro.infer.init_to_value``).

    Parameters
    ----------
    name : str
        The prefix for the parameter names (e.g., "theta").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_genotype`` : (int) number of non-wt genotypes

    Returns
    -------
    dict
        A dictionary mapping parameter names to their initial
        guess values.
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
    guesses[f"{name}_offset"] = -0.8240460108562919*jnp.ones(data.num_genotype,dtype=float)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
    