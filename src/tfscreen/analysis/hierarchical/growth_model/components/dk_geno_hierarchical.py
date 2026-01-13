import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from .genotype_utils import (sample_genotype_parameter, 
                             sample_genotype_parameter_guide,
                             get_genotype_parameter_guesses)

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
        The prefix for all Numpyro sample sites (e.g., "dk_geno").
    data : DataClass
        A data object containing metadata, primarily:
        - ``data.num_genotype``
        - ``data.wt_indexes``
        - ``data.map_genotype``
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model, including:
        - ``priors.dk_geno_hyper_shift_loc``
        - ``priors.dk_geno_hyper_shift_scale``
        - ``priors.dk_geno_hyper_loc_loc``
        - ``priors.dk_geno_hyper_loc_scale``
        - ``priors.dk_geno_hyper_scale_loc``

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

    def sample_dk_geno_per_unit(site_name, size):
        dk_geno_offset = pyro.sample(f"{site_name}_offset", dist.Normal(0.0, 1.0))
        
        # Guard against full-sized array substitution during initialization or re-runs 
        # with full-sized initial values (only relevant in genotype mode)
        if data.epistasis_mode == "genotype":
            if dk_geno_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
                dk_geno_offset = dk_geno_offset[..., data.batch_idx]
        
        lognormal_values = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale), max=1e30)
        return dk_geno_hyper_shift - lognormal_values

    dk_geno_per_genotype = sample_genotype_parameter(name, data, sample_dk_geno_per_unit)

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
    Guide corresponding to the hierarchical dk_geno model.

    This guide defines the variational family for the pleiotropic growth
    effect model. It handles:
    - Normal distributions for the hyper-location and shift.
    - LogNormal distribution for the hyper-scale.
    - Normal distributions for the per-genotype offsets.
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

    # --- Local Parameters (Per Genotype/Mutation) ---
    
    # We use num_genotype as the total size for local parameters even in mutation mode
    # for simplicity in parameter mapping, or we could use num_mutation.
    # Actually, the guide needs to know the correct size.
    # In genotype_utils, we handle the size.
    
    def guide_dk_geno_per_unit(site_name, size):
        # Determine actual size needed for param initialization
        actual_size = data.num_genotype if data.epistasis_mode == "genotype" else data.num_mutation
        
        g_offset_locs = pyro.param(f"{site_name}_offset_locs", jnp.zeros(actual_size))
        g_offset_scales = pyro.param(f"{site_name}_offset_scales", jnp.ones(actual_size), 
                                     constraint=dist.constraints.positive)
        
        # In genotype mode, we index by batch_idx
        if data.epistasis_mode == "genotype":
            batch_locs = g_offset_locs[data.batch_idx]
            batch_scales = g_offset_scales[data.batch_idx]
        else:
            # In mutation mode, sample_genotype_parameter_guide calls this inside 
            # shared_mutation_plate, so it already handles the "size" dimension.
            batch_locs = g_offset_locs
            batch_scales = g_offset_scales

        dk_geno_offset = pyro.sample(f"{site_name}_offset", dist.Normal(batch_locs, batch_scales))

        # Guard against full-sized array substitution during initialization or
        # re-runs with full-sized initial values
        if dk_geno_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
             dk_geno_offset = dk_geno_offset[..., data.batch_idx]

        # Replicate the Shift - LogNormal logic
        dk_geno_lognormal_values = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale), max=1e30)
        return dk_geno_hyper_shift - dk_geno_lognormal_values

    dk_geno_per_genotype = sample_genotype_parameter_guide(name, data, guide_dk_geno_per_unit)

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

    def guess_dk_geno_per_unit(site_name, size):
        return {f"{site_name}_offset": -0.8240460108562919 * jnp.ones(size)}

    guesses = {}
    guesses[f"{name}_hyper_loc"] = -3.5
    guesses[f"{name}_hyper_scale"] = 0.5
    guesses[f"{name}_shift"] = 0.02
    
    guesses.update(get_genotype_parameter_guesses(name, data, guess_dk_geno_per_unit))

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
    