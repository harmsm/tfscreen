import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the ln_cfu0 model.

    Attributes
    ----------
    ln_cfu0_hyper_loc_loc : float
        Mean of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_loc_scale : float
        Standard deviation of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of ln_cfu0.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the hierarchical model for initial cell counts (ln_cfu0).

    This model treats the ``ln_cfu0`` for each independent experimental
    group (e.g., each genotype/replicate combination) as being drawn
    from a shared, pooled Normal distribution. The location and scale
    of this distribution are learned hyper-parameters.

    This function defines the non-centered parameterization for these
    parameters and returns the final `ln_cfu0` values expanded to match
    the full set of observations.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites in this
        component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_ln_cfu0`` : (int) The number of independent ln_cfu0
          groups.
        - ``data.map_ln_cfu0`` : (jnp.ndarray) Index array to map
          per-group parameters to the full set of observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for
        the pooled priors.

    Returns
    -------
    jnp.ndarray
        The sampled ``ln_cfu0`` values, expanded to match the shape of
        the observations via ``data.map_ln_cfu0``.
    """
    
    # Define hyper-priors for the pooled distribution
    ln_cfu0_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.ln_cfu0_hyper_loc_loc,
                    priors.ln_cfu0_hyper_loc_scale)
    )
    ln_cfu0_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.ln_cfu0_hyper_scale_loc)
    )
    
    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate",data.num_replicate,dim=-3):
        with pyro.plate(f"{name}_condition_pre",data.num_condition_pre,dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = ln_cfu0_hyper_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Register deterministic values for inspection
    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide
    """

    # -------------------------------------------------------------------------
    # Global parameters

    # Hyper Loc (Normal posterior approximation)
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.ln_cfu0_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.ln_cfu0_hyper_loc_scale), constraint=dist.constraints.positive)
    ln_cfu0_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal posterior approximation for positive variable)
    # We use LogNormal because HalfNormal (prior) support is positive.
    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0)) # Init small
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    ln_cfu0_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))
    
    # -------------------------------------------------------------------------
    # Genotype-specific parameter

    param_shape = (data.num_replicate, data.num_condition_pre, data.num_genotype)
    offset_locs = pyro.param(f"{name}_offset_locs",
                             jnp.zeros(param_shape,dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales",
                               jnp.ones(param_shape,dtype=float),
                               constraint=dist.constraints.positive)

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate",data.num_replicate,dim=-3):
        with pyro.plate(f"{name}_condition_pre",data.num_condition_pre,dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    
                    batch_locs = offset_locs[...,data.batch_idx]
                    batch_scales = offset_scales[...,data.batch_idx]
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(batch_locs,batch_scales))

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = ln_cfu0_hyper_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_loc"] = -2.5
    parameters["ln_cfu0_hyper_loc_scale"] = 3.0
    parameters["ln_cfu0_hyper_scale_loc"] = 2.0
               
    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `numpyro.handlers.substitute` for testing
    or initializing inference. The offsets are set to zero, meaning
    all per-group ``ln_cfu0`` values will equal the ``_hyper_loc`` guess.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A Pytree containing data metadata, used to determine the
        shape of the guess arrays. Requires:
        - ``data.num_ln_cfu0``

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names (e.g., "my_model_offset")
        to JAX arrays of guess values.
    """
    
    guesses = {}
    guesses[f"{name}_hyper_loc"] = -2.5
    guesses[f"{name}_hyper_scale"] = 3.0
    guesses[f"{name}_offset"] = jnp.zeros((data.num_replicate,
                                           data.num_condition_pre,
                                           data.num_genotype),dtype=float)

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