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
        Mean of the prior for the hyper-location of ln_cfu0 (library genotypes).
    ln_cfu0_hyper_loc_scale : float
        Standard deviation of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of ln_cfu0.
    ln_cfu0_spiked_loc_loc : float
        Mean of the Normal prior for the ln_cfu0 location of spiked genotypes.
    ln_cfu0_spiked_loc_scale : float
        Standard deviation of the Normal prior for the spiked genotype location.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float
    ln_cfu0_spiked_loc_loc: float
    ln_cfu0_spiked_loc_scale: float

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Defines the hierarchical model for initial cell counts (ln_cfu0).

    Library genotypes share a pooled Normal distribution whose location and
    scale are learned hyper-parameters.  Spiked genotypes (flagged via
    ``data.ln_cfu0_spiked_mask``) share a separate scalar location parameter,
    allowing them to have a very different prior mean (e.g. ln_cfu0 ~ 10)
    without distorting the library hierarchy.  All genotypes share the same
    hyper-scale and per-genotype non-centered offsets.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites.
    data : GrowthData
        Pytree containing experimental data and metadata. Uses:
        - ``data.num_replicate``, ``data.num_condition_pre``, ``data.batch_size``
        - ``data.batch_idx`` : index array for the current mini-batch
        - ``data.scale_vector`` : importance-weight vector for subsampling
        - ``data.ln_cfu0_spiked_mask`` : bool array (num_genotype,), True = spiked
    priors : ModelPriors
        Pytree containing the hyperparameters.

    Returns
    -------
    jnp.ndarray
        The sampled ``ln_cfu0`` values expanded to observation shape
        ``(num_replicate, 1, num_condition_pre, 1, 1, 1, batch_size)``.
    """

    # Hyper-priors for the library-genotype pooled distribution
    ln_cfu0_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.ln_cfu0_hyper_loc_loc,
                    priors.ln_cfu0_hyper_loc_scale)
    )
    ln_cfu0_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.ln_cfu0_hyper_scale_loc)
    )

    # Separate scalar location for spiked genotypes
    ln_cfu0_spiked_loc = pyro.sample(
        f"{name}_spiked_loc",
        dist.Normal(priors.ln_cfu0_spiked_loc_loc,
                    priors.ln_cfu0_spiked_loc_scale)
    )

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate",data.num_replicate,dim=-3):
        with pyro.plate(f"{name}_condition_pre",data.num_condition_pre,dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
                with pyro.handlers.scale(scale=data.scale_vector):
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Per-genotype location: spiked genotypes use spiked_loc, library uses hyper_loc
    batch_spiked_mask = data.ln_cfu0_spiked_mask[data.batch_idx]
    per_geno_loc = jnp.where(batch_spiked_mask, ln_cfu0_spiked_loc, ln_cfu0_hyper_loc)

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Register deterministic values for inspection
    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide corresponding to the hierarchical ln_cfu0 model.

    Uses Normal variational distributions for the hyper-location and spiked
    location, LogNormal for the hyper-scale, and Normal for per-genotype offsets.
    """

    # -------------------------------------------------------------------------
    # Global parameters

    # Hyper Loc — library genotypes (Normal posterior)
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.ln_cfu0_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.ln_cfu0_hyper_loc_scale), constraint=dist.constraints.positive)
    ln_cfu0_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal posterior approximation for positive variable)
    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0)) # Init small
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    ln_cfu0_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))

    # Spiked Loc — spiked genotypes (Normal posterior)
    s_loc_loc = pyro.param(f"{name}_spiked_loc_loc", jnp.array(priors.ln_cfu0_spiked_loc_loc))
    s_loc_scale = pyro.param(f"{name}_spiked_loc_scale", jnp.array(priors.ln_cfu0_spiked_loc_scale), constraint=dist.constraints.positive)
    ln_cfu0_spiked_loc = pyro.sample(f"{name}_spiked_loc", dist.Normal(s_loc_loc, s_loc_scale))

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

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Per-genotype location: spiked genotypes use spiked_loc, library uses hyper_loc
    batch_spiked_mask = data.ln_cfu0_spiked_mask[data.batch_idx]
    per_geno_loc = jnp.where(batch_spiked_mask, ln_cfu0_spiked_loc, ln_cfu0_hyper_loc)

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = per_geno_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]

    return ln_cfu0

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their default values.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_loc"] = -2.5
    parameters["ln_cfu0_hyper_loc_scale"] = 3.0
    parameters["ln_cfu0_hyper_scale_loc"] = 2.0
    parameters["ln_cfu0_spiked_loc_loc"] = 10.0
    parameters["ln_cfu0_spiked_loc_scale"] = 3.0

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.

    Offsets are set to zero (all genotypes start at their respective location
    prior mean).  The ``spiked_loc`` guess is set to the prior mean so that
    spiked genotypes initialise near ln_cfu0 = 10.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites.
    data : GrowthData
        Pytree containing data metadata. Requires ``data.num_replicate``,
        ``data.num_condition_pre``, ``data.num_genotype``.

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names to JAX arrays of guess values.
    """

    guesses = {}
    guesses[f"{name}_hyper_loc"] = -2.5
    guesses[f"{name}_hyper_scale"] = 3.0
    guesses[f"{name}_spiked_loc"] = 10.0
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
