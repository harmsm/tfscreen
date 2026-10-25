import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field
from typing import Dict, Any, Mapping

from tfscreen.tfmodel.data_class import GrowthData
from tfscreen.tfmodel.generative.components._population import per_genotype
from tfscreen.tfmodel.generative.components._pinning import (
    _hyper,
    _pinned_value,
)


# Hyperparameter suffixes that may be pinned via ModelPriors.pinned.
_PINNABLE_SUFFIXES = (
    "hyper_loc", "hyper_scale",
)


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the hierarchical activity model.

    Attributes
    ----------
    hyper_loc_loc : float
        Mean of the prior for the hyper-location of log(activity).
    hyper_loc_scale : float
        Std dev of the prior for the hyper-location of log(activity).
    hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of log(activity).
    pinned : Mapping[str, float]
        Optional mapping from suffix (in ``_PINNABLE_SUFFIXES``) to a
        constant value used in place of sampling that hyperparameter.
    """

    hyper_loc_loc: float
    hyper_loc_scale: float
    hyper_scale_loc: float

    pinned: Mapping[str, float] = field(
        pytree_node=False, default_factory=dict
    )

def _activity_values(hyper_loc, hyper_scale, wt_indexes):
    """Per-genotype activity from log-space offsets; wt pinned to 1."""
    def compute(activity_offset, genotype_idx):
        log_activity = hyper_loc + activity_offset * hyper_scale
        activity = jnp.clip(jnp.exp(log_activity), max=1e30)
        return jnp.where(jnp.isin(genotype_idx, wt_indexes), 1.0, activity)
    return compute


def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors,
                 return_population: bool = False) -> jnp.ndarray:
    """
    Defines the hierarchical model for genotype-specific activity.

    Activity is the scale of transcriptional output (range 0 to infinity).
    This model assumes the wild-type genotype has a fixed activity of 1.0.
    The activities of all other (mutant) genotypes are modeled as being
    drawn from a shared, pooled LogNormal distribution (i.e., a Normal
    distribution in log-space).

    This function returns the final `activity` values expanded to match
    the full set of observations.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites in this
        component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_genotype`` : (int) The total number of genotypes.
        - ``data.wt_indexes`` : (jnp.ndarray) integer indexes of wt elements
        - ``data.map_genotype`` : (jnp.ndarray) Index array to map
          per-genotype parameters to the full set of observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for
        the pooled priors.
    return_population : bool, default False
        Also return the library-ordered per-genotype values (shape
        ``(num_genotype,)``, or ``None`` when the latents arrived as a
        batch-sized substitution), for callers that look genotypes up by
        library index (the congression mixture).

    Returns
    -------
    jnp.ndarray
        The sampled `activity` values, expanded to match the shape of
        the observations via ``data.map_genotype``. With
        ``return_population``, a ``(tensor, population)`` tuple.
    """

    pinned = priors.pinned

    # Priors are on log(activity), so their mean is log(1.0) = 0.0
    log_activity_hyper_loc = _hyper(
        name, "hyper_loc",
        dist.Normal(priors.hyper_loc_loc,
                    priors.hyper_loc_scale),
        pinned,
    )
    log_activity_hyper_scale = _hyper(
        name, "hyper_scale",
        dist.HalfNormal(priors.hyper_scale_loc),
        pinned,
    )

    # Sample non-centered offsets for mutant genotypes only
    with pyro.plate(f"{name}_genotype_plate", data.num_genotype, dim=-1):
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Sampled at library size: slice to this batch's genotypes (a value of
    # any other size is an already-sliced substitution from the posterior
    # forward pass). Computed in log-space; wt activity is 1.0.
    activity, population = per_genotype(
        _activity_values(log_activity_hyper_loc, log_activity_hyper_scale,
                         data.wt_indexes),
        [activity_offset], data, return_population)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity)  

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:] 

    if return_population:
        return activity, population
    return activity

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors,
          return_population: bool = False) -> jnp.ndarray:
    """
    Guide corresponding to the hierarchical activity model.

    This guide uses:
    - Normal distributions for the hyper-location mean.
    - LogNormal distributions for hyper-scale and positive scale parameters.
    - A non-centered parameterization for the per-genotype offsets.
    """

    pinned = priors.pinned

    pinned_hl = _pinned_value("hyper_loc", pinned)
    if pinned_hl is not None:
        log_activity_hyper_loc = pinned_hl
    else:
        a_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.hyper_loc_loc))
        a_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.hyper_loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        log_activity_hyper_loc = pyro.sample(
            f"{name}_hyper_loc",
            dist.Normal(a_loc_loc, a_loc_scale)
        )

    pinned_hs = _pinned_value("hyper_scale", pinned)
    if pinned_hs is not None:
        log_activity_hyper_scale = pinned_hs
    else:
        a_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0))
        a_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1),
                                   constraint=dist.constraints.greater_than(1e-4))
        log_activity_hyper_scale = pyro.sample(
            f"{name}_hyper_scale",
            dist.LogNormal(a_scale_loc, a_scale_scale)
        )

    offset_locs = pyro.param(f"{name}_offset_locs",
                             jnp.zeros(data.num_genotype,dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales",
                               jnp.ones(data.num_genotype,dtype=float),
                               constraint=dist.constraints.positive)

    # Sample non-centered offsets for mutant genotypes only
    with pyro.plate(f"{name}_genotype_plate", data.num_genotype, dim=-1):
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(offset_locs, offset_scales))

    # Sampled at library size: slice to this batch's genotypes (a value of
    # any other size is an already-sliced substitution from the posterior
    # forward pass). Computed in log-space; wt activity is 1.0.
    activity, population = per_genotype(
        _activity_values(log_activity_hyper_loc, log_activity_hyper_scale,
                         data.wt_indexes),
        [activity_offset], data, return_population)

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:] 

    if return_population:
        return activity, population
    return activity


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    The hyper-location is centered at 0.0, corresponding to log(1.0),
    so the prior is centered on the wild-type activity.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their
        default values.
    """
    
    parameters = {}
    parameters["hyper_loc_loc"] = 0.0
    parameters["hyper_loc_scale"] = 0.01
    parameters["hyper_scale_loc"] = 0.1

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `numpyro.handlers.substitute` for testing
    or initializing inference. The offsets are set to zero, meaning
    all mutant activities will be guessed as 1.0 (same as wild-type).

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A Pytree containing data metadata, used to determine the
        shape of the guess arrays. Requires:
        - ``data.num_genotype``

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names to JAX arrays of
        guess values.
    """

    guesses = {}
    guesses[f"{name}_hyper_loc"] = 0.0
    guesses[f"{name}_hyper_scale"] = 0.1
    guesses[f"{name}_offset"] = jnp.zeros(data.num_genotype,dtype=float)

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


def get_extract_specs(ctx):
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["activity"],
        map_column="map_genotype",
        get_columns=["genotype"],
        in_run_prefix="",
    )]