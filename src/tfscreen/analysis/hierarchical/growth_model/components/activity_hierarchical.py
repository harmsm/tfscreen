import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the hierarchical activity model.

    Attributes
    ----------
    activity_hyper_loc_loc : float
        Mean of the prior for the hyper-location of log(activity).
    activity_hyper_loc_scale : float
        Std dev of the prior for the hyper-location of log(activity).
    activity_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of log(activity).
    """

    activity_hyper_loc_loc: float
    activity_hyper_loc_scale: float
    activity_hyper_scale_loc: float

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> jnp.ndarray:
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
        - ``data.wt_indexes`` : (jnp.ndarray) A boolean mask that is
          `True` for non-wild-type genotypes.
        - ``data.map_genotype`` : (jnp.ndarray) Index array to map
          per-genotype parameters to the full set of observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for
        the pooled priors.

    Returns
    -------
    jnp.ndarray
        The sampled `activity` values, expanded to match the shape of
        the observations via ``data.map_genotype``.
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

    # Sample non-centered offsets for mutant genotypes only
    with pyro.plate("shared_genotype_plate", size=data.num_genotype,subsample_size=data.batch_size,dim=-1):
        activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0, 1))
    
    # Calculate in log-space, then exponentiate
    log_activity_mutant_dists = log_activity_hyper_loc + activity_offset * log_activity_hyper_scale
    activity = jnp.clip(jnp.exp(log_activity_mutant_dists), max=1e30)

    # Set wildtype activity to 1.0
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    activity = jnp.where(is_wt_mask, 1.0, activity)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity)  

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:] 

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
    parameters["activity_hyper_loc_loc"] = 0.0
    parameters["activity_hyper_loc_scale"] = 0.01
    parameters["activity_hyper_scale_loc"] = 0.1

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
    guesses[f"{name}_log_hyper_loc"] = 0.0
    guesses[f"{name}_log_hyper_scale"] = 0.1
    guesses[f"{name}_offset"] = jnp.zeros(data.num_genotype)

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