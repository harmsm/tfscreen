import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple, Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the independent growth model.

    Attributes
    ----------
    growth_k_hyper_loc_loc : jnp.ndarray
        Mean of the prior for the hyper-location of k (per-condition).
    growth_k_hyper_loc_scale : jnp.ndarray
        Standard deviation of the prior for the hyper-location of k 
        (per-condition).
    growth_k_hyper_scale : jnp.ndarray
        Scale of the HalfNormal prior for the hyper-scale of k 
        (per-condition).
    growth_m_hyper_loc_loc : jnp.ndarray
        Mean of the prior for the hyper-location of m (per-condition).
    growth_m_hyper_loc_scale : jnp.ndarray
        Standard deviation of the prior for the hyper-location of m 
        (per-condition).
    growth_m_hyper_scale : jnp.ndarray
        Scale of the HalfNormal prior for the hyper-scale of m 
        (per-condition).
    """

    # dims are num_conditions long
    growth_k_hyper_loc_loc: jnp.ndarray
    growth_k_hyper_loc_scale: jnp.ndarray
    growth_k_hyper_scale: jnp.ndarray

    growth_m_hyper_loc_loc: jnp.ndarray
    growth_m_hyper_loc_scale: jnp.ndarray
    growth_m_hyper_scale: jnp.ndarray

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> Tuple[jnp.ndarray, ...]:
    """
    Defines growth parameters k and m with independent priors per condition.

    This model defines growth parameters k (basal growth) and m (theta-dependent
    growth) where k and m are modeled as `k = k_hyper_loc + k_offset * k_hyper_scale` (and similarly for m).

    In this "independent" model, the hyper-parameters (`_hyper_loc`, 
    `_hyper_scale`) are sampled independently for each experimental condition,
    and then all replicates within that condition share those hyper-parameters.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample/deterministic sites in this
        component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_condition`` : (int) Number of experimental conditions.
        - ``data.num_replicate`` : (int) Number of replicates per condition.
        - ``data.map_condition_pre`` : (jnp.ndarray) Index array to map
          per-condition/replicate parameters to pre-selection observations.
        - ``data.map_condition_sel`` : (jnp.ndarray) Index array to map
          per-condition/replicate parameters to post-selection observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for the
        priors. All attributes are ``jnp.ndarray``s of shape
        ``(data.num_condition,)``.
        - priors.growth_k_hyper_loc_loc
        - priors.growth_k_hyper_loc_scale
        - priors.growth_k_hyper_scale
        - priors.growth_m_hyper_loc_loc
        - priors.growth_m_hyper_loc_scale
        - priors.growth_m_hyper_scale

    Returns
    -------
    k_pre : jnp.ndarray
        Basal growth rate `k` for pre-selection, expanded to match
        observations.
    m_pre : jnp.ndarray
        Theta-dependent growth rate `m` for pre-selection, expanded to match
        observations.
    k_sel : jnp.ndarray
        Basal growth rate `k` for post-selection, expanded to match
        observations.
    m_sel : jnp.ndarray
        Theta-dependent growth rate `m` for post-selection, expanded to match
        observations.
    """

    # Loop over conditions. NOTE THE FLIPPED PLATES. I need each condition to 
    # have its own priors (outer loop) for each replicate (inner loop). The 
    # data are ordered in the parameters as rep0, cond0 \ rep0, cond1 \ etc.
    # which means they ravel with these dimensions. 
    with pyro.plate(f"{name}_condition_parameters",data.num_condition,dim=-1):

        growth_k_hyper_loc = pyro.sample(
            f"{name}_k_hyper_loc",
            dist.Normal(priors.growth_k_hyper_loc_loc,
                        priors.growth_k_hyper_loc_scale)
        )
        growth_k_hyper_scale = pyro.sample(
            f"{name}_k_hyper_scale",
            dist.HalfNormal(priors.growth_k_hyper_scale)
        )

        growth_m_hyper_loc = pyro.sample(
            f"{name}_m_hyper_loc",
            dist.Normal(priors.growth_m_hyper_loc_loc,
                        priors.growth_m_hyper_loc_scale)
        )
        growth_m_hyper_scale = pyro.sample(
            f"{name}_m_hyper_scale",
            dist.HalfNormal(priors.growth_m_hyper_scale)
        )

        # Loop over replicates
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate,dim=-2):
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0, 1))
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0, 1))
    
        growth_k_dist = growth_k_hyper_loc + k_offset * growth_k_hyper_scale
        growth_m_dist = growth_m_hyper_loc + m_offset * growth_m_hyper_scale
    
    # Flatten array
    growth_k_dist_1d = growth_k_dist.ravel()
    growth_m_dist_1d = growth_m_dist.ravel()

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_dist_1d)
    pyro.deterministic(f"{name}_m", growth_m_dist_1d)

    # Expand to full-sized tensors
    k_pre = growth_k_dist_1d[data.map_condition_pre]
    m_pre = growth_m_dist_1d[data.map_condition_pre]
    k_sel = growth_k_dist_1d[data.map_condition_sel]
    m_sel = growth_m_dist_1d[data.map_condition_sel]

    return k_pre, m_pre, k_sel, m_sel

def get_hyperparameters(num_condition: int) -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Parameters
    ----------
    num_condition : int
        The number of experimental conditions, used to shape the
        hyperparameter arrays.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values (JAX arrays).
    """

    parameters = {}
    parameters["growth_k_hyper_loc_loc"] = jnp.ones(num_condition)*0.025
    parameters["growth_k_hyper_loc_scale"] = jnp.ones(num_condition)*0.1
    parameters["growth_k_hyper_scale"] = jnp.ones(num_condition)*1.0
    parameters["growth_m_hyper_loc_loc"] = jnp.zeros(num_condition)*0.0
    parameters["growth_m_hyper_loc_scale"] = jnp.ones(num_condition)*0.01
    parameters["growth_m_hyper_scale"] = jnp.ones(num_condition)*1.0

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `numpyro.handlers.substitute` for testing
    or initializing inference.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A Pytree containing data metadata, used to determine the
        shape of the guess arrays. Requires:
        - ``data.num_condition``
        - ``data.num_replicate``

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping sample site names (e.g., "my_model_k_offset")
        to JAX arrays of guess values.
    
    Notes
    -----
    The shapes of the guesses are critical:
    - ``_hyper_loc``/``_hyper_scale`` sites are sampled within the
      ``condition_parameters`` plate, so their shape must be
      ``(data.num_condition, 1)``.
    - ``_offset`` sites are sampled within both plates, so their shape
      must be ``(data.num_condition, data.num_replicate)``.
    """

    shape = (data.num_condition, data.num_replicate)

    # Shape for hyper-parameters sampled inside the condition plate
    hyper_shape = (data.num_condition, 1) 

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = jnp.ones(hyper_shape) * 1.0
    guesses[f"{name}_k_hyper_scale"] = jnp.ones(hyper_shape) * 0.1
    guesses[f"{name}_m_hyper_loc"] = jnp.ones(hyper_shape) * 1.0
    guesses[f"{name}_m_hyper_scale"] = jnp.ones(hyper_shape) * 0.1
    
    guesses[f"{name}_k_offset"] = jnp.zeros(shape)
    guesses[f"{name}_m_offset"] = jnp.zeros(shape)

    return guesses

def get_priors(num_condition: int) -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Parameters
    ----------
    num_condition : int
        The number of experimental conditions, which is required by
        `get_hyperparameters`.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    # Call the imported get_hyperparameters
    params = get_hyperparameters(num_condition)
    return ModelPriors(**params)

    