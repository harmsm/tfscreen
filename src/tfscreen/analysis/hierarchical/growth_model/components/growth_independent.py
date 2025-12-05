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

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> Tuple[jnp.ndarray, ...]:
    """
    Guide corresponding to the independent condition/replicate model.
    """

    # --- 1. Global Parameters (Per Condition) ---
    
    # K Hyper Loc (Normal)
    k_hl_loc = pyro.param(f"{name}_k_hyper_loc_loc", jnp.array(priors.growth_k_hyper_loc_loc))
    k_hl_scale = pyro.param(f"{name}_k_hyper_loc_scale", jnp.array(priors.growth_k_hyper_loc_scale),
                            constraint=dist.constraints.positive)
    
    # K Hyper Scale (LogNormal guide for HalfNormal prior)
    k_hs_loc = pyro.param(f"{name}_k_hyper_scale_loc", jnp.array(-1.0))
    k_hs_scale = pyro.param(f"{name}_k_hyper_scale_scale", jnp.array(0.1),
                            constraint=dist.constraints.positive)

    # M Hyper Loc (Normal)
    m_hl_loc = pyro.param(f"{name}_m_hyper_loc_loc", jnp.array(priors.growth_m_hyper_loc_loc))
    m_hl_scale = pyro.param(f"{name}_m_hyper_loc_scale", jnp.array(priors.growth_m_hyper_loc_scale),
                            constraint=dist.constraints.positive)
    
    # M Hyper Scale (LogNormal guide for HalfNormal prior)
    m_hs_loc = pyro.param(f"{name}_m_hyper_scale_loc", jnp.array(-1.0))
    m_hs_scale = pyro.param(f"{name}_m_hyper_scale_scale", jnp.array(0.1),
                            constraint=dist.constraints.positive)

    # --- 2. Local Parameters (Per Replicate AND Condition) ---
    # Shape: (num_replicate, num_condition)
    # Note: dim 0 is replicate (-2), dim 1 is condition (-1)
    
    local_shape = (data.num_replicate, data.num_condition)

    k_offset_locs = pyro.param(f"{name}_k_offset_locs", jnp.zeros(local_shape,dtype=float))
    k_offset_scales = pyro.param(f"{name}_k_offset_scales", jnp.ones(local_shape,dtype=float),
                                 constraint=dist.constraints.positive)

    m_offset_locs = pyro.param(f"{name}_m_offset_locs", jnp.zeros(local_shape,dtype=float))
    m_offset_scales = pyro.param(f"{name}_m_offset_scales", jnp.ones(local_shape,dtype=float),
                                 constraint=dist.constraints.positive)


    # --- 3. Sampling with Nested Plates ---
    
    # Outer Loop: Conditions (dim=-1)
    with pyro.plate(f"{name}_condition_parameters", data.num_condition, dim=-1) as idx_c:

        # Sample Hypers (Sliced by Condition)
        growth_k_hyper_loc = pyro.sample(f"{name}_k_hyper_loc", 
                                         dist.Normal(k_hl_loc[idx_c], k_hl_scale[idx_c]))
        
        growth_k_hyper_scale = pyro.sample(f"{name}_k_hyper_scale", 
                                           dist.LogNormal(k_hs_loc[idx_c], k_hs_scale[idx_c]))

        growth_m_hyper_loc = pyro.sample(f"{name}_m_hyper_loc", 
                                         dist.Normal(m_hl_loc[idx_c], m_hl_scale[idx_c]))
        
        growth_m_hyper_scale = pyro.sample(f"{name}_m_hyper_scale", 
                                           dist.LogNormal(m_hs_loc[idx_c], m_hs_scale[idx_c]))

        # Inner Loop: Replicates (dim=-2)
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate, dim=-2) as idx_r:
            
            # Slice Locals: 
            # We must broadcast row indices (idx_r) against col indices (idx_c)
            # idx_r[:, None] gives shape (Batch_R, 1)
            # idx_c          gives shape (Batch_C)
            # Result         gives shape (Batch_R, Batch_C) matching the plates
            
            k_batch_locs = k_offset_locs[idx_r[:, None], idx_c]
            k_batch_scales = k_offset_scales[idx_r[:, None], idx_c]
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(k_batch_locs, k_batch_scales))

            m_batch_locs = m_offset_locs[idx_r[:, None], idx_c]
            m_batch_scales = m_offset_scales[idx_r[:, None], idx_c]
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(m_batch_locs, m_batch_scales))
    
    # --- 4. Reconstruction ---
    # Note: Broadcasting handles the shape mismatch between Hypers (Batch_C,) and Offsets (Batch_R, Batch_C)
    growth_k_dist = growth_k_hyper_loc + k_offset * growth_k_hyper_scale
    growth_m_dist = growth_m_hyper_loc + m_offset * growth_m_hyper_scale
    
    # Flatten array (ravel uses C-style order: row0, row1...)
    # This matches the "rep0, cond0 \ rep0, cond1" order if cond is the last axis.
    growth_k_dist_1d = growth_k_dist.ravel()
    growth_m_dist_1d = growth_m_dist.ravel()

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
    parameters["growth_k_hyper_loc_loc"] = jnp.ones(num_condition,dtype=float)*0.025
    parameters["growth_k_hyper_loc_scale"] = jnp.ones(num_condition,dtype=float)*0.1
    parameters["growth_k_hyper_scale"] = jnp.ones(num_condition,dtype=float)
    parameters["growth_m_hyper_loc_loc"] = jnp.zeros(num_condition,dtype=float)
    parameters["growth_m_hyper_loc_scale"] = jnp.ones(num_condition,dtype=float)*0.01
    parameters["growth_m_hyper_scale"] = jnp.ones(num_condition,dtype=float)

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
    guesses[f"{name}_k_hyper_loc"] = jnp.ones(hyper_shape,dtype=float)
    guesses[f"{name}_k_hyper_scale"] = jnp.ones(hyper_shape,dtype=float) * 0.1
    guesses[f"{name}_m_hyper_loc"] = jnp.ones(hyper_shape,dtype=float) 
    guesses[f"{name}_m_hyper_scale"] = jnp.ones(hyper_shape,dtype=float) * 0.1
    
    guesses[f"{name}_k_offset"] = jnp.zeros(shape,dtype=float)
    guesses[f"{name}_m_offset"] = jnp.zeros(shape,dtype=float)

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

    