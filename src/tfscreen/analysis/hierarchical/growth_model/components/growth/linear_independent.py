import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple, Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)

@dataclass(frozen=True)
class LinearParams:
    """
    Holds linear growth parameters (intercept and slope) for pre-selection 
    and selection phases.
    """
    k_pre: jnp.ndarray
    m_pre: jnp.ndarray
    k_sel: jnp.ndarray
    m_sel: jnp.ndarray

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the independent growth model.

    Attributes
    ----------
    k_hyper_loc_loc : jnp.ndarray
        Mean of the prior for the hyper-location of k (per-condition).
    k_hyper_loc_scale : jnp.ndarray
        Standard deviation of the prior for the hyper-location of k
        (per-condition).
    k_hyper_scale_loc : jnp.ndarray
        Scale of the HalfNormal prior for the hyper-scale of k
        (per-condition).
    m_hyper_loc_loc : jnp.ndarray
        Mean of the prior for the hyper-location of m (per-condition).
    m_hyper_loc_scale : jnp.ndarray
        Standard deviation of the prior for the hyper-location of m
        (per-condition).
    m_hyper_scale_loc : jnp.ndarray
        Scale of the HalfNormal prior for the hyper-scale of m
        (per-condition).
    """

    # dims are num_conditions long
    k_hyper_loc_loc: jnp.ndarray
    k_hyper_loc_scale: jnp.ndarray
    k_hyper_scale_loc: jnp.ndarray

    m_hyper_loc_loc: jnp.ndarray
    m_hyper_loc_scale: jnp.ndarray
    m_hyper_scale_loc: jnp.ndarray

def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> LinearParams:
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
        - ``data.num_condition_rep`` : (int) Number of experimental conditions.
        - ``data.num_replicate`` : (int) Number of replicates per condition.
        - ``data.map_condition_pre`` : (jnp.ndarray) Index array to map
          per-condition/replicate parameters to pre-selection observations.
        - ``data.map_condition_sel`` : (jnp.ndarray) Index array to map
          per-condition/replicate parameters to post-selection observations.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for the
        priors. All attributes are ``jnp.ndarray``s of shape
        ``(data.num_condition_rep,)``.
        - priors.k_hyper_loc_loc
        - priors.k_hyper_loc_scale
        - priors.k_hyper_scale_loc
        - priors.m_hyper_loc_loc
        - priors.m_hyper_loc_scale
        - priors.m_hyper_scale_loc

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """

    # Data assertions
    if data.growth_shares_replicates:
        raise ValueError("linear_independent cannot be used with growth_shares_replicates=True. Use 'linear' instead.")

    # Loop over conditions. NOTE THE FLIPPED PLATES. I need each condition to 
    # have its own priors (outer loop) for each replicate (inner loop). The 
    # data are ordered in the parameters as rep0, cond0 \ rep0, cond1 \ etc.
    # which means they ravel with these dimensions. 
    with pyro.plate(f"{name}_condition_parameters",data.num_condition_rep,dim=-1):

        growth_k_hyper_loc = pyro.sample(
            f"{name}_k_hyper_loc",
            dist.Normal(priors.k_hyper_loc_loc,
                        priors.k_hyper_loc_scale)
        )
        growth_k_hyper_scale = pyro.sample(
            f"{name}_k_hyper_scale",
            dist.HalfNormal(priors.k_hyper_scale_loc)
        )

        growth_m_hyper_loc = pyro.sample(
            f"{name}_m_hyper_loc",
            dist.Normal(priors.m_hyper_loc_loc,
                        priors.m_hyper_loc_scale)
        )
        growth_m_hyper_scale = pyro.sample(
            f"{name}_m_hyper_scale",
            dist.HalfNormal(priors.m_hyper_scale_loc)
        )

        # Loop over replicates
        with pyro.plate(f"{name}_replicate_parameters", data.num_replicate,dim=-2):
            k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0.0, 1.0))
            m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0.0, 1.0))
    
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

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors) -> LinearParams:
    """
    Guide function for the independent condition/replicate growth model.

    This function defines the variational distributions (guide) for the 
    independent growth model, specifying the parameterization of the 
    variational family for SVI inference. It registers variational parameters 
    for all global (per-condition) and local (per-replicate and per-condition) 
    latent variables, and samples from the corresponding distributions using 
    nested plates.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample and parameter sites in this guide.
    data : GrowthData
        Pytree (Flax dataclass) containing experimental data and metadata.
        Used to determine the number of conditions and replicates, and to 
        provide mapping arrays for expanding parameters to observations.
    priors : ModelPriors
        Pytree (Flax dataclass) containing the prior hyperparameters for the 
        model. Used to initialize the variational parameters.

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.

    Notes
    -----
    - The guide uses nested plates: the outer plate is over experimental 
      conditions, and the inner plate is over replicates within each condition.
    - All variational parameters are registered using `pyro.param` and are 
      initialized from the provided priors or with default values.
    - The returned arrays are flattened and then expanded to match the 
      observation indices using the mapping arrays in `data`.
    """

    # --- 1. Global Parameters (Per Condition) ---
    if data.growth_shares_replicates:
        raise ValueError("linear_independent cannot be used with growth_shares_replicates=True. Use 'linear' instead.")
    
    # K Hyper Loc (Normal)
    k_hl_loc = pyro.param(f"{name}_k_hyper_loc_loc", jnp.array(priors.k_hyper_loc_loc))
    k_hl_scale = pyro.param(f"{name}_k_hyper_loc_scale", jnp.array(priors.k_hyper_loc_scale),
                            constraint=dist.constraints.greater_than(1e-4))

    # K Hyper Scale (LogNormal guide for HalfNormal prior)
    k_hs_loc = pyro.param(f"{name}_k_hyper_scale_loc",
                          jnp.full(data.num_condition_rep, -1.0))
    k_hs_scale = pyro.param(f"{name}_k_hyper_scale_scale",
                            jnp.full(data.num_condition_rep, 0.1),
                            constraint=dist.constraints.greater_than(1e-4))

    # M Hyper Loc (Normal)
    m_hl_loc = pyro.param(f"{name}_m_hyper_loc_loc", jnp.array(priors.m_hyper_loc_loc))
    m_hl_scale = pyro.param(f"{name}_m_hyper_loc_scale", jnp.array(priors.m_hyper_loc_scale),
                            constraint=dist.constraints.greater_than(1e-4))

    # M Hyper Scale (LogNormal guide for HalfNormal prior)
    m_hs_loc = pyro.param(f"{name}_m_hyper_scale_loc",
                          jnp.full(data.num_condition_rep, -1.0))
    m_hs_scale = pyro.param(f"{name}_m_hyper_scale_scale",
                            jnp.full(data.num_condition_rep, 0.1),
                            constraint=dist.constraints.greater_than(1e-4))

    # --- 2. Local Parameters (Per Replicate AND Condition) ---
    # Shape: (num_replicate, num_condition_rep)
    # Note: dim 0 is replicate (-2), dim 1 is condition (-1)
    
    local_shape = (data.num_replicate, data.num_condition_rep)

    k_offset_locs = pyro.param(f"{name}_k_offset_locs", jnp.zeros(local_shape,dtype=float))
    k_offset_scales = pyro.param(f"{name}_k_offset_scales", jnp.ones(local_shape,dtype=float),
                                 constraint=dist.constraints.positive)

    m_offset_locs = pyro.param(f"{name}_m_offset_locs", jnp.zeros(local_shape,dtype=float))
    m_offset_scales = pyro.param(f"{name}_m_offset_scales", jnp.ones(local_shape,dtype=float),
                                 constraint=dist.constraints.positive)


    # --- 3. Sampling with Nested Plates ---
    
    # Outer Loop: Conditions (dim=-1)
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep, dim=-1) as idx_c:

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

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def calculate_growth(params: LinearParams,
                     dk_geno: jnp.ndarray,
                     activity: jnp.ndarray,
                     theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the growth rates for pre-selection and selection phases.

    Parameters
    ----------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, m_sel. 
    dk_geno : jnp.ndarray
        Genotype-specific death rate. 
    activity : jnp.ndarray
        Genotype activity. 
    theta : jnp.ndarray
        Occupancy/binding probability. 

    Returns
    -------
    g_pre : jnp.ndarray
        Pre-selection growth rate tensor.
    g_sel : jnp.ndarray
        Selection growth rate tensor.
    """
    
    g_pre = params.k_pre + dk_geno + activity * params.m_pre * theta
    g_sel = params.k_sel + dk_geno + activity * params.m_sel * theta
    
    return g_pre, g_sel


def get_hyperparameters(num_condition_rep: int=1) -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Parameters
    ----------
    num_condition_rep : int
        The number of experimental conditions, used to shape the
        hyperparameter arrays.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values (JAX arrays).
    """

    parameters = {}
    parameters["k_hyper_loc_loc"] = jnp.ones(num_condition_rep,dtype=float)*0.025
    parameters["k_hyper_loc_scale"] = jnp.ones(num_condition_rep,dtype=float)*0.1
    parameters["k_hyper_scale_loc"] = jnp.ones(num_condition_rep,dtype=float)
    parameters["m_hyper_loc_loc"] = jnp.zeros(num_condition_rep,dtype=float)
    parameters["m_hyper_loc_scale"] = jnp.ones(num_condition_rep,dtype=float)*0.01
    parameters["m_hyper_scale_loc"] = jnp.ones(num_condition_rep,dtype=float)

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get empirical guess values for the model's latent parameters.

    For each condition, ``k_hyper_loc`` is estimated as the median OLS slope
    of ln_cfu versus t_sel across replicates.  Per-replicate ``k_offset``
    values capture the within-condition deviation.  Falls back to hard-coded
    defaults when ``data.ln_cfu`` or ``data.t_sel`` are not 7-D tensors (e.g.
    mock data in tests).

    Parameters
    ----------
    name : str
        Prefix for all sample sites (e.g., ``"condition_growth"``).
    data : GrowthData
        Pytree with experimental data.  Requires:
        ``num_condition_rep``, ``num_replicate``, ``map_condition_sel``,
        ``ln_cfu``, ``t_sel``, ``good_mask``.

    Returns
    -------
    dict[str, jnp.ndarray]
        Keys and shapes:

        - ``{name}_k_hyper_loc``   : ``(num_condition_rep, 1)``
        - ``{name}_k_hyper_scale`` : ``(num_condition_rep, 1)``
        - ``{name}_m_hyper_loc``   : ``(num_condition_rep, 1)``
        - ``{name}_m_hyper_scale`` : ``(num_condition_rep, 1)``
        - ``{name}_k_offset``      : ``(num_condition_rep, num_replicate)``
        - ``{name}_m_offset``      : ``(num_condition_rep, num_replicate)``
    """

    _DEFAULT_HYPER_SCALE = 0.1
    _DEFAULT_K = 0.025

    num_cond = data.num_condition_rep   # N: unique conditions
    num_rep  = data.num_replicate       # M: replicates per condition

    hyper_shape = (num_cond, 1)
    local_shape = (num_cond, num_rep)

    ln_cfu           = np.array(data.ln_cfu)
    t_sel            = np.array(data.t_sel)
    good_mask        = np.array(data.good_mask)
    map_condition_sel = np.array(data.map_condition_sel)

    if ln_cfu.ndim != 7 or t_sel.ndim != 7:
        guesses = {}
        guesses[f"{name}_k_hyper_loc"]   = jnp.full(hyper_shape, _DEFAULT_K,           dtype=float)
        guesses[f"{name}_k_hyper_scale"] = jnp.full(hyper_shape, _DEFAULT_HYPER_SCALE,  dtype=float)
        guesses[f"{name}_m_hyper_loc"]   = jnp.zeros(hyper_shape,                       dtype=float)
        guesses[f"{name}_m_hyper_scale"] = jnp.full(hyper_shape, 0.01,                  dtype=float)
        guesses[f"{name}_k_offset"]      = jnp.zeros(local_shape,                       dtype=float)
        guesses[f"{name}_m_offset"]      = jnp.zeros(local_shape,                       dtype=float)
        return guesses

    # OLS slope of ln_cfu vs t_sel (vectorised over all axes except time).
    # Matches the approach used in linear.get_guesses.
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)
    t_sel_valid  = np.where(good_mask, t_sel,  np.nan)

    t_moved = np.moveaxis(t_sel_valid,  1, -1)
    y_moved = np.moveaxis(ln_cfu_valid, 1, -1)

    valid   = ~np.isnan(t_moved)
    n_valid = np.sum(valid, axis=-1)
    denom   = np.maximum(n_valid, 1)
    t_sum   = np.sum(np.where(valid, t_moved, 0.0), axis=-1)
    y_sum   = np.sum(np.where(valid, y_moved, 0.0), axis=-1)
    t_mean  = (t_sum / denom)[..., np.newaxis]
    y_mean  = (y_sum / denom)[..., np.newaxis]
    t_diff  = np.where(valid, t_moved - t_mean, 0.0)
    y_diff  = np.where(valid, y_moved - y_mean, 0.0)
    cov_ty  = np.sum(t_diff * y_diff, axis=-1)
    var_t   = np.sum(t_diff * t_diff, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = np.where((n_valid >= 2) & (var_t > 1e-20), cov_ty / var_t, np.nan)

    if map_condition_sel.ndim == 7:
        map_cr = map_condition_sel[:, 0, ...]
    else:
        map_cr = map_condition_sel

    # Collect one k estimate per flat (condition × replicate) index.
    # Flat indices run [0, N*M) in C-order of shape (N, M): index = c*M + r.
    total   = num_cond * num_rep
    k_flat  = np.full(total, np.nan)
    for flat_idx in range(total):
        if map_cr.ndim == slopes.ndim:
            cr_slopes = slopes[map_cr == flat_idx]
        else:
            cr_slopes = slopes[:, :, map_cr == flat_idx, ...]
        n_valid_cr = np.sum(~np.isnan(cr_slopes))
        if n_valid_cr > 0:
            k_flat[flat_idx] = np.nansum(cr_slopes) / n_valid_cr

    # Reshape to (num_cond, num_rep) so rows = conditions, cols = replicates.
    k_per_cond_rep = k_flat.reshape(num_cond, num_rep)

    # Per-condition k_hyper_loc: median across replicates.
    k_hyper_loc = np.nanmedian(k_per_cond_rep, axis=1)       # (N,)
    valid_k  = k_hyper_loc[~np.isnan(k_hyper_loc)]
    global_k = float(np.nanmedian(valid_k)) if len(valid_k) > 0 else _DEFAULT_K
    k_hyper_loc = np.where(np.isnan(k_hyper_loc), global_k, k_hyper_loc)

    # Per-(condition, replicate) offsets.
    k_offsets = (k_per_cond_rep - k_hyper_loc[:, np.newaxis]) / _DEFAULT_HYPER_SCALE
    k_offsets = np.where(np.isnan(k_offsets), 0.0, k_offsets)

    guesses = {}
    guesses[f"{name}_k_hyper_loc"]   = jnp.array(k_hyper_loc.reshape(num_cond, 1), dtype=float)
    guesses[f"{name}_k_hyper_scale"] = jnp.full(hyper_shape, _DEFAULT_HYPER_SCALE, dtype=float)
    guesses[f"{name}_m_hyper_loc"]   = jnp.zeros(hyper_shape,                      dtype=float)
    guesses[f"{name}_m_hyper_scale"] = jnp.full(hyper_shape, 0.01,                 dtype=float)
    guesses[f"{name}_k_offset"]      = jnp.array(k_offsets,                        dtype=float)
    guesses[f"{name}_m_offset"]      = jnp.zeros(local_shape,                      dtype=float)
    return guesses

def get_priors(num_condition_rep: int=1) -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Parameters
    ----------
    num_condition_rep : int, optional
        The number of experimental conditions, which is required by
        `get_hyperparameters`. Default is 1.

    Returns
    -------
    ModelPriors
        A populated Pytree (Flax dataclass) of hyperparameters.
    """
    # Call the imported get_hyperparameters
    params = get_hyperparameters(num_condition_rep)
    return ModelPriors(**params)

    