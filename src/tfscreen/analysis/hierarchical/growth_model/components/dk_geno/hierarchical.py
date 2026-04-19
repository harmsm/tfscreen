import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field
from typing import Mapping

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
)


# Hyperparameter suffixes that may be pinned via ModelPriors.pinned.
_PINNABLE_SUFFIXES = (
    "hyper_loc", "hyper_scale", "shift",
)


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

    pinned: Mapping[str, float] = field(
        pytree_node=False, default_factory=dict
    )
    

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

    pinned = priors.pinned

    dk_geno_hyper_loc = _hyper(
        name, "hyper_loc",
        dist.Normal(priors.dk_geno_hyper_loc_loc,
                    priors.dk_geno_hyper_loc_scale),
        pinned,
    )
    dk_geno_hyper_scale = _hyper(
        name, "hyper_scale",
        dist.HalfNormal(priors.dk_geno_hyper_scale_loc),
        pinned,
    )

    dk_geno_hyper_shift = _hyper(
        name, "shift",
        dist.Normal(priors.dk_geno_hyper_shift_loc,
                    priors.dk_geno_hyper_shift_scale),
        pinned,
    )

    with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
        with pyro.handlers.scale(scale=data.scale_vector):
            dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs 
    # with full-sized initial values
    if dk_geno_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        dk_geno_offset = dk_geno_offset[..., data.batch_idx]
    
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
    Guide corresponding to the hierarchical dk_geno model.

    This guide defines the variational family for the pleiotropic growth
    effect model. It handles:
    - Normal distributions for the hyper-location and shift.
    - LogNormal distribution for the hyper-scale.
    - Normal distributions for the per-genotype offsets.
    """

    pinned = priors.pinned

    # --- Global Parameters ---

    # Hyper Loc (Normal guide for Normal prior)
    pinned_hl = _pinned_value("hyper_loc", pinned)
    if pinned_hl is not None:
        dk_geno_hyper_loc = pinned_hl
    else:
        h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", jnp.array(priors.dk_geno_hyper_loc_loc))
        h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", jnp.array(priors.dk_geno_hyper_loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        dk_geno_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal guide for HalfNormal prior)
    # Initialized to -1.0 (approx 0.37) to start with reasonable spread
    pinned_hs = _pinned_value("hyper_scale", pinned)
    if pinned_hs is not None:
        dk_geno_hyper_scale = pinned_hs
    else:
        h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0))
        h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1),
                                   constraint=dist.constraints.greater_than(1e-4))
        dk_geno_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))

    # Shift (Normal guide for Normal prior)
    pinned_sh = _pinned_value("shift", pinned)
    if pinned_sh is not None:
        dk_geno_hyper_shift = pinned_sh
    else:
        shift_loc = pyro.param(f"{name}_shift_loc", jnp.array(priors.dk_geno_hyper_shift_loc))
        shift_scale = pyro.param(f"{name}_shift_scale", jnp.array(priors.dk_geno_hyper_shift_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        dk_geno_hyper_shift = pyro.sample(f"{name}_shift", dist.Normal(shift_loc, shift_scale))

    # --- Local Parameters (Per Genotype) ---
    
    offset_locs = pyro.param(f"{name}_offset_locs", jnp.zeros(data.num_genotype,dtype=float))
    offset_scales = pyro.param(f"{name}_offset_scales", jnp.ones(data.num_genotype,dtype=float), 
                               constraint=dist.constraints.positive)

    # --- Batching ---
    with pyro.plate("shared_genotype_plate", size=data.batch_size,dim=-1):
        with pyro.handlers.scale(scale=data.scale_vector):
        
            batch_locs = offset_locs[...,data.batch_idx]
            batch_scales = offset_scales[...,data.batch_idx]

            dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(batch_locs, batch_scales))

    # Guard against full-sized array substitution during initialization or re-runs 
    # with full-sized initial values
    if dk_geno_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        dk_geno_offset = dk_geno_offset[..., data.batch_idx]

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
    

def get_guesses(name, data):
    """
    Get initial guess values derived empirically from observed ln_cfu data.

    For each genotype, the pleiotropic growth effect (dk_geno) is estimated
    as the difference between that genotype's growth rate and the wildtype
    growth rate, averaged across replicates and pre-selection conditions.
    Growth rates are computed via ordinary-least-squares regression of
    ``data.ln_cfu`` against ``data.t_sel`` (selection-phase time) over all
    valid observations for each (replicate, condition_pre, genotype) group.

    The per-genotype dk_geno estimate is then inverted through the model's
    parameterisation ``dk_geno = shift - exp(hyper_loc + offset * hyper_scale)``
    to obtain the non-centred offset guess.

    Falls back to the hard-coded neutral offset (corresponding to dk_geno = 0)
    for any genotype that lacks sufficient valid data (fewer than 2 distinct
    time points), and for the whole function when ``data.ln_cfu`` is not a
    7-D tensor (e.g. when called with a mock object during unrelated tests).

    Parameters
    ----------
    name : str
        The prefix for the parameter names.
    data : GrowthData
        Pytree containing data and metadata.  Uses ``data.ln_cfu``,
        ``data.t_sel``, ``data.good_mask``, ``data.wt_indexes``, and
        ``data.num_genotype``.

    Returns
    -------
    dict
        A dictionary mapping parameter names to their initial guess values.

    Notes
    -----
    ``data.ln_cfu`` and ``data.t_sel`` are expected to share the shape
    ``(num_replicate, num_time, num_condition_pre, num_condition_sel,
    num_titrant_name, num_titrant_conc, num_genotype)``.
    The OLS slope is computed by reducing over axes 1, 3, 4, 5 (time,
    condition_sel, titrant_name, titrant_conc), yielding one slope per
    (replicate, condition_pre, genotype).
    """

    _DEFAULT_HYPER_LOC   = -3.5
    _DEFAULT_HYPER_SCALE = 0.5
    _DEFAULT_SHIFT       = 0.02

    # Offset that maps to dk_geno = 0 under the default parameterisation
    _NEUTRAL_OFFSET = (np.log(_DEFAULT_SHIFT) - _DEFAULT_HYPER_LOC) / _DEFAULT_HYPER_SCALE

    num_geno = data.num_genotype

    ln_cfu    = np.array(data.ln_cfu)
    t_sel     = np.array(data.t_sel)
    good_mask = np.array(data.good_mask)
    wt_indexes = np.array(data.wt_indexes)

    # Guard: fall back when tensors are not the expected 7-D shape (e.g. mocked data)
    if ln_cfu.ndim != 7 or t_sel.ndim != 7:
        guesses = {}
        guesses[f"{name}_hyper_loc"]   = _DEFAULT_HYPER_LOC
        guesses[f"{name}_hyper_scale"] = _DEFAULT_HYPER_SCALE
        guesses[f"{name}_shift"]       = _DEFAULT_SHIFT
        guesses[f"{name}_offset"]      = _NEUTRAL_OFFSET * jnp.ones(num_geno, dtype=float)
        return guesses

    # Mask invalid observations
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)
    t_sel_valid  = np.where(good_mask, t_sel,  np.nan)

    # Rearrange axes so the dimensions to reduce are last:
    # (rep=0, time=1, cond_pre=2, cond_sel=3, tname=4, tconc=5, geno=6)
    # → (rep, cond_pre, geno, time, cond_sel, tname, tconc)
    t_moved = np.moveaxis(t_sel_valid,  [1, 3, 4, 5], [-4, -3, -2, -1])
    y_moved = np.moveaxis(ln_cfu_valid, [1, 3, 4, 5], [-4, -3, -2, -1])

    num_rep, num_cond_pre, num_geno_tensor = t_moved.shape[:3]
    n_obs = int(np.prod(t_moved.shape[3:]))

    t_flat = t_moved.reshape(num_rep, num_cond_pre, num_geno_tensor, n_obs)
    y_flat = y_moved.reshape(num_rep, num_cond_pre, num_geno_tensor, n_obs)

    # Vectorised OLS: slope = Σ(tᵢ-t̄)(yᵢ-ȳ) / Σ(tᵢ-t̄)²
    valid   = ~np.isnan(t_flat)                            # NaN pattern is same for t and y
    n_valid = np.sum(valid, axis=-1)                       # (rep, cond_pre, geno)
    denom   = np.maximum(n_valid, 1)

    t_sum   = np.sum(np.where(valid, t_flat, 0.0), axis=-1)
    y_sum   = np.sum(np.where(valid, y_flat, 0.0), axis=-1)
    t_mean  = (t_sum / denom)[..., np.newaxis]
    y_mean  = (y_sum / denom)[..., np.newaxis]

    t_diff  = np.where(valid, t_flat - t_mean, 0.0)
    y_diff  = np.where(valid, y_flat - y_mean, 0.0)

    cov_ty  = np.sum(t_diff * y_diff, axis=-1)            # (rep, cond_pre, geno)
    var_t   = np.sum(t_diff * t_diff, axis=-1)

    with np.errstate(invalid="ignore", divide="ignore"):
        slopes  = np.where((n_valid >= 2) & (var_t > 1e-20),
                           cov_ty / var_t, np.nan)        # (rep, cond_pre, geno)

    # WT reference slope per (rep, cond_pre)
    wt_geno_mask = np.zeros(num_geno_tensor, dtype=bool)
    if len(wt_indexes) > 0:
        wt_geno_mask[wt_indexes] = True

    # WT reference slope per (rep, cond_pre): NaN-safe mean avoids RuntimeWarnings
    # from nanmean on all-NaN slices by using nansum / valid-count instead.
    if wt_geno_mask.any():
        wt_slopes   = slopes[:, :, wt_geno_mask]                    # (rep, cond_pre, n_wt)
        wt_count    = np.sum(~np.isnan(wt_slopes), axis=2)
        g_wt        = np.where(wt_count > 0,
                               np.nansum(wt_slopes, axis=2) / np.maximum(wt_count, 1),
                               np.nan)                               # (rep, cond_pre)
    else:
        g_wt = np.zeros((num_rep, num_cond_pre))

    # dk_geno = genotype slope − WT slope, averaged over (rep, cond_pre)
    dk_empirical  = slopes - g_wt[:, :, np.newaxis]                 # (rep, cond_pre, geno)
    dk_count      = np.sum(~np.isnan(dk_empirical), axis=(0, 1))    # (num_geno,)
    dk_per_geno   = np.where(dk_count > 0,
                             np.nansum(dk_empirical, axis=(0, 1)) / np.maximum(dk_count, 1),
                             np.nan)                                 # (num_geno,)
    dk_per_geno      = np.where(np.isnan(dk_per_geno), 0.0, dk_per_geno)

    # Invert: offset = (log(shift − dk_geno) − hyper_loc) / hyper_scale
    lognormal_arg    = np.clip(_DEFAULT_SHIFT - dk_per_geno, 1e-6, None)
    offset_per_geno  = (np.log(lognormal_arg) - _DEFAULT_HYPER_LOC) / _DEFAULT_HYPER_SCALE

    guesses = {}
    guesses[f"{name}_hyper_loc"]   = _DEFAULT_HYPER_LOC
    guesses[f"{name}_hyper_scale"] = _DEFAULT_HYPER_SCALE
    guesses[f"{name}_shift"]       = _DEFAULT_SHIFT
    guesses[f"{name}_offset"]      = jnp.array(offset_per_geno, dtype=float)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
    