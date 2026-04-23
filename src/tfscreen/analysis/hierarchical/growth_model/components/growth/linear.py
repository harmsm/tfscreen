import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    JAX Pytree holding data needed to specify model priors.

    Attributes
    ----------
    k_loc : float
        Mean of the Normal prior on the per-condition baseline growth rate k.
    k_scale : float
        Standard deviation of the Normal prior on k.
    m_loc : float
        Mean of the Normal prior on the per-condition occupancy slope m.
    m_scale : float
        Standard deviation of the Normal prior on m.
    """
    k_loc: float
    k_scale: float
    m_loc: float
    m_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> LinearParams:
    """
    Growth parameters k and m per condition with simple Normal priors.

    Each condition_rep gets its own k and m drawn independently from
    Normal(k_loc, k_scale) and Normal(m_loc, m_scale).

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree containing the prior parameters.

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep):
        growth_k = pyro.sample(f"{name}_k", dist.Normal(priors.k_loc, priors.k_scale))
        growth_m = pyro.sample(f"{name}_m", dist.Normal(priors.m_loc, priors.m_scale))

    k_pre = growth_k[data.map_condition_pre]
    m_pre = growth_m[data.map_condition_pre]
    k_sel = growth_k[data.map_condition_sel]
    m_sel = growth_m[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> LinearParams:
    """
    Guide for the linear growth model with simple Normal priors.

    Maintains per-condition variational parameters for k and m.
    """
    k_locs = pyro.param(f"{name}_k_locs",
                        jnp.full(data.num_condition_rep, priors.k_loc, dtype=float))
    k_scales = pyro.param(f"{name}_k_scales",
                          jnp.full(data.num_condition_rep, priors.k_scale, dtype=float),
                          constraint=dist.constraints.positive)
    m_locs = pyro.param(f"{name}_m_locs",
                        jnp.full(data.num_condition_rep, priors.m_loc, dtype=float))
    m_scales = pyro.param(f"{name}_m_scales",
                          jnp.full(data.num_condition_rep, priors.m_scale, dtype=float),
                          constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        growth_k = pyro.sample(f"{name}_k",
                               dist.Normal(k_locs[..., idx], k_scales[..., idx]))
        growth_m = pyro.sample(f"{name}_m",
                               dist.Normal(m_locs[..., idx], m_scales[..., idx]))

    k_pre = growth_k[data.map_condition_pre]
    m_pre = growth_m[data.map_condition_pre]
    k_sel = growth_k[data.map_condition_sel]
    m_sel = growth_m[data.map_condition_sel]

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


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """
    parameters = {}
    parameters["k_loc"] = 0.025
    parameters["k_scale"] = 0.1
    parameters["m_loc"] = 0.0
    parameters["m_scale"] = 0.05

    return parameters


def get_guesses(name, data):
    """
    Get guesses for the model parameters.

    Estimates the per-condition k from OLS slopes of ln_cfu versus t_sel,
    averaged over replicates, titrants, and genotypes.  m_locs are set to
    zero (cannot be estimated without knowing theta).

    Falls back to hard-coded defaults when ``data.ln_cfu`` or ``data.t_sel``
    are not 7-D tensors.
    """
    _DEFAULT_K = 0.025
    _DEFAULT_SCALE = 0.01

    num_cond_rep = data.num_condition_rep

    ln_cfu = np.array(data.ln_cfu)
    t_sel  = np.array(data.t_sel)
    good_mask = np.array(data.good_mask)
    map_condition_sel = np.array(data.map_condition_sel)

    if ln_cfu.ndim != 7 or t_sel.ndim != 7:
        guesses = {}
        guesses[f"{name}_k_locs"] = jnp.full(num_cond_rep, _DEFAULT_K, dtype=float)
        guesses[f"{name}_k_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
        guesses[f"{name}_m_locs"] = jnp.zeros(num_cond_rep, dtype=float)
        guesses[f"{name}_m_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
        return guesses

    # Mask invalid observations
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)
    t_sel_valid  = np.where(good_mask, t_sel,  np.nan)

    # Move time axis (1) to last position for vectorised OLS
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
        slopes = np.where((n_valid >= 2) & (var_t > 1e-20),
                          cov_ty / var_t, np.nan)

    # map_condition_sel may be 7D or 1D
    if map_condition_sel.ndim == 7:
        map_cr = map_condition_sel[:, 0, ...]
    else:
        map_cr = map_condition_sel

    # Group slopes by condition_rep and average
    k_per_cond_rep = np.full(num_cond_rep, np.nan)
    for cr in range(num_cond_rep):
        if map_cr.ndim == slopes.ndim:
            cr_slopes = slopes[map_cr == cr]
        else:
            cr_slopes = slopes[:, :, map_cr == cr, ...]
        n_valid_cr = np.sum(~np.isnan(cr_slopes))
        if n_valid_cr > 0:
            k_per_cond_rep[cr] = np.nansum(cr_slopes) / n_valid_cr

    # Fill any NaN cond_reps with the global median
    valid_k  = k_per_cond_rep[~np.isnan(k_per_cond_rep)]
    global_k = float(np.nanmedian(valid_k)) if len(valid_k) > 0 else _DEFAULT_K
    k_per_cond_rep = np.where(np.isnan(k_per_cond_rep), global_k, k_per_cond_rep)

    guesses = {}
    guesses[f"{name}_k_locs"] = jnp.array(k_per_cond_rep, dtype=float)
    guesses[f"{name}_k_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    guesses[f"{name}_m_locs"] = jnp.zeros(num_cond_rep, dtype=float)
    guesses[f"{name}_m_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    return guesses


def get_priors():
    return ModelPriors(**get_hyperparameters())
