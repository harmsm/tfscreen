import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field
from typing import Tuple, Mapping

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)
from tfscreen.analysis.hierarchical.growth_model.components._pinning import (
    _hyper,
    _pinned_value,
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
    growth_k_hyper_loc_loc, growth_k_hyper_loc_scale : float
        Mean and standard deviation of the Normal prior on the hyper-location
        of the per-condition baseline growth rate ``k``.
    growth_k_hyper_scale : float
        Scale of the HalfNormal prior on the hyper-scale of ``k``.
    growth_m_hyper_loc_loc, growth_m_hyper_loc_scale : float
        Mean and standard deviation of the Normal prior on the hyper-location
        of the per-condition occupancy slope ``m``.
    growth_m_hyper_scale : float
        Scale of the HalfNormal prior on the hyper-scale of ``m``.
    pinned : dict[str, float], optional
        Map from hyper-site *suffix* to the constant value at which that
        site should be pinned.  Recognised suffixes are ``"k_hyper_loc"``,
        ``"k_hyper_scale"``, ``"m_hyper_loc"``, ``"m_hyper_scale"``.  Any
        suffix listed here bypasses both the model's ``pyro.sample`` and
        the guide's variational parameters; the model registers a
        ``pyro.deterministic`` at the pinned value.  Stored as a static
        (non-pytree) field so branching happens at trace time.
    """

    growth_k_hyper_loc_loc: float
    growth_k_hyper_loc_scale: float
    growth_k_hyper_scale: float

    growth_m_hyper_loc_loc: float
    growth_m_hyper_loc_scale: float
    growth_m_hyper_scale: float

    pinned: Mapping[str, float] = field(pytree_node=False, default_factory=dict)


# Suffixes recognised by the pinning machinery.  Anything in ``priors.pinned``
# whose key is not in this set is silently ignored at trace time.
_PINNABLE_SUFFIXES = ("k_hyper_loc", "k_hyper_scale",
                      "m_hyper_loc", "m_hyper_scale")


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> LinearParams:
    """
    Growth parameters k_xx and m_xx versus condition, where xx are things like
    pheS+4CP, kanR-kan, etc. These go into the model as k + m*theta. Assigns
    each condition/replicate a normal prior. Returns full k_pre, m_pre, k_sel
    and m_sel tensors.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "theta").
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_condition_rep``
        - ``data.num_replicate``
        - ``data.map_condition_pre``
        - ``data.map_condition_sel``
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model, including:
        - ``priors.growth_k_hyper_loc_loc``
        - ``priors.growth_k_hyper_loc_scale``
        - ``priors.growth_k_hyper_scale``
        - ``priors.growth_m_hyper_loc_loc``
        - ``priors.growth_m_hyper_loc_scale``
        - ``priors.growth_m_hyper_scale``
        - ``priors.pinned`` (optional pinning dict; see ``ModelPriors``)

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """

    pinned = priors.pinned

    growth_k_hyper_loc = _hyper(
        name, "k_hyper_loc",
        dist.Normal(priors.growth_k_hyper_loc_loc,
                    priors.growth_k_hyper_loc_scale),
        pinned,
    )
    growth_k_hyper_scale = _hyper(
        name, "k_hyper_scale",
        dist.HalfNormal(priors.growth_k_hyper_scale),
        pinned,
    )

    growth_m_hyper_loc = _hyper(
        name, "m_hyper_loc",
        dist.Normal(priors.growth_m_hyper_loc_loc,
                    priors.growth_m_hyper_loc_scale),
        pinned,
    )
    growth_m_hyper_scale = _hyper(
        name, "m_hyper_scale",
        dist.HalfNormal(priors.growth_m_hyper_scale),
        pinned,
    )

    # Loop over conditions and replicates
    with pyro.plate(f"{name}_condition_parameters",data.num_condition_rep):
        growth_k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(0.0, 1.0))
        growth_m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(0.0, 1.0))

    growth_k_per_condition = growth_k_hyper_loc + growth_k_offset * growth_k_hyper_scale
    growth_m_per_condition = growth_m_hyper_loc + growth_m_offset * growth_m_hyper_scale

    # Register dists
    pyro.deterministic(f"{name}_k", growth_k_per_condition)
    pyro.deterministic(f"{name}_m", growth_m_per_condition)

    # Expand to full-sized tensors
    k_pre = growth_k_per_condition[data.map_condition_pre]
    m_pre = growth_m_per_condition[data.map_condition_pre]
    k_sel = growth_k_per_condition[data.map_condition_sel]
    m_sel = growth_m_per_condition[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> LinearParams:
    """
    Guide corresponding to the pooled growth model.

    This guide defines the variational family for the growth parameters `k`
    and `m`, assuming a single, shared pool across all conditions. It uses:
    - Normal distributions for hyper-location means.
    - LogNormal distributions for hyper-scales.
    - Normal distributions for per-condition offsets.

    When ``priors.pinned`` contains any of the recognised hyper-site
    suffixes, the corresponding variational parameters and sample sites are
    omitted entirely; the value is taken straight from the pinned constant.
    """

    pinned = priors.pinned

    # k_hyper_loc -----------------------------------------------------------
    pinned_k_hyper_loc = _pinned_value("k_hyper_loc", pinned)
    if pinned_k_hyper_loc is not None:
        growth_k_hyper_loc = pinned_k_hyper_loc
    else:
        k_loc_loc = pyro.param(f"{name}_k_hyper_loc_loc", jnp.array(priors.growth_k_hyper_loc_loc))
        k_loc_scale = pyro.param(f"{name}_k_hyper_loc_scale", jnp.array(priors.growth_k_hyper_loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        growth_k_hyper_loc = pyro.sample(
            f"{name}_k_hyper_loc",
            dist.Normal(k_loc_loc, k_loc_scale)
        )

    # k_hyper_scale ---------------------------------------------------------
    pinned_k_hyper_scale = _pinned_value("k_hyper_scale", pinned)
    if pinned_k_hyper_scale is not None:
        growth_k_hyper_scale = pinned_k_hyper_scale
    else:
        k_scale_loc = pyro.param(f"{name}_k_hyper_scale_loc", jnp.array(-1.0))
        k_scale_scale = pyro.param(f"{name}_k_hyper_scale_scale",jnp.array(0.1),
                                   constraint=dist.constraints.greater_than(1e-4))
        growth_k_hyper_scale = pyro.sample(
            f"{name}_k_hyper_scale",
            dist.LogNormal(k_scale_loc, k_scale_scale)
        )

    # m_hyper_loc -----------------------------------------------------------
    pinned_m_hyper_loc = _pinned_value("m_hyper_loc", pinned)
    if pinned_m_hyper_loc is not None:
        growth_m_hyper_loc = pinned_m_hyper_loc
    else:
        m_loc_loc = pyro.param(f"{name}_m_hyper_loc_loc", jnp.array(priors.growth_m_hyper_loc_loc))
        m_loc_scale = pyro.param(f"{name}_m_hyper_loc_scale", jnp.array(priors.growth_m_hyper_loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        growth_m_hyper_loc = pyro.sample(
            f"{name}_m_hyper_loc",
            dist.Normal(m_loc_loc, m_loc_scale)
        )

    # m_hyper_scale ---------------------------------------------------------
    pinned_m_hyper_scale = _pinned_value("m_hyper_scale", pinned)
    if pinned_m_hyper_scale is not None:
        growth_m_hyper_scale = pinned_m_hyper_scale
    else:
        m_scale_loc = pyro.param(f"{name}_m_hyper_scale_loc", jnp.array(-1.0))
        m_scale_scale = pyro.param(f"{name}_m_hyper_scale_scale",jnp.array(0.1),
                                   constraint=dist.constraints.greater_than(1e-4))
        growth_m_hyper_scale = pyro.sample(
            f"{name}_m_hyper_scale",
            dist.LogNormal(m_scale_loc, m_scale_scale)
        )

    k_offset_locs = pyro.param(f"{name}_k_offset_locs",
                               jnp.zeros(data.num_condition_rep,dtype=float))
    k_offset_scales = pyro.param(f"{name}_k_offset_scales",
                                 jnp.ones(data.num_condition_rep,dtype=float),
                                 constraint=dist.constraints.positive)


    m_offset_locs = pyro.param(f"{name}_m_offset_locs",
                               jnp.zeros(data.num_condition_rep,dtype=float))
    m_offset_scales = pyro.param(f"{name}_m_offset_scales",
                                 jnp.ones(data.num_condition_rep,dtype=float),
                                 constraint=dist.constraints.positive)


    # Loop over conditions and replicates
    with pyro.plate(f"{name}_condition_parameters",data.num_condition_rep) as idx:

        k_batch_locs = k_offset_locs[...,idx]
        k_batch_scales = k_offset_scales[...,idx]
        m_batch_locs = m_offset_locs[...,idx]
        m_batch_scales = m_offset_scales[...,idx]

        growth_k_offset = pyro.sample(f"{name}_k_offset", dist.Normal(k_batch_locs,k_batch_scales))
        growth_m_offset = pyro.sample(f"{name}_m_offset", dist.Normal(m_batch_locs,m_batch_scales))

    growth_k_per_condition = growth_k_hyper_loc + growth_k_offset * growth_k_hyper_scale
    growth_m_per_condition = growth_m_hyper_loc + growth_m_offset * growth_m_hyper_scale

    # Expand to full-sized tensors
    k_pre = growth_k_per_condition[data.map_condition_pre]
    m_pre = growth_m_per_condition[data.map_condition_pre]
    k_sel = growth_k_per_condition[data.map_condition_sel]
    m_sel = growth_m_per_condition[data.map_condition_sel]

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
    parameters["growth_k_hyper_loc_loc"] = 0.025
    parameters["growth_k_hyper_loc_scale"] = 0.1
    parameters["growth_k_hyper_scale"] = 0.1

    parameters["growth_m_hyper_loc_loc"] = 0.0
    parameters["growth_m_hyper_loc_scale"] = 0.05
    parameters["growth_m_hyper_scale"] = 0.1

    return parameters

def get_guesses(name, data):
    """
    Get guesses for the model parameters.

    Estimates the baseline growth rate k per condition_rep from the OLS slope
    of ln_cfu versus t_sel, averaged over replicates, titrants, and genotypes,
    then grouped by condition_rep via map_condition_sel.  m offsets are left at
    zero because m cannot be estimated without knowing theta.

    Falls back to the hard-coded defaults when ``data.ln_cfu`` or ``data.t_sel``
    are not 7-D tensors (e.g. when called with a mock object during unrelated
    tests).
    """

    _DEFAULT_HYPER_SCALE = 0.1
    _DEFAULT_K = 0.025

    num_cond_rep = data.num_condition_rep
    shape = num_cond_rep

    ln_cfu = np.array(data.ln_cfu)
    t_sel  = np.array(data.t_sel)
    good_mask = np.array(data.good_mask)
    map_condition_sel = np.array(data.map_condition_sel)

    if ln_cfu.ndim != 7 or t_sel.ndim != 7:
        guesses = {}
        guesses[f"{name}_k_hyper_loc"]   = _DEFAULT_K
        guesses[f"{name}_k_hyper_scale"] = _DEFAULT_HYPER_SCALE
        guesses[f"{name}_m_hyper_loc"]   = 0.0
        guesses[f"{name}_m_hyper_scale"] = 0.01
        guesses[f"{name}_k_offset"] = jnp.zeros(shape, dtype=float)
        guesses[f"{name}_m_offset"] = jnp.zeros(shape, dtype=float)
        return guesses

    # Mask invalid observations
    ln_cfu_valid = np.where(good_mask, ln_cfu, np.nan)
    t_sel_valid  = np.where(good_mask, t_sel,  np.nan)

    # Move time axis (1) to last position for vectorised OLS
    # (rep, time, cond_pre, cond_sel, tname, tconc, geno)
    # → (rep, cond_pre, cond_sel, tname, tconc, geno, time)
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
    # slopes shape: (num_rep, num_cond_pre, num_cond_sel, num_tname, num_tconc, num_geno)

    # map_condition_sel may be 7D (real GrowthData) or 1D (unit-test mock).
    # Drop the time axis from the 7D tensor to match the slopes shape.
    if map_condition_sel.ndim == 7:
        # (rep, time, cond_pre, cond_sel, tname, tconc, geno) → drop time (axis 1)
        map_cr = map_condition_sel[:, 0, ...]
    else:
        # 1D (num_cond_sel,): cond_sel is axis 2 of slopes; other axes will be broadcast
        map_cr = map_condition_sel

    # Group slopes by condition_rep and average
    k_per_cond_rep = np.full(num_cond_rep, np.nan)
    for cr in range(num_cond_rep):
        if map_cr.ndim == slopes.ndim:
            cr_slopes = slopes[map_cr == cr]
        else:
            # 1D map_cr: boolean index along cond_sel axis (axis 2)
            cr_slopes = slopes[:, :, map_cr == cr, ...]
        n_valid_cr = np.sum(~np.isnan(cr_slopes))
        if n_valid_cr > 0:
            k_per_cond_rep[cr] = np.nansum(cr_slopes) / n_valid_cr

    # Fill any NaN cond_reps with the global median (or hard-coded fallback)
    valid_k  = k_per_cond_rep[~np.isnan(k_per_cond_rep)]
    global_k = float(np.nanmedian(valid_k)) if len(valid_k) > 0 else _DEFAULT_K
    k_per_cond_rep = np.where(np.isnan(k_per_cond_rep), global_k, k_per_cond_rep)

    k_hyper_loc = float(np.nanmedian(k_per_cond_rep))
    k_offsets   = (k_per_cond_rep - k_hyper_loc) / _DEFAULT_HYPER_SCALE

    guesses = {}
    guesses[f"{name}_k_hyper_loc"]   = k_hyper_loc
    guesses[f"{name}_k_hyper_scale"] = _DEFAULT_HYPER_SCALE
    guesses[f"{name}_m_hyper_loc"]   = 0.0
    guesses[f"{name}_m_hyper_scale"] = 0.01
    guesses[f"{name}_k_offset"] = jnp.array(k_offsets, dtype=float)
    guesses[f"{name}_m_offset"] = jnp.zeros(num_cond_rep, dtype=float)
    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
