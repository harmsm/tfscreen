import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple

from tfscreen.tfmodel.data_class import (
    GrowthData
)


@dataclass(frozen=True)
class PowerParams:
    """
    Holds power-law growth parameters for pre-selection and selection phases.
    Growth = k + m * (theta**n)
    """
    k_pre: jnp.ndarray
    m_pre: jnp.ndarray
    n_pre: jnp.ndarray
    k_sel: jnp.ndarray
    m_sel: jnp.ndarray
    n_sel: jnp.ndarray

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.

    Attributes
    ----------
    k_loc, k_scale : float or array
        Normal prior parameters for per-condition baseline growth rate k.
        Scalar (broadcast to every condition) or a length-``num_condition_rep``
        array of per-condition values (written by the pre-fit calibration).
    m_loc, m_scale : float or array
        Normal prior parameters for per-condition occupancy slope m.
        Scalar or per-condition array.
    n_loc, n_scale : float or array
        Normal prior parameters for log(n), the power-law exponent.
        The actual exponent n = exp(sample), so n > 0 always.
        Scalar or per-condition array.
    """
    k_loc: float
    k_scale: float
    m_loc: float
    m_scale: float
    n_loc: float
    n_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> PowerParams:
    """
    Growth parameters k, m, n per condition with simple Normal priors.

    n is sampled in log-space (Normal prior) and exponentiated to enforce
    positivity: n = exp(Normal(n_loc, n_scale)).

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
    params : PowerParams
        A dataclass containing k, m, and n for pre and sel.
    """
    num_cr = data.num_condition_rep

    # Broadcast scalar-or-array priors to per-condition arrays so each
    # parameter can be pinned condition-by-condition (see prefit calibration).
    # A scalar prior broadcasts to every condition (previous behaviour).
    k_loc_arr = jnp.broadcast_to(jnp.asarray(priors.k_loc, dtype=float), (num_cr,))
    k_scale_arr = jnp.broadcast_to(jnp.asarray(priors.k_scale, dtype=float), (num_cr,))
    m_loc_arr = jnp.broadcast_to(jnp.asarray(priors.m_loc, dtype=float), (num_cr,))
    m_scale_arr = jnp.broadcast_to(jnp.asarray(priors.m_scale, dtype=float), (num_cr,))
    n_loc_arr = jnp.broadcast_to(jnp.asarray(priors.n_loc, dtype=float), (num_cr,))
    n_scale_arr = jnp.broadcast_to(jnp.asarray(priors.n_scale, dtype=float), (num_cr,))

    with pyro.plate(f"{name}_k_condition_parameters", num_cr) as idx:
        k_per_condition = pyro.sample(f"{name}_k",
                                      dist.Normal(k_loc_arr[idx], k_scale_arr[idx]))

    with pyro.plate(f"{name}_m_condition_parameters", num_cr) as idx:
        m_per_condition = pyro.sample(f"{name}_m",
                                      dist.Normal(m_loc_arr[idx], m_scale_arr[idx]))

    with pyro.plate(f"{name}_n_condition_parameters", num_cr) as idx:
        ln_n = pyro.sample(f"{name}_n",
                           dist.Normal(n_loc_arr[idx], n_scale_arr[idx]))
    n_per_condition = jnp.exp(ln_n)

    k_pre = k_per_condition[data.map_condition_pre]
    m_pre = m_per_condition[data.map_condition_pre]
    n_pre = n_per_condition[data.map_condition_pre]

    k_sel = k_per_condition[data.map_condition_sel]
    m_sel = m_per_condition[data.map_condition_sel]
    n_sel = n_per_condition[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> PowerParams:
    """
    Guide for the power growth model with simple Normal priors.
    """
    num_cr = data.num_condition_rep
    k_locs = pyro.param(f"{name}_k_locs",
                        jnp.broadcast_to(jnp.asarray(priors.k_loc, dtype=float), (num_cr,)))
    k_scales = pyro.param(f"{name}_k_scales",
                          jnp.broadcast_to(jnp.asarray(priors.k_scale, dtype=float), (num_cr,)),
                          constraint=dist.constraints.positive)
    m_locs = pyro.param(f"{name}_m_locs",
                        jnp.broadcast_to(jnp.asarray(priors.m_loc, dtype=float), (num_cr,)))
    m_scales = pyro.param(f"{name}_m_scales",
                          jnp.broadcast_to(jnp.asarray(priors.m_scale, dtype=float), (num_cr,)),
                          constraint=dist.constraints.positive)
    n_locs = pyro.param(f"{name}_n_locs",
                        jnp.broadcast_to(jnp.asarray(priors.n_loc, dtype=float), (num_cr,)))
    n_scales = pyro.param(f"{name}_n_scales",
                          jnp.broadcast_to(jnp.asarray(priors.n_scale, dtype=float), (num_cr,)),
                          constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_k_condition_parameters", data.num_condition_rep) as idx:
        k_per_condition = pyro.sample(f"{name}_k",
                                      dist.Normal(k_locs[..., idx], k_scales[..., idx]))

    with pyro.plate(f"{name}_m_condition_parameters", data.num_condition_rep) as idx:
        m_per_condition = pyro.sample(f"{name}_m",
                                      dist.Normal(m_locs[..., idx], m_scales[..., idx]))

    with pyro.plate(f"{name}_n_condition_parameters", data.num_condition_rep) as idx:
        ln_n = pyro.sample(f"{name}_n",
                           dist.Normal(n_locs[..., idx], n_scales[..., idx]))
    n_per_condition = jnp.exp(ln_n)

    k_pre = k_per_condition[data.map_condition_pre]
    m_pre = m_per_condition[data.map_condition_pre]
    n_pre = n_per_condition[data.map_condition_pre]

    k_sel = k_per_condition[data.map_condition_sel]
    m_sel = m_per_condition[data.map_condition_sel]
    n_sel = n_per_condition[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)


def calculate_growth(params: PowerParams,
                     dk_geno: jnp.ndarray,
                     activity: jnp.ndarray,
                     theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the growth rates for pre-selection and selection phases.
    Growth = k + dk_geno + activity * m * (theta**n)
    """
    g_pre = params.k_pre + dk_geno + activity * params.m_pre * (theta**params.n_pre)
    g_sel = params.k_sel + dk_geno + activity * params.m_sel * (theta**params.n_sel)

    return g_pre, g_sel


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """
    parameters = {}
    parameters["k_loc"] = 0.020
    parameters["k_scale"] = 0.1
    parameters["m_loc"] = 0.0
    parameters["m_scale"] = 0.01
    parameters["n_loc"] = 0.0   # ln(1.0); n=1 by default
    parameters["n_scale"] = 0.5

    return parameters


def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 0.01

    guesses = {}
    guesses[f"{name}_k_locs"] = jnp.full(num_cond_rep, 0.020, dtype=float)
    guesses[f"{name}_k_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    guesses[f"{name}_m_locs"] = jnp.zeros(num_cond_rep, dtype=float)
    guesses[f"{name}_m_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    guesses[f"{name}_n_locs"] = jnp.zeros(num_cond_rep, dtype=float)
    guesses[f"{name}_n_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)

    return guesses


def get_priors():
    return ModelPriors(**get_hyperparameters())


def get_scale_bounds():
    """
    Per-condition prior-scale bounds for the pre-fit calibration.

    See ``linear.get_scale_bounds`` for the contract.  ``k`` is the
    theta-independent baseline (tight floor); ``n`` is a log-space exponent,
    so its bounds are looser than the rate parameters.

    Returns
    -------
    dict
        ``{suffix: {"floor": float, "ceiling": float, "scale_field": str}}``.
    """
    return {
        "k": {"floor": 0.002, "ceiling": 0.1,  "scale_field": "k_scale"},
        "m": {"floor": 0.001, "ceiling": 0.01, "scale_field": "m_scale"},
        "n": {"floor": 0.05,  "ceiling": 0.5,  "scale_field": "n_scale"},
    }


def get_extract_specs(ctx):
    if "condition_rep" not in ctx.growth_tm.map_groups:
        return []
    cond_rep_cols = (["condition_rep"] if ctx.growth_shares_replicates
                     else ["replicate", "condition_rep"])
    return [dict(
        input_df=ctx.growth_tm.map_groups["condition_rep"],
        params_to_get=["growth_k", "growth_m", "growth_n"],
        map_column="map_condition_rep",
        get_columns=cond_rep_cols,
        in_run_prefix="condition_",
    )]
