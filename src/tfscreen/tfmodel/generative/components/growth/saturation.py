import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple

from tfscreen.tfmodel.data_class import (
    GrowthData
)


@dataclass(frozen=True)
class SaturationParams:
    """
    Holds saturation growth parameters for pre-selection and selection phases.
    Growth = min + (max - min) * (theta / (1 + theta))
    """
    min_pre: jnp.ndarray
    max_pre: jnp.ndarray
    min_sel: jnp.ndarray
    max_sel: jnp.ndarray

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.

    Attributes
    ----------
    min_loc, min_scale : float or array
        Normal prior parameters for per-condition minimum growth rate.
        Scalar (broadcast to every condition) or a length-``num_condition_rep``
        array of per-condition values (written by the pre-fit calibration).
    max_loc, max_scale : float or array
        Normal prior parameters for per-condition maximum growth rate.
        Scalar or per-condition array.
    """
    min_loc: float
    min_scale: float
    max_loc: float
    max_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> SaturationParams:
    """
    Growth parameters min and max per condition with simple Normal priors.

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
    params : SaturationParams
        A dataclass containing min and max for pre and sel.
    """
    num_cr = data.num_condition_rep

    # Broadcast scalar-or-array priors to per-condition arrays so min, max can
    # be pinned condition-by-condition (see prefit calibration).  A scalar
    # prior broadcasts to every condition (previous behaviour).
    min_loc_arr = jnp.broadcast_to(jnp.asarray(priors.min_loc, dtype=float), (num_cr,))
    min_scale_arr = jnp.broadcast_to(jnp.asarray(priors.min_scale, dtype=float), (num_cr,))
    max_loc_arr = jnp.broadcast_to(jnp.asarray(priors.max_loc, dtype=float), (num_cr,))
    max_scale_arr = jnp.broadcast_to(jnp.asarray(priors.max_scale, dtype=float), (num_cr,))

    with pyro.plate(f"{name}_min_condition_parameters", num_cr) as idx:
        min_per_condition = pyro.sample(f"{name}_min",
                                        dist.Normal(min_loc_arr[idx], min_scale_arr[idx]))

    with pyro.plate(f"{name}_max_condition_parameters", num_cr) as idx:
        max_per_condition = pyro.sample(f"{name}_max",
                                        dist.Normal(max_loc_arr[idx], max_scale_arr[idx]))

    min_pre = min_per_condition[data.map_condition_pre]
    max_pre = max_per_condition[data.map_condition_pre]
    min_sel = min_per_condition[data.map_condition_sel]
    max_sel = max_per_condition[data.map_condition_sel]

    return SaturationParams(min_pre=min_pre, max_pre=max_pre,
                            min_sel=min_sel, max_sel=max_sel)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> SaturationParams:
    """
    Guide for the saturation growth model with simple Normal priors.
    """
    num_cr = data.num_condition_rep
    min_locs = pyro.param(f"{name}_min_locs",
                          jnp.broadcast_to(jnp.asarray(priors.min_loc, dtype=float), (num_cr,)))
    min_scales = pyro.param(f"{name}_min_scales",
                            jnp.broadcast_to(jnp.asarray(priors.min_scale, dtype=float), (num_cr,)),
                            constraint=dist.constraints.positive)
    max_locs = pyro.param(f"{name}_max_locs",
                          jnp.broadcast_to(jnp.asarray(priors.max_loc, dtype=float), (num_cr,)))
    max_scales = pyro.param(f"{name}_max_scales",
                            jnp.broadcast_to(jnp.asarray(priors.max_scale, dtype=float), (num_cr,)),
                            constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_min_condition_parameters", data.num_condition_rep) as idx:
        min_per_condition = pyro.sample(f"{name}_min",
                                        dist.Normal(min_locs[..., idx], min_scales[..., idx]))

    with pyro.plate(f"{name}_max_condition_parameters", data.num_condition_rep) as idx:
        max_per_condition = pyro.sample(f"{name}_max",
                                        dist.Normal(max_locs[..., idx], max_scales[..., idx]))

    min_pre = min_per_condition[data.map_condition_pre]
    max_pre = max_per_condition[data.map_condition_pre]
    min_sel = min_per_condition[data.map_condition_sel]
    max_sel = max_per_condition[data.map_condition_sel]

    return SaturationParams(min_pre=min_pre, max_pre=max_pre,
                            min_sel=min_sel, max_sel=max_sel)


def calculate_growth(params: SaturationParams,
                     dk_geno: jnp.ndarray,
                     activity: jnp.ndarray,
                     theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the growth rates for pre-selection and selection phases.
    Growth = min + (max - min) * (theta / (1 + theta))
    """
    g_pre = params.min_pre + dk_geno + activity * (params.max_pre - params.min_pre) * (theta / (1.0 + theta))
    g_sel = params.min_sel + dk_geno + activity * (params.max_sel - params.min_sel) * (theta / (1.0 + theta))

    return g_pre, g_sel


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """
    parameters = {}
    parameters["min_loc"] = 0.020
    parameters["min_scale"] = 0.1
    parameters["max_loc"] = 0.030
    parameters["max_scale"] = 0.1

    return parameters


def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 0.01

    guesses = {}
    guesses[f"{name}_min_locs"] = jnp.full(num_cond_rep, 0.020, dtype=float)
    guesses[f"{name}_min_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    guesses[f"{name}_max_locs"] = jnp.full(num_cond_rep, 0.030, dtype=float)
    guesses[f"{name}_max_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)

    return guesses


def get_priors():
    return ModelPriors(**get_hyperparameters())


def get_scale_bounds():
    """
    Per-condition prior-scale bounds for the pre-fit calibration.

    See ``linear.get_scale_bounds`` for the contract.  ``min`` is the
    theta-independent baseline that trades off against ``dk_geno`` (tight
    floor); ``max`` is a rate too and gets the same bounds.

    Returns
    -------
    dict
        ``{suffix: {"floor": float, "ceiling": float, "scale_field": str}}``.
    """
    return {
        "min": {"floor": 0.002, "ceiling": 0.1, "scale_field": "min_scale"},
        "max": {"floor": 0.002, "ceiling": 0.1, "scale_field": "max_scale"},
    }


def get_extract_specs(ctx):
    if "condition_rep" not in ctx.growth_tm.map_groups:
        return []
    cond_rep_cols = (["condition_rep"] if ctx.growth_shares_replicates
                     else ["replicate", "condition_rep"])
    return [dict(
        input_df=ctx.growth_tm.map_groups["condition_rep"],
        params_to_get=["growth_min", "growth_max"],
        map_column="map_condition_rep",
        get_columns=cond_rep_cols,
        in_run_prefix="condition_",
    )]
