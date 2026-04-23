import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import (
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
    min_loc, min_scale : float
        Normal prior parameters for per-condition minimum growth rate.
    max_loc, max_scale : float
        Normal prior parameters for per-condition maximum growth rate.
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
    with pyro.plate(f"{name}_min_condition_parameters", data.num_condition_rep):
        min_per_condition = pyro.sample(f"{name}_min",
                                        dist.Normal(priors.min_loc, priors.min_scale))

    with pyro.plate(f"{name}_max_condition_parameters", data.num_condition_rep):
        max_per_condition = pyro.sample(f"{name}_max",
                                        dist.Normal(priors.max_loc, priors.max_scale))

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
    min_locs = pyro.param(f"{name}_min_locs",
                          jnp.full(data.num_condition_rep, priors.min_loc, dtype=float))
    min_scales = pyro.param(f"{name}_min_scales",
                            jnp.full(data.num_condition_rep, priors.min_scale, dtype=float),
                            constraint=dist.constraints.positive)
    max_locs = pyro.param(f"{name}_max_locs",
                          jnp.full(data.num_condition_rep, priors.max_loc, dtype=float))
    max_scales = pyro.param(f"{name}_max_scales",
                            jnp.full(data.num_condition_rep, priors.max_scale, dtype=float),
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
    parameters["min_loc"] = 0.025
    parameters["min_scale"] = 0.1
    parameters["max_loc"] = 0.025
    parameters["max_scale"] = 0.1

    return parameters


def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 0.01

    guesses = {}
    guesses[f"{name}_min_locs"] = jnp.full(num_cond_rep, 0.025, dtype=float)
    guesses[f"{name}_min_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)
    guesses[f"{name}_max_locs"] = jnp.full(num_cond_rep, 0.025, dtype=float)
    guesses[f"{name}_max_scales"] = jnp.full(num_cond_rep, _DEFAULT_SCALE, dtype=float)

    return guesses


def get_priors():
    return ModelPriors(**get_hyperparameters())
