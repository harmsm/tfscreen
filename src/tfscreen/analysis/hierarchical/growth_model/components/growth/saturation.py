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


# Hyperparameter suffixes that may be pinned via ModelPriors.pinned.
_PINNABLE_SUFFIXES = (
    "min_hyper_loc", "min_hyper_scale",
    "max_hyper_loc", "max_hyper_scale",
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
    """

    min_hyper_loc_loc: float
    min_hyper_loc_scale: float
    min_hyper_scale_loc: float

    max_hyper_loc_loc: float
    max_hyper_loc_scale: float
    max_hyper_scale_loc: float

    pinned: Mapping[str, float] = field(
        pytree_node=False, default_factory=dict
    )


def define_model(name: str, 
                 data: GrowthData, 
                 priors: ModelPriors) -> SaturationParams:
    """
    Growth parameters min_xx and max_xx versus condition. These go into 
    the model as min + (max - min)*(theta/(1 + theta)). Returns full 
    min_pre, max_pre, min_sel, and max_sel tensors.

    Parameters
    ----------
    name : str
        The prefix for all Numpyro sample sites (e.g., "saturation").
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree containing all hyperparameters for the model.

    Returns
    -------
    params : SaturationParams
        A dataclass containing min and max for pre and sel.
    """

    pinned = priors.pinned

    def sample_param(param_name, loc_loc, loc_scale, hyper_scale):
        hyper_loc = _hyper(
            name, f"{param_name}_hyper_loc",
            dist.Normal(loc_loc, loc_scale), pinned,
        )
        hyper_scale_val = _hyper(
            name, f"{param_name}_hyper_scale",
            dist.HalfNormal(hyper_scale), pinned,
        )
        
        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep):
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(0.0, 1.0))
        
        param_per_condition = hyper_loc + offset * hyper_scale_val
        pyro.deterministic(f"{name}_{param_name}", param_per_condition)
        return param_per_condition

    min_per_condition = sample_param("min", priors.min_hyper_loc_loc, priors.min_hyper_loc_scale, priors.min_hyper_scale_loc)
    max_per_condition = sample_param("max", priors.max_hyper_loc_loc, priors.max_hyper_loc_scale, priors.max_hyper_scale_loc)

    # Expand to full-sized tensors
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
    Guide corresponding to the pooled saturation growth model.
    """

    pinned = priors.pinned

    def sample_guide_param(param_name, loc_loc, loc_scale):
        loc_suffix = f"{param_name}_hyper_loc"
        scale_suffix = f"{param_name}_hyper_scale"

        pinned_loc = _pinned_value(loc_suffix, pinned)
        if pinned_loc is not None:
            hyper_loc = pinned_loc
        else:
            p_loc_loc = pyro.param(f"{name}_{loc_suffix}_loc", jnp.array(loc_loc))
            p_loc_scale = pyro.param(
                f"{name}_{loc_suffix}_scale", jnp.array(loc_scale),
                constraint=dist.constraints.greater_than(1e-4),
            )
            hyper_loc = pyro.sample(
                f"{name}_{loc_suffix}", dist.Normal(p_loc_loc, p_loc_scale)
            )

        pinned_scale = _pinned_value(scale_suffix, pinned)
        if pinned_scale is not None:
            hyper_scale = pinned_scale
        else:
            p_scale_loc = pyro.param(f"{name}_{scale_suffix}_loc", jnp.array(-1.0))
            p_scale_scale = pyro.param(
                f"{name}_{scale_suffix}_scale", jnp.array(0.1),
                constraint=dist.constraints.greater_than(1e-4),
            )
            hyper_scale = pyro.sample(
                f"{name}_{scale_suffix}",
                dist.LogNormal(p_scale_loc, p_scale_scale),
            )

        offset_locs = pyro.param(f"{name}_{param_name}_offset_locs", jnp.zeros(data.num_condition_rep, dtype=float))
        offset_scales = pyro.param(f"{name}_{param_name}_offset_scales", jnp.ones(data.num_condition_rep, dtype=float),
                                   constraint=dist.constraints.positive)

        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep) as idx:
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(offset_locs[idx], offset_scales[idx]))
        
        param_per_condition = hyper_loc + offset * hyper_scale
        return param_per_condition

    min_per_condition = sample_guide_param("min", priors.min_hyper_loc_loc, priors.min_hyper_loc_scale)
    max_per_condition = sample_guide_param("max", priors.max_hyper_loc_loc, priors.max_hyper_loc_scale)

    # Expand to full-sized tensors
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
    parameters["min_hyper_loc_loc"] = 0.025
    parameters["min_hyper_loc_scale"] = 0.1
    parameters["min_hyper_scale_loc"] = 0.1

    parameters["max_hyper_loc_loc"] = 0.025
    parameters["max_hyper_loc_scale"] = 0.1
    parameters["max_hyper_scale_loc"] = 0.1

    return parameters

def get_guesses(name, data):
    """
    Get guesses for the model parameters. 
    """

    shape = data.num_condition_rep

    guesses = {}
    guesses[f"{name}_min_hyper_loc"] = 0.025
    guesses[f"{name}_min_hyper_scale"] = 0.1
    guesses[f"{name}_max_hyper_loc"] = 0.025
    guesses[f"{name}_max_hyper_scale"] = 0.1
    
    guesses[f"{name}_min_offset"] = jnp.zeros(shape, dtype=float)
    guesses[f"{name}_max_offset"] = jnp.zeros(shape, dtype=float)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
