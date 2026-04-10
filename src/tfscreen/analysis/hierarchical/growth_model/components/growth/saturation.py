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
    """

    growth_min_hyper_loc_loc: float
    growth_min_hyper_loc_scale: float
    growth_min_hyper_scale: float
    
    growth_max_hyper_loc_loc: float
    growth_max_hyper_loc_scale: float
    growth_max_hyper_scale: float


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

    def sample_param(param_name, loc_loc, loc_scale, hyper_scale):
        hyper_loc = pyro.sample(
            f"{name}_{param_name}_hyper_loc",
            dist.Normal(loc_loc, loc_scale)
        )
        hyper_scale_val = pyro.sample(
            f"{name}_{param_name}_hyper_scale",
            dist.HalfNormal(hyper_scale)
        )
        
        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep):
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(0.0, 1.0))
        
        param_per_condition = hyper_loc + offset * hyper_scale_val
        pyro.deterministic(f"{name}_{param_name}", param_per_condition)
        return param_per_condition

    min_per_condition = sample_param("min", priors.growth_min_hyper_loc_loc, priors.growth_min_hyper_loc_scale, priors.growth_min_hyper_scale)
    max_per_condition = sample_param("max", priors.growth_max_hyper_loc_loc, priors.growth_max_hyper_loc_scale, priors.growth_max_hyper_scale)

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

    def sample_guide_param(param_name, loc_loc, loc_scale):
        p_loc_loc = pyro.param(f"{name}_{param_name}_hyper_loc_loc", jnp.array(loc_loc))
        p_loc_scale = pyro.param(f"{name}_{param_name}_hyper_loc_scale", jnp.array(loc_scale),
                                 constraint=dist.constraints.greater_than(1e-4))
        hyper_loc = pyro.sample(f"{name}_{param_name}_hyper_loc", dist.Normal(p_loc_loc, p_loc_scale))

        p_scale_loc = pyro.param(f"{name}_{param_name}_hyper_scale_loc", jnp.array(-1.0))
        p_scale_scale = pyro.param(f"{name}_{param_name}_hyper_scale_scale", jnp.array(0.1),
                                   constraint=dist.constraints.greater_than(1e-4))
        hyper_scale = pyro.sample(f"{name}_{param_name}_hyper_scale", dist.LogNormal(p_scale_loc, p_scale_scale))
        
        offset_locs = pyro.param(f"{name}_{param_name}_offset_locs", jnp.zeros(data.num_condition_rep, dtype=float))
        offset_scales = pyro.param(f"{name}_{param_name}_offset_scales", jnp.ones(data.num_condition_rep, dtype=float),
                                   constraint=dist.constraints.positive)

        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep) as idx:
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(offset_locs[idx], offset_scales[idx]))
        
        param_per_condition = hyper_loc + offset * hyper_scale
        return param_per_condition

    min_per_condition = sample_guide_param("min", priors.growth_min_hyper_loc_loc, priors.growth_min_hyper_loc_scale)
    max_per_condition = sample_guide_param("max", priors.growth_max_hyper_loc_loc, priors.growth_max_hyper_loc_scale)

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
    parameters["growth_min_hyper_loc_loc"] = 0.025
    parameters["growth_min_hyper_loc_scale"] = 0.1
    parameters["growth_min_hyper_scale"] = 0.1

    parameters["growth_max_hyper_loc_loc"] = 0.025
    parameters["growth_max_hyper_loc_scale"] = 0.1
    parameters["growth_max_hyper_scale"] = 0.1

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
