import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)

@dataclass
class SaturationParams:
    """
    Holds saturation growth parameters for pre-selection and selection phases.
    Growth = min + (max - min) * (theta / (1 + theta))
    """
    min_pre: torch.Tensor
    max_pre: torch.Tensor
    min_sel: torch.Tensor
    max_sel: torch.Tensor

@dataclass
class ModelPriors:
    """
    Holds data needed to specify model priors.
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
        The prefix for all Pyro sample sites (e.g., "saturation").
    data : GrowthData
        A dataclass containing experimental data and metadata.
    priors : ModelPriors
        A dataclass containing all hyperparameters for the model.

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

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    min_1d = min_per_condition.reshape(-1)
    max_1d = max_per_condition.reshape(-1)
    min_pre = min_1d[data.map_condition_pre]
    max_pre = max_1d[data.map_condition_pre]
    min_sel = min_1d[data.map_condition_sel]
    max_sel = max_1d[data.map_condition_sel]

    return SaturationParams(min_pre=min_pre, max_pre=max_pre,
                            min_sel=min_sel, max_sel=max_sel)

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> SaturationParams:
    """
    Guide corresponding to the pooled saturation growth model.
    """

    def sample_guide_param(param_name, loc_loc, loc_scale):
        p_loc_loc = pyro.param(f"{name}_{param_name}_hyper_loc_loc", torch.tensor(loc_loc))
        p_loc_scale = pyro.param(f"{name}_{param_name}_hyper_loc_scale", torch.tensor(loc_scale),
                                 constraint=torch.distributions.constraints.positive)
        hyper_loc = pyro.sample(f"{name}_{param_name}_hyper_loc", dist.Normal(p_loc_loc, p_loc_scale))

        p_scale_loc = pyro.param(f"{name}_{param_name}_hyper_scale_loc", torch.tensor(-1.0))
        p_scale_scale = pyro.param(f"{name}_{param_name}_hyper_scale_scale", torch.tensor(0.1),
                                   constraint=torch.distributions.constraints.positive)
        hyper_scale = pyro.sample(f"{name}_{param_name}_hyper_scale", dist.LogNormal(p_scale_loc, p_scale_scale))

        offset_locs = pyro.param(f"{name}_{param_name}_offset_locs", torch.zeros(data.num_condition_rep))
        offset_scales = pyro.param(f"{name}_{param_name}_offset_scales", torch.ones(data.num_condition_rep),
                                   constraint=torch.distributions.constraints.positive)

        with pyro.plate(f"{name}_{param_name}_condition_parameters", data.num_condition_rep) as idx:
            offset = pyro.sample(f"{name}_{param_name}_offset", dist.Normal(offset_locs[idx], offset_scales[idx]))

        param_per_condition = hyper_loc + offset * hyper_scale
        return param_per_condition

    min_per_condition = sample_guide_param("min", priors.growth_min_hyper_loc_loc, priors.growth_min_hyper_loc_scale)
    max_per_condition = sample_guide_param("max", priors.growth_max_hyper_loc_loc, priors.growth_max_hyper_loc_scale)

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    min_1d = min_per_condition.reshape(-1)
    max_1d = max_per_condition.reshape(-1)
    min_pre = min_1d[data.map_condition_pre]
    max_pre = max_1d[data.map_condition_pre]
    min_sel = min_1d[data.map_condition_sel]
    max_sel = max_1d[data.map_condition_sel]

    return SaturationParams(min_pre=min_pre, max_pre=max_pre,
                            min_sel=min_sel, max_sel=max_sel)

def calculate_growth(params: SaturationParams,
                     dk_geno: torch.Tensor,
                     activity: torch.Tensor,
                     theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    guesses[f"{name}_min_offset"] = torch.zeros(shape)
    guesses[f"{name}_max_offset"] = torch.zeros(shape)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
