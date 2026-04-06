import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)

@dataclass
class PowerParams:
    """
    Holds power-law growth parameters for pre-selection and selection phases.
    Growth = k + m * (theta**n)
    """
    k_pre: torch.Tensor
    m_pre: torch.Tensor
    n_pre: torch.Tensor
    k_sel: torch.Tensor
    m_sel: torch.Tensor
    n_sel: torch.Tensor

@dataclass
class ModelPriors:
    """
    Holds data needed to specify model priors.
    """

    growth_k_hyper_loc_loc: float
    growth_k_hyper_loc_scale: float
    growth_k_hyper_scale: float

    growth_m_hyper_loc_loc: float
    growth_m_hyper_loc_scale: float
    growth_m_hyper_scale: float

    growth_n_hyper_loc_loc: float
    growth_n_hyper_loc_scale: float
    growth_n_hyper_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> PowerParams:
    """
    Growth parameters k_xx, m_xx, n_xx versus condition. These go into
    the model as k + m*(theta**n). Returns full k_pre, m_pre, n_pre,
    k_sel, m_sel, and n_sel tensors.

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample sites (e.g., "power").
    data : GrowthData
        A dataclass containing experimental data and metadata.
    priors : ModelPriors
        A dataclass containing all hyperparameters for the model.

    Returns
    -------
    params : PowerParams
        A dataclass containing k, m, and n for pre and sel.
    """

    def sample_param(param_name, loc_loc, loc_scale, hyper_scale, is_positive=False):
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
        if is_positive:
            param_per_condition = torch.exp(param_per_condition)

        pyro.deterministic(f"{name}_{param_name}", param_per_condition)
        return param_per_condition

    k_per_condition = sample_param("k", priors.growth_k_hyper_loc_loc, priors.growth_k_hyper_loc_scale, priors.growth_k_hyper_scale)
    m_per_condition = sample_param("m", priors.growth_m_hyper_loc_loc, priors.growth_m_hyper_loc_scale, priors.growth_m_hyper_scale)
    n_per_condition = sample_param("n", priors.growth_n_hyper_loc_loc, priors.growth_n_hyper_loc_scale, priors.growth_n_hyper_scale, is_positive=True)

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    k_1d = k_per_condition.reshape(-1)
    m_1d = m_per_condition.reshape(-1)
    n_1d = n_per_condition.reshape(-1)
    k_pre = k_1d[data.map_condition_pre]
    m_pre = m_1d[data.map_condition_pre]
    n_pre = n_1d[data.map_condition_pre]
    k_sel = k_1d[data.map_condition_sel]
    m_sel = m_1d[data.map_condition_sel]
    n_sel = n_1d[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> PowerParams:
    """
    Guide corresponding to the pooled power growth model.
    """

    def sample_guide_param(param_name, loc_loc, loc_scale, is_positive=False):
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
        if is_positive:
            param_per_condition = torch.exp(param_per_condition)
        return param_per_condition

    k_per_condition = sample_guide_param("k", priors.growth_k_hyper_loc_loc, priors.growth_k_hyper_loc_scale)
    m_per_condition = sample_guide_param("m", priors.growth_m_hyper_loc_loc, priors.growth_m_hyper_loc_scale)
    n_per_condition = sample_guide_param("n", priors.growth_n_hyper_loc_loc, priors.growth_n_hyper_loc_scale, is_positive=True)

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    k_1d = k_per_condition.reshape(-1)
    m_1d = m_per_condition.reshape(-1)
    n_1d = n_per_condition.reshape(-1)
    k_pre = k_1d[data.map_condition_pre]
    m_pre = m_1d[data.map_condition_pre]
    n_pre = n_1d[data.map_condition_pre]
    k_sel = k_1d[data.map_condition_sel]
    m_sel = m_1d[data.map_condition_sel]
    n_sel = n_1d[data.map_condition_sel]

    return PowerParams(k_pre=k_pre, m_pre=m_pre, n_pre=n_pre,
                       k_sel=k_sel, m_sel=m_sel, n_sel=n_sel)

def calculate_growth(params: PowerParams,
                     dk_geno: torch.Tensor,
                     activity: torch.Tensor,
                     theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    parameters["growth_k_hyper_loc_loc"] = 0.025
    parameters["growth_k_hyper_loc_scale"] = 0.1
    parameters["growth_k_hyper_scale"] = 0.1

    parameters["growth_m_hyper_loc_loc"] = 0.0
    parameters["growth_m_hyper_loc_scale"] = 0.01
    parameters["growth_m_hyper_scale"] = 0.1

    parameters["growth_n_hyper_loc_loc"] = 0.0  # ln(1.0)
    parameters["growth_n_hyper_loc_scale"] = 0.5
    parameters["growth_n_hyper_scale"] = 0.1

    return parameters

def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """

    shape = data.num_condition_rep

    guesses = {}
    guesses[f"{name}_k_hyper_loc"] = 0.025
    guesses[f"{name}_k_hyper_scale"] = 0.1
    guesses[f"{name}_m_hyper_loc"] = 0.0
    guesses[f"{name}_m_hyper_scale"] = 0.01
    guesses[f"{name}_n_hyper_loc"] = 0.0
    guesses[f"{name}_n_hyper_scale"] = 0.1

    guesses[f"{name}_k_offset"] = torch.zeros(shape)
    guesses[f"{name}_m_offset"] = torch.zeros(shape)
    guesses[f"{name}_n_offset"] = torch.zeros(shape)

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())
