import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    GrowthData
)

@dataclass
class LinearParams:
    """
    Holds linear growth parameters (intercept and slope) for pre-selection
    and selection phases.
    """
    k_pre: torch.Tensor
    m_pre: torch.Tensor
    k_sel: torch.Tensor
    m_sel: torch.Tensor

@dataclass
class ModelPriors:
    """
    Holds data needed to specify model priors.
    """

    growth_k_per_cond_rep: torch.Tensor
    growth_m_per_cond_rep: torch.Tensor

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
        The prefix for all Pyro sample sites.
    data : GrowthData
        A dataclass containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_condition_rep``
        - ``data.num_replicate``
        - ``data.map_condition_pre``
        - ``data.map_condition_sel``
    priors : ModelPriors
        A dataclass containing all hyperparameters for the model, including:
        - ``priors.growth_k_per_cond_rep``
        - ``priors.growth_m_per_cond_rep``

    Returns
    -------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, and m_sel.
    """

    # Tile priors to cover data.num_condition_rep if the provided arrays are
    # shorter (e.g., when calibration values don't match the data structure).
    import math
    n = data.num_condition_rep
    k_vals = priors.growth_k_per_cond_rep
    m_vals = priors.growth_m_per_cond_rep
    if k_vals.shape[0] < n:
        k_vals = k_vals.repeat(math.ceil(n / k_vals.shape[0]))[:n]
        m_vals = m_vals.repeat(math.ceil(n / m_vals.shape[0]))[:n]

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    k_1d = k_vals.reshape(-1)
    m_1d = m_vals.reshape(-1)
    k_pre = k_1d[data.map_condition_pre]
    m_pre = m_1d[data.map_condition_pre]
    k_sel = k_1d[data.map_condition_sel]
    m_sel = m_1d[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> LinearParams:
    """
    Guide for the fixed growth model.

    This guide simply returns the fixed per-condition/replicate values
    defined in the priors, without registering any learnable parameters.
    """

    import math
    n = data.num_condition_rep
    k_vals = priors.growth_k_per_cond_rep
    m_vals = priors.growth_m_per_cond_rep
    if k_vals.shape[0] < n:
        k_vals = k_vals.repeat(math.ceil(n / k_vals.shape[0]))[:n]
        m_vals = m_vals.repeat(math.ceil(n / m_vals.shape[0]))[:n]

    # Flatten to 1D before indexing to handle extra leading singleton dims
    # added by AutoDelta's plate broadcasting during Predictive.
    k_1d = k_vals.reshape(-1)
    m_1d = m_vals.reshape(-1)
    k_pre = k_1d[data.map_condition_pre]
    m_pre = m_1d[data.map_condition_pre]
    k_sel = k_1d[data.map_condition_sel]
    m_sel = m_1d[data.map_condition_sel]

    return LinearParams(k_pre=k_pre, m_pre=m_pre, k_sel=k_sel, m_sel=m_sel)

def calculate_growth(params: LinearParams,
                     dk_geno: torch.Tensor,
                     activity: torch.Tensor,
                     theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the growth rates for pre-selection and selection phases.

    Parameters
    ----------
    params : LinearParams
        A dataclass containing k_pre, m_pre, k_sel, m_sel.
    dk_geno : torch.Tensor
        Genotype-specific death rate.
    activity : torch.Tensor
        Genotype activity.
    theta : torch.Tensor
        Occupancy/binding probability.

    Returns
    -------
    g_pre : torch.Tensor
        Pre-selection growth rate tensor.
    g_sel : torch.Tensor
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
    parameters["growth_k_per_cond_rep"] = torch.tensor([0.010696, 0.015366, 0.021437, 0.028558])
    parameters["growth_m_per_cond_rep"] = torch.tensor([-0.009933, 0.000808, 0.006226, -0.000344])

    return parameters

def get_guesses(name, data):
    """
    Get guesses for the model parameters.
    """

    return {}

def get_priors():
    return ModelPriors(**get_hyperparameters())
