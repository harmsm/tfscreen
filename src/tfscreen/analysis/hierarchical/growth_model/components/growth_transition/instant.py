import torch
import pyro
from dataclasses import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any

@dataclass(frozen=True)
class ModelPriors:
    """
    Holds hyperparameters for the instant growth transition model.
    Currently, there are no priors needed for an instant transition.
    """
    pass

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: torch.Tensor,
                 g_sel: torch.Tensor,
                 t_pre: torch.Tensor,
                 t_sel: torch.Tensor,
                 theta: torch.Tensor = None) -> torch.Tensor:
    """
    Combines the pre-selection and selection growth phases with an instant
    transition between them.

    Parameters
    ----------
    name : str
        The prefix for Pyro sample/deterministic sites in this component.
    data : GrowthData
        Container holding experimental data and metadata.
    priors : ModelPriors
        Hyperparameters.
    g_pre : torch.Tensor
        Pre-selection growth rate tensor.
    g_sel : torch.Tensor
        Selection growth rate tensor.
    t_pre : torch.Tensor
        Pre-selection time tensor.
    t_sel : torch.Tensor
        Selection time tensor.

    Returns
    -------
    total_growth : torch.Tensor
        The total growth over both phases.
    """

    return g_pre * t_pre + g_sel * t_sel


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: torch.Tensor,
          g_sel: torch.Tensor,
          t_pre: torch.Tensor,
          t_sel: torch.Tensor,
          theta: torch.Tensor = None) -> torch.Tensor:
    """
    Guide corresponding to the instant growth transition model.
    """

    return g_pre * t_pre + g_sel * t_sel


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {}

def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.
    """
    return {}

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors()
