import torch
import pyro
from dataclasses import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the fixed activity model.
    This is an empty placeholder, as this model has no priors.
    """
    pass

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> torch.Tensor:
    """
    Defines the fixed model for genotype-specific activity.

    This model provides a fixed, non-learned value for activity. It sets
    the activity for all genotypes to 1.0, effectively making the
    growth rate dependent only on `k` and `m*theta` without any
    additional scaling.

    Parameters
    ----------
    name : str
        The prefix for all Pyro deterministic sites.
    data : GrowthData
        A data object containing experimental data and metadata.
        This function primarily uses:
        - ``data.batch_size`` : (int) The total number of observations.
    priors : ModelPriors
        A dataclass of hyperparameters. (Unused in this model).

    Returns
    -------
    torch.Tensor
        An array of 1.0s, expanded to match the shape of the observations.
    """

    # Set activity for all genotypes to 1.0
    activity_dists = torch.ones(data.batch_size)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity_dists)

    # Broadcast to full-sized tensor
    activity = activity_dists.reshape(-1)[None,None,None,None,None,None,:]

    return activity


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> torch.Tensor:
    """
    Guide for the fixed activity model.

    Since all parameters are fixed and deterministic, this guide simply
    returns the fixed values (1.0) and does not register any learnable
    parameters.
    """

    # Set activity for all genotypes to 1.0
    activity_dists = torch.ones(data.batch_size)

    # Broadcast to full-sized tensor
    activity = activity_dists.reshape(-1)[None,None,None,None,None,None,:]

    return activity

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no hyperparameters.
    """
    return {}


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    """
    Get guess values for the model's latent parameters.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites. (Unused).
    data : GrowthData
        A data object containing data metadata. (Unused).

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """
    return {}

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        An empty, populated dataclass.
    """
    return ModelPriors(**get_hyperparameters())
