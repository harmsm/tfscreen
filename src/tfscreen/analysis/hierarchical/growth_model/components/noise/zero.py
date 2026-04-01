import torch
from dataclasses import dataclass
from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

@dataclass(frozen=True)
class ModelPriors:
    """
    Holds hyperparameters for the no-noise model.
    This is an empty placeholder, as this model has no priors.
    """

    pass


def define_model(name: str,
                 fx_calc: torch.Tensor,
                 priors: ModelPriors) -> torch.Tensor:
    """
    A null model that applies no noise to the input.

    This function serves as a placeholder in the model architecture.
    It simply returns the deterministic input array ``fx_calc``
    without adding any stochasticity or sampling any parameters.

    Parameters
    ----------
    name : str
        The prefix for Pyro sites. (Unused in this model).
    fx_calc : torch.Tensor
        The deterministically calculated input array (e.g., fractional
        occupancy).
    priors : ModelPriors
        Hyperparameters. (Unused in this model).

    Returns
    -------
    torch.Tensor
        The original, unmodified ``fx_calc`` array.
    """

    return fx_calc

def guide(name: str,
          fx_calc: torch.Tensor,
          priors: ModelPriors) -> torch.Tensor:
    """
    Guide for the no-noise model.

    This guide corresponds to the deterministic (null) noise model.
    It simply returns the input calculated value and registers no parameters.
    """

    return fx_calc


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no hyperparameters.
    """

    parameters = {}

    return parameters


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """
    Get guess values for the model's latent parameters.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites. (Unused).
    data : DataClass
        A container holding data metadata. (Unused).

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """

    guesses = {}

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        An empty, populated dataclass.
    """
    return ModelPriors(**get_hyperparameters())
