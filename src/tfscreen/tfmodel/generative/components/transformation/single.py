"""
No congression: every cell of a genotype carries only its own plasmid.
"""

from flax.struct import dataclass

from tfscreen.tfmodel.data_class import GrowthData
from tfscreen.tfmodel.generative.components.transformation._classes import (
    CellClasses,
    single_class,
)


# One class built from the genotype's own values; no population needed.
NEEDS_POPULATION = False


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.

    No priors needed for single transformation (dummy).
    """
    pass


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors):
    """No latent parameters."""
    return None


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors):
    """No latent parameters."""
    return None


def cell_classes(focal, population, params, data: GrowthData) -> CellClasses:
    """
    One class, weight 1, built from the genotype's own theta, activity and
    dk_geno. ``population`` and ``params`` are ignored.
    """
    theta, activity, dk_geno = focal
    return single_class(theta, activity, dk_geno)


def get_hyperparameters():
    """
    No hyperparameters for single transformation.
    """
    return {}


def get_guesses(name, data):
    """
    No guesses needed.
    """
    return {}


def get_priors():
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    return []
