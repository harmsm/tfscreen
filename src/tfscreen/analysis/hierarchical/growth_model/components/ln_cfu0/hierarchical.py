import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass

from typing import Dict, Any
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData


@dataclass(frozen=True)
class ModelPriors:
    """
    Holds hyperparameters for the ln_cfu0 model.

    Attributes
    ----------
    ln_cfu0_hyper_loc_loc : float
        Mean of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_loc_scale : float
        Standard deviation of the prior for the hyper-location of ln_cfu0.
    ln_cfu0_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of ln_cfu0.
    """

    ln_cfu0_hyper_loc_loc: float
    ln_cfu0_hyper_loc_scale: float
    ln_cfu0_hyper_scale_loc: float

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> torch.Tensor:
    """
    Defines the hierarchical model for initial cell counts (ln_cfu0).

    This model treats the ``ln_cfu0`` for each independent experimental
    group (e.g., each genotype/replicate combination) as being drawn
    from a shared, pooled Normal distribution. The location and scale
    of this distribution are learned hyper-parameters.

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample/deterministic sites in this
        component.
    data : GrowthData
        Container holding experimental data and metadata.
    priors : ModelPriors
        Hyperparameters for the pooled priors.

    Returns
    -------
    torch.Tensor
        The sampled ``ln_cfu0`` values, expanded to match the shape of
        the observations via ``data.map_ln_cfu0``.
    """

    # Define hyper-priors for the pooled distribution
    ln_cfu0_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.ln_cfu0_hyper_loc_loc,
                    priors.ln_cfu0_hyper_loc_scale)
    )
    ln_cfu0_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.ln_cfu0_hyper_scale_loc)
    )

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate", data.num_replicate, dim=-3):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.poutine.scale(scale=data.scale_vector):
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = ln_cfu0_hyper_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Register deterministic values for inspection
    pyro.deterministic(name, ln_cfu0_per_rep_cond_geno)

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]

    return ln_cfu0

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> torch.Tensor:
    """
    Guide corresponding to the hierarchical ln_cfu0 model.

    This guide defines the variational family for the initial cell count
    model. It uses:
    - Normal distributions for the hyper-location mean.
    - LogNormal distributions for the hyper-scale.
    - Normal distributions for the per-replicate/group offsets.
    """

    # -------------------------------------------------------------------------
    # Global parameters

    # Hyper Loc (Normal posterior approximation)
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc", torch.tensor(priors.ln_cfu0_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale", torch.tensor(priors.ln_cfu0_hyper_loc_scale),
                             constraint=torch.distributions.constraints.positive)
    ln_cfu0_hyper_loc = pyro.sample(f"{name}_hyper_loc", dist.Normal(h_loc_loc, h_loc_scale))

    # Hyper Scale (LogNormal posterior approximation for positive variable)
    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", torch.tensor(-1.0))
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", torch.tensor(0.1),
                               constraint=torch.distributions.constraints.positive)
    ln_cfu0_hyper_scale = pyro.sample(f"{name}_hyper_scale", dist.LogNormal(h_scale_loc, h_scale_scale))

    # -------------------------------------------------------------------------
    # Genotype-specific parameter

    param_shape = (data.num_replicate, data.num_condition_pre, data.num_genotype)
    offset_locs = pyro.param(f"{name}_offset_locs", torch.zeros(param_shape))
    offset_scales = pyro.param(f"{name}_offset_scales", torch.ones(param_shape),
                               constraint=torch.distributions.constraints.positive)

    # Sample non-centered offsets for each ln_cfu0 group
    with pyro.plate(f"{name}_replicate", data.num_replicate, dim=-3):
        with pyro.plate(f"{name}_condition_pre", data.num_condition_pre, dim=-2):
            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                with pyro.poutine.scale(scale=data.scale_vector):
                    batch_locs = offset_locs[..., data.batch_idx]
                    batch_scales = offset_scales[..., data.batch_idx]
                    ln_cfu0_offsets = pyro.sample(f"{name}_offset", dist.Normal(batch_locs, batch_scales))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if ln_cfu0_offsets.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        ln_cfu0_offsets = ln_cfu0_offsets[..., data.batch_idx]

    # Calculate the per-group ln_cfu0 values
    ln_cfu0_per_rep_cond_geno = ln_cfu0_hyper_loc + ln_cfu0_offsets * ln_cfu0_hyper_scale

    # Expand tensor to match all observations
    ln_cfu0 = ln_cfu0_per_rep_cond_geno[:, None, :, None, None, None, :]

    return ln_cfu0

def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names (as strings) to their
        default values.
    """

    parameters = {}

    parameters["ln_cfu0_hyper_loc_loc"] = -2.5
    parameters["ln_cfu0_hyper_loc_scale"] = 3.0
    parameters["ln_cfu0_hyper_scale_loc"] = 2.0

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        Container holding data metadata.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary mapping sample site names to guess values.
    """

    guesses = {}
    guesses[f"{name}_hyper_loc"] = -2.5
    guesses[f"{name}_hyper_scale"] = 3.0
    guesses[f"{name}_offset"] = torch.zeros((data.num_replicate,
                                             data.num_condition_pre,
                                             data.num_genotype))

    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.

    Returns
    -------
    ModelPriors
        A populated dataclass of hyperparameters.
    """
    return ModelPriors(**get_hyperparameters())
