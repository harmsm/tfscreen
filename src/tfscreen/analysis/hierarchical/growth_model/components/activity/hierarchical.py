import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData

@dataclass
class ModelPriors:
    """
    Holds hyperparameters for the hierarchical activity model.

    Attributes
    ----------
    activity_hyper_loc_loc : float
        Mean of the prior for the hyper-location of log(activity).
    activity_hyper_loc_scale : float
        Std dev of the prior for the hyper-location of log(activity).
    activity_hyper_scale_loc : float
        Scale of the HalfNormal prior for the hyper-scale of log(activity).
    """

    activity_hyper_loc_loc: float
    activity_hyper_loc_scale: float
    activity_hyper_scale_loc: float

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> torch.Tensor:
    """
    Defines the hierarchical model for genotype-specific activity.

    Activity is the scale of transcriptional output (range 0 to infinity).
    This model assumes the wild-type genotype has a fixed activity of 1.0.
    The activities of all other (mutant) genotypes are modeled as being
    drawn from a shared, pooled LogNormal distribution (i.e., a Normal
    distribution in log-space).

    Parameters
    ----------
    name : str
        The prefix for all Pyro sample/deterministic sites in this component.
    data : GrowthData
        A data object containing experimental data and metadata.
        This function primarily uses:
        - ``data.num_genotype`` : (int) The total number of genotypes.
        - ``data.wt_indexes`` : (torch.Tensor) integer indexes of wt elements
        - ``data.batch_idx`` : (torch.Tensor) Index array mapping observations to genotypes.
    priors : ModelPriors
        A dataclass containing the hyperparameters for the pooled priors.

    Returns
    -------
    torch.Tensor
        The sampled `activity` values, expanded to match the shape of
        the observations.
    """

    # Priors are on log(activity), so their mean is log(1.0) = 0.0
    log_activity_hyper_loc = pyro.sample(
        f"{name}_log_hyper_loc",
        dist.Normal(priors.activity_hyper_loc_loc,
                    priors.activity_hyper_loc_scale)
    )
    log_activity_hyper_scale = pyro.sample(
        f"{name}_log_hyper_scale",
        dist.HalfNormal(priors.activity_hyper_scale_loc)
    )

    # Sample non-centered offsets for mutant genotypes only
    with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
        with pyro.poutine.scale(scale=data.scale_vector):
            activity_offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if activity_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        activity_offset = activity_offset[..., data.batch_idx]

    # Calculate in log-space, then exponentiate
    log_activity_mutant_dists = log_activity_hyper_loc + activity_offset * log_activity_hyper_scale
    activity = torch.clamp(torch.exp(log_activity_mutant_dists), max=1e30)

    # Set wildtype activity to 1.0
    is_wt_mask = torch.isin(data.batch_idx, data.wt_indexes)
    activity = torch.where(is_wt_mask, torch.tensor(1.0), activity)

    # Register per-genotype values for inspection
    pyro.deterministic(name, activity)

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:]

    return activity

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> torch.Tensor:
    """
    Guide corresponding to the hierarchical activity model.

    This guide uses:
    - Normal distributions for the hyper-location mean.
    - LogNormal distributions for hyper-scale and positive scale parameters.
    - A non-centered parameterization for the per-genotype offsets.
    """

    a_loc_loc = pyro.param(f"{name}_a_hyper_loc_loc", torch.tensor(priors.activity_hyper_loc_loc))
    a_loc_scale = pyro.param(f"{name}_a_hyper_loc_scale", torch.tensor(priors.activity_hyper_loc_scale),
                             constraint=torch.distributions.constraints.positive)
    log_activity_hyper_loc = pyro.sample(
        f"{name}_log_hyper_loc",
        dist.Normal(a_loc_loc, a_loc_scale)
    )

    a_scale_loc = pyro.param(f"{name}_a_hyper_scale_loc", torch.tensor(-1.0))
    a_scale_scale = pyro.param(f"{name}_a_hyper_scale_scale", torch.tensor(0.1),
                               constraint=torch.distributions.constraints.positive)
    log_activity_hyper_scale = pyro.sample(
        f"{name}_log_hyper_scale",
        dist.LogNormal(a_scale_loc, a_scale_scale)
    )

    offset_locs = pyro.param(f"{name}_offset_locs", torch.zeros(data.num_genotype))
    offset_scales = pyro.param(f"{name}_offset_scales", torch.ones(data.num_genotype),
                               constraint=torch.distributions.constraints.positive)

    # Sample non-centered offsets for mutant genotypes only
    with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
        with pyro.poutine.scale(scale=data.scale_vector):

            batch_locs = offset_locs[data.batch_idx]
            batch_scales = offset_scales[data.batch_idx]

            activity_offset = pyro.sample(f"{name}_offset", dist.Normal(batch_locs, batch_scales))

    # Guard against full-sized array substitution during initialization or re-runs
    # with full-sized initial values
    if activity_offset.shape[-1] == data.num_genotype and data.batch_size < data.num_genotype:
        activity_offset = activity_offset[..., data.batch_idx]

    # Calculate in log-space, then exponentiate
    log_activity_mutant_dists = log_activity_hyper_loc + activity_offset * log_activity_hyper_scale
    activity = torch.clamp(torch.exp(log_activity_mutant_dists), max=1e30)

    # Set wildtype activity to 1.0
    is_wt_mask = torch.isin(data.batch_idx, data.wt_indexes)
    activity = torch.where(is_wt_mask, torch.tensor(1.0), activity)

    # Broadcast to full-sized tensor
    activity = activity[None,None,None,None,None,None,:]

    return activity


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.

    The hyper-location is centered at 0.0, corresponding to log(1.0),
    so the prior is centered on the wild-type activity.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping hyperparameter names to their default values.
    """

    parameters = {}
    parameters["activity_hyper_loc_loc"] = 0.0
    parameters["activity_hyper_loc_scale"] = 0.01
    parameters["activity_hyper_scale_loc"] = 0.1

    return parameters


def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.

    These values are used in `pyro.poutine.condition` for testing
    or initializing inference. The offsets are set to zero, meaning
    all mutant activities will be guessed as 1.0 (same as wild-type).

    Parameters
    ----------
    name : str
        The prefix used for all sample sites (e.g., "my_model").
    data : GrowthData
        A data object containing data metadata. Requires:
        - ``data.num_genotype``

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary mapping sample site names to tensors of guess values.
    """

    guesses = {}
    guesses[f"{name}_log_hyper_loc"] = 0.0
    guesses[f"{name}_log_hyper_scale"] = 0.1
    guesses[f"{name}_offset"] = torch.zeros(data.num_genotype)

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
