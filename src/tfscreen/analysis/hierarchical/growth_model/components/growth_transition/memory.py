import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any

@dataclass(frozen=True)
class ModelPriors:
    """
    Holds hyperparameters for the memory-dependent growth transition model.
    """
    tau0_hyper_loc_loc: float
    tau0_hyper_loc_scale: float
    tau0_hyper_scale: float

    k1_hyper_loc_loc: float
    k1_hyper_loc_scale: float
    k1_hyper_scale: float

    k2_hyper_loc_loc: float
    k2_hyper_loc_scale: float
    k2_hyper_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: torch.Tensor,
                 g_sel: torch.Tensor,
                 t_pre: torch.Tensor,
                 t_sel: torch.Tensor,
                 theta: torch.Tensor) -> torch.Tensor:
    """
    Combines the pre-selection and selection growth phases with a transition
    that depends on theta (memory effect).

    tau = tau0 + (k1 / (theta + k2))

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
    theta : torch.Tensor
        Fractional occupancy tensor.

    Returns
    -------
    total_growth : torch.Tensor
        The total growth over both phases.
    """

    # Hierarchical tau0
    tau0_hyper_loc = pyro.sample(
        f"{name}_tau0_hyper_loc",
        dist.Normal(priors.tau0_hyper_loc_loc, priors.tau0_hyper_loc_scale)
    )
    tau0_hyper_scale = pyro.sample(
        f"{name}_tau0_hyper_scale",
        dist.HalfNormal(priors.tau0_hyper_scale)
    )

    # Hierarchical k1
    k1_hyper_loc = pyro.sample(
        f"{name}_k1_hyper_loc",
        dist.Normal(priors.k1_hyper_loc_loc, priors.k1_hyper_loc_scale)
    )
    k1_hyper_scale = pyro.sample(
        f"{name}_k1_hyper_scale",
        dist.HalfNormal(priors.k1_hyper_scale)
    )

    # Hierarchical k2
    k2_hyper_loc = pyro.sample(
        f"{name}_k2_hyper_loc",
        dist.Normal(priors.k2_hyper_loc_loc, priors.k2_hyper_loc_scale)
    )
    k2_hyper_scale = pyro.sample(
        f"{name}_k2_hyper_scale",
        dist.HalfNormal(priors.k2_hyper_scale)
    )

    # Plate over conditions
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep):
        tau0_offset = pyro.sample(f"{name}_tau0_offset", dist.Normal(0.0, 1.0))
        k1_offset = pyro.sample(f"{name}_k1_offset", dist.Normal(0.0, 1.0))
        k2_offset = pyro.sample(f"{name}_k2_offset", dist.Normal(0.0, 1.0))

    tau0_per_condition = tau0_hyper_loc + tau0_offset * tau0_hyper_scale
    k1_per_condition = k1_hyper_loc + k1_offset * k1_hyper_scale
    k2_per_condition = k2_hyper_loc + k2_offset * k2_hyper_scale

    # Register deterministic sites
    pyro.deterministic(f"{name}_tau0", tau0_per_condition)
    pyro.deterministic(f"{name}_k1", k1_per_condition)
    pyro.deterministic(f"{name}_k2", k2_per_condition)

    # Expand to match g_pre/theta shape using map_condition_pre
    tau0_expanded = tau0_per_condition[data.map_condition_pre]
    k1_expanded = k1_per_condition[data.map_condition_pre]
    k2_expanded = k2_per_condition[data.map_condition_pre]

    # Calculate transition lag tau
    tau = tau0_expanded + (k1_expanded / (theta + k2_expanded))

    # Calculate growth in each phase
    dln_cfu_pre = g_pre * t_pre

    # Selection phase growth with transition at t_sel = tau
    # dln_cfu_sel = g_pre * t_sel if t_sel < tau
    # dln_cfu_sel = g_pre * tau + g_sel * (t_sel - tau) if t_sel >= tau
    dln_cfu_sel = torch.where(
        t_sel < tau,
        g_pre * t_sel,
        g_pre * tau + g_sel * (t_sel - tau)
    )

    return dln_cfu_pre + dln_cfu_sel


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: torch.Tensor,
          g_sel: torch.Tensor,
          t_pre: torch.Tensor,
          t_sel: torch.Tensor,
          theta: torch.Tensor) -> torch.Tensor:
    """
    Guide corresponding to the memory-dependent growth transition model.
    """

    # tau0 hyper
    tau0_loc_loc = pyro.param(f"{name}_tau0_hyper_loc_loc", torch.tensor(priors.tau0_hyper_loc_loc))
    tau0_loc_scale = pyro.param(f"{name}_tau0_hyper_loc_scale", torch.tensor(priors.tau0_hyper_loc_scale),
                                constraint=torch.distributions.constraints.positive)
    tau0_hyper_loc = pyro.sample(f"{name}_tau0_hyper_loc", dist.Normal(tau0_loc_loc, tau0_loc_scale))

    tau0_scale_loc = pyro.param(f"{name}_tau0_hyper_scale_loc", torch.tensor(-1.0))
    tau0_scale_scale = pyro.param(f"{name}_tau0_hyper_scale_scale", torch.tensor(0.1),
                                  constraint=torch.distributions.constraints.positive)
    tau0_hyper_scale = pyro.sample(f"{name}_tau0_hyper_scale", dist.LogNormal(tau0_scale_loc, tau0_scale_scale))

    # k1 hyper
    k1_loc_loc = pyro.param(f"{name}_k1_hyper_loc_loc", torch.tensor(priors.k1_hyper_loc_loc))
    k1_loc_scale = pyro.param(f"{name}_k1_hyper_loc_scale", torch.tensor(priors.k1_hyper_loc_scale),
                               constraint=torch.distributions.constraints.positive)
    k1_hyper_loc = pyro.sample(f"{name}_k1_hyper_loc", dist.Normal(k1_loc_loc, k1_loc_scale))

    k1_scale_loc = pyro.param(f"{name}_k1_hyper_scale_loc", torch.tensor(-1.0))
    k1_scale_scale = pyro.param(f"{name}_k1_hyper_scale_scale", torch.tensor(0.1),
                                 constraint=torch.distributions.constraints.positive)
    k1_hyper_scale = pyro.sample(f"{name}_k1_hyper_scale", dist.LogNormal(k1_scale_loc, k1_scale_scale))

    # k2 hyper
    k2_loc_loc = pyro.param(f"{name}_k2_hyper_loc_loc", torch.tensor(priors.k2_hyper_loc_loc))
    k2_loc_scale = pyro.param(f"{name}_k2_hyper_loc_scale", torch.tensor(priors.k2_hyper_loc_scale),
                               constraint=torch.distributions.constraints.positive)
    k2_hyper_loc = pyro.sample(f"{name}_k2_hyper_loc", dist.Normal(k2_loc_loc, k2_loc_scale))

    k2_scale_loc = pyro.param(f"{name}_k2_hyper_scale_loc", torch.tensor(-1.0))
    k2_scale_scale = pyro.param(f"{name}_k2_hyper_scale_scale", torch.tensor(0.1),
                                 constraint=torch.distributions.constraints.positive)
    k2_hyper_scale = pyro.sample(f"{name}_k2_hyper_scale", dist.LogNormal(k2_scale_loc, k2_scale_scale))

    # Offsets
    tau0_offset_locs = pyro.param(f"{name}_tau0_offset_locs", torch.zeros(data.num_condition_rep))
    tau0_offset_scales = pyro.param(f"{name}_tau0_offset_scales", torch.ones(data.num_condition_rep),
                                    constraint=torch.distributions.constraints.positive)

    k1_offset_locs = pyro.param(f"{name}_k1_offset_locs", torch.zeros(data.num_condition_rep))
    k1_offset_scales = pyro.param(f"{name}_k1_offset_scales", torch.ones(data.num_condition_rep),
                                   constraint=torch.distributions.constraints.positive)

    k2_offset_locs = pyro.param(f"{name}_k2_offset_locs", torch.zeros(data.num_condition_rep))
    k2_offset_scales = pyro.param(f"{name}_k2_offset_scales", torch.ones(data.num_condition_rep),
                                   constraint=torch.distributions.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau0_offset = pyro.sample(f"{name}_tau0_offset", dist.Normal(tau0_offset_locs[idx], tau0_offset_scales[idx]))
        k1_offset = pyro.sample(f"{name}_k1_offset", dist.Normal(k1_offset_locs[idx], k1_offset_scales[idx]))
        k2_offset = pyro.sample(f"{name}_k2_offset", dist.Normal(k2_offset_locs[idx], k2_offset_scales[idx]))

    tau0_per_condition = tau0_hyper_loc + tau0_offset * tau0_hyper_scale
    k1_per_condition = k1_hyper_loc + k1_offset * k1_hyper_scale
    k2_per_condition = k2_hyper_loc + k2_offset * k2_hyper_scale

    tau0_expanded = tau0_per_condition[data.map_condition_pre]
    k1_expanded = k1_per_condition[data.map_condition_pre]
    k2_expanded = k2_per_condition[data.map_condition_pre]

    tau = tau0_expanded + (k1_expanded / (theta + k2_expanded))

    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = torch.where(
        t_sel < tau,
        g_pre * t_sel,
        g_pre * tau + g_sel * (t_sel - tau)
    )

    return dln_cfu_pre + dln_cfu_sel


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "tau0_hyper_loc_loc": 1.0,
        "tau0_hyper_loc_scale": 0.5,
        "tau0_hyper_scale": 0.5,

        "k1_hyper_loc_loc": 0.0,
        "k1_hyper_loc_scale": 0.1,
        "k1_hyper_scale": 0.1,

        "k2_hyper_loc_loc": 0.1,
        "k2_hyper_loc_scale": 0.05,
        "k2_hyper_scale": 0.05,
    }

def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.
    """
    guesses = {
        f"{name}_tau0_hyper_loc": 1.0,
        f"{name}_tau0_hyper_scale": 0.1,
        f"{name}_k1_hyper_loc": 0.0,
        f"{name}_k1_hyper_scale": 0.01,
        f"{name}_k2_hyper_loc": 0.1,
        f"{name}_k2_hyper_scale": 0.01,
        f"{name}_tau0_offset": torch.zeros(data.num_condition_rep),
        f"{name}_k1_offset": torch.zeros(data.num_condition_rep),
        f"{name}_k2_offset": torch.zeros(data.num_condition_rep),
    }
    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
