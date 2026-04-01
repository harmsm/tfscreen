import torch
import pyro
import pyro.distributions as dist
from dataclasses import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any

@dataclass(frozen=True)
class ModelPriors:
    """
    Holds hyperparameters for the Baranyi growth transition model.
    """
    tau_lag_hyper_loc_loc: float
    tau_lag_hyper_loc_scale: float
    tau_lag_hyper_scale: float

    k_sharp_hyper_loc_loc: float
    k_sharp_hyper_loc_scale: float
    k_sharp_hyper_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: torch.Tensor,
                 g_sel: torch.Tensor,
                 t_pre: torch.Tensor,
                 t_sel: torch.Tensor,
                 theta: torch.Tensor = None) -> torch.Tensor:
    """
    Combines the pre-selection and selection growth phases with a Baranyi-style
    transition using an integrated sigmoid.

    integrated_sigmoid = (logaddexp(0, k_sharp*(t_sel - tau_lag)) -
                          logaddexp(0, -k_sharp*tau_lag)) / k_sharp

    dln_cfu_sel = g_pre*t_sel + (g_sel - g_pre)*integrated_sigmoid

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
    theta : torch.Tensor, optional
        Fractional occupancy tensor (unused in this model).

    Returns
    -------
    total_growth : torch.Tensor
        The total growth over both phases.
    """

    # Hierarchical tau_lag
    tau_lag_hyper_loc = pyro.sample(
        f"{name}_tau_lag_hyper_loc",
        dist.Normal(priors.tau_lag_hyper_loc_loc, priors.tau_lag_hyper_loc_scale)
    )
    tau_lag_hyper_scale = pyro.sample(
        f"{name}_tau_lag_hyper_scale",
        dist.HalfNormal(priors.tau_lag_hyper_scale)
    )

    # Hierarchical k_sharp
    k_sharp_hyper_loc = pyro.sample(
        f"{name}_k_sharp_hyper_loc",
        dist.Normal(priors.k_sharp_hyper_loc_loc, priors.k_sharp_hyper_loc_scale)
    )
    k_sharp_hyper_scale = pyro.sample(
        f"{name}_k_sharp_hyper_scale",
        dist.HalfNormal(priors.k_sharp_hyper_scale)
    )

    # Plate over conditions
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep):
        tau_lag_offset = pyro.sample(f"{name}_tau_lag_offset", dist.Normal(0.0, 1.0))
        k_sharp_offset = pyro.sample(f"{name}_k_sharp_offset", dist.Normal(0.0, 1.0))

    tau_lag_per_condition = tau_lag_hyper_loc + tau_lag_offset * tau_lag_hyper_scale
    k_sharp_per_condition = torch.exp(k_sharp_hyper_loc + k_sharp_offset * k_sharp_hyper_scale)

    # Register deterministic sites
    pyro.deterministic(f"{name}_tau_lag", tau_lag_per_condition)
    pyro.deterministic(f"{name}_k_sharp", k_sharp_per_condition)

    # Expand to match g_pre shape
    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    # Calculate transition components
    term1 = torch.logaddexp(torch.tensor(0.0), k_sharp * (t_sel - tau_lag))
    term0 = torch.logaddexp(torch.tensor(0.0), -k_sharp * tau_lag)
    integrated_sigmoid = (term1 - term0) / k_sharp

    # dln_cfu_sel = g_pre*t_sel + (g_sel - g_pre)*integrated_sigmoid
    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = g_pre * t_sel + (g_sel - g_pre) * integrated_sigmoid

    return dln_cfu_pre + dln_cfu_sel


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: torch.Tensor,
          g_sel: torch.Tensor,
          t_pre: torch.Tensor,
          t_sel: torch.Tensor,
          theta: torch.Tensor = None) -> torch.Tensor:
    """
    Guide corresponding to the Baranyi growth transition model.
    """

    # tau_lag hyper
    tau_lag_loc_loc = pyro.param(f"{name}_tau_lag_hyper_loc_loc", torch.tensor(priors.tau_lag_hyper_loc_loc))
    tau_lag_loc_scale = pyro.param(f"{name}_tau_lag_hyper_loc_scale", torch.tensor(priors.tau_lag_hyper_loc_scale),
                                   constraint=torch.distributions.constraints.positive)
    tau_lag_hyper_loc = pyro.sample(f"{name}_tau_lag_hyper_loc", dist.Normal(tau_lag_loc_loc, tau_lag_loc_scale))

    tau_lag_scale_loc = pyro.param(f"{name}_tau_lag_hyper_scale_loc", torch.tensor(-1.0))
    tau_lag_scale_scale = pyro.param(f"{name}_tau_lag_hyper_scale_scale", torch.tensor(0.1),
                                     constraint=torch.distributions.constraints.positive)
    tau_lag_hyper_scale = pyro.sample(f"{name}_tau_lag_hyper_scale", dist.LogNormal(tau_lag_scale_loc, tau_lag_scale_scale))

    # k_sharp hyper
    k_sharp_loc_loc = pyro.param(f"{name}_k_sharp_hyper_loc_loc", torch.tensor(priors.k_sharp_hyper_loc_loc))
    k_sharp_loc_scale = pyro.param(f"{name}_k_sharp_hyper_loc_scale", torch.tensor(priors.k_sharp_hyper_loc_scale),
                                   constraint=torch.distributions.constraints.positive)
    k_sharp_hyper_loc = pyro.sample(f"{name}_k_sharp_hyper_loc", dist.Normal(k_sharp_loc_loc, k_sharp_loc_scale))

    k_sharp_scale_loc = pyro.param(f"{name}_k_sharp_hyper_scale_loc", torch.tensor(-1.0))
    k_sharp_scale_scale = pyro.param(f"{name}_k_sharp_hyper_scale_scale", torch.tensor(0.1),
                                     constraint=torch.distributions.constraints.positive)
    k_sharp_hyper_scale = pyro.sample(f"{name}_k_sharp_hyper_scale", dist.LogNormal(k_sharp_scale_loc, k_sharp_scale_scale))

    # Offsets
    tau_lag_offset_locs = pyro.param(f"{name}_tau_lag_offset_locs", torch.zeros(data.num_condition_rep))
    tau_lag_offset_scales = pyro.param(f"{name}_tau_lag_offset_scales", torch.ones(data.num_condition_rep),
                                       constraint=torch.distributions.constraints.positive)

    k_sharp_offset_locs = pyro.param(f"{name}_k_sharp_offset_locs", torch.zeros(data.num_condition_rep))
    k_sharp_offset_scales = pyro.param(f"{name}_k_sharp_offset_scales", torch.ones(data.num_condition_rep),
                                       constraint=torch.distributions.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau_lag_offset = pyro.sample(f"{name}_tau_lag_offset",
                                     dist.Normal(tau_lag_offset_locs[idx], tau_lag_offset_scales[idx]))
        k_sharp_offset = pyro.sample(f"{name}_k_sharp_offset",
                                     dist.Normal(k_sharp_offset_locs[idx], k_sharp_offset_scales[idx]))

    tau_lag_per_condition = tau_lag_hyper_loc + tau_lag_offset * tau_lag_hyper_scale
    k_sharp_per_condition = torch.exp(k_sharp_hyper_loc + k_sharp_offset * k_sharp_hyper_scale)

    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    term1 = torch.logaddexp(torch.tensor(0.0), k_sharp * (t_sel - tau_lag))
    term0 = torch.logaddexp(torch.tensor(0.0), -k_sharp * tau_lag)
    integrated_sigmoid = (term1 - term0) / k_sharp

    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = g_pre * t_sel + (g_sel - g_pre) * integrated_sigmoid

    return dln_cfu_pre + dln_cfu_sel


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "tau_lag_hyper_loc_loc": 1.0,
        "tau_lag_hyper_loc_scale": 0.5,
        "tau_lag_hyper_scale": 0.5,

        # k_sharp is modeled in log-space: exp(1.0) approx 2.7
        "k_sharp_hyper_loc_loc": 1.0,
        "k_sharp_hyper_loc_scale": 1.0,
        "k_sharp_hyper_scale": 1.0,
    }

def get_guesses(name: str, data: GrowthData) -> Dict[str, torch.Tensor]:
    """
    Get guess values for the model's latent parameters.
    """
    guesses = {
        f"{name}_tau_lag_hyper_loc": 1.0,
        f"{name}_tau_lag_hyper_scale": 0.1,
        f"{name}_k_sharp_hyper_loc": 1.0,
        f"{name}_k_sharp_hyper_scale": 0.1,
        f"{name}_tau_lag_offset": torch.zeros(data.num_condition_rep),
        f"{name}_k_sharp_offset": torch.zeros(data.num_condition_rep),
    }
    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
