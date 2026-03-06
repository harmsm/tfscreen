import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding hyperparameters for the Baranyi growth transition model.
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
                 g_pre: jnp.ndarray,
                 g_sel: jnp.ndarray,
                 t_pre: jnp.ndarray,
                 t_sel: jnp.ndarray,
                 theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Combines the pre-selection and selection growth phases with a Baranyi-style
    transition using an integrated sigmoid.

    integrated_sigmoid = (logaddexp(0, k_sharp*(t_sel - tau_lag)) - 
                          logaddexp(0, -k_sharp*tau_lag)) / k_sharp
    
    dln_cfu_sel = g_pre*t_sel + (g_sel - g_pre)*integrated_sigmoid

    Parameters
    ----------
    name : str
        The prefix for Numpyro sample/deterministic sites in this component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the hyperparameters for the priors.
    g_pre : jnp.ndarray
        Pre-selection growth rate tensor.
    g_sel : jnp.ndarray
        Selection growth rate tensor.
    t_pre : jnp.ndarray
        Pre-selection time tensor.
    t_sel : jnp.ndarray
        Selection time tensor.
    theta : jnp.ndarray, optional
        Fractional occupancy tensor (unused in this model).

    Returns
    -------
    total_growth : jnp.ndarray
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
    k_sharp_per_condition = jnp.exp(k_sharp_hyper_loc + k_sharp_offset * k_sharp_hyper_scale)

    # Register deterministic sites
    pyro.deterministic(f"{name}_tau_lag", tau_lag_per_condition)
    pyro.deterministic(f"{name}_k_sharp", k_sharp_per_condition)

    # Expand to match g_pre shape
    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    # Calculate transition components
    term1 = jnp.logaddexp(0.0, k_sharp * (t_sel - tau_lag))
    term0 = jnp.logaddexp(0.0, -k_sharp * tau_lag)
    integrated_sigmoid = (term1 - term0) / k_sharp

    # dln_cfu_sel = g_pre*t_sel + (g_sel - g_pre)*integrated_sigmoid
    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = g_pre * t_sel + (g_sel - g_pre) * integrated_sigmoid

    return dln_cfu_pre + dln_cfu_sel


def guide(name: str, 
          data: GrowthData, 
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray,
          theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Guide corresponding to the Baranyi growth transition model.
    """

    # tau_lag hyper
    tau_lag_loc_loc = pyro.param(f"{name}_tau_lag_hyper_loc_loc", jnp.array(priors.tau_lag_hyper_loc_loc))
    tau_lag_loc_scale = pyro.param(f"{name}_tau_lag_hyper_loc_scale", jnp.array(priors.tau_lag_hyper_loc_scale),
                                constraint=dist.constraints.positive)
    tau_lag_hyper_loc = pyro.sample(f"{name}_tau_lag_hyper_loc", dist.Normal(tau_lag_loc_loc, tau_lag_loc_scale))

    tau_lag_scale_loc = pyro.param(f"{name}_tau_lag_hyper_scale_loc", jnp.array(-1.0))
    tau_lag_scale_scale = pyro.param(f"{name}_tau_lag_hyper_scale_scale", jnp.array(0.1),
                                  constraint=dist.constraints.positive)
    tau_lag_hyper_scale = pyro.sample(f"{name}_tau_lag_hyper_scale", dist.LogNormal(tau_lag_scale_loc, tau_lag_scale_scale))

    # k_sharp hyper
    k_sharp_loc_loc = pyro.param(f"{name}_k_sharp_hyper_loc_loc", jnp.array(priors.k_sharp_hyper_loc_loc))
    k_sharp_loc_scale = pyro.param(f"{name}_k_sharp_hyper_loc_scale", jnp.array(priors.k_sharp_hyper_loc_scale),
                                constraint=dist.constraints.positive)
    k_sharp_hyper_loc = pyro.sample(f"{name}_k_sharp_hyper_loc", dist.Normal(k_sharp_loc_loc, k_sharp_loc_scale))

    k_sharp_scale_loc = pyro.param(f"{name}_k_sharp_hyper_scale_loc", jnp.array(-1.0))
    k_sharp_scale_scale = pyro.param(f"{name}_k_sharp_hyper_scale_scale", jnp.array(0.1),
                                 constraint=dist.constraints.positive)
    k_sharp_hyper_scale = pyro.sample(f"{name}_k_sharp_hyper_scale", dist.LogNormal(k_sharp_scale_loc, k_sharp_scale_scale))

    # Offsets
    tau_lag_offset_locs = pyro.param(f"{name}_tau_lag_offset_locs", jnp.zeros(data.num_condition_rep))
    tau_lag_offset_scales = pyro.param(f"{name}_tau_lag_offset_scales", jnp.ones(data.num_condition_rep),
                                    constraint=dist.constraints.positive)
    
    k_sharp_offset_locs = pyro.param(f"{name}_k_sharp_offset_locs", jnp.zeros(data.num_condition_rep))
    k_sharp_offset_scales = pyro.param(f"{name}_k_sharp_offset_scales", jnp.ones(data.num_condition_rep),
                                   constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau_lag_offset = pyro.sample(f"{name}_tau_lag_offset", dist.Normal(tau_lag_offset_locs[idx], tau_lag_offset_scales[idx]))
        k_sharp_offset = pyro.sample(f"{name}_k_sharp_offset", dist.Normal(k_sharp_offset_locs[idx], k_sharp_offset_scales[idx]))

    tau_lag_per_condition = tau_lag_hyper_loc + tau_lag_offset * tau_lag_hyper_scale
    k_sharp_per_condition = jnp.exp(k_sharp_hyper_loc + k_sharp_offset * k_sharp_hyper_scale)

    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    term1 = jnp.logaddexp(0.0, k_sharp * (t_sel - tau_lag))
    term0 = jnp.logaddexp(0.0, -k_sharp * tau_lag)
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

def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    guesses = {
        f"{name}_tau_lag_hyper_loc": 1.0,
        f"{name}_tau_lag_hyper_scale": 0.1,
        f"{name}_k_sharp_hyper_loc": 1.0,
        f"{name}_k_sharp_hyper_scale": 0.1,
        f"{name}_tau_lag_offset": jnp.zeros(data.num_condition_rep),
        f"{name}_k_sharp_offset": jnp.zeros(data.num_condition_rep),
    }
    return guesses

def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
