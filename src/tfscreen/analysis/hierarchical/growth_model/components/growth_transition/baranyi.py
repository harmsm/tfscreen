import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition._baranyi import compute_growth
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding prior parameters for the Baranyi growth transition model.

    Attributes
    ----------
    tau_lag_loc, tau_lag_scale : float
        Normal prior parameters for per-condition lag time tau_lag.
    k_sharp_loc, k_sharp_scale : float
        Normal prior parameters for log(k_sharp). The sharpness parameter
        k_sharp = exp(Normal(k_sharp_loc, k_sharp_scale)), enforcing k_sharp > 0.
    """
    tau_lag_loc: float
    tau_lag_scale: float
    k_sharp_loc: float
    k_sharp_scale: float


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

    The instantaneous rate during selection is:
        r(t) = g_pre + (g_sel - g_pre) · expit(k_sharp · (t - tau_lag))

    Integrated from 0 to t_sel:
        integrated_sigmoid = (logaddexp(0, k_sharp·(t_sel - tau_lag))
                              - logaddexp(0, -k_sharp·tau_lag)) / k_sharp

        dln_cfu_sel = g_pre·t_sel + (g_sel - g_pre) · integrated_sigmoid

    Parameters
    ----------
    name : str
        The prefix for Numpyro sample/deterministic sites in this component.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and metadata.
    priors : ModelPriors
        A Pytree (Flax dataclass) containing the prior parameters.
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
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep):
        tau_lag_per_condition = pyro.sample(
            f"{name}_tau_lag", dist.Normal(priors.tau_lag_loc, priors.tau_lag_scale)
        )
        ln_k_sharp = pyro.sample(
            f"{name}_k_sharp", dist.Normal(priors.k_sharp_loc, priors.k_sharp_scale)
        )
    k_sharp_per_condition = jnp.exp(ln_k_sharp)

    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    return compute_growth(g_pre, g_sel, t_pre, t_sel, tau_lag, k_sharp)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray,
          theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Guide for the Baranyi growth transition model with simple Normal priors.
    """
    tau_lag_locs = pyro.param(f"{name}_tau_lag_locs",
                              jnp.full(data.num_condition_rep, priors.tau_lag_loc))
    tau_lag_scales = pyro.param(f"{name}_tau_lag_scales",
                                jnp.full(data.num_condition_rep, priors.tau_lag_scale),
                                constraint=dist.constraints.positive)
    k_sharp_locs = pyro.param(f"{name}_k_sharp_locs",
                              jnp.full(data.num_condition_rep, priors.k_sharp_loc))
    k_sharp_scales = pyro.param(f"{name}_k_sharp_scales",
                                jnp.full(data.num_condition_rep, priors.k_sharp_scale),
                                constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau_lag_per_condition = pyro.sample(
            f"{name}_tau_lag",
            dist.Normal(tau_lag_locs[..., idx], tau_lag_scales[..., idx])
        )
        ln_k_sharp = pyro.sample(
            f"{name}_k_sharp",
            dist.Normal(k_sharp_locs[..., idx], k_sharp_scales[..., idx])
        )
    k_sharp_per_condition = jnp.exp(ln_k_sharp)

    tau_lag = tau_lag_per_condition[data.map_condition_pre]
    k_sharp = k_sharp_per_condition[data.map_condition_pre]

    return compute_growth(g_pre, g_sel, t_pre, t_sel, tau_lag, k_sharp)


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "tau_lag_loc": 100.0, # 100 minutes
        "tau_lag_scale": 100,
        "k_sharp_loc": 1.0,    
        "k_sharp_scale": 1.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 1.0

    return {
        f"{name}_tau_lag_locs": jnp.full(num_cond_rep, 100.0),
        f"{name}_tau_lag_scales": jnp.full(num_cond_rep, 100),
        f"{name}_k_sharp_locs": jnp.full(num_cond_rep, 1.0),
        f"{name}_k_sharp_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
    }


def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    cond_rep_cols = (["condition_rep"] if ctx.growth_shares_replicates
                     else ["replicate", "condition_rep"])
    return [dict(
        input_df=ctx.growth_tm.map_groups["condition_rep"],
        params_to_get=["growth_transition_tau_lag",
                       "growth_transition_k_sharp"],
        map_column="map_condition_rep",
        get_columns=cond_rep_cols,
        in_run_prefix="",
    )]
