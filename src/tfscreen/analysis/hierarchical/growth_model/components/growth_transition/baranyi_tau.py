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
    JAX Pytree holding prior parameters for the baranyi_tau growth transition model.

    Attributes
    ----------
    tau_0_loc, tau_0_scale : float
        Normal prior parameters for per-condition base transition midpoint tau_0.
    ln_k0_loc, ln_k0_scale : float
        Normal prior parameters for log(k0). k0 = exp(Normal(...)) enforces k0 > 0.
        k0 scales how much the rate difference delays the transition midpoint:
        tau = tau_0 + k0 * |g_sel - g_pre|
    ln_k_loc, ln_k_scale : float
        Normal prior parameters for log(k). k = exp(Normal(...)) enforces k > 0.
        k is the fixed sigmoid sharpness (steepness of the transition).
    """
    tau_0_loc: float
    tau_0_scale: float
    ln_k0_loc: float
    ln_k0_scale: float
    ln_k_loc: float
    ln_k_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: jnp.ndarray,
                 g_sel: jnp.ndarray,
                 t_pre: jnp.ndarray,
                 t_sel: jnp.ndarray,
                 theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Combines pre-selection and selection growth phases using a Baranyi integrated-
    sigmoid transition whose midpoint is delayed by the growth rate difference.

    The instantaneous rate during selection is:
        r(t) = g_pre + (g_sel - g_pre) · expit(k · (t - tau))

    with tau = tau_0 + k0 · |g_sel - g_pre|, so larger rate differences push
    the transition midpoint later (more inertia to change growth strategy).

    Integrated from 0 to t_sel:
        integrated_sigmoid = (logaddexp(0, k·(t_sel - tau))
                              - logaddexp(0, -k·tau)) / k
        dln_cfu_sel = g_pre·t_sel + (g_sel - g_pre) · integrated_sigmoid

    Parameters
    ----------
    name : str
        Prefix for Numpyro sample/deterministic sites in this component.
    data : GrowthData
        Pytree containing experimental data and metadata.
    priors : ModelPriors
        Pytree containing prior parameters.
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
        tau_0_per_condition = pyro.sample(
            f"{name}_tau_0", dist.Normal(priors.tau_0_loc, priors.tau_0_scale)
        )
        ln_k0 = pyro.sample(
            f"{name}_ln_k0", dist.Normal(priors.ln_k0_loc, priors.ln_k0_scale)
        )
        ln_k = pyro.sample(
            f"{name}_ln_k", dist.Normal(priors.ln_k_loc, priors.ln_k_scale)
        )

    k0_per_condition = jnp.exp(ln_k0)
    k_per_condition = jnp.exp(ln_k)

    tau_0 = tau_0_per_condition[data.map_condition_pre]
    k0 = k0_per_condition[data.map_condition_pre]
    k = k_per_condition[data.map_condition_pre]

    delta_g = jnp.abs(g_sel - g_pre)
    tau = tau_0 + k0 * delta_g

    return compute_growth(g_pre, g_sel, t_pre, t_sel, tau, k)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray,
          theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Guide for the baranyi_tau growth transition model with simple Normal variational posteriors.
    """
    _DEFAULT_SCALE = 0.1

    tau_0_locs = pyro.param(f"{name}_tau_0_locs",
                            jnp.full(data.num_condition_rep, priors.tau_0_loc))
    tau_0_scales = pyro.param(f"{name}_tau_0_scales",
                              jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                              constraint=dist.constraints.positive)
    ln_k0_locs = pyro.param(f"{name}_ln_k0_locs",
                            jnp.full(data.num_condition_rep, priors.ln_k0_loc))
    ln_k0_scales = pyro.param(f"{name}_ln_k0_scales",
                              jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                              constraint=dist.constraints.positive)
    ln_k_locs = pyro.param(f"{name}_ln_k_locs",
                           jnp.full(data.num_condition_rep, priors.ln_k_loc))
    ln_k_scales = pyro.param(f"{name}_ln_k_scales",
                             jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                             constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau_0_per_condition = pyro.sample(
            f"{name}_tau_0",
            dist.Normal(tau_0_locs[..., idx], tau_0_scales[..., idx])
        )
        ln_k0 = pyro.sample(
            f"{name}_ln_k0",
            dist.Normal(ln_k0_locs[..., idx], ln_k0_scales[..., idx])
        )
        ln_k = pyro.sample(
            f"{name}_ln_k",
            dist.Normal(ln_k_locs[..., idx], ln_k_scales[..., idx])
        )

    k0_per_condition = jnp.exp(ln_k0)
    k_per_condition = jnp.exp(ln_k)

    tau_0 = tau_0_per_condition[data.map_condition_pre]
    k0 = k0_per_condition[data.map_condition_pre]
    k = k_per_condition[data.map_condition_pre]

    delta_g = jnp.abs(g_sel - g_pre)
    tau = tau_0 + k0 * delta_g

    return compute_growth(g_pre, g_sel, t_pre, t_sel, tau, k)


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "tau_0_loc": 100.0,
        "tau_0_scale": 100.0,
        "ln_k0_loc": 0.0,   # k0 = exp(0) = 1.0
        "ln_k0_scale": 1.0,
        "ln_k_loc": 0.0,    # k  = exp(0) = 1.0
        "ln_k_scale": 1.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 1.0

    return {
        f"{name}_tau_0_locs": jnp.full(num_cond_rep, 100.0),
        f"{name}_tau_0_scales": jnp.full(num_cond_rep, 100),
        f"{name}_ln_k0_locs": jnp.full(num_cond_rep, 0.0),
        f"{name}_ln_k0_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
        f"{name}_ln_k_locs": jnp.full(num_cond_rep, 0.0),
        f"{name}_ln_k_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
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
        params_to_get=["growth_transition_tau_0",
                       "growth_transition_ln_k0",
                       "growth_transition_ln_k"],
        map_column="map_condition_rep",
        get_columns=cond_rep_cols,
        in_run_prefix="",
    )]
