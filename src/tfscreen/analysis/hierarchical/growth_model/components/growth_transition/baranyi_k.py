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
    JAX Pytree holding prior parameters for the baranyi_k growth transition model.

    Attributes
    ----------
    tau_loc, tau_scale : float
        Normal prior parameters for per-condition transition midpoint tau.
    ln_k0_loc, ln_k0_scale : float
        Normal prior parameters for log(k0). k0 = exp(Normal(...)) enforces k0 > 0.
        k0 is the base sigmoid sharpness when growth rates are identical.
    ln_gamma_loc, ln_gamma_scale : float
        Normal prior parameters for log(gamma). gamma = exp(Normal(...)) enforces gamma > 0.
        gamma scales how much the rate difference suppresses transition sharpness:
        k = k0 / (1 + gamma * |g_sel - g_pre|)
    """
    tau_loc: float
    tau_scale: float
    ln_k0_loc: float
    ln_k0_scale: float
    ln_gamma_loc: float
    ln_gamma_scale: float


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
    sigmoid transition whose sharpness is modulated by the growth rate difference.

    The instantaneous rate during selection is:
        r(t) = g_pre + (g_sel - g_pre) · expit(k · (t - tau))

    with k = k0 / (1 + gamma · |g_sel - g_pre|), so larger rate differences
    produce a slower (more inertial) transition.

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
        tau_per_condition = pyro.sample(
            f"{name}_tau", dist.Normal(priors.tau_loc, priors.tau_scale)
        )
        ln_k0 = pyro.sample(
            f"{name}_ln_k0", dist.Normal(priors.ln_k0_loc, priors.ln_k0_scale)
        )
        ln_gamma = pyro.sample(
            f"{name}_ln_gamma", dist.Normal(priors.ln_gamma_loc, priors.ln_gamma_scale)
        )

    k0_per_condition = jnp.exp(ln_k0)
    gamma_per_condition = jnp.exp(ln_gamma)

    tau = tau_per_condition[data.map_condition_pre]
    k0 = k0_per_condition[data.map_condition_pre]
    gamma = gamma_per_condition[data.map_condition_pre]

    delta_g = jnp.abs(g_sel - g_pre)
    k = k0 / (1.0 + gamma * delta_g)

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
    Guide for the baranyi_k growth transition model with simple Normal variational posteriors.
    """
    _DEFAULT_SCALE = 0.1

    tau_locs = pyro.param(f"{name}_tau_locs",
                          jnp.full(data.num_condition_rep, priors.tau_loc))
    tau_scales = pyro.param(f"{name}_tau_scales",
                            jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                            constraint=dist.constraints.positive)
    ln_k0_locs = pyro.param(f"{name}_ln_k0_locs",
                            jnp.full(data.num_condition_rep, priors.ln_k0_loc))
    ln_k0_scales = pyro.param(f"{name}_ln_k0_scales",
                              jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                              constraint=dist.constraints.positive)
    ln_gamma_locs = pyro.param(f"{name}_ln_gamma_locs",
                               jnp.full(data.num_condition_rep, priors.ln_gamma_loc))
    ln_gamma_scales = pyro.param(f"{name}_ln_gamma_scales",
                                 jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
                                 constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau_per_condition = pyro.sample(
            f"{name}_tau",
            dist.Normal(tau_locs[..., idx], tau_scales[..., idx])
        )
        ln_k0 = pyro.sample(
            f"{name}_ln_k0",
            dist.Normal(ln_k0_locs[..., idx], ln_k0_scales[..., idx])
        )
        ln_gamma = pyro.sample(
            f"{name}_ln_gamma",
            dist.Normal(ln_gamma_locs[..., idx], ln_gamma_scales[..., idx])
        )

    k0_per_condition = jnp.exp(ln_k0)
    gamma_per_condition = jnp.exp(ln_gamma)

    tau = tau_per_condition[data.map_condition_pre]
    k0 = k0_per_condition[data.map_condition_pre]
    gamma = gamma_per_condition[data.map_condition_pre]

    delta_g = jnp.abs(g_sel - g_pre)
    k = k0 / (1.0 + gamma * delta_g)

    return compute_growth(g_pre, g_sel, t_pre, t_sel, tau, k)


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "tau_loc": 100.0,
        "tau_scale": 100,
        "ln_k0_loc": 0.0,    # k0 = exp(0) = 1.0
        "ln_k0_scale": 1.0,
        "ln_gamma_loc": 0.0,  # gamma = exp(0) = 1.0
        "ln_gamma_scale": 1.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 1.0

    return {
        f"{name}_tau_locs": jnp.full(num_cond_rep, 100.0),
        f"{name}_tau_scales": jnp.full(num_cond_rep, 100),
        f"{name}_ln_k0_locs": jnp.full(num_cond_rep, 0.0),
        f"{name}_ln_k0_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
        f"{name}_ln_gamma_locs": jnp.full(num_cond_rep, 0.0),
        f"{name}_ln_gamma_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
    }


def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
