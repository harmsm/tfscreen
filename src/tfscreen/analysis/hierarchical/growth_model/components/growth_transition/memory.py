import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding prior parameters for the memory-dependent growth
    transition model.

    Attributes
    ----------
    tau0_loc, tau0_scale : float
        Normal prior parameters for per-condition baseline lag time tau0.
    k1_loc, k1_scale : float
        Normal prior parameters for per-condition memory coefficient k1.
    k2_loc, k2_scale : float
        Normal prior parameters for per-condition memory offset k2.
    """
    tau0_loc: float
    tau0_scale: float
    k1_loc: float
    k1_scale: float
    k2_loc: float
    k2_scale: float


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: jnp.ndarray,
                 g_sel: jnp.ndarray,
                 t_pre: jnp.ndarray,
                 t_sel: jnp.ndarray,
                 theta: jnp.ndarray) -> jnp.ndarray:
    """
    Combines the pre-selection and selection growth phases with a transition
    that depends on theta (memory effect).

    tau = tau0 + (k1 / (theta + k2))

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
    theta : jnp.ndarray
        Fractional occupancy tensor.

    Returns
    -------
    total_growth : jnp.ndarray
        The total growth over both phases.
    """
    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep):
        tau0_per_condition = pyro.sample(
            f"{name}_tau0", dist.Normal(priors.tau0_loc, priors.tau0_scale)
        )
        k1_per_condition = pyro.sample(
            f"{name}_k1", dist.Normal(priors.k1_loc, priors.k1_scale)
        )
        k2_per_condition = pyro.sample(
            f"{name}_k2", dist.Normal(priors.k2_loc, priors.k2_scale)
        )

    tau0_expanded = tau0_per_condition[data.map_condition_pre]
    k1_expanded = k1_per_condition[data.map_condition_pre]
    k2_expanded = k2_per_condition[data.map_condition_pre]

    # Clamp denominator away from zero so neither the forward pass nor the
    # backward pass produces inf/NaN.  jnp.where evaluates both branches in
    # the backward pass, so a bare division by near-zero produces inf
    # gradients even when the condition masks it out.
    denom = theta + k2_expanded
    safe_denom = jnp.where(jnp.abs(denom) > 1e-6, denom, jnp.sign(denom + 1e-30) * 1e-6)
    tau = tau0_expanded + (k1_expanded / safe_denom)

    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = jnp.where(
        t_sel < tau,
        g_pre * t_sel,
        g_pre * tau + g_sel * (t_sel - tau)
    )

    return dln_cfu_pre + dln_cfu_sel


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray,
          theta: jnp.ndarray) -> jnp.ndarray:
    """
    Guide for the memory-dependent growth transition model with simple Normal priors.
    """
    tau0_locs = pyro.param(f"{name}_tau0_locs",
                           jnp.full(data.num_condition_rep, priors.tau0_loc))
    tau0_scales = pyro.param(f"{name}_tau0_scales",
                             jnp.full(data.num_condition_rep, priors.tau0_scale),
                             constraint=dist.constraints.positive)
    k1_locs = pyro.param(f"{name}_k1_locs",
                         jnp.full(data.num_condition_rep, priors.k1_loc))
    k1_scales = pyro.param(f"{name}_k1_scales",
                           jnp.full(data.num_condition_rep, priors.k1_scale),
                           constraint=dist.constraints.positive)
    k2_locs = pyro.param(f"{name}_k2_locs",
                         jnp.full(data.num_condition_rep, priors.k2_loc))
    k2_scales = pyro.param(f"{name}_k2_scales",
                           jnp.full(data.num_condition_rep, priors.k2_scale),
                           constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        tau0_per_condition = pyro.sample(
            f"{name}_tau0",
            dist.Normal(tau0_locs[..., idx], tau0_scales[..., idx])
        )
        k1_per_condition = pyro.sample(
            f"{name}_k1",
            dist.Normal(k1_locs[..., idx], k1_scales[..., idx])
        )
        k2_per_condition = pyro.sample(
            f"{name}_k2",
            dist.Normal(k2_locs[..., idx], k2_scales[..., idx])
        )

    tau0_expanded = tau0_per_condition[data.map_condition_pre]
    k1_expanded = k1_per_condition[data.map_condition_pre]
    k2_expanded = k2_per_condition[data.map_condition_pre]

    denom = theta + k2_expanded
    safe_denom = jnp.where(jnp.abs(denom) > 1e-6, denom, jnp.sign(denom + 1e-30) * 1e-6)
    tau = tau0_expanded + (k1_expanded / safe_denom)

    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = jnp.where(
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
        "tau0_loc": 100.0, # takes about 100 minutes to turn over
        "tau0_scale": 50,
        "k1_loc": 1.0,
        "k1_scale": 10, # no strong constraints on k_2
        "k2_loc": 1,
        "k2_scale": 10, # no strong constraints on k2
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 1.0

    return {
        f"{name}_tau0_locs": jnp.full(num_cond_rep, 100),
        f"{name}_tau0_scales": jnp.full(num_cond_rep, 100),
        f"{name}_k1_locs": jnp.full(num_cond_rep,1.0),
        f"{name}_k1_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
        f"{name}_k2_locs": jnp.full(num_cond_rep, 1.0),
        f"{name}_k2_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
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
        params_to_get=["growth_transition_tau0",
                       "growth_transition_k1",
                       "growth_transition_k2"],
        map_column="map_condition_rep",
        get_columns=cond_rep_cols,
        in_run_prefix="",
    )]
