import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from typing import Dict, Any


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding prior parameters for the two_pop growth transition model.

    Attributes
    ----------
    ln_k_trans_loc, ln_k_trans_scale : float
        Normal prior parameters for log(k_trans).
        k_trans = exp(Normal(...)) enforces k_trans > 0.
        k_trans is the rate at which cells transition from pre-selection to
        selection growth mode.

    Notes
    -----
    The formula is valid only when D = g_pre - g_sel - k_trans > 0, i.e.
    k_trans < g_pre - g_sel.  When D <= 0 the function falls back to the
    k_trans -> 0 limit (growth at the pre-selection rate) so that the forward
    pass and all gradients remain finite during MAP/SVI optimisation.
    """
    ln_k_trans_loc: float
    ln_k_trans_scale: float


def _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans):
    """
    Compute total growth for the two-population transition model.

    The model tracks two sub-populations: cells growing at rate g_pre that
    transition irreversibly at rate k_trans to cells growing at rate g_sel.
    The total count is:

        N_tot(t) = N_0 * [k_trans * exp(g_sel * t)
                          + D * exp((g_pre - k_trans) * t)] / D

    where D = g_pre - g_sel - k_trans, so:

        dln_cfu_sel = ln(N_tot / N_0)
                    = ln[k_trans*exp(a) + D*exp(b)] - ln(D)

    with a = g_sel*t_sel, b = (g_pre - k_trans)*t_sel.

    For numerical stability the two-term sum is evaluated by factoring out
    exp(max(a, b)) before taking the log.

    Limit k_trans -> 0: dln_cfu_sel -> g_pre * t_sel (no transition, pre rate).

    Parameters
    ----------
    g_pre, g_sel : jnp.ndarray
        Pre- and selection-phase growth rate tensors.
    t_pre, t_sel : jnp.ndarray
        Pre- and selection-phase time tensors.
    k_trans : jnp.ndarray
        Transition rate from pre- to selection-phase growth mode.
        Must satisfy k_trans < g_pre - g_sel (D > 0) for valid output.

    The formula is valid only when D = g_pre - g_sel - k_trans > 0.  When D <= 0
    (either because g_pre <= g_sel or because k_trans is too large) the function
    falls back to the k_trans -> 0 limit:
        dln_cfu_sel = g_pre * t_sel   (growth stays at the pre-selection rate)

    Implementation note — JAX NaN-gradient safety
    ----------------------------------------------
    ``jnp.where`` evaluates BOTH branches before selecting, so a naive
    implementation would still compute ``log(D)`` with D <= 0 in the invalid
    branch, producing NaN that propagates through the backward pass via
    ``NaN * 0 = NaN``.

    The fix uses ``safe_D``: a version of D where invalid values are replaced
    with 1.0 (an arbitrary positive placeholder) *before* any log is taken.
    ``safe_D`` is used in both the numerator and denominator so that
    ``log(scaled_num)`` and ``log(safe_D)`` are always finite regardless of
    which branch is selected.  The two-pop branch result is discarded by the
    final ``jnp.where`` when D <= 0.

    Returns
    -------
    jnp.ndarray
        Total growth: dln_cfu_pre + dln_cfu_sel.
    """
    D = g_pre - g_sel - k_trans
    valid = D > 0

    # Substitute safe_D = 1.0 wherever D <= 0.  This keeps log(safe_D) and
    # log(scaled_num) finite in the invalid branch so that JAX gradients
    # never encounter NaN * 0.  The two-pop result is discarded by the final
    # jnp.where when valid is False.
    safe_D = jnp.where(valid, D, jnp.ones_like(D))

    a = g_sel * t_sel                        # exponent of the k_trans term
    b = (g_pre - k_trans) * t_sel            # exponent of the D term

    # Numerator = k_trans*exp(a) + safe_D*exp(b).
    # Factor out exp(m) where m = max(a, b) for numerical stability.
    # Using safe_D in the numerator keeps scaled_num > 0 when valid=False.
    m = jnp.maximum(a, b)
    scaled_num = k_trans * jnp.exp(a - m) + safe_D * jnp.exp(b - m)

    two_pop_sel = m + jnp.log(scaled_num) - jnp.log(safe_D)
    fallback_sel = g_pre * t_sel   # k_trans -> 0 limit: no transition

    dln_cfu_sel = jnp.where(valid, two_pop_sel, fallback_sel)
    dln_cfu_pre = g_pre * t_pre

    return dln_cfu_pre + dln_cfu_sel


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors,
                 g_pre: jnp.ndarray,
                 g_sel: jnp.ndarray,
                 t_pre: jnp.ndarray,
                 t_sel: jnp.ndarray,
                 theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Combines pre-selection and selection growth using a two-population ODE model.

    Two sub-populations start at t_sel=0 all in the pre-selection growth mode
    (rate g_pre).  Cells transition irreversibly to selection growth mode (rate
    g_sel) at rate k_trans.  The total ln-population change during selection is:

        dln_cfu_sel = ln[k_trans*exp(g_sel*t) + D*exp((g_pre-k_trans)*t)] - ln(D)

    where D = g_pre - g_sel - k_trans.  Valid when D > 0.

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
        ln_k_trans = pyro.sample(
            f"{name}_ln_k_trans",
            dist.Normal(priors.ln_k_trans_loc, priors.ln_k_trans_scale)
        )

    k_trans_per_condition = jnp.exp(ln_k_trans)
    k_trans = k_trans_per_condition[data.map_condition_pre]

    return _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans)


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors,
          g_pre: jnp.ndarray,
          g_sel: jnp.ndarray,
          t_pre: jnp.ndarray,
          t_sel: jnp.ndarray,
          theta: jnp.ndarray = None) -> jnp.ndarray:
    """
    Guide for the two_pop growth transition model with simple Normal variational posteriors.
    """
    _DEFAULT_SCALE = 0.1

    ln_k_trans_locs = pyro.param(
        f"{name}_ln_k_trans_locs",
        jnp.full(data.num_condition_rep, priors.ln_k_trans_loc)
    )
    ln_k_trans_scales = pyro.param(
        f"{name}_ln_k_trans_scales",
        jnp.full(data.num_condition_rep, _DEFAULT_SCALE),
        constraint=dist.constraints.positive
    )

    with pyro.plate(f"{name}_condition_parameters", data.num_condition_rep) as idx:
        ln_k_trans = pyro.sample(
            f"{name}_ln_k_trans",
            dist.Normal(ln_k_trans_locs[..., idx], ln_k_trans_scales[..., idx])
        )

    k_trans_per_condition = jnp.exp(ln_k_trans)
    k_trans = k_trans_per_condition[data.map_condition_pre]

    return _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans)


def get_hyperparameters() -> Dict[str, Any]:
    """
    Get default values for the model hyperparameters.
    """
    return {
        "ln_k_trans_loc": -15.0,   # k_trans = exp(-2) ≈ 0.14; small relative to typical growth rates
        "ln_k_trans_scale": 5.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, jnp.ndarray]:
    """
    Get guess values for the model's latent parameters.
    """
    num_cond_rep = data.num_condition_rep
    _DEFAULT_SCALE = 5.0

    return {
        f"{name}_ln_k_trans_locs": jnp.full(num_cond_rep, -15.0),
        f"{name}_ln_k_trans_scales": jnp.full(num_cond_rep, _DEFAULT_SCALE),
    }


def get_priors() -> ModelPriors:
    """
    Utility function to create a populated ModelPriors object.
    """
    return ModelPriors(**get_hyperparameters())
