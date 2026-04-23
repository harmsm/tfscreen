"""
Shared thermodynamic functions for the lac-dimer partition-function θ model.

Four-state partition function for the TF homodimer:

    State      Weight
    H          1                               (apo, high-affinity conformation)
    H·op       K_op · [op_free]                (operator-bound, repressing state)
    L          K_HL                            (low-affinity conformation)
    L·E²       K_HL · K_E · [E_free]²          (effector-bound, n=2 hardcoded)

    Z     = 1 + K_op·[op_free] + K_HL + K_HL·K_E·[E_free]²
    θ     = K_op·[op_free] / Z

Units: all concentrations in nM; K_op in nM⁻¹; K_E in nM⁻²; K_HL dimensionless.

Operator depletion approximation
---------------------------------
[TF_total] ≈ 650 nM, [op_total] ≈ 25 nM
(Sochor, PeerJ 2014, https://doi.org/10.7717/peerj.498),
giving [TF]/[op] ≈ 26.  Because the TF is in large excess we treat
[op_free] ≈ [op_total] = const.

Free effector (Newton solve)
----------------------------
Free-effector concentration satisfies the cubic mass balance:

    a·x³ + a·(2·[TF_total] − [E_total])·x² + Z₀·x − Z₀·[E_total] = 0

where  a  = K_HL·K_E (shape T×G)  and  Z₀ = 1 + K_op·[op_total] + K_HL (shape G).
Eight Newton iterations from x₀ = [E_total] converge for all
biophysically plausible parameter combinations.

This module is imported by all K-assembly variants
(lnK_additive, nn_additive, …). It contains no Numpyro sample sites.
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass

# Number of Newton iterations for the free-effector cubic solve.
NEWTON_ITERATIONS = 8


# ---------------------------------------------------------------------------
# ThetaParam pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """Assembled partition-function parameters passed between K-assembly and thermo."""

    ln_K_op:  jnp.ndarray   # (G,)    — per-genotype log protein-DNA affinity
    ln_K_HL:  jnp.ndarray   # (G,)    — per-genotype log conformational equilibrium
    ln_K_E:   jnp.ndarray   # (T, G)  — per-genotype log effector binding constant
    tf_total: float          # nM
    op_total: float          # nM
    mu:       jnp.ndarray   # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray   # (T, C, 1) — population std  logit-theta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solve_free_effector(e_total, tf_total, a, Z0):
    """
    Solve the cubic mass-balance equation for free effector concentration.

        a·x³ + a·(2·tf_total − e_total)·x² + Z₀·x − Z₀·e_total = 0

    Uses NEWTON_ITERATIONS Newton iterations starting from x₀ = e_total.

    Parameters
    ----------
    e_total : (1, C, 1)  — total effector concentration grid
    tf_total : scalar    — total TF concentration (nM)
    a : (T, 1, G)        — K_HL · K_E
    Z0 : (1, 1, G)       — 1 + K_op·[op_total] + K_HL

    Returns
    -------
    x_free : (T, C, G)   — free effector concentration, clipped to [0, e_total]
    """
    coeff_a = a                                     # (T, 1, G)
    coeff_b = a * (2.0 * tf_total - e_total)        # (T, C, G)
    coeff_c = Z0                                    # (1, 1, G)
    coeff_d = -Z0 * e_total                         # (1, C, G)

    def f(x):
        return coeff_a * x**3 + coeff_b * x**2 + coeff_c * x + coeff_d

    def df(x):
        return 3.0 * coeff_a * x**2 + 2.0 * coeff_b * x + coeff_c

    def step(i, x):
        dfx = df(x)
        return x - f(x) / jnp.where(jnp.abs(dfx) < 1e-30, 1e-30, dfx)

    x0 = jnp.broadcast_to(e_total, coeff_b.shape)
    x_free = jax.lax.fori_loop(0, NEWTON_ITERATIONS, step, x0)
    return jnp.clip(x_free, 0.0, e_total)


def _compute_theta(ln_K_op, ln_K_HL, ln_K_E, titrant_conc, tf_total, op_total):
    """
    Evaluate the partition-function theta for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_op : (G,)
    ln_K_HL : (G,)
    ln_K_E  : (T, G)
    titrant_conc : (C,)  — effector concentrations in nM
    tf_total : scalar
    op_total : scalar

    Returns
    -------
    theta : (T, C, G)
    """
    K_op = jnp.exp(ln_K_op)[None, None, :]     # (1, 1, G)
    K_HL = jnp.exp(ln_K_HL)[None, None, :]     # (1, 1, G)
    K_E  = jnp.exp(ln_K_E)[:, None, :]         # (T, 1, G)
    e_total = titrant_conc[None, :, None]       # (1, C, 1)

    Z0 = 1.0 + K_op * op_total + K_HL          # (1, 1, G)
    a  = K_HL * K_E                             # (T, 1, G)

    e_free = _solve_free_effector(e_total, tf_total, a, Z0)  # (T, C, G)

    w_Hop = K_op * op_total                     # (1, 1, G)
    w_LE  = K_HL * K_E * e_free**2             # (T, C, G)
    Z     = 1.0 + w_Hop + K_HL + w_LE          # (T, C, G)

    return w_Hop / Z                            # (T, C, G)


def _population_moments(theta, data):
    """
    Compute per-concentration population moments of logit(theta) over genotypes.

    Parameters
    ----------
    theta : (T, C, G)
    data  : GrowthData or BindingData (unused; retained for API consistency)

    Returns
    -------
    mu    : (T, C, 1)
    sigma : (T, C, 1)
    """
    eps = 1e-6
    theta_clipped = jnp.clip(theta, eps, 1.0 - eps)
    logit_theta = jax.scipy.special.logit(theta_clipped)           # (T, C, G)
    mu    = jnp.mean(logit_theta, axis=-1, keepdims=True)          # (T, C, 1)
    sigma = jnp.std(logit_theta,  axis=-1, keepdims=True)          # (T, C, 1)
    return mu, sigma


# ---------------------------------------------------------------------------
# Public interface used by the growth/binding model
# ---------------------------------------------------------------------------

def run_model(theta_param: ThetaParam, data) -> jnp.ndarray:
    """
    Evaluate theta at the concentrations and genotypes present in ``data``.

    Recomputes theta from stored equilibrium constants so the function works
    for both growth and binding data (which may have different concentration
    grids).

    Returns
    -------
    theta_calc : (T, C, num_obs)  or  (1,1,1,1,T,C,num_obs)
    """
    theta_all = _compute_theta(
        theta_param.ln_K_op,
        theta_param.ln_K_HL,
        theta_param.ln_K_E,
        data.titrant_conc,
        theta_param.tf_total,
        theta_param.op_total,
    )                                              # (T, C, G)

    theta_calc = theta_all[:, :, data.geno_theta_idx]   # (T, C, num_obs)

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data) -> tuple:
    return theta_param.mu, theta_param.sigma
