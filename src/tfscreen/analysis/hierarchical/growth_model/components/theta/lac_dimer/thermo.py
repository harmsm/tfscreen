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

Units: all concentrations in M; K_op in M⁻¹; K_E in M⁻²; K_HL dimensionless.

Operator depletion approximation
---------------------------------
[TF_total] ≈ 6.5e-7 M, [op_total] ≈ 2.5e-8 M
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
import numpy as np
import pandas as pd
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
    tf_total: float          # M
    op_total: float          # M
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
    tf_total : scalar    — total TF concentration (M)
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
    titrant_conc : (C,)  — effector concentrations in M
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


# ---------------------------------------------------------------------------
# Numpy extraction helpers (shared by lac_dimer_mut and lac_dimer_nn_mut)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration grid DataFrame for theta curve extraction.

    Shared by ``lac_dimer_mut`` and ``lac_dimer_nn_mut``.

    Returns
    -------
    calc_df : pd.DataFrame
        Rows for each (genotype, titrant_name, titrant_conc) combination,
        including internal ``genotype_idx`` and ``titrant_name_idx`` columns.
    internal_cols : list of str
        Columns to strip before returning results to the caller.
    extra_kwargs : dict
        ``tf_total`` and ``op_total`` scalars (in M) required by
        ``compute_theta_samples``.
    """
    import tfscreen.util.dataframe

    if manual_titrant_df is None:
        calc_df = (model.growth_tm.df[["genotype", "titrant_name", "titrant_conc",
                                       "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                              required_columns=["titrant_name", "titrant_conc"])
        if "genotype" not in manual_titrant_df.columns:
            genotypes = model.growth_tm.df["genotype"].unique()
            dfs = [manual_titrant_df.assign(genotype=g) for g in genotypes]
            calc_df = pd.concat(dfs).reset_index(drop=True)
        else:
            calc_df = manual_titrant_df.copy()

        geno_map = (model.growth_tm.df[["genotype", "genotype_idx"]]
                    .drop_duplicates()
                    .set_index("genotype")["genotype_idx"]
                    .to_dict())
        titrant_map = (model.growth_tm.df[["titrant_name", "titrant_name_idx"]]
                       .drop_duplicates()
                       .set_index("titrant_name")["titrant_name_idx"]
                       .to_dict())
        calc_df["genotype_idx"] = calc_df["genotype"].map(geno_map)
        calc_df["titrant_name_idx"] = calc_df["titrant_name"].map(titrant_map)

        missing = calc_df["genotype_idx"].isna() | calc_df["titrant_name_idx"].isna()
        if missing.any():
            bad = calc_df[missing][["genotype", "titrant_name"]].drop_duplicates()
            raise ValueError(
                "Some (genotype, titrant_name) pairs in manual_titrant_df "
                "were not found in the model data: "
                f"{bad.values}"
            )

    extra = {
        "tf_total": float(model.priors.theta.theta_tf_total_M),
        "op_total": float(model.priors.theta.theta_op_total_M),
    }
    return calc_df, ["genotype_idx", "titrant_name_idx"], extra


def compute_theta_samples(calc_df, param_posteriors, *, tf_total, op_total):
    """
    Compute posterior theta samples for the lac_dimer partition-function model.

    Uses a pure-numpy Newton solve identical in structure to the JAX version
    in ``_compute_theta``, operating on flattened per-row posterior slices
    rather than the full (T, C, G) grid.

    Parameters
    ----------
    calc_df : pd.DataFrame
        Output of ``build_calc_df``; must contain ``titrant_conc``,
        ``genotype_idx``, and ``titrant_name_idx`` columns.
    param_posteriors : dict-like
        Posterior samples keyed by parameter name (with ``theta_`` prefix).
    tf_total : float
        Total TF concentration in M.
    op_total : float
        Total operator concentration in M.

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
        Posterior theta at each row of ``calc_df``.
    """
    from tfscreen.analysis.hierarchical.posteriors import get_posterior_samples

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    conc = calc_df["titrant_conc"].values.copy().astype(float)   # (N,)
    conc[conc == 0] = _ZERO_CONC_VALUE

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v

    ln_K_op_all = _load("theta_ln_K_op")   # (S, G)
    ln_K_HL_all = _load("theta_ln_K_HL")   # (S, G)
    ln_K_E_all  = _load("theta_ln_K_E")    # (S, T, G)

    # Index to per-row parameters: (S, N)
    ln_K_op_pts = ln_K_op_all[:, geno_indices]
    ln_K_HL_pts = ln_K_HL_all[:, geno_indices]
    ln_K_E_pts  = ln_K_E_all[:, titrant_indices, geno_indices]

    K_op = np.exp(ln_K_op_pts)   # (S, N)
    K_HL = np.exp(ln_K_HL_pts)   # (S, N)
    K_E  = np.exp(ln_K_E_pts)    # (S, N)

    e_total  = conc[np.newaxis, :]              # (1, N)
    Z0       = 1.0 + K_op * op_total + K_HL    # (S, N)
    a        = K_HL * K_E                       # (S, N)
    coeff_b  = a * (2.0 * tf_total - e_total)   # (S, N)

    e_free = e_total * np.ones_like(a)   # (S, N)
    for _ in range(NEWTON_ITERATIONS):
        f  = a * e_free**3 + coeff_b * e_free**2 + Z0 * e_free - Z0 * e_total
        df = 3.0 * a * e_free**2 + 2.0 * coeff_b * e_free + Z0
        e_free = e_free - f / np.where(np.abs(df) < 1e-30, 1e-30, df)
    e_free = np.clip(e_free, 0.0, e_total)

    w_Hop = K_op * op_total          # (S, N)
    w_LE  = a * e_free**2            # (S, N)
    Z     = 1.0 + w_Hop + K_HL + w_LE
    return w_Hop / Z                 # (S, N)
