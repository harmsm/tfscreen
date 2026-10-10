"""
Shared thermodynamic functions for the M2 lac-dimer θ model (O2_C4 U0).

M2 is the standard MWC dimer of ``thermo/mwc.py`` restricted to K_h_e = 0
(H does not bind effector) and K_l_o = 0 (L does not bind operator). The
dimer is in conformation H, which binds operator, or L, which binds two
effector molecules at independent, equivalent sites. Effector pulls the
dimer into L and lowers operator occupancy θ. Used by
``thermo.O2_C4_K3_U0_a.PK``, ``.PddG`` and ``.PnnC``.

Three constants, named for M3's (O2_C12) constants they equal:

    K_op = K_h_o   H + O → HO (M⁻¹)
    K_HL = K_h_l   [L]/[H] (dimensionless)
    K_E  = K_l_e   per-site effector association constant of L (M⁻¹)

Species not bound to operator, with weights relative to free H (h):

    H        1
    L        K_HL
    LE       2·K_HL·K_E·e         (two ways to fill one site)
    LE2      K_HL·K_E²·e²

    F(e) = 1 + K_HL·(1 + K_E·e)²

TF (1e-5 M monomer by default) is about 400-fold in excess over operator, so
operator-bound TF is left out of the TF mass balance and operator total does
not enter the model. With r = tf_total/2 (tf_total in MONOMER units):

    h = r / F(e),    W = h·K_op,    θ = W / (1 + W)

Free effector e solves mwc.py's cubic with B0 = 1 + K_HL,
B1 = 2·K_HL·K_E and B2 = K_HL·K_E². θ here is M3's θ with K_h_e and K_l_o at
0, computed by the same function (``mwc.theta_c4`` calls ``mwc.mwc_theta``),
so the nesting is exact.

History. Before 2026-10-11 M2 had a different partition function,
Z = 1 + K_op·[op_total] + K_HL + K_HL·K_E·e² with K_E in M⁻² for a one-step
L + 2E → LE2, operator total as a fixed free-operator concentration, and a
mass balance that counted operator-bound TF. The C4/K3 labels in the
registry keys are historical: the model now has the species above, three
constants and no operator-bound species in the TF mass balance.

Units
-----
Data concentrations (``titrant_conc``) are mM; ``theta_conc_unit_scale``
(1e-3) converts them to M, so K_op and K_E are in M⁻¹ and tf_total in M.

This module contains no Numpyro sample sites.
"""

import jax
import jax.numpy as jnp
import numpy as np

import pandas as pd
from flax.struct import dataclass, field

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.mwc import (  # noqa: F401
    DEFAULT_CONC_UNIT_SCALE,
    DEFAULT_TF_TOTAL_M,
    DEFAULT_LN_K_H_O,
    DEFAULT_LN_K_H_L,
    DEFAULT_LN_K_E_C4,
    NEWTON_ITERATIONS,
)

# Default wt constants (see mwc.py): Sochor's K_h_o and K_h_l, and the per-site
# K_E that puts the default wt midpoint at 80 µM IPTG.
DEFAULT_LN_K_OP_WT_LOC = DEFAULT_LN_K_H_O
DEFAULT_LN_K_HL_WT_LOC = DEFAULT_LN_K_H_L
DEFAULT_LN_K_E_WT_LOC = DEFAULT_LN_K_E_C4


# ---------------------------------------------------------------------------
# ThetaParam pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """Assembled partition-function parameters passed between K-assembly and thermo."""

    ln_K_op:  jnp.ndarray   # (G,)    — per-genotype log K_op (= K_h_o), M⁻¹
    ln_K_HL:  jnp.ndarray   # (G,)    — per-genotype log K_HL (= K_h_l)
    ln_K_E:   jnp.ndarray   # (T, G)  — per-genotype log per-site K_E (= K_l_e), M⁻¹
    tf_total: float          # M, monomer units
    mu:       jnp.ndarray   # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray   # (T, C, 1) — population std  logit-theta
    # Factor that converts data.titrant_conc (mM) to M before it enters the
    # thermodynamic equations (theta_conc_unit_scale, 1e-3 by default).
    conc_unit_scale: float = field(pytree_node=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_theta(ln_K_op, ln_K_HL, ln_K_E, titrant_conc, tf_total):
    """
    Evaluate θ for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_op : (G,)
    ln_K_HL : (G,)
    ln_K_E  : (T, G)
    titrant_conc : (C,)  — effector concentrations in M (already scaled)
    tf_total : scalar   — total TF in M, monomer units

    Returns
    -------
    theta : (T, C, G)
    """
    return mwc.theta_c4(ln_K_op[None, None, :],
                        ln_K_HL[None, None, :],
                        ln_K_E[:, None, :],
                        None,
                        titrant_conc[None, :, None],
                        tf_total)


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
        data.titrant_conc * theta_param.conc_unit_scale,
        theta_param.tf_total,
    )                                              # (T, C, G)

    geno_idx = data.batch_idx[data.geno_theta_idx]
    theta_calc = theta_all[:, :, geno_idx]   # (T, C, num_obs)

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data) -> tuple:
    return theta_param.mu, theta_param.sigma


# ---------------------------------------------------------------------------
# Numpy extraction helpers (shared across lac_dimer theta components)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def _solve_theta_np(ln_K_op, ln_K_HL, ln_K_E, conc, tf_total):
    """
    Numpy θ at per-row K values (the same ``mwc.theta_c4`` as the JAX path).

    Parameters
    ----------
    ln_K_op : (S, N)
    ln_K_HL : (S, N)
    ln_K_E  : (S, N)
    conc    : (N,)  — effector concentrations in M (zero already replaced)
    tf_total : float — M, monomer units

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    return mwc.theta_c4(np.asarray(ln_K_op, float), np.asarray(ln_K_HL, float),
                        np.asarray(ln_K_E, float), None,
                        np.asarray(conc, float)[np.newaxis, :], tf_total,
                        xp=np)


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration grid DataFrame for theta curve extraction.

    Shared by ``lac_dimer_mut`` and ``lac_dimer_lnK_nn_prior``.

    Returns
    -------
    calc_df : pd.DataFrame
        Rows for each (genotype, titrant_name, titrant_conc) combination,
        including internal ``genotype_idx`` and ``titrant_name_idx`` columns.
    internal_cols : list of str
        Columns to strip before returning results to the caller.
    extra_kwargs : dict
        ``tf_total`` (M, monomer units) and ``conc_unit_scale`` required
        by ``compute_theta_samples``.
    """
    import tfscreen.util.dataframe

    if manual_titrant_df is None:
        calc_df = (model.training_tm.df[["genotype", "titrant_name", "titrant_conc",
                                       "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    else:
        tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                              required_columns=["titrant_name", "titrant_conc"])
        if "genotype" not in manual_titrant_df.columns:
            genotypes = model.training_tm.df["genotype"].unique()
            dfs = [manual_titrant_df.assign(genotype=g) for g in genotypes]
            calc_df = pd.concat(dfs).reset_index(drop=True)
        else:
            calc_df = manual_titrant_df.copy()

        geno_map = (model.training_tm.df[["genotype", "genotype_idx"]]
                    .drop_duplicates()
                    .set_index("genotype")["genotype_idx"]
                    .to_dict())
        titrant_map = (model.training_tm.df[["titrant_name", "titrant_name_idx"]]
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
        "conc_unit_scale": float(model.priors.theta.theta_conc_unit_scale),
    }
    return calc_df, ["genotype_idx", "titrant_name_idx"], extra


def compute_theta_samples(calc_df, param_posteriors, *, tf_total,
                          conc_unit_scale=1.0):
    """
    Compute posterior theta samples for the lac_dimer partition-function model.

    Uses the same equations as the JAX path (``mwc.theta_c4``) in numpy,
    on flattened per-row posterior slices rather than the full (T, C, G)
    grid.

    Parameters
    ----------
    calc_df : pd.DataFrame
        Output of ``build_calc_df``; must contain ``titrant_conc``,
        ``genotype_idx``, and ``titrant_name_idx`` columns.
    param_posteriors : dict-like
        Posterior samples keyed by parameter name (with ``theta_`` prefix).
    tf_total : float
        Total TF concentration in M, monomer units.
    conc_unit_scale : float, optional
        Multiply ``titrant_conc`` by this factor to get M. ``build_calc_df``
        passes ``theta_conc_unit_scale`` (1e-3 for mM data).

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
        Posterior theta at each row of ``calc_df``.
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale   # (N,)
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

    return _solve_theta_np(ln_K_op_pts, ln_K_HL_pts, ln_K_E_pts,
                           conc, tf_total)
