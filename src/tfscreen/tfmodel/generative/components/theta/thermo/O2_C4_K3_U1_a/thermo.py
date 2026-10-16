"""
Shared thermodynamic functions for the M2 lac-dimer θ model with unfolding (O2_C4 U1).

Extends M2 (``O2_C4_K3_U0_a/thermo.py``) with a globally unfolded state U
that binds neither operator nor effector. K_U = [U]/[H]. In wt K_U << 1;
destabilizing mutations raise it, shift the dimer into U and pull θ toward
zero at every effector concentration. Used by ``thermo.O2_C4_K3_U1_a.PK``,
``.PddG`` and ``.PnnC``.

U adds K_U to the ligand-free part of the TF polynomial:

    F(e) = 1 + K_HL·(1 + K_E·e)² + K_U
    h = r / F(e),    W = h·K_op,    θ = W / (1 + W)

It is M3 with unfolding (``O2_C12_K5_U1_a``) restricted to K_h_e = K_l_o = 0,
computed by the same ``mwc.mwc_theta``. Constants, units, bookkeeping and the
free-effector cubic are those of the U0 model and ``thermo/mwc.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.struct import dataclass, field

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
# Unit convention and the defaults are shared with the U0 model.
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (  # noqa: F401
    DEFAULT_CONC_UNIT_SCALE,
    DEFAULT_LN_K_E_WT_LOC,
    DEFAULT_LN_K_HL_WT_LOC,
    DEFAULT_LN_K_OP_WT_LOC,
    DEFAULT_TF_TOTAL_M,
    NEWTON_ITERATIONS,
)


# ---------------------------------------------------------------------------
# ThetaParam pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """Assembled partition-function parameters passed between K-assembly and thermo."""

    ln_K_op:  jnp.ndarray   # (G,)    — per-genotype log K_op (= K_h_o), M⁻¹
    ln_K_HL:  jnp.ndarray   # (G,)    — per-genotype log K_HL (= K_h_l)
    ln_K_E:   jnp.ndarray   # (T, G)  — per-genotype log per-site K_E (= K_l_e), M⁻¹
    ln_K_U:   jnp.ndarray   # (G,)    — per-genotype log H→U unfolding equilibrium
    tf_total: float          # M, monomer units
    mu:       jnp.ndarray   # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray   # (T, C, 1) — population std  logit-theta
    # Factor that converts data.titrant_conc (mM) to M before it enters the
    # thermodynamic equations (theta_conc_unit_scale, 1e-3 by default).
    conc_unit_scale: float = field(pytree_node=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, titrant_conc, tf_total):
    """
    Evaluate θ with the unfolded state for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_op : (G,)
    ln_K_HL : (G,)
    ln_K_E  : (T, G)
    ln_K_U  : (G,)
    titrant_conc : (C,)  — effector concentrations in M (already scaled)
    tf_total : scalar   — total TF in M, monomer units

    Returns
    -------
    theta : (T, C, G)
    """
    return mwc.theta_c4(ln_K_op[None, None, :],
                        ln_K_HL[None, None, :],
                        ln_K_E[:, None, :],
                        ln_K_U[None, None, :],
                        titrant_conc[None, :, None],
                        tf_total)


def _population_moments(theta, data):
    """
    Compute per-concentration population moments of logit(theta) over genotypes.

    Returns
    -------
    mu    : (T, C, 1)
    sigma : (T, C, 1)
    """
    eps = 1e-6
    theta_clipped = jnp.clip(theta, eps, 1.0 - eps)
    logit_theta = jax.scipy.special.logit(theta_clipped)
    mu    = jnp.mean(logit_theta, axis=-1, keepdims=True)
    sigma = jnp.std(logit_theta,  axis=-1, keepdims=True)
    return mu, sigma


# ---------------------------------------------------------------------------
# Public interface used by the growth/binding model
# ---------------------------------------------------------------------------

def run_model(theta_param: ThetaParam, data) -> jnp.ndarray:
    """
    Evaluate theta at the concentrations and genotypes present in ``data``.

    Returns
    -------
    theta_calc : (T, C, num_obs)  or  (1,1,1,1,T,C,num_obs)
    """
    theta_all = _compute_theta(
        theta_param.ln_K_op,
        theta_param.ln_K_HL,
        theta_param.ln_K_E,
        theta_param.ln_K_U,
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
# Numpy extraction helpers (shared across lac_dimer_unfolded theta components)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def _solve_theta_np(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, conc, tf_total):
    """
    Numpy θ at per-row K values (the same ``mwc.theta_c4`` as the JAX path).

    Parameters
    ----------
    ln_K_op : (S, N)
    ln_K_HL : (S, N)
    ln_K_E  : (S, N)
    ln_K_U  : (S, N)  or broadcastable to (S, N)
    conc    : (N,)  — effector concentrations in M (zero already replaced)
    tf_total : float — M, monomer units

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    return mwc.theta_c4(np.asarray(ln_K_op, float), np.asarray(ln_K_HL, float),
                        np.asarray(ln_K_E, float), np.asarray(ln_K_U, float),
                        np.asarray(conc, float)[np.newaxis, :], tf_total,
                        xp=np)


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration grid DataFrame for theta curve extraction.

    Shared by all lac_dimer_unfolded theta components.

    Returns
    -------
    calc_df : pd.DataFrame
    internal_cols : list of str
    extra_kwargs : dict
        ``tf_total`` (M, monomer units) and ``conc_unit_scale``.
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
    Compute posterior theta samples for the lac_dimer_unfolded model.

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
        Multiply ``titrant_conc`` by this factor to get M (1e-3 for mM data).

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale
    conc[conc == 0] = _ZERO_CONC_VALUE

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v

    ln_K_op_all = _load("theta_ln_K_op")   # (S, G)
    ln_K_HL_all = _load("theta_ln_K_HL")   # (S, G)
    ln_K_E_all  = _load("theta_ln_K_E")    # (S, T, G)
    ln_K_U_all  = _load("theta_ln_K_U")    # (S, G)

    ln_K_op_pts = ln_K_op_all[:, geno_indices]
    ln_K_HL_pts = ln_K_HL_all[:, geno_indices]
    ln_K_E_pts  = ln_K_E_all[:, titrant_indices, geno_indices]
    ln_K_U_pts  = ln_K_U_all[:, geno_indices]

    return _solve_theta_np(ln_K_op_pts, ln_K_HL_pts, ln_K_E_pts, ln_K_U_pts,
                           conc, tf_total)
