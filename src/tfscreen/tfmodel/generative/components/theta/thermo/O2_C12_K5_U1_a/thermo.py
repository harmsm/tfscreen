"""
Shared thermodynamic functions for the M3 MWC lac-dimer θ model with unfolding (O2_C12 U1).

Extends the five-constant MWC model (``O2_C12_K5_U0_a/thermo.py``) with a
globally unfolded state U that binds neither operator nor effector.
Destabilizing mutations shift the dimer into U and pull θ toward zero at all
effector concentrations. Used by ``thermo.O2_C12_K5_U1_a.PK``, ``.PddG`` and
``.PnnC``.

Species not bound to operator, with weights relative to free H (h), for two
independent, equivalent effector sites per conformation:

    State       Weight relative to H
    H           1                           (high-affinity conformation)
    H·E         2·K_h_e·e
    H·E²        K_h_e²·e²
    L           K_h_l                       (low-affinity conformation)
    L·E         2·K_h_l·K_l_e·e
    L·E²        K_h_l·K_l_e²·e²
    U           K_u                         (globally unfolded; no binding)

The operator-bound species HO·E_n and LO·E_n carry the extra factor K_h_o·[O]
or K_l_o·[O]. K_u = [U]/[H]; in wt K_u << 1. K_u adds to B0, the ligand-free
coefficient of F(e), which enters the free-effector cubic and h:

    B0 = 1 + K_h_l + K_u
    F(e) = B0 + B1·e + B2·e²
    h = r / F(e_free)
    W = h·(K_h_o·P_H + K_h_l·K_l_o·P_L)
    θ = W / (1 + W)

Everything else (polynomial, bookkeeping, Newton solve, monomer units for
tf_total, conc_unit_scale, defaults) is that of the U0 model and
``thermo/mwc.py``.
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
    DEFAULT_LN_K_H_L,
    DEFAULT_LN_K_H_O,
    DEFAULT_LN_K_H_E,
    DEFAULT_LN_K_L_O,
    DEFAULT_LN_K_L_E,
    NEWTON_ITERATIONS,
)


# ---------------------------------------------------------------------------
# ThetaParam pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """Assembled MWC-unfolded partition-function parameters."""

    ln_K_h_l: jnp.ndarray   # (G,)    — log conformational equilibrium
    ln_K_h_o: jnp.ndarray   # (G,)    — log H-state DNA-binding constant
    ln_K_h_e: jnp.ndarray   # (T, G)  — log per-site H-state effector constant
    ln_K_l_o: jnp.ndarray   # (G,)    — log L-state DNA-binding constant
    ln_K_l_e: jnp.ndarray   # (T, G)  — log per-site L-state effector constant
    ln_K_u:   jnp.ndarray   # (G,)    — log H→U unfolding equilibrium constant
    tf_total: float           # M — monomer units; r = tf_total/2 is the dimer concentration
    mu:       jnp.ndarray    # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray    # (T, C, 1) — population std  logit-theta
    conc_unit_scale: float = field(pytree_node=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
                   titrant_conc, tf_total):
    """
    Evaluate MWC-unfolded operator-occupancy for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_h_l : (G,)
    ln_K_h_o : (G,)
    ln_K_h_e : (T, G)
    ln_K_l_o : (G,)
    ln_K_l_e : (T, G)
    ln_K_u   : (G,)
    titrant_conc : (C,)  — M (already scaled)
    tf_total : scalar — total TF in **monomer** units (M); dimer = tf_total/2

    Returns
    -------
    theta : (T, C, G)
    """
    return mwc.theta_c12(ln_K_h_l[None, None, :],
                         ln_K_h_o[None, None, :],
                         ln_K_h_e[:, None, :],
                         ln_K_l_o[None, None, :],
                         ln_K_l_e[:, None, :],
                         ln_K_u[None, None, :],
                         titrant_conc[None, :, None],
                         tf_total)


def _population_moments(theta, data):
    """
    Compute per-(T, C) population moments of logit(theta) over genotypes.

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
        theta_param.ln_K_h_l,
        theta_param.ln_K_h_o,
        theta_param.ln_K_h_e,
        theta_param.ln_K_l_o,
        theta_param.ln_K_l_e,
        theta_param.ln_K_u,
        data.titrant_conc * theta_param.conc_unit_scale,
        theta_param.tf_total,
    )                                               # (T, C, G)

    geno_idx = data.batch_idx[data.geno_theta_idx]
    theta_calc = theta_all[:, :, geno_idx]   # (T, C, num_obs)

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data) -> tuple:
    return theta_param.mu, theta_param.sigma


# ---------------------------------------------------------------------------
# Numpy extraction helpers (shared across mwc_dimer_unfolded theta components)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def _solve_theta_np(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
                    conc, tf_total):
    """
    Numpy MWC-dimer-unfolded θ at per-row K values (same ``mwc.theta_c12``).

    Parameters
    ----------
    ln_K_h_l : (S, N)
    ln_K_h_o : (S, N)
    ln_K_h_e : (S, N)
    ln_K_l_o : (S, N)
    ln_K_l_e : (S, N)
    ln_K_u   : (S, N)  or broadcastable to (S, N)
    conc     : (N,)  — effector concentrations in M (zero already replaced)
    tf_total : float — monomer units

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    f = lambda v: np.asarray(v, float)  # noqa: E731
    return mwc.theta_c12(f(ln_K_h_l), f(ln_K_h_o), f(ln_K_h_e), f(ln_K_l_o),
                         f(ln_K_l_e), f(ln_K_u), f(conc)[np.newaxis, :],
                         tf_total, xp=np)


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration-grid DataFrame for theta curve extraction.

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
    Compute posterior theta samples for the MWC-dimer-unfolded model.

    Parameters
    ----------
    calc_df : pd.DataFrame
    param_posteriors : dict-like
    tf_total : float
        Total TF concentration in M, monomer units.
    conc_unit_scale : float

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

    ln_K_h_l_all = _load("theta_ln_K_h_l")   # (S, G)
    ln_K_h_o_all = _load("theta_ln_K_h_o")   # (S, G)
    ln_K_h_e_all = _load("theta_ln_K_h_e")   # (S, T, G)
    ln_K_l_o_all = _load("theta_ln_K_l_o")   # (S, G)
    ln_K_l_e_all = _load("theta_ln_K_l_e")   # (S, T, G)
    ln_K_u_all   = _load("theta_ln_K_u")     # (S, G)

    ln_K_h_l_pts = ln_K_h_l_all[:, geno_indices]
    ln_K_h_o_pts = ln_K_h_o_all[:, geno_indices]
    ln_K_h_e_pts = ln_K_h_e_all[:, titrant_indices, geno_indices]
    ln_K_l_o_pts = ln_K_l_o_all[:, geno_indices]
    ln_K_l_e_pts = ln_K_l_e_all[:, titrant_indices, geno_indices]
    ln_K_u_pts   = ln_K_u_all[:, geno_indices]

    return _solve_theta_np(ln_K_h_l_pts, ln_K_h_o_pts, ln_K_h_e_pts,
                           ln_K_l_o_pts, ln_K_l_e_pts, ln_K_u_pts,
                           conc, tf_total)
