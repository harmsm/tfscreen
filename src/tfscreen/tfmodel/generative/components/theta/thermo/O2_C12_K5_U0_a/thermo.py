"""
Shared thermodynamic functions for the M3 MWC lac-dimer θ model (O2_C12 U0).

The standard Monod-Wyman-Changeux (MWC) two-conformation dimer with five
equilibrium constants (Sochor 2014, PeerJ 2:e498, for the constants and
values). Unlike M2 (``O2_C4``), both conformations may bind operator and
effector. Used by ``thermo.O2_C12_K5_U0_a.PK``, ``.PddG`` and ``.PnnC``.
The equations are in ``thermo/mwc.py`` and shared with M2, which is this
model with K_h_e = K_l_o = 0. **All concentrations and K values are in M.**

    K_h_l  — [L]/[H] (dimensionless)
    K_h_o  — H + O → HO  (M⁻¹)
    K_h_e  — per-site effector association constant of H (M⁻¹)
    K_l_o  — L + O → LO  (M⁻¹); ≈ 0 for wild-type lac repressor
    K_l_e  — per-site effector association constant of L (M⁻¹)

Each conformation has two independent, equivalent effector sites:

    P_H(e) = (1 + K_h_e·e)²,    P_L(e) = (1 + K_l_e·e)²

Until 2026-10-11 the model used Sochor's polynomial 1 + K·e + K²·e², which
fixes the ratio of macroscopic constants within a conformation at
K2/K1 = 1 instead of the independent-site 1/4 (see ``mwc.py``).

Wild-type defaults
------------------
K_h_l = 6.3, K_h_o = 4.2×10⁸ M⁻¹ and K_l_o = 0.1 M⁻¹ are Sochor's (Table 1).
The effector selectivity K_l_e/K_h_e = ``mwc.MWC_EFFECTOR_SELECTIVITY``
≈ 79.6, wider than Sochor's fitted 13.6, puts θ at saturating effector at
0.05 for tf_total = 1e-5 M. K_h_e and K_l_e (≈ 3.7×10³ and 3.0×10⁵ M⁻¹)
then share the scale ``mwc.MWC_EFFECTOR_SCALE`` ≈ 0.39 on Sochor's K_l_e,
which puts the midpoint of the default wt θ(IPTG) curve at 80 µM with
effector depletion. The derivation is in ``mwc.py``.

K_h_l > 1 is required for effector-induced derepression. If K_h_l << 1,
F(e) ≈ P_H(e) and h·P_H is about constant, so θ is flat in effector.

Bookkeeping
-----------
tf_total is in **monomer** units (default 1e-5 M); r = tf_total/2 is the
dimer concentration. TF is about 400-fold in excess over operator, so
operator-bound TF is left out of the TF mass balance and operator total does
not enter the model:

    h = r / F(e)
    F(e) = P_H(e) + K_h_l·P_L(e) = B0 + B1·e + B2·e²
    W = h·(K_h_o·P_H + K_h_l·K_l_o·P_L)
    θ = W / (1 + W)

with B0 = 1 + K_h_l, B1 = 2·(K_h_e + K_h_l·K_l_e), B2 = K_h_e² + K_h_l·K_l_e².
Free effector solves the cubic in ``mwc.py``:

    B2·x³ + (B1 + B2·(2r − e_total))·x²
          + (B0 + B1·(r − e_total))·x − B0·e_total = 0
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
    """Assembled MWC partition-function parameters passed to run_model."""

    ln_K_h_l: jnp.ndarray   # (G,)    — log conformational equilibrium
    ln_K_h_o: jnp.ndarray   # (G,)    — log H-state DNA-binding constant
    ln_K_h_e: jnp.ndarray   # (T, G)  — log per-site H-state effector constant
    ln_K_l_o: jnp.ndarray   # (G,)    — log L-state DNA-binding constant
    ln_K_l_e: jnp.ndarray   # (T, G)  — log per-site L-state effector constant
    tf_total: float           # M — monomer units; r = tf_total/2 is the dimer concentration
    mu:       jnp.ndarray    # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray    # (T, C, 1) — population std  logit-theta
    # Scale factor applied to data.titrant_conc before entering thermodynamic
    # equations.  Use 1e-3 when titrant concentrations are in mM and K values
    # are in M⁻¹ (the recommended convention for this model).
    conc_unit_scale: float = field(pytree_node=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e,
                   titrant_conc, tf_total):
    """
    Evaluate MWC operator-occupancy for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_h_l : (G,)
    ln_K_h_o : (G,)
    ln_K_h_e : (T, G)
    ln_K_l_o : (G,)
    ln_K_l_e : (T, G)
    titrant_conc : (C,)  — M (already scaled)
    tf_total : scalar — total TF in **monomer** units (M); dimer units = tf_total/2

    Returns
    -------
    theta : (T, C, G)
    """
    return mwc.theta_c12(ln_K_h_l[None, None, :],
                         ln_K_h_o[None, None, :],
                         ln_K_h_e[:, None, :],
                         ln_K_l_o[None, None, :],
                         ln_K_l_e[:, None, :],
                         None,
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
# Numpy extraction helpers (shared across mwc_dimer theta components)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def _solve_theta_np(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e,
                    conc, tf_total):
    """
    Numpy MWC-dimer θ at per-row K values (the same ``mwc.theta_c12`` as JAX).

    Parameters
    ----------
    ln_K_h_l : (S, N)
    ln_K_h_o : (S, N)
    ln_K_h_e : (S, N)
    ln_K_l_o : (S, N)
    ln_K_l_e : (S, N)
    conc     : (N,)  — effector concentrations in M (zero already replaced)
    tf_total : float  — monomer units; r = tf_total / 2 is the dimer concentration

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    f = lambda v: np.asarray(v, float)  # noqa: E731
    return mwc.theta_c12(f(ln_K_h_l), f(ln_K_h_o), f(ln_K_h_e), f(ln_K_l_o),
                         f(ln_K_l_e), None, f(conc)[np.newaxis, :], tf_total,
                         xp=np)


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration-grid DataFrame for theta curve extraction.

    Returns
    -------
    calc_df : pd.DataFrame
    internal_cols : list of str
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
    Compute posterior theta samples for the MWC dimer model.

    Uses the same equations as the JAX path (``mwc.theta_c12``) in numpy.

    Parameters
    ----------
    calc_df : pd.DataFrame
        Output of ``build_calc_df``; must have ``titrant_conc``,
        ``genotype_idx``, and ``titrant_name_idx`` columns.
    param_posteriors : dict-like
        Posterior samples keyed by parameter name (with ``theta_`` prefix).
    tf_total : float
        Total TF concentration in M, monomer units.
    conc_unit_scale : float, optional
        Multiply ``titrant_conc`` by this factor before passing to the
        thermodynamic equations.  Use ``1e-3`` when concentrations are in
        mM and K values are in M⁻¹ (the default convention).

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale  # (N,)
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

    ln_K_h_l_pts = ln_K_h_l_all[:, geno_indices]
    ln_K_h_o_pts = ln_K_h_o_all[:, geno_indices]
    ln_K_h_e_pts = ln_K_h_e_all[:, titrant_indices, geno_indices]
    ln_K_l_o_pts = ln_K_l_o_all[:, geno_indices]
    ln_K_l_e_pts = ln_K_l_e_all[:, titrant_indices, geno_indices]

    return _solve_theta_np(ln_K_h_l_pts, ln_K_h_o_pts, ln_K_h_e_pts,
                           ln_K_l_o_pts, ln_K_l_e_pts,
                           conc, tf_total)
