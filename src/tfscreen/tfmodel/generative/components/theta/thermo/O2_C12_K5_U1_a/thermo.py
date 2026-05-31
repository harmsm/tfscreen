"""
Shared thermodynamic functions for the MWC lac-dimer-unfolded theta model.

Extends the MWC two-state model (Sochor 2014, PeerJ 2:e498) with an explicit
unfolded state U:

    State       Weight relative to H
    H           1                           (high-affinity conformation)
    H·op        K_h_o · [op_free]
    H·E         K_h_e · [e_free]
    H·E²        K_h_e² · [e_free]²
    L           K_h_l                       (low-affinity conformation)
    L·op        K_h_l · K_l_o · [op_free]
    L·E         K_h_l · K_l_e · [e_free]
    L·E²        K_h_l · K_l_e² · [e_free]²
    U           K_u                         (globally unfolded; no binding)

K_u = [U]/[H] is the H→U equilibrium constant.  In WT K_u << 1.  Mutations
accumulate destabilising effects as additive shifts in ln(K_u); when K_u
dominates the partition function, θ collapses toward zero.

U does not bind operator or effector, so the thermodynamic projection matrix
for all other equilibrium constants is unchanged.  The only modification to
the MWC partition function is that K_u is added to B0 (the ligand-free
coefficient), which propagates into the free-effector cubic and into F(e):

    B0 = 1 + K_h_l + K_u
    F(e) = B0 + B1·e + B2·e²
    h_free = r / F(e_free)
    W = h_free · (K_h_o·P_H + K_h_l·K_l_o·P_L)
    θ = W / (1 + W)

All other details (operator-depletion approximation, eight Newton steps,
monomer-unit convention for tf_total, conc_unit_scale) are identical to
mwc_dimer/thermo.py.  See that module for full derivations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.struct import dataclass, field

NEWTON_ITERATIONS = 8


# ---------------------------------------------------------------------------
# ThetaParam pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """Assembled MWC-unfolded partition-function parameters."""

    ln_K_h_l: jnp.ndarray   # (G,)    — log conformational equilibrium
    ln_K_h_o: jnp.ndarray   # (G,)    — log H-state DNA-binding constant
    ln_K_h_e: jnp.ndarray   # (T, G)  — log H-state effector-binding constant
    ln_K_l_o: jnp.ndarray   # (G,)    — log L-state DNA-binding constant
    ln_K_l_e: jnp.ndarray   # (T, G)  — log L-state effector-binding constant
    ln_K_u:   jnp.ndarray   # (G,)    — log H→U unfolding equilibrium constant
    tf_total: float           # M — monomer units; r = tf_total/2 is the dimer concentration
    op_total: float           # M
    mu:       jnp.ndarray    # (T, C, 1) — population mean logit-theta
    sigma:    jnp.ndarray    # (T, C, 1) — population std  logit-theta
    conc_unit_scale: float = field(pytree_node=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u,
                   titrant_conc, tf_total, op_total):
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
    titrant_conc : (C,)
    tf_total : scalar — total TF in **monomer** units (M); dimer = tf_total/2
    op_total : scalar

    Returns
    -------
    theta : (T, C, G)
    """
    K_h_l = jnp.exp(ln_K_h_l)[None, None, :]    # (1, 1, G)
    K_h_o = jnp.exp(ln_K_h_o)[None, None, :]    # (1, 1, G)
    K_h_e = jnp.exp(ln_K_h_e)[:, None, :]       # (T, 1, G)
    K_l_o = jnp.exp(ln_K_l_o)[None, None, :]    # (1, 1, G)
    K_l_e = jnp.exp(ln_K_l_e)[:, None, :]       # (T, 1, G)
    K_u   = jnp.exp(ln_K_u)[None, None, :]      # (1, 1, G)
    e_total = titrant_conc[None, :, None]        # (1, C, 1)

    r = tf_total / 2.0   # dimer units

    # B0 includes K_u; B1, B2 are unchanged (U has no effector binding)
    B0 = 1.0 + K_h_l + K_u
    B1 = K_h_e + K_h_l * K_l_e
    B2 = K_h_e ** 2 + K_h_l * K_l_e ** 2

    c3 = B2
    c2 = B1 + B2 * (2.0 * r - e_total)
    c1 = B0 + B1 * (r - e_total)
    c0 = -B0 * e_total

    def f(x):
        return c3 * x ** 3 + c2 * x ** 2 + c1 * x + c0

    def df(x):
        return 3.0 * c3 * x ** 2 + 2.0 * c2 * x + c1

    def step(_, x):
        dfx = df(x)
        return x - f(x) / jnp.where(jnp.abs(dfx) < 1e-30, 1e-30, dfx)

    x0 = jnp.broadcast_to(e_total, c2.shape)
    e_free = jax.lax.fori_loop(0, NEWTON_ITERATIONS, step, x0)
    e_free = jnp.clip(e_free, 0.0, e_total)

    P_H = 1.0 + K_h_e * e_free + K_h_e ** 2 * e_free ** 2
    P_L = 1.0 + K_l_e * e_free + K_l_e ** 2 * e_free ** 2

    F = B0 + B1 * e_free + B2 * e_free ** 2
    h_free = r / jnp.where(F < 1e-30, 1e-30, F)

    W = h_free * (K_h_o * P_H + K_h_l * K_l_o * P_L)
    return W / (1.0 + W)


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
        theta_param.op_total,
    )                                               # (T, C, G)

    theta_calc = theta_all[:, :, data.geno_theta_idx]   # (T, C, num_obs)

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
                    conc, tf_total, op_total):
    """
    Pure-numpy Newton solve for MWC-dimer-unfolded theta at per-row K values.

    Parameters
    ----------
    ln_K_h_l : (S, N)
    ln_K_h_o : (S, N)
    ln_K_h_e : (S, N)
    ln_K_l_o : (S, N)
    ln_K_l_e : (S, N)
    ln_K_u   : (S, N)  or broadcastable to (S, N)
    conc     : (N,)
    tf_total : float — monomer units
    op_total : float — unused (operator-depletion approx.)

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    K_h_l = np.exp(ln_K_h_l)
    K_h_o = np.exp(ln_K_h_o)
    K_h_e = np.exp(ln_K_h_e)
    K_l_o = np.exp(ln_K_l_o)
    K_l_e = np.exp(ln_K_l_e)
    K_u   = np.exp(ln_K_u)

    e_total = conc[np.newaxis, :]
    r = tf_total / 2.0

    B0 = 1.0 + K_h_l + K_u
    B1 = K_h_e + K_h_l * K_l_e
    B2 = K_h_e ** 2 + K_h_l * K_l_e ** 2

    c3 = B2
    c2 = B1 + B2 * (2.0 * r - e_total)
    c1 = B0 + B1 * (r - e_total)
    c0 = -B0 * e_total

    e_free = np.broadcast_to(e_total, c2.shape).copy()
    for _ in range(NEWTON_ITERATIONS):
        f  = c3 * e_free**3 + c2 * e_free**2 + c1 * e_free + c0
        df = 3.0 * c3 * e_free**2 + 2.0 * c2 * e_free + c1
        e_free = e_free - f / np.where(np.abs(df) < 1e-30, 1e-30, df)
    e_free = np.clip(e_free, 0.0, e_total)

    P_H = 1.0 + K_h_e * e_free + K_h_e ** 2 * e_free ** 2
    P_L = 1.0 + K_l_e * e_free + K_l_e ** 2 * e_free ** 2

    F = B0 + B1 * e_free + B2 * e_free ** 2
    h_free = r / np.where(F < 1e-30, 1e-30, F)

    W = h_free * (K_h_o * P_H + K_h_l * K_l_o * P_L)
    return W / (1.0 + W)


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration-grid DataFrame for theta curve extraction.

    Returns
    -------
    calc_df : pd.DataFrame
    internal_cols : list of str
    extra_kwargs : dict
        ``tf_total``, ``op_total``, and ``conc_unit_scale``.
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
        "op_total": float(model.priors.theta.theta_op_total_M),
        "conc_unit_scale": float(model.priors.theta.theta_conc_unit_scale),
    }
    return calc_df, ["genotype_idx", "titrant_name_idx"], extra


def compute_theta_samples(calc_df, param_posteriors, *, tf_total, op_total,
                          conc_unit_scale=1.0):
    """
    Compute posterior theta samples for the MWC-dimer-unfolded model.

    Parameters
    ----------
    calc_df : pd.DataFrame
    param_posteriors : dict-like
    tf_total : float
    op_total : float
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
                           conc, tf_total, op_total)
