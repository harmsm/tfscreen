"""
Shared thermodynamic functions for the lac-dimer-unfolded partition-function θ model.

Extends the four-state lac-dimer model with a globally unfolded state U.
Destabilising mutations shift population into U, where neither operator nor
effector can bind, collapsing θ toward zero at all effector concentrations.
Used by ``theta_lac_dimer_unfolded_lnK_mut`` (PK),
``theta_lac_dimer_unfolded_lnK_ddG_prior`` (PddG), and
``theta_lac_dimer_unfolded_lnK_nn_prior`` (PnnC).

Five-state partition function for the TF homodimer:

    State      Weight
    H          1                               (apo, high-affinity conformation)
    H·op       K_op · [op_free]                (operator-bound, repressing state)
    L          K_HL                            (low-affinity conformation)
    L·E²       K_HL · K_E · [E_free]²          (effector-bound, n=2 hardcoded)
    U          K_U                             (globally unfolded; no DNA/effector binding)

    Z     = 1 + K_op·[op_free] + K_HL + K_HL·K_E·[E_free]² + K_U
    θ     = K_op·[op_free] / Z

The U state is the H→U equilibrium: K_U = [U]/[H].  In WT K_U << 1 (U is
strongly disfavoured).  Mutations accumulate destabilising effects as additive
shifts in ln(K_U); when K_U dominates Z, θ collapses toward zero for all
conditions regardless of effector concentration.

U has no DNA-binding or effector-binding interactions, so the thermodynamic
projection matrix for the folded states is unchanged.  The only modification
to the Newton solve for free effector is that K_U appears in Z₀ (the
ligand-free part of the partition function), which enters the cubic mass
balance for free effector.

Operator depletion approximation, free-effector cubic, and units are identical
to lac_dimer/thermo.py.  See that module for details.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.struct import dataclass

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
    ln_K_U:   jnp.ndarray   # (G,)    — per-genotype log H→U unfolding equilibrium
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

    Identical to lac_dimer/thermo.py; Z0 now includes the K_U term.
    """
    coeff_a = a
    coeff_b = a * (2.0 * tf_total - e_total)
    coeff_c = Z0
    coeff_d = -Z0 * e_total

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


def _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, titrant_conc, tf_total, op_total):
    """
    Evaluate the five-state partition-function theta for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_op : (G,)
    ln_K_HL : (G,)
    ln_K_E  : (T, G)
    ln_K_U  : (G,)
    titrant_conc : (C,)
    tf_total : scalar
    op_total : scalar

    Returns
    -------
    theta : (T, C, G)
    """
    K_op = jnp.exp(ln_K_op)[None, None, :]     # (1, 1, G)
    K_HL = jnp.exp(ln_K_HL)[None, None, :]     # (1, 1, G)
    K_E  = jnp.exp(ln_K_E)[:, None, :]         # (T, 1, G)
    K_U  = jnp.exp(ln_K_U)[None, None, :]      # (1, 1, G)
    e_total = titrant_conc[None, :, None]       # (1, C, 1)

    # Z₀ includes K_U: the ligand-free part of the partition function
    Z0 = 1.0 + K_op * op_total + K_HL + K_U    # (1, 1, G)
    a  = K_HL * K_E                             # (T, 1, G)

    e_free = _solve_free_effector(e_total, tf_total, a, Z0)  # (T, C, G)

    w_Hop = K_op * op_total                     # (1, 1, G)
    w_LE  = K_HL * K_E * e_free**2             # (T, C, G)
    Z     = 1.0 + w_Hop + K_HL + w_LE + K_U   # (T, C, G)

    return w_Hop / Z                            # (T, C, G)


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
# Numpy extraction helpers (shared across lac_dimer_unfolded theta components)
# ---------------------------------------------------------------------------

_ZERO_CONC_VALUE = 1e-20


def _solve_theta_np(ln_K_op, ln_K_HL, ln_K_E, ln_K_U, conc, tf_total, op_total):
    """
    Pure-numpy Newton solve for lac_dimer_unfolded theta at per-row K values.

    Parameters
    ----------
    ln_K_op : (S, N)
    ln_K_HL : (S, N)
    ln_K_E  : (S, N)
    ln_K_U  : (S, N)  or broadcastable to (S, N)
    conc    : (N,)  — effector concentrations in M (zero already replaced)
    tf_total : float
    op_total : float

    Returns
    -------
    theta : np.ndarray, shape (S, N)
    """
    K_op = np.exp(ln_K_op)
    K_HL = np.exp(ln_K_HL)
    K_E  = np.exp(ln_K_E)
    K_U  = np.exp(ln_K_U)

    e_total = conc[np.newaxis, :]
    Z0      = 1.0 + K_op * op_total + K_HL + K_U
    a       = K_HL * K_E
    coeff_b = a * (2.0 * tf_total - e_total)

    e_free = e_total * np.ones_like(a)
    for _ in range(NEWTON_ITERATIONS):
        f  = a * e_free**3 + coeff_b * e_free**2 + Z0 * e_free - Z0 * e_total
        df = 3.0 * a * e_free**2 + 2.0 * coeff_b * e_free + Z0
        e_free = e_free - f / np.where(np.abs(df) < 1e-30, 1e-30, df)
    e_free = np.clip(e_free, 0.0, e_total)

    w_Hop = K_op * op_total
    w_LE  = a * e_free**2
    Z     = 1.0 + w_Hop + K_HL + w_LE + K_U
    return w_Hop / Z


def build_calc_df(model, manual_titrant_df):
    """
    Build the concentration grid DataFrame for theta curve extraction.

    Shared by all lac_dimer_unfolded theta components.

    Returns
    -------
    calc_df : pd.DataFrame
    internal_cols : list of str
    extra_kwargs : dict
        ``tf_total`` and ``op_total`` scalars (in M).
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
    }
    return calc_df, ["genotype_idx", "titrant_name_idx"], extra


def compute_theta_samples(calc_df, param_posteriors, *, tf_total, op_total):
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
    op_total : float

    Returns
    -------
    theta_samples : np.ndarray, shape (S, N)
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    geno_indices    = calc_df["genotype_idx"].values.astype(int)
    titrant_indices = calc_df["titrant_name_idx"].values.astype(int)

    conc = calc_df["titrant_conc"].values.copy().astype(float)
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
                           conc, tf_total, op_total)
