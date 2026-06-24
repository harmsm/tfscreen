"""
PnnC — mutation effects on equilibrium constants from NN-predicted ΔΔG prior means (lac-dimer-unfolded).

Extends the lac-dimer PnnC model with a WT-level unfolding constant (ln_K_U_wt).
Per-mutation effects on K_op, K_HL, and K_E use MLP-predicted ΔΔG prior means
from LigandMPNN structural features; K_U is held at its WT value for all
genotypes (d_ln_K_U = 0) because NN features describe local folded-state
environments, not global stability.  Registry name:
``theta_lac_dimer_unfolded_lnK_nn_prior``.

Contrast with PK (all K shifts inferred from data, including K_U per mutation)
and PddG (ΔΔG prior means from a user-supplied CSV rather than a trained NN).

Identical to lac_dimer/lnK_nn_prior.py with one additional scalar parameter:

    ln_K_U_wt  — WT H→U unfolding equilibrium constant (same for all genotypes)

Per-mutation effects on the unfolded state are deliberately absent: the NN
features describe the local structural environment of the *folded* protein and
have no information about global unfolding stability.  Setting d_ln_K_U = 0
for all mutations is an explicit design choice — use lac_dimer_unfolded/lnK_mut
if per-mutation unfolding effects should be inferred from the data.

The assembled ln_K_U is constant across genotypes:
    ln_K_U[g] = ln_K_U_wt    for all g
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import pandas as pd
from flax.struct import dataclass, field
from functools import partial
from typing import Dict, Any, Union

from tfscreen.tfmodel.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix
from tfscreen.tfmodel.generative.components.theta.thermo.nn import (
    compute_nn_predictions,
    _DEFAULT_HIDDEN_SIZE,
)
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    sample_ddG,
)
from tfscreen.tfmodel.generative.components.theta.thermo.horseshoe import (
    sample_pair_ddG,
    _DEFAULT_D0,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U1_a.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

STRUCTURE_NAMES = ('H', 'HD', 'L', 'LE2')

_PROJ = jnp.array([[1., -1.,  0.,  0.],
                   [1.,  0., -1.,  0.],
                   [0.,  0.,  1., -1.]])


# ──────────────────────────────────────────────────────────────────────────────
# Priors pytree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the lnK_nn_prior lac-dimer-unfolded theta model."""

    theta_ln_K_op_wt_loc:   float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc:   float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc:    float
    theta_ln_K_E_wt_scale:  float
    theta_ln_K_U_wt_loc:    float
    theta_ln_K_U_wt_scale:  float

    theta_tf_total_M: float
    theta_op_total_M: float

    theta_nn_hidden_size: int = field(pytree_node=False)

    theta_epi_tau_scale:  float
    theta_epi_slab_scale: float
    theta_epi_slab_df:    float
    theta_epi_d0:         float


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_struct_names(data):
    if data.struct_names != STRUCTURE_NAMES:
        raise ValueError(
            f"lnK_nn_prior (lac_dimer_unfolded) requires struct_names == {STRUCTURE_NAMES}; "
            f"got {data.struct_names}."
        )


def _project_ddG(ddG):
    return ddG @ _PROJ.T


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_nn_prior lac-dimer-unfolded theta model.

    Adds ``ln_K_U_wt`` to the standard lnK_nn_prior model.  Per-mutation
    effects on K_U are fixed at zero (no d_ln_K_U parameter).
    """
    _check_struct_names(data)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    features = jnp.array(data.struct_features)
    n_chains = data.struct_n_chains
    has_epi  = (data.num_pair > 0
                and data.struct_contact_distances is not None)

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

    # ── WT equilibrium constants ─────────────────────────────────────────────
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt",
        dist.Normal(priors.theta_ln_K_op_wt_loc, priors.theta_ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt",
        dist.Normal(priors.theta_ln_K_HL_wt_loc, priors.theta_ln_K_HL_wt_scale))
    ln_K_U_wt = pyro.sample(
        f"{name}_ln_K_U_wt",
        dist.Normal(priors.theta_ln_K_U_wt_loc, priors.theta_ln_K_U_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(priors.theta_ln_K_E_wt_loc, priors.theta_ln_K_E_wt_scale))

    # ── NN prior means ────────────────────────────────────────────────────────
    nn_means = compute_nn_predictions(
        name, features, list(data.struct_names), n_chains,
        hidden_size=priors.theta_nn_hidden_size,
    )

    # ── Per-mutation per-structure ΔΔG latent variables ──────────────────────
    ddG = sample_ddG(name, list(data.struct_names), num_mut, nn_means)

    delta_lnK = _project_ddG(ddG)
    d_ln_K_op = delta_lnK[:, 0]
    d_ln_K_HL = delta_lnK[:, 1]
    d_ln_K_E  = delta_lnK[:, 2]

    pyro.deterministic(f"{name}_d_ln_K_op", d_ln_K_op)
    pyro.deterministic(f"{name}_d_ln_K_HL", d_ln_K_HL)
    pyro.deterministic(f"{name}_d_ln_K_E",  d_ln_K_E)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    ln_K_op = ln_K_op_wt + mut_scatter(d_ln_K_op)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_ln_K_HL)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_ln_K_E)[None, :]
    # K_U is homogeneous across genotypes — no per-mutation effects
    ln_K_U  = jnp.full(data.num_genotype, ln_K_U_wt)

    # ── Optional pairwise epistasis (folded-state K values only) ─────────────
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        contact_distances = jnp.array(data.struct_contact_distances)

        epi_ddG = sample_pair_ddG(
            name, list(data.struct_names), contact_distances,
            tau_scale=priors.theta_epi_tau_scale,
            slab_scale=priors.theta_epi_slab_scale,
            slab_df=priors.theta_epi_slab_df,
            d0=priors.theta_epi_d0,
        )

        epi_delta_lnK = _project_ddG(epi_ddG)
        epi_ln_K_op   = epi_delta_lnK[:, 0]
        epi_ln_K_HL   = epi_delta_lnK[:, 1]
        epi_ln_K_E    = epi_delta_lnK[:, 2]

        pyro.deterministic(f"{name}_epi_ln_K_op", epi_ln_K_op)
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_ln_K_HL)
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_ln_K_E)

        ln_K_op = ln_K_op + pair_scatter(epi_ln_K_op)
        ln_K_HL = ln_K_HL + pair_scatter(epi_ln_K_HL)
        ln_K_E  = ln_K_E  + pair_scatter(epi_ln_K_E)[None, :]

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)
    pyro.deterministic(f"{name}_ln_K_U",  ln_K_U)

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                data.titrant_conc, tf_total, op_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Guide
# ──────────────────────────────────────────────────────────────────────────────

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the lnK_nn_prior lac-dimer-unfolded theta model."""

    _check_struct_names(data)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    S       = data.num_struct
    features = jnp.array(data.struct_features)
    n_chains = data.struct_n_chains
    has_epi  = (data.num_pair > 0
                and data.struct_contact_distances is not None)

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

    ln_K_op_wt_loc   = pyro.param(f"{name}_ln_K_op_wt_loc",   jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(f"{name}_ln_K_op_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_HL_wt_loc   = pyro.param(f"{name}_ln_K_HL_wt_loc",   jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(f"{name}_ln_K_HL_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_U_wt_loc    = pyro.param(f"{name}_ln_K_U_wt_loc",    jnp.array(priors.theta_ln_K_U_wt_loc))
    ln_K_U_wt_scale  = pyro.param(f"{name}_ln_K_U_wt_scale",  jnp.array(1.0), constraint=constraints.positive)
    ln_K_E_wt_locs   = pyro.param(f"{name}_ln_K_E_wt_locs",   jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(f"{name}_ln_K_E_wt_scales", jnp.ones(T), constraint=constraints.positive)

    nn_means = compute_nn_predictions(
        name, features, list(data.struct_names), n_chains,
        hidden_size=priors.theta_nn_hidden_size,
    )

    pyro.param(f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive)

    ddG_offset_locs   = pyro.param(f"{name}_ddG_offset_locs",   jnp.zeros((S, num_mut)))
    ddG_offset_scales = pyro.param(f"{name}_ddG_offset_scales", jnp.ones((S, num_mut)), constraint=constraints.positive)

    if has_epi:
        num_pair = data.num_pair
        pyro.param(f"{name}_epi_tau_loc",    jnp.array(-2.0))
        pyro.param(f"{name}_epi_tau_scale",  jnp.array(0.5),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_c2_loc",     jnp.array(1.4))
        pyro.param(f"{name}_epi_c2_scale",   jnp.array(0.5),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_lambda_locs",  jnp.zeros((S, num_pair)))
        pyro.param(f"{name}_epi_lambda_scales",jnp.ones((S, num_pair)),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_offset_locs",  jnp.zeros((S, num_pair)))
        pyro.param(f"{name}_epi_offset_scales",jnp.ones((S, num_pair)),  constraint=constraints.positive)

    ln_K_op_wt = pyro.sample(f"{name}_ln_K_op_wt", dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(f"{name}_ln_K_HL_wt", dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    ln_K_U_wt  = pyro.sample(f"{name}_ln_K_U_wt",  dist.Normal(ln_K_U_wt_loc,  ln_K_U_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(f"{name}_ln_K_E_wt", dist.Normal(ln_K_E_wt_locs, ln_K_E_wt_scales))

    with pyro.plate(f"{name}_struct_plate", S, dim=-2):
        with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
            offsets = pyro.sample(
                f"{name}_ddG_offset",
                dist.Normal(ddG_offset_locs, ddG_offset_scales),
            )

    sigma_s = pyro.param(f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive)
    ddG_SM = nn_means.T + sigma_s[:, None] * offsets
    ddG    = ddG_SM.T

    if has_epi:
        tau_loc   = pyro.param(f"{name}_epi_tau_loc",    jnp.array(-2.0))
        tau_scale = pyro.param(f"{name}_epi_tau_scale",  jnp.array(0.5),  constraint=constraints.positive)
        c2_loc    = pyro.param(f"{name}_epi_c2_loc",     jnp.array(1.4))
        c2_scale  = pyro.param(f"{name}_epi_c2_scale",   jnp.array(0.5),  constraint=constraints.positive)
        lam_locs  = pyro.param(f"{name}_epi_lambda_locs",  jnp.zeros((S, num_pair)))
        lam_scales= pyro.param(f"{name}_epi_lambda_scales",jnp.ones((S, num_pair)), constraint=constraints.positive)
        epi_locs  = pyro.param(f"{name}_epi_offset_locs",  jnp.zeros((S, num_pair)))
        epi_scales= pyro.param(f"{name}_epi_offset_scales",jnp.ones((S, num_pair)), constraint=constraints.positive)

        tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_loc, tau_scale))
        c2_epi  = pyro.sample(f"{name}_epi_c2",  dist.LogNormal(c2_loc,  c2_scale))

        with pyro.plate(f"{name}_struct_epi_plate", S, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                lam = pyro.sample(f"{name}_epi_lambda", dist.LogNormal(lam_locs, lam_scales))
                epi_off = pyro.sample(f"{name}_epi_offset", dist.Normal(epi_locs, epi_scales))

    delta_lnK = _project_ddG(ddG)
    d_ln_K_op = delta_lnK[:, 0]
    d_ln_K_HL = delta_lnK[:, 1]
    d_ln_K_E  = delta_lnK[:, 2]

    ln_K_op = ln_K_op_wt + mut_scatter(d_ln_K_op)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_ln_K_HL)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_ln_K_E)[None, :]
    ln_K_U  = jnp.full(data.num_genotype, ln_K_U_wt)

    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)

        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_SP    = epi_off * tau_epi * _lam_tilde(lam)
        epi_ddG   = epi_SP.T
        epi_delta = _project_ddG(epi_ddG)

        ln_K_op = ln_K_op + pair_scatter(epi_delta[:, 0])
        ln_K_HL = ln_K_HL + pair_scatter(epi_delta[:, 1])
        ln_K_E  = ln_K_E  + pair_scatter(epi_delta[:, 2])[None, :]

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                data.titrant_conc, tf_total, op_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    p["theta_ln_K_op_wt_loc"]   = 23.0
    p["theta_ln_K_op_wt_scale"] = 2.0
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    p["theta_ln_K_E_wt_loc"]    = 33.4
    p["theta_ln_K_E_wt_scale"]  = 3.0
    p["theta_ln_K_U_wt_loc"]    = -12.0
    p["theta_ln_K_U_wt_scale"]  = 3.0
    p["theta_tf_total_M"] = 6.5e-7
    p["theta_op_total_M"] = 2.5e-8
    p["theta_nn_hidden_size"]   = _DEFAULT_HIDDEN_SIZE
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    p["theta_epi_d0"]          = float(_DEFAULT_D0)
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    S = data.num_struct
    g = {}
    g[f"{name}_ln_K_op_wt"]  = jnp.array(23.0)
    g[f"{name}_ln_K_HL_wt"]  = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]   = jnp.full(T, 33.4)
    g[f"{name}_ln_K_U_wt"]   = jnp.array(-12.0)
    g[f"{name}_ddG_offset"]  = jnp.zeros((S, M))
    if data.num_pair > 0 and data.struct_contact_distances is not None:
        P = data.num_pair
        g[f"{name}_epi_tau"]      = jnp.array(0.05)
        g[f"{name}_epi_c2"]       = jnp.array(4.0)
        g[f"{name}_epi_lambda"]   = jnp.ones((S, P)) * 0.5
        g[f"{name}_epi_offset"]   = jnp.zeros((S, P))
    return g


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def predict_unmeasured(
    target_genotypes,
    titrant_names,
    manual_titrant_df,
    mut_labels,
    pair_labels,
    param_posteriors,
    q_to_get,
    *,
    tf_total,
    op_total,
):
    """
    Predict theta for unmeasured genotypes (lac_dimer_unfolded/lnK_nn_prior).

    K_U is homogeneous across genotypes: ln_K_U_wt is broadcast to all rows.
    """
    import numpy as np
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )
    from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U1_a.thermo import (
        _solve_theta_np,
        _ZERO_CONC_VALUE,
    )

    target_genotypes = list(target_genotypes)
    mut_mat, pair_mat, is_valid = _build_genotype_indicators(
        target_genotypes, mut_labels, pair_labels
    )
    calc_df, geno_idx, titrant_idx = _build_prediction_grid(
        target_genotypes, titrant_names, manual_titrant_df
    )

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return np.array(v)

    ln_K_op_wt = _load("theta_ln_K_op_wt")   # (S,)
    ln_K_HL_wt = _load("theta_ln_K_HL_wt")   # (S,)
    ln_K_E_wt  = _load("theta_ln_K_E_wt")    # (S, T)
    ln_K_U_wt  = _load("theta_ln_K_U_wt")    # (S,)

    d_op = _load("theta_d_ln_K_op")   # (S, M)
    d_HL = _load("theta_d_ln_K_HL")   # (S, M)
    d_E  = _load("theta_d_ln_K_E")    # (S, M)

    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    ln_K_E_geno  = (ln_K_E_wt[:, :, None]
                    + np.einsum("sm,nm->sn", d_E, mut_mat)[:, None, :])
    # K_U is constant across genotypes — broadcast over rows
    ln_K_U_rows  = ln_K_U_wt[:, None]   # (S, 1) → broadcasts to (S, N)

    if len(pair_labels) > 0:
        epi_op = _load("theta_epi_ln_K_op")
        epi_HL = _load("theta_epi_ln_K_HL")
        epi_E  = _load("theta_epi_ln_K_E")
        ln_K_op_geno += np.einsum("sp,np->sn", epi_op, pair_mat)
        ln_K_HL_geno += np.einsum("sp,np->sn", epi_HL, pair_mat)
        ln_K_E_geno  += np.einsum("sp,np->sn", epi_E, pair_mat)[:, None, :]

    ln_K_op_rows = ln_K_op_geno[:, geno_idx]
    ln_K_HL_rows = ln_K_HL_geno[:, geno_idx]
    ln_K_E_rows  = ln_K_E_geno[:, titrant_idx, geno_idx]

    conc = calc_df["titrant_conc"].values.copy().astype(float)
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_op_rows, ln_K_HL_rows, ln_K_E_rows, ln_K_U_rows,
        conc, tf_total, op_total,
    )

    theta_samples[:, ~is_valid[geno_idx]] = np.nan

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df


def get_extract_specs(ctx):
    geno_dim   = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    titrant_dim = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    num_mut = len(ctx.mut_labels)

    geno_df = (ctx.growth_tm.df[["genotype", "genotype_idx"]]
               .drop_duplicates().copy())
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(
        input_df=geno_df,
        params_to_get=["ln_K_op", "ln_K_HL", "ln_K_U"],
        map_column="map_geno",
        get_columns=["genotype"],
        in_run_prefix="theta_",
    )]

    theta_KE_df = (ctx.growth_tm.df[["genotype", "titrant_name",
                                     "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates().copy())
    theta_KE_df["map_theta_KE"] = (theta_KE_df["titrant_name_idx"] * num_genotype
                                   + theta_KE_df["genotype_idx"])
    specs.append(dict(
        input_df=theta_KE_df,
        params_to_get=["ln_K_E"],
        map_column="map_theta_KE",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    ))

    mut_df = pd.DataFrame({"mutation": ctx.mut_labels, "map_mut": range(num_mut)})
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_op", "d_ln_K_HL", "d_ln_K_E"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        num_pair = len(ctx.pair_labels)
        pair_df = pd.DataFrame({"pair": ctx.pair_labels, "map_pair": range(num_pair)})
        specs.append(dict(
            input_df=pair_df,
            params_to_get=["epi_ln_K_op", "epi_ln_K_HL", "epi_ln_K_E"],
            map_column="map_pair",
            get_columns=["pair"],
            in_run_prefix="theta_",
        ))

    return specs


from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U1_a.thermo import (
    build_calc_df,
    compute_theta_samples,
)
