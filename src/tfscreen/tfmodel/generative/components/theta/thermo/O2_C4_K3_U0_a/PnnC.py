"""
PnnC — mutation effects on equilibrium constants from NN-predicted ΔΔG prior means (lac-dimer).

Identical to PddG, but prior means for the per-structure ΔΔG latent variables
are predicted by per-structure two-layer MLPs trained on LigandMPNN structural
features rather than read from a user-supplied CSV.  The MLPs are optimised
jointly with the model (empirical Bayes).  Requires a structural ensemble HDF5
file as input.  Registry name: ``theta_lac_dimer_lnK_nn_prior``.

Contrast with PK (Δln_K inferred from data alone, no structure) and PddG
(ΔΔG prior means from a user-supplied CSV rather than a trained NN).

Per-mutation per-structure ΔΔG latents with MLP-predicted prior means
(empirical-Bayes via LigandMPNN structural features):

    ΔΔG[m, s]  ~ Normal( MLP_s(features[m, s, :]) · n_chains[s],  σ_s )
    σ_s        — per-structure trust scale (``pyro.param``, init=1.0)

When σ_s → 0 the posterior collapses to the deterministic MLP output.
When σ_s → ∞ the prior is flat and ΔΔG is unconstrained (like ``lnK_mut``).  The data learn the balance per structure.

The four per-structure ΔΔG values are projected onto Δln_K via the fixed
thermodynamic matrix:

    Δln_K_op[m] = ΔΔG[m, H]  − ΔΔG[m, HD]
    Δln_K_HL[m] = ΔΔG[m, H]  − ΔΔG[m, L]
    Δln_K_E[m]  = ΔΔG[m, L]  − ΔΔG[m, LE2]

Epistasis (when ``data.num_pair > 0`` and ``data.struct_contact_distances``
is not None) uses a distance-dependent regularised horseshoe applied at the
ΔΔG level per structure, projected to Δln_K in the same way.  Distant pairs
are strongly shrunk toward zero.

Data requirements
-----------------
``data`` must have (all stored as ``pytree_node=False`` statics):
    struct_names            — tuple ('H', 'HD', 'L', 'LE2') in that order
    struct_features         — (M, S, 60) float32
    struct_n_chains         — (S,) int32
    mut_geno_matrix         — (M, G) for mutation → genotype scatter
    num_mutation            — M
    num_struct              — S (must equal 4 for lac dimer)

Optional (for distance-dependent epistasis):
    struct_contact_distances — (P, S) float32
    struct_contact_pair_idx  — (P, 2) int32 (not used in model, stored for
                                reference only)
    pair_nnz_pair_idx        — (nnz,) int32
    pair_nnz_geno_idx        — (nnz,) int32
    num_pair                 — P
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
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# Expected structure names — column order matches _PROJ below.
STRUCTURE_NAMES = ('H', 'HD', 'L', 'LE2')

# Fixed thermodynamic projection: rows = (K_op, K_HL, K_E), cols = (H, HD, L, LE2).
# Δln_K = ddG @ _PROJ.T   where ddG is (..., 4) → result (..., 3)
_PROJ = jnp.array([[1., -1.,  0.,  0.],   # K_op = ΔΔG_H − ΔΔG_HD
                   [1.,  0., -1.,  0.],    # K_HL = ΔΔG_H − ΔΔG_L
                   [0.,  0.,  1., -1.]])   # K_E  = ΔΔG_L − ΔΔG_LE2  (S=4, K=3)


# ──────────────────────────────────────────────────────────────────────────────
# Priors pytree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the lnK_nn_prior lac-dimer theta model."""

    # WT equilibrium constants (Normal in ln-K space)
    theta_ln_K_op_wt_loc:   float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc:   float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc:    float
    theta_ln_K_E_wt_scale:  float

    # Physical concentrations (M); Sochor, PeerJ 2014
    theta_tf_total_M: float    # ≈ 6.5e-7 M (650 nM)
    theta_op_total_M: float    # ≈ 2.5e-8 M (25 nM)

    # Per-structure MLP hidden width (static shape — not a float parameter)
    theta_nn_hidden_size: int = field(pytree_node=False)

    # Regularised horseshoe for pairwise epistasis
    theta_epi_tau_scale:  float
    theta_epi_slab_scale: float
    theta_epi_slab_df:    float

    # Distance scale for distance-dependent λ prior (Å)
    theta_epi_d0: float


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_struct_names(data):
    if data.struct_names != STRUCTURE_NAMES:
        raise ValueError(
            f"lnK_nn_prior requires struct_names == {STRUCTURE_NAMES}; "
            f"got {data.struct_names}."
        )


def _project_ddG(ddG):
    """
    Project per-structure ΔΔG to Δln_K via the fixed thermodynamic matrix.

    Parameters
    ----------
    ddG : jnp.ndarray, shape (..., 4)
        ΔΔG for each structure (H, HD, L, LE2 in that order).

    Returns
    -------
    jnp.ndarray, shape (..., 3)
        Δln_K for (K_op, K_HL, K_E).
    """
    return ddG @ _PROJ.T


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_nn_prior lac-dimer theta model.

    Samples WT equilibrium constants plus per-mutation per-structure ΔΔG
    latent variables with LigandMPNN NN-predicted prior means.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample/param sites.
    data : GrowthData or BindingData
        Must carry ``struct_features``, ``struct_names``, ``struct_n_chains``,
        ``mut_geno_matrix``, and ``num_mutation``.
    priors : ModelPriors

    Returns
    -------
    ThetaParam
    """
    _check_struct_names(data)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    features = jnp.array(data.struct_features)             # (M, S, 60)
    n_chains = data.struct_n_chains                         # (S,) int
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
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(priors.theta_ln_K_E_wt_loc, priors.theta_ln_K_E_wt_scale))

    # ── NN prior means (pyro.param — shared with guide) ──────────────────────
    nn_means = compute_nn_predictions(
        name, features, list(data.struct_names), n_chains,
        hidden_size=priors.theta_nn_hidden_size,
    )  # (M, S)

    # ── Per-mutation per-structure ΔΔG latent variables ──────────────────────
    # sigma_s (pyro.param) controls trust in the NN; sample_ddG also records it.
    ddG = sample_ddG(name, list(data.struct_names), num_mut, nn_means)  # (M, S)

    # Project (M, 4) → (M, 3) Δln_K
    delta_lnK = _project_ddG(ddG)        # (M, 3)
    d_ln_K_op = delta_lnK[:, 0]          # (M,)
    d_ln_K_HL = delta_lnK[:, 1]          # (M,)
    d_ln_K_E  = delta_lnK[:, 2]          # (M,) — uniform across effectors

    pyro.deterministic(f"{name}_d_ln_K_op",   d_ln_K_op)
    pyro.deterministic(f"{name}_d_ln_K_HL",   d_ln_K_HL)
    pyro.deterministic(f"{name}_d_ln_K_E",    d_ln_K_E)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    ln_K_op = ln_K_op_wt + mut_scatter(d_ln_K_op)                          # (G,)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_ln_K_HL)                          # (G,)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_ln_K_E)[None, :]          # (T, G)

    # ── Optional pairwise epistasis ───────────────────────────────────────────
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        contact_distances = jnp.array(data.struct_contact_distances)  # (P, S)

        epi_ddG      = sample_pair_ddG(
            name, list(data.struct_names), contact_distances,
            tau_scale=priors.theta_epi_tau_scale,
            slab_scale=priors.theta_epi_slab_scale,
            slab_df=priors.theta_epi_slab_df,
            d0=priors.theta_epi_d0,
        )  # (P, S)

        epi_delta_lnK = _project_ddG(epi_ddG)     # (P, 3)
        epi_ln_K_op   = epi_delta_lnK[:, 0]       # (P,)
        epi_ln_K_HL   = epi_delta_lnK[:, 1]       # (P,)
        epi_ln_K_E    = epi_delta_lnK[:, 2]       # (P,) — uniform across effectors

        pyro.deterministic(f"{name}_epi_ln_K_op", epi_ln_K_op)
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_ln_K_HL)
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_ln_K_E)

        ln_K_op = ln_K_op + pair_scatter(epi_ln_K_op)
        ln_K_HL = ln_K_HL + pair_scatter(epi_ln_K_HL)
        ln_K_E  = ln_K_E  + pair_scatter(epi_ln_K_E)[None, :]

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)

    # ── Population moments for transformation model ───────────────────────────
    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                data.titrant_conc, tf_total, op_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Guide
# ──────────────────────────────────────────────────────────────────────────────

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the lnK_nn_prior lac-dimer theta model."""

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

    # ── Variational params for WT K values ────────────────────────────────────
    ln_K_op_wt_loc   = pyro.param(f"{name}_ln_K_op_wt_loc",   jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(f"{name}_ln_K_op_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_HL_wt_loc   = pyro.param(f"{name}_ln_K_HL_wt_loc",   jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(f"{name}_ln_K_HL_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_E_wt_locs   = pyro.param(f"{name}_ln_K_E_wt_locs",   jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(f"{name}_ln_K_E_wt_scales", jnp.ones(T), constraint=constraints.positive)

    # ── NN forward pass (registers same pyro.param sites as model) ────────────
    nn_means = compute_nn_predictions(
        name, features, list(data.struct_names), n_chains,
        hidden_size=priors.theta_nn_hidden_size,
    )  # (M, S)

    # ── Register sigma_s so the param store includes it before sampling ───────
    pyro.param(
        f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive,
    )

    # ── Variational params for ddG offsets, shape (S, M) in plate layout ─────
    ddG_offset_locs   = pyro.param(f"{name}_ddG_offset_locs",
                                   jnp.zeros((S, num_mut)))
    ddG_offset_scales = pyro.param(f"{name}_ddG_offset_scales",
                                   jnp.ones((S, num_mut)),
                                   constraint=constraints.positive)

    # ── Optional epistasis variational params ─────────────────────────────────
    if has_epi:
        num_pair = data.num_pair
        tau_loc   = pyro.param(f"{name}_epi_tau_loc",    jnp.array(-2.0))
        tau_scale = pyro.param(f"{name}_epi_tau_scale",  jnp.array(0.5),  constraint=constraints.positive)
        c2_loc    = pyro.param(f"{name}_epi_c2_loc",     jnp.array(1.4))
        c2_scale  = pyro.param(f"{name}_epi_c2_scale",   jnp.array(0.5),  constraint=constraints.positive)
        lam_locs  = pyro.param(f"{name}_epi_lambda_locs",  jnp.zeros((S, num_pair)))
        lam_scales= pyro.param(f"{name}_epi_lambda_scales",jnp.ones((S, num_pair)),  constraint=constraints.positive)
        epi_locs  = pyro.param(f"{name}_epi_offset_locs",  jnp.zeros((S, num_pair)))
        epi_scales= pyro.param(f"{name}_epi_offset_scales",jnp.ones((S, num_pair)),  constraint=constraints.positive)

    # ── Sample WT K values ────────────────────────────────────────────────────
    ln_K_op_wt = pyro.sample(f"{name}_ln_K_op_wt", dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(f"{name}_ln_K_HL_wt", dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(f"{name}_ln_K_E_wt", dist.Normal(ln_K_E_wt_locs, ln_K_E_wt_scales))

    # ── Sample ddG offsets (matches model's prior.sample_ddG plate layout) ────
    with pyro.plate(f"{name}_struct_plate", S, dim=-2):
        with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
            offsets = pyro.sample(
                f"{name}_ddG_offset",
                dist.Normal(ddG_offset_locs, ddG_offset_scales),
            )  # (S, M)

    # Reconstruct ddG from guide offsets and current sigma_s
    sigma_s = pyro.param(
        f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive,
    )  # (S,) — already registered above; second call returns same tensor
    ddG_SM = nn_means.T + sigma_s[:, None] * offsets   # (S, M)
    ddG    = ddG_SM.T                                   # (M, S)

    # ── Sample epistasis ──────────────────────────────────────────────────────
    if has_epi:
        tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_loc, tau_scale))
        c2_epi  = pyro.sample(f"{name}_epi_c2",  dist.LogNormal(c2_loc,  c2_scale))

        with pyro.plate(f"{name}_struct_epi_plate", S, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                lam = pyro.sample(
                    f"{name}_epi_lambda",
                    dist.LogNormal(lam_locs, lam_scales),
                )
                epi_off = pyro.sample(
                    f"{name}_epi_offset",
                    dist.Normal(epi_locs, epi_scales),
                )

    # ── Assemble per-genotype K values ────────────────────────────────────────
    delta_lnK = _project_ddG(ddG)
    d_ln_K_op = delta_lnK[:, 0]
    d_ln_K_HL = delta_lnK[:, 1]
    d_ln_K_E  = delta_lnK[:, 2]

    ln_K_op = ln_K_op_wt + mut_scatter(d_ln_K_op)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_ln_K_HL)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_ln_K_E)[None, :]

    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)

        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_SP       = epi_off * tau_epi * _lam_tilde(lam)   # (S, P)
        epi_ddG      = epi_SP.T                              # (P, S)
        epi_delta    = _project_ddG(epi_ddG)                 # (P, 3)

        ln_K_op = ln_K_op + pair_scatter(epi_delta[:, 0])
        ln_K_HL = ln_K_HL + pair_scatter(epi_delta[:, 1])
        ln_K_E  = ln_K_E  + pair_scatter(epi_delta[:, 2])[None, :]

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                data.titrant_conc, tf_total, op_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # WT K priors (same values as lnK_mut for continuity)
    p["theta_ln_K_op_wt_loc"]   = 23.0
    p["theta_ln_K_op_wt_scale"] = 2.0
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    p["theta_ln_K_E_wt_loc"]    = 33.4
    p["theta_ln_K_E_wt_scale"]  = 3.0
    # Physical concentrations
    p["theta_tf_total_M"] = 6.5e-7
    p["theta_op_total_M"] = 2.5e-8
    # NN architecture
    p["theta_nn_hidden_size"]   = _DEFAULT_HIDDEN_SIZE
    # Horseshoe hyperparameters for pairwise epistasis
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    p["theta_epi_d0"]          = float(_DEFAULT_D0)
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T       = data.num_titrant_name
    M       = data.num_mutation
    S       = data.num_struct
    g = {}
    g[f"{name}_ln_K_op_wt"]  = jnp.array(23.0)
    g[f"{name}_ln_K_HL_wt"]  = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]   = jnp.full(T, 33.4)
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
    Predict theta for unmeasured genotypes using NN-prior ln-K assembly.

    Identical interface to the lnK_mut variant.  The key difference is that
    ``d_ln_K_E`` has shape ``(S, M)`` (no T dimension) — the NN prior learns a
    single per-mutation effect broadcast uniformly across effector species.

    Parameters
    ----------
    target_genotypes : list[str]
    titrant_names : list[str]
    manual_titrant_df : pd.DataFrame
    mut_labels : list[str]
    pair_labels : list[str]
    param_posteriors : dict-like
    q_to_get : dict
    tf_total : float
    op_total : float

    Returns
    -------
    pd.DataFrame
    """
    import numpy as np
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )
    from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (
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

    d_op = _load("theta_d_ln_K_op")   # (S, M)
    d_HL = _load("theta_d_ln_K_HL")   # (S, M)
    d_E  = _load("theta_d_ln_K_E")    # (S, M) — scalar per mutation, no T dim

    # scalar Ks: (S, N)
    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    # T-dim K: broadcast scalar d_E across T → (S, T, N)
    ln_K_E_geno = (ln_K_E_wt[:, :, None]
                   + np.einsum("sm,nm->sn", d_E, mut_mat)[:, None, :])

    if len(pair_labels) > 0:
        epi_op = _load("theta_epi_ln_K_op")   # (S, P)
        epi_HL = _load("theta_epi_ln_K_HL")   # (S, P)
        epi_E  = _load("theta_epi_ln_K_E")    # (S, P) — scalar, no T dim
        ln_K_op_geno += np.einsum("sp,np->sn", epi_op, pair_mat)
        ln_K_HL_geno += np.einsum("sp,np->sn", epi_HL, pair_mat)
        ln_K_E_geno  += np.einsum("sp,np->sn", epi_E, pair_mat)[:, None, :]

    ln_K_op_rows = ln_K_op_geno[:, geno_idx]
    ln_K_HL_rows = ln_K_HL_geno[:, geno_idx]
    ln_K_E_rows  = ln_K_E_geno[:, titrant_idx, geno_idx]

    conc = calc_df["titrant_conc"].values.copy().astype(float)
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_op_rows, ln_K_HL_rows, ln_K_E_rows, conc, tf_total, op_total
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
        params_to_get=["ln_K_op", "ln_K_HL"],
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

    # Per-mutation projected ΔΔG → Δln_K
    mut_df = pd.DataFrame({
        "mutation": ctx.mut_labels,
        "map_mut":  range(num_mut),
    })
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_op", "d_ln_K_HL", "d_ln_K_E"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        num_pair = len(ctx.pair_labels)
        pair_df = pd.DataFrame({
            "pair":     ctx.pair_labels,
            "map_pair": range(num_pair),
        })
        specs.append(dict(
            input_df=pair_df,
            params_to_get=["epi_ln_K_op", "epi_ln_K_HL", "epi_ln_K_E"],
            map_column="map_pair",
            get_columns=["pair"],
            in_run_prefix="theta_",
        ))

    return specs


from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (
    build_calc_df,
    compute_theta_samples,
)
