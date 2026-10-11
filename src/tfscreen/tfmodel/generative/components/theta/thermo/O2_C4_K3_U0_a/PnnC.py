"""
PnnC — mutation effects on equilibrium constants from NN-predicted ΔΔG prior means (lac-dimer).

Identical to PddG, but prior means for the per-structure ΔΔG latent variables
are predicted by per-structure two-layer MLPs trained on LigandMPNN structural
features rather than read from a user-supplied CSV.  The MLPs are optimized
jointly with the model (empirical Bayes).  Requires a structural ensemble HDF5
file as input.  Registry name: ``theta_lac_dimer_lnK_nn_prior``.

Contrast with PK (Δln_K inferred from data alone, no structure) and PddG
(ΔΔG prior means from a user-supplied CSV rather than a trained NN).

Per-mutation per-structure ΔΔG latents with MLP-predicted prior means
(empirical-Bayes via LigandMPNN structural features):

    σ_s        ~ LogNormal(theta_ddG_sigma_loc, theta_ddG_sigma_scale), one per
                 structure (SD of the true ΔΔG about its prior mean)
    Δln K[m]   ~ MVN( P · μ[m],  P · diag(σ_s²) · Pᵀ )      (site theta_dlnK)
    μ[m, s]    = MLP_s(features[m, s, :]) · n_chains[s]

with P = ``DLNK_PROJ``, the projection below. This is the distribution the
per-structure model ΔΔG[m, s] ~ Normal(MLP_s(features[m, s, :]) ·
n_chains[s], σ_s) induces on the constants, which are all the data see
(thermo/prior.py). The per-structure ΔΔG is the deterministic site
``theta_ddG``: the conditional mean under that per-structure prior, whose
common shift across structures comes from the prior means, not the data (no
unfolded state anchors it; score per-structure ΔΔG only for U1 models).

When σ_s → 0 the posterior collapses to the deterministic MLP output.
When σ_s → ∞ the prior is flat and ΔΔG is unconstrained (like ``lnK_mut``).  The data set
the balance per structure.

The four per-structure ΔΔG values are projected onto Δln_K via the fixed
thermodynamic matrix:

    Δln_K_op[m] = ΔΔG[m, H]  − ΔΔG[m, HD]
    Δln_K_HL[m] = ΔΔG[m, H]  − ΔΔG[m, L]
    Δln_K_E[m]  = (ΔΔG[m, L] − ΔΔG[m, LE2]) / 2

ΔΔG_X adds to state X's free energy, and Δln K_{A→B} = ΔΔG_A − ΔΔG_B. K_E
is the per-site constant of L (M⁻¹); LE2 has weight K_E²·e² relative to L, so
ΔΔG_L − ΔΔG_LE2 = 2·Δln K_E. The model is M2, the MWC dimer of
``thermo/mwc.py`` with K_h_e = K_l_o = 0.

Epistasis (when ``data.num_pair > 0`` and ``data.struct_contact_distances``
is not None) uses a distance-dependent regularized horseshoe applied at the
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
from tfscreen.tfmodel.generative.components._horseshoe import regularized_scale
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
from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    ddG_extract_specs,
    dlnK_guesses,
    guide_dlnK,
    projection,
    sample_dlnK,
)
from tfscreen.tfmodel.generative.components.theta.thermo.horseshoe import (
    sample_pair_ddG,
    _DEFAULT_D0,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (
    DEFAULT_CONC_UNIT_SCALE,
    DEFAULT_LN_K_E_WT_LOC,
    DEFAULT_LN_K_HL_WT_LOC,
    DEFAULT_LN_K_OP_WT_LOC,
    DEFAULT_TF_TOTAL_M,
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# Structure basis and projection rows (thermo/mwc.py). Convention: ΔΔG_X
# adds to state X's free energy (kT), Δln K_{A→B} = ΔΔG_A − ΔΔG_B, and the
# effector rows are per site, half of ΔΔG_X − ΔΔG_XE2.
STRUCTURE_NAMES = mwc.STRUCTURE_NAMES_C4

# Δln_K = ddG @ _PROJ.T; rows = constants, cols = STRUCTURE_NAMES.
_PROJ = jnp.asarray(mwc.PROJ_C4)

# Δln K (sampled, M × 3) = ΔΔG @ DLNK_PROJ.T (thermo/prior.py).
DLNK_PROJ = projection(mwc.PROJ_C4)


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

    # Total TF (M, monomer units); see thermo.py Units
    theta_tf_total_M: float   # M, monomer units; default 1e-5 (thermo/mwc.py)
    # Factor converting data titrant_conc (mM) to M (1e-3); see thermo.py.
    theta_conc_unit_scale: float

    # Per-structure MLP hidden width (static shape — not a float parameter)
    theta_nn_hidden_size: int = field(pytree_node=False)

    # Regularized horseshoe for pairwise epistasis
    theta_epi_tau_scale:  float
    theta_epi_slab_scale: float
    theta_epi_slab_df:    float

    # Distance scale for distance-dependent λ prior (Å)
    theta_epi_d0: float

    # LogNormal prior on the per-structure ΔΔG scale σ_s (log kT); see
    # thermo/prior.py.
    theta_ddG_sigma_loc:   float
    theta_ddG_sigma_scale: float


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
    conc_unit_scale = priors.theta_conc_unit_scale

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

    # ── Per-mutation Δln K under the structural prior ─────────────────────────
    # Δln K[m] ~ MVN(P · prior_mean[m], P · diag(σ_s²) · Pᵀ), σ_s per state
    # (thermo/prior.py); the post-hoc per-state ΔΔG is the site {name}_ddG.
    dlnK = sample_dlnK(name, list(data.struct_names), num_mut, nn_means, DLNK_PROJ,
                       sigma_loc=priors.theta_ddG_sigma_loc,
                       sigma_scale=priors.theta_ddG_sigma_scale)  # (M, C)
    delta_lnK = dlnK[:, :3]
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
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


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
    conc_unit_scale = priors.theta_conc_unit_scale

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

    # ── Sample σ_s and Δln K (matches prior.sample_dlnK) ────────────────────
    dlnK = guide_dlnK(name, num_mut, S, DLNK_PROJ, jnp.zeros((num_mut, S)))   # (M, C)

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
    delta_lnK = dlnK[:, :3]
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
            return regularized_scale(lam, tau_epi, c2_epi)

        epi_SP       = epi_off * tau_epi * _lam_tilde(lam)   # (S, P)
        epi_ddG      = epi_SP.T                              # (P, S)
        epi_delta    = _project_ddG(epi_ddG)                 # (P, 3)

        ln_K_op = ln_K_op + pair_scatter(epi_delta[:, 0])
        ln_K_HL = ln_K_HL + pair_scatter(epi_delta[:, 1])
        ln_K_E  = ln_K_E  + pair_scatter(epi_delta[:, 2])[None, :]

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # Wild-type defaults: see thermo/mwc.py (Sochor 2014 constants and the
    # 80 µM midpoint calibration at tf_total = 1e-5 M monomer).
    p["theta_ln_K_op_wt_loc"]   = DEFAULT_LN_K_OP_WT_LOC   # = K_h_o; Sochor: 4.2e8 M⁻¹
    p["theta_ln_K_op_wt_scale"] = 2.0
    p["theta_ln_K_HL_wt_loc"]   = DEFAULT_LN_K_HL_WT_LOC   # = K_h_l; Sochor: 6.3
    p["theta_ln_K_HL_wt_scale"] = 3.0
    p["theta_ln_K_E_wt_loc"]    = DEFAULT_LN_K_E_WT_LOC   # = K_l_e, per site (M⁻¹); midpoint 80 µM
    p["theta_ln_K_E_wt_scale"]  = 3.0
    p["theta_tf_total_M"] = DEFAULT_TF_TOTAL_M   # M, monomer units (5e-6 M dimer)
    p["theta_conc_unit_scale"] = DEFAULT_CONC_UNIT_SCALE   # titrant_conc mM → M
    # NN architecture
    p["theta_nn_hidden_size"]   = _DEFAULT_HIDDEN_SIZE
    # Horseshoe hyperparameters for pairwise epistasis
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    p["theta_epi_d0"]          = float(_DEFAULT_D0)
    p["theta_ddG_sigma_loc"]   = DEFAULT_DDG_SIGMA_LOC     # median σ_s 1 kT
    p["theta_ddG_sigma_scale"] = DEFAULT_DDG_SIGMA_SCALE   # 95%: 0.1-10 kT
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T       = data.num_titrant_name
    M       = data.num_mutation
    S       = data.num_struct
    g = {}
    g[f"{name}_ln_K_op_wt"]  = jnp.array(DEFAULT_LN_K_OP_WT_LOC)
    g[f"{name}_ln_K_HL_wt"]  = jnp.array(DEFAULT_LN_K_HL_WT_LOC)
    g[f"{name}_ln_K_E_wt"]   = jnp.full(T, DEFAULT_LN_K_E_WT_LOC)
    # Δln K starts at P · (ΔΔG prior means), σ_s at 1 kT.
    g.update(dlnK_guesses(name, jnp.zeros((M, S)), DLNK_PROJ))
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
    conc_unit_scale=1.0,
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
    conc_unit_scale : float
        Converts titrant_conc to M (1e-3 for mM data).

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

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_op_rows, ln_K_HL_rows, ln_K_E_rows, conc, tf_total
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

    # Per-structure ddG and its prior mean, in STRUCTURE_NAMES order (the
    # order the model samples in and the simulator labels its truth with).
    specs.extend(ddG_extract_specs(ctx, STRUCTURE_NAMES))

    return specs


from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.thermo import (
    build_calc_df,
    compute_theta_samples,
)
