"""
PddG — mutation effects on equilibrium constants from user-supplied ΔΔG prior means (MWC dimer unfolded).

Extends the MWC dimer PddG model with a WT-level unfolding constant (ln_K_u_wt).
Per-mutation effects on the five folded-state K values (K_h_l, K_h_o, K_l_o,
K_h_e, K_l_e) and on K_u are derived from user-supplied structural ΔΔG (CSV:
columns mut, H, HO, L, LO, HE2, LE2).  Registry name:
``theta_mwc_dimer_unfolded_lnK_ddG_prior``.

Contrast with PK (all K shifts inferred from data, including K_u per mutation)
and PnnC (ΔΔG prior means from a per-structure MLP rather than a user CSV).

Implements the standard MWC two-state lac-dimer model (``thermo/mwc.py``;
constants and values from Sochor 2014) with five
equilibrium constants plus a global unfolding equilibrium, using six structural
states as the free-energy basis:

    Structures: H, HO, L, LO, HE2, LE2

    σ_s        ~ LogNormal(theta_ddG_sigma_loc, theta_ddG_sigma_scale), one per
                 structure (SD of the true ΔΔG about its prior mean)
    Δln K[m]   ~ MVN( P · ddG_prior[m],  P · diag(σ_s²) · Pᵀ )      (site theta_dlnK)

with P = ``DLNK_PROJ``, the projection below plus the unfolding row. This is
the distribution the per-structure model ΔΔG[m, s] ~ Normal(ddG_prior[m, s],
σ_s) induces on the constants, which are all the data see (thermo/prior.py).
The per-structure ΔΔG is the deterministic site ``theta_ddG``: P⁻¹ · Δln K
exactly (ΔΔG_U = 0; the unfolding row makes P square and invertible).

where ``ddG_prior`` is a (M, S) float32 array loaded from a CSV spreadsheet
(columns: ``mut``, ``H``, ``HO``, ``L``, ``LO``, ``HE2``, ``LE2``).

Projection from per-structure ΔΔG to Δln_K is identical to lnK_nn_prior
(rows = K, cols = structures):

    Δln_K_h_l = ΔΔG_H  − ΔΔG_L
    Δln_K_h_o = ΔΔG_H  − ΔΔG_HO
    Δln_K_h_e = (ΔΔG_H − ΔΔG_HE2) / 2
    Δln_K_l_o = ΔΔG_L  − ΔΔG_LO
    Δln_K_l_e = (ΔΔG_L − ΔΔG_LE2) / 2

Unfolding extension
-------------------
A WT-level unfolding constant ``ln_K_u_wt`` is sampled, and each mutation
shifts it through the H column of its ΔΔG.  A mutation shifts the free energy
of each folded structure X by ΔΔG_X and leaves the unfolded state alone
(ΔΔG_U = 0, the standard assumption).  Because K_u = [U]/[H], the projection
convention above (Δln K_{A→B} = ΔΔG_A − ΔΔG_B) gives

    Δln_K_u   = ΔΔG_H − ΔΔG_U = ΔΔG_H
    ln_K_u_wt ~ Normal(theta_ln_K_u_wt_loc, theta_ln_K_u_wt_scale)
    ln_K_u[g] = ln_K_u_wt + Σ_m ΔΔG_H[m]   (plus the pairwise ΔΔG_H epistasis)

A mutation that raises every folded state by the same ΔΔG leaves the five
folded-state constants unchanged and raises K_u by ΔΔG: a pure stability
effect.  No extra structure is needed.  The per-mutation shift is exposed as
``d_ln_K_u`` (and ``epi_ln_K_u`` per pair), the same name the PK model uses
for its free per-mutation unfolding shift.

Data requirements
-----------------
``data`` must have (stored as ``pytree_node=False`` statics):
    struct_names            — tuple containing at least {'H', 'HO', 'L', 'LO', 'HE2', 'LE2'} (extras ignored)
    struct_features         — (M, S) float32  per-mutation per-structure ΔΔG prior means
    mut_geno_matrix         — (M, G) for mutation → genotype scatter
    num_mutation            — M
    num_struct              — S (at least 6)

Input CSV format (passed via ``thermo_data``)
------------------------------------------------------
A CSV with a mutation column (``mut``, ``Mutation`` or ``mutation``) and one
column per structure (``H``, ``HO``, ``L``, ``LO``, ``HE2``, ``LE2``, any order; other columns are
ignored, and ``thermo_options`` ``column_map`` can name the column for a
structure).  ΔΔG_X is the shift a mutation makes to state X's free energy,
so positive destabilizes X, and Δln K_{A→B} = ΔΔG_A − ΔΔG_B.  The model
needs kT; ``thermo_options`` ``ddG_units: kcal/mol`` (or ``kJ/mol``, with
``temperature_K``) converts at load, and the default ``kT`` uses the values
as-is.  The loader (``thermo.io.load_ddG_prior_csv``) reports coverage, errors
on zero coverage or a wt-residue mismatch (numbering offset; see
``residue_offset``), and gives absent mutations a prior mean of 0.0 with a
warning when more than ``max_missing_fraction`` are absent.
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
from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    ddG_sigma_fixed,
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
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U1_a.thermo import (
    DEFAULT_CONC_UNIT_SCALE,
    DEFAULT_LN_K_H_E,
    DEFAULT_LN_K_H_L,
    DEFAULT_LN_K_H_O,
    DEFAULT_LN_K_L_E,
    DEFAULT_LN_K_L_O,
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
STRUCTURE_NAMES = mwc.STRUCTURE_NAMES_C12

# Δln_K = ddG @ _PROJ.T; rows = constants, cols = STRUCTURE_NAMES.
_PROJ = jnp.asarray(mwc.PROJ_C12)

# Unfolding row. K_u = [U]/[H] and ΔΔG_U = 0, so Δln K_u = ΔΔG_H.
_PROJ_U = jnp.asarray(mwc.PROJ_U_C12)

# Δln K (sampled, M × 6) = ΔΔG @ DLNK_PROJ.T: the folded rows, then the
# unfolding row; square and invertible (thermo/prior.py).
DLNK_PROJ = projection(mwc.PROJ_C12, mwc.PROJ_U_C12)


# ──────────────────────────────────────────────────────────────────────────────
# Priors pytree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the lnK_ddG_prior mwc-dimer-unfolded theta model."""

    # WT equilibrium constants (Normal in ln-K space)
    theta_ln_K_h_l_wt_loc:   float
    theta_ln_K_h_l_wt_scale: float
    theta_ln_K_h_o_wt_loc:   float
    theta_ln_K_h_o_wt_scale: float
    theta_ln_K_h_e_wt_loc:   float
    theta_ln_K_h_e_wt_scale: float
    theta_ln_K_l_o_wt_loc:   float
    theta_ln_K_l_o_wt_scale: float
    theta_ln_K_l_e_wt_loc:   float
    theta_ln_K_l_e_wt_scale: float

    # WT unfolding constant (Normal in ln-K space)
    theta_ln_K_u_wt_loc:   float
    theta_ln_K_u_wt_scale: float

    # Total TF (M, monomer units)
    theta_tf_total_M: float   # M, monomer units; default 1e-5 (thermo/mwc.py)

    # Unit-conversion factor applied to data.titrant_conc before thermodynamics.
    # Default 1e-3 assumes concentrations are in mM and K values are in M⁻¹.
    theta_conc_unit_scale: float

    # Regularized horseshoe for pairwise epistasis
    theta_epi_tau_scale:  float
    theta_epi_slab_scale: float
    theta_epi_slab_df:    float
    theta_epi_d0:         float

    # LogNormal prior on the per-structure ΔΔG scale σ_s (log kT); see
    # thermo/prior.py.
    theta_ddG_sigma_loc:   float
    theta_ddG_sigma_scale: float
    # > 0 holds every σ_s at this value (kT) instead of sampling it;
    # 0 (default) samples σ_s (thermo/prior.py, "Holding σ_s").
    theta_ddG_sigma_fixed: float = field(pytree_node=False, default=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_struct_perm(data):
    """
    Verify struct_names contains every required structure (extras are
    ignored) and return the index of each STRUCTURE_NAMES entry in it.
    """
    data_names = tuple(data.struct_names or ())
    missing = [s for s in STRUCTURE_NAMES if s not in data_names]
    if missing:
        raise ValueError(
            f"lnK_ddG_prior (mwc_dimer_unfolded) needs structures {STRUCTURE_NAMES}; "
            f"struct_names {data_names} lacks {missing}. Extra "
            f"structures are ignored."
        )
    name_to_idx = {n: i for i, n in enumerate(data_names)}
    return [name_to_idx[s] for s in STRUCTURE_NAMES]


def _project_ddG(ddG):
    """
    Project per-structure ΔΔG to Δln_K via the fixed thermodynamic matrix.

    Parameters
    ----------
    ddG : jnp.ndarray, shape (..., 6)

    Returns
    -------
    jnp.ndarray, shape (..., 5)
    """
    return ddG @ _PROJ.T


def _project_unfolding(ddG):
    """Per-mutation Δln K_u = ΔΔG_H (ΔΔG_U = 0); (..., 6) → (...,)."""
    return ddG @ _PROJ_U


def _assemble_K_u(ln_K_u_wt, d_ln_K_u, mut_scatter,
                  epi_ln_K_u=None, pair_scatter=None):
    """ln_K_u (G,) from the wt value plus scattered per-mutation/pair shifts."""
    ln_K_u = ln_K_u_wt + mut_scatter(d_ln_K_u)
    if epi_ln_K_u is not None and pair_scatter is not None:
        ln_K_u = ln_K_u + pair_scatter(epi_ln_K_u)
    return ln_K_u


def _assemble_K(ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
                ln_K_l_o_wt, ln_K_l_e_wt,
                delta_lnK, mut_scatter,
                epi_delta_lnK=None, pair_scatter=None):
    """
    Assemble per-genotype K values from WT scalars + per-mutation deltas.

    Parameters
    ----------
    delta_lnK : (M, 5)
    mut_scatter : callable (M,) -> (G,)
    epi_delta_lnK : (P, 5) or None
    pair_scatter : callable (P,) -> (G,) or None

    Returns
    -------
    Tuple: ln_K_h_l (G,), ln_K_h_o (G,), ln_K_h_e (T, G),
           ln_K_l_o (G,), ln_K_l_e (T, G)
    """
    d_h_l = delta_lnK[:, 0]
    d_h_o = delta_lnK[:, 1]
    d_h_e = delta_lnK[:, 2]
    d_l_o = delta_lnK[:, 3]
    d_l_e = delta_lnK[:, 4]

    ln_K_h_l = ln_K_h_l_wt + mut_scatter(d_h_l)
    ln_K_h_o = ln_K_h_o_wt + mut_scatter(d_h_o)
    ln_K_h_e = ln_K_h_e_wt[:, None] + mut_scatter(d_h_e)[None, :]
    ln_K_l_o = ln_K_l_o_wt + mut_scatter(d_l_o)
    ln_K_l_e = ln_K_l_e_wt[:, None] + mut_scatter(d_l_e)[None, :]

    if epi_delta_lnK is not None and pair_scatter is not None:
        ln_K_h_l = ln_K_h_l + pair_scatter(epi_delta_lnK[:, 0])
        ln_K_h_o = ln_K_h_o + pair_scatter(epi_delta_lnK[:, 1])
        ln_K_h_e = ln_K_h_e + pair_scatter(epi_delta_lnK[:, 2])[None, :]
        ln_K_l_o = ln_K_l_o + pair_scatter(epi_delta_lnK[:, 3])
        ln_K_l_e = ln_K_l_e + pair_scatter(epi_delta_lnK[:, 4])[None, :]

    return ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_ddG_prior mwc-dimer-unfolded theta model.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample/param sites.
    data : GrowthData or BindingData
        Must carry ``struct_features`` (M, S) float32 ddG prior means,
        ``struct_names``, ``mut_geno_matrix``, and ``num_mutation``.
    priors : ModelPriors
    """
    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    S       = len(STRUCTURE_NAMES)
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut  = data.num_mutation
    has_epi  = data.num_pair > 0
    # Reorder columns to canonical STRUCTURE_NAMES order
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]   # (M, S)

    tf_total        = priors.theta_tf_total_M
    conc_unit_scale = priors.theta_conc_unit_scale

    # ── WT equilibrium constants ─────────────────────────────────────────────
    ln_K_h_l_wt = pyro.sample(
        f"{name}_ln_K_h_l_wt",
        dist.Normal(priors.theta_ln_K_h_l_wt_loc, priors.theta_ln_K_h_l_wt_scale))
    ln_K_h_o_wt = pyro.sample(
        f"{name}_ln_K_h_o_wt",
        dist.Normal(priors.theta_ln_K_h_o_wt_loc, priors.theta_ln_K_h_o_wt_scale))
    ln_K_l_o_wt = pyro.sample(
        f"{name}_ln_K_l_o_wt",
        dist.Normal(priors.theta_ln_K_l_o_wt_loc, priors.theta_ln_K_l_o_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_h_e_wt = pyro.sample(
            f"{name}_ln_K_h_e_wt",
            dist.Normal(priors.theta_ln_K_h_e_wt_loc, priors.theta_ln_K_h_e_wt_scale))
        ln_K_l_e_wt = pyro.sample(
            f"{name}_ln_K_l_e_wt",
            dist.Normal(priors.theta_ln_K_l_e_wt_loc, priors.theta_ln_K_l_e_wt_scale))

    # ── WT unfolding constant ─────────────────────────────────────────────────
    ln_K_u_wt = pyro.sample(
        f"{name}_ln_K_u_wt",
        dist.Normal(priors.theta_ln_K_u_wt_loc, priors.theta_ln_K_u_wt_scale))

    # ── Per-mutation Δln K under the structural prior ─────────────────────────
    # Δln K[m] ~ MVN(P · prior_mean[m], P · diag(σ_s²) · Pᵀ), σ_s per state
    # (thermo/prior.py); the post-hoc per-state ΔΔG is the site {name}_ddG.
    dlnK = sample_dlnK(name, list(STRUCTURE_NAMES), num_mut, ddG_prior_means, DLNK_PROJ,
                       sigma_loc=priors.theta_ddG_sigma_loc,
                       sigma_scale=priors.theta_ddG_sigma_scale,
                       sigma_fixed=ddG_sigma_fixed(priors))  # (M, C)
    delta_lnK  = dlnK[:, :5]

    pyro.deterministic(f"{name}_d_ln_K_h_l", delta_lnK[:, 0])
    pyro.deterministic(f"{name}_d_ln_K_h_o", delta_lnK[:, 1])
    pyro.deterministic(f"{name}_d_ln_K_h_e", delta_lnK[:, 2])
    pyro.deterministic(f"{name}_d_ln_K_l_o", delta_lnK[:, 3])
    pyro.deterministic(f"{name}_d_ln_K_l_e", delta_lnK[:, 4])
    d_ln_K_u = dlnK[:, 5]
    pyro.deterministic(f"{name}_d_ln_K_u",   d_ln_K_u)

    # ── Optional pairwise epistasis ───────────────────────────────────────────
    epi_delta_lnK = None
    epi_ln_K_u    = None
    pair_scatter  = None
    if has_epi:
        P = data.num_pair
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        contact_distances = jnp.zeros((P, S))
        epi_ddG = sample_pair_ddG(
            name, list(STRUCTURE_NAMES), contact_distances,
            tau_scale=priors.theta_epi_tau_scale,
            slab_scale=priors.theta_epi_slab_scale,
            slab_df=priors.theta_epi_slab_df,
            d0=priors.theta_epi_d0,
        )  # (P, S)
        epi_delta_lnK = _project_ddG(epi_ddG)   # (P, 5)
        pyro.deterministic(f"{name}_epi_ln_K_h_l", epi_delta_lnK[:, 0])
        pyro.deterministic(f"{name}_epi_ln_K_h_o", epi_delta_lnK[:, 1])
        pyro.deterministic(f"{name}_epi_ln_K_h_e", epi_delta_lnK[:, 2])
        pyro.deterministic(f"{name}_epi_ln_K_l_o", epi_delta_lnK[:, 3])
        pyro.deterministic(f"{name}_epi_ln_K_l_e", epi_delta_lnK[:, 4])
        epi_ln_K_u = _project_unfolding(epi_ddG)   # (P,)
        pyro.deterministic(f"{name}_epi_ln_K_u",   epi_ln_K_u)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e = _assemble_K(
        ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
        ln_K_l_o_wt, ln_K_l_e_wt,
        delta_lnK, mut_scatter,
        epi_delta_lnK=epi_delta_lnK, pair_scatter=pair_scatter,
    )

    pyro.deterministic(f"{name}_ln_K_h_l", ln_K_h_l)
    pyro.deterministic(f"{name}_ln_K_h_o", ln_K_h_o)
    pyro.deterministic(f"{name}_ln_K_h_e", ln_K_h_e)
    pyro.deterministic(f"{name}_ln_K_l_o", ln_K_l_o)
    pyro.deterministic(f"{name}_ln_K_l_e", ln_K_l_e)
    ln_K_u = _assemble_K_u(ln_K_u_wt, d_ln_K_u, mut_scatter,
                           epi_ln_K_u=epi_ln_K_u, pair_scatter=pair_scatter)
    pyro.deterministic(f"{name}_ln_K_u",   ln_K_u)

    # ── Population moments for transformation model ───────────────────────────
    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                ln_K_u,
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e, ln_K_u=ln_K_u,
                      tf_total=tf_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Guide
# ──────────────────────────────────────────────────────────────────────────────

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the lnK_ddG_prior mwc-dimer-unfolded theta model."""

    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    S       = len(STRUCTURE_NAMES)
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]   # (M, S)

    tf_total        = priors.theta_tf_total_M
    conc_unit_scale = priors.theta_conc_unit_scale
    has_epi         = data.num_pair > 0

    # ── Variational params for WT K values ────────────────────────────────────
    ln_K_h_l_wt_loc   = pyro.param(f"{name}_ln_K_h_l_wt_loc",   jnp.array(priors.theta_ln_K_h_l_wt_loc))
    ln_K_h_l_wt_scale = pyro.param(f"{name}_ln_K_h_l_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_h_o_wt_loc   = pyro.param(f"{name}_ln_K_h_o_wt_loc",   jnp.array(priors.theta_ln_K_h_o_wt_loc))
    ln_K_h_o_wt_scale = pyro.param(f"{name}_ln_K_h_o_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_l_o_wt_loc   = pyro.param(f"{name}_ln_K_l_o_wt_loc",   jnp.array(priors.theta_ln_K_l_o_wt_loc))
    ln_K_l_o_wt_scale = pyro.param(f"{name}_ln_K_l_o_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_h_e_wt_locs   = pyro.param(f"{name}_ln_K_h_e_wt_locs",   jnp.full(T, priors.theta_ln_K_h_e_wt_loc))
    ln_K_h_e_wt_scales = pyro.param(f"{name}_ln_K_h_e_wt_scales", jnp.ones(T), constraint=constraints.positive)
    ln_K_l_e_wt_locs   = pyro.param(f"{name}_ln_K_l_e_wt_locs",   jnp.full(T, priors.theta_ln_K_l_e_wt_loc))
    ln_K_l_e_wt_scales = pyro.param(f"{name}_ln_K_l_e_wt_scales", jnp.ones(T), constraint=constraints.positive)

    # ── Variational params for WT unfolding constant ──────────────────────────
    ln_K_u_wt_loc   = pyro.param(f"{name}_ln_K_u_wt_loc",   jnp.array(priors.theta_ln_K_u_wt_loc))
    ln_K_u_wt_scale = pyro.param(f"{name}_ln_K_u_wt_scale", jnp.array(1.0), constraint=constraints.positive)

    # ── Optional epistasis variational params ─────────────────────────────────
    if has_epi:
        num_pair = data.num_pair
        pyro.param(f"{name}_epi_tau_loc",    jnp.array(-2.0))
        pyro.param(f"{name}_epi_tau_scale",  jnp.array(0.5),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_c2_loc",     jnp.array(1.4))
        pyro.param(f"{name}_epi_c2_scale",   jnp.array(0.5),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_lambda_locs",   jnp.zeros((S, num_pair)))
        pyro.param(f"{name}_epi_lambda_scales", jnp.ones((S, num_pair)),  constraint=constraints.positive)
        pyro.param(f"{name}_epi_offset_locs",   jnp.zeros((S, num_pair)))
        pyro.param(f"{name}_epi_offset_scales", jnp.ones((S, num_pair)),  constraint=constraints.positive)

    # ── Sample WT K values ────────────────────────────────────────────────────
    ln_K_h_l_wt = pyro.sample(f"{name}_ln_K_h_l_wt", dist.Normal(ln_K_h_l_wt_loc, ln_K_h_l_wt_scale))
    ln_K_h_o_wt = pyro.sample(f"{name}_ln_K_h_o_wt", dist.Normal(ln_K_h_o_wt_loc, ln_K_h_o_wt_scale))
    ln_K_l_o_wt = pyro.sample(f"{name}_ln_K_l_o_wt", dist.Normal(ln_K_l_o_wt_loc, ln_K_l_o_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_h_e_wt = pyro.sample(f"{name}_ln_K_h_e_wt", dist.Normal(ln_K_h_e_wt_locs, ln_K_h_e_wt_scales))
        ln_K_l_e_wt = pyro.sample(f"{name}_ln_K_l_e_wt", dist.Normal(ln_K_l_e_wt_locs, ln_K_l_e_wt_scales))

    # ── Sample WT unfolding constant ──────────────────────────────────────────
    ln_K_u_wt = pyro.sample(f"{name}_ln_K_u_wt", dist.Normal(ln_K_u_wt_loc, ln_K_u_wt_scale))

    # ── Sample σ_s and Δln K (matches prior.sample_dlnK) ────────────────────
    dlnK = guide_dlnK(name, num_mut, S, DLNK_PROJ, ddG_prior_means,
                      sigma_fixed=ddG_sigma_fixed(priors))   # (M, C)

    # ── Sample epistasis ──────────────────────────────────────────────────────
    epi_delta_lnK = None
    epi_ln_K_u    = None
    pair_scatter  = None
    if has_epi:
        tau_loc    = pyro.param(f"{name}_epi_tau_loc",    jnp.array(-2.0))
        tau_scale  = pyro.param(f"{name}_epi_tau_scale",  jnp.array(0.5),  constraint=constraints.positive)
        c2_loc     = pyro.param(f"{name}_epi_c2_loc",     jnp.array(1.4))
        c2_scale   = pyro.param(f"{name}_epi_c2_scale",   jnp.array(0.5),  constraint=constraints.positive)
        lam_locs   = pyro.param(f"{name}_epi_lambda_locs",   jnp.zeros((S, num_pair)))
        lam_scales = pyro.param(f"{name}_epi_lambda_scales", jnp.ones((S, num_pair)),  constraint=constraints.positive)
        epi_locs   = pyro.param(f"{name}_epi_offset_locs",   jnp.zeros((S, num_pair)))
        epi_scales = pyro.param(f"{name}_epi_offset_scales", jnp.ones((S, num_pair)),  constraint=constraints.positive)

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

        def _lam_tilde(l):
            return regularized_scale(l, tau_epi, c2_epi)

        epi_SP        = epi_off * tau_epi * _lam_tilde(lam)   # (S, P)
        epi_ddG       = epi_SP.T                               # (P, S)
        epi_delta_lnK = _project_ddG(epi_ddG)                 # (P, 5)
        epi_ln_K_u    = _project_unfolding(epi_ddG)           # (P,)

        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    delta_lnK = dlnK[:, :5]
    ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e = _assemble_K(
        ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
        ln_K_l_o_wt, ln_K_l_e_wt,
        delta_lnK, mut_scatter,
        epi_delta_lnK=epi_delta_lnK, pair_scatter=pair_scatter,
    )
    ln_K_u = _assemble_K_u(ln_K_u_wt, dlnK[:, 5], mut_scatter,
                           epi_ln_K_u=epi_ln_K_u, pair_scatter=pair_scatter)

    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                ln_K_u,
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e, ln_K_u=ln_K_u,
                      tf_total=tf_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # Wild-type defaults: see thermo/mwc.py (Sochor 2014 constants and the
    # 80 µM midpoint calibration at tf_total = 1e-5 M monomer).
    p["theta_ln_K_h_l_wt_loc"]   = DEFAULT_LN_K_H_L   # Sochor (2014) Table 1: K_h_l = 6.3
    p["theta_ln_K_h_l_wt_scale"] = 2.0
    p["theta_ln_K_h_o_wt_loc"]   = DEFAULT_LN_K_H_O   # Sochor: K_h_o = 4.2e8 M⁻¹
    p["theta_ln_K_h_o_wt_scale"] = 2.0
    p["theta_ln_K_h_e_wt_loc"]   = DEFAULT_LN_K_H_E   # per site; ≈ 3.7e3 M⁻¹ = K_l_e/79.6, theta_sat 0.05
    p["theta_ln_K_h_e_wt_scale"] = 2.0
    p["theta_ln_K_l_o_wt_loc"]   = DEFAULT_LN_K_L_O   # Sochor: K_l_o = 0.1 M⁻¹
    p["theta_ln_K_l_o_wt_scale"] = 3.0
    p["theta_ln_K_l_e_wt_loc"]   = DEFAULT_LN_K_L_E   # per site; ≈ 3.0e5 M⁻¹ = 0.39 × Sochor, midpoint 80 µM
    p["theta_ln_K_l_e_wt_scale"] = 2.0
    # WT unfolded state: K_u << 1 in WT (ln_K_u ≈ -12 → K_u ≈ 6e-6)
    p["theta_ln_K_u_wt_loc"]     = -12.0
    p["theta_ln_K_u_wt_scale"]   = 3.0
    p["theta_tf_total_M"]        = DEFAULT_TF_TOTAL_M   # M, monomer units (5e-6 M dimer)
    p["theta_conc_unit_scale"]   = DEFAULT_CONC_UNIT_SCALE   # titrant_conc mM → M
    p["theta_epi_tau_scale"]     = 0.1
    p["theta_epi_slab_scale"]    = 2.0
    p["theta_epi_slab_df"]       = 4.0
    p["theta_epi_d0"]            = float(_DEFAULT_D0)
    p["theta_ddG_sigma_loc"]   = DEFAULT_DDG_SIGMA_LOC     # median σ_s 1 kT
    p["theta_ddG_sigma_scale"] = DEFAULT_DDG_SIGMA_SCALE   # 95%: 0.1-10 kT
    p["theta_ddG_sigma_fixed"] = 0.0   # > 0 holds σ_s (kT)
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    S = len(STRUCTURE_NAMES)
    g = {}
    g[f"{name}_ln_K_h_l_wt"]  = jnp.array(DEFAULT_LN_K_H_L)
    g[f"{name}_ln_K_h_o_wt"]  = jnp.array(DEFAULT_LN_K_H_O)
    g[f"{name}_ln_K_h_e_wt"]  = jnp.full(T, DEFAULT_LN_K_H_E)
    g[f"{name}_ln_K_l_o_wt"]  = jnp.array(DEFAULT_LN_K_L_O)
    g[f"{name}_ln_K_l_e_wt"]  = jnp.full(T, DEFAULT_LN_K_L_E)
    g[f"{name}_ln_K_u_wt"]    = jnp.array(-12.0)
    # Δln K starts at P · (ΔΔG prior means), σ_s at 1 kT.
    g.update(dlnK_guesses(name, jnp.asarray(data.struct_features)[:, jnp.array(_get_struct_perm(data))], DLNK_PROJ))
    if data.num_pair > 0:
        P = data.num_pair
        g[f"{name}_epi_tau"]    = jnp.array(0.05)
        g[f"{name}_epi_c2"]     = jnp.array(4.0)
        g[f"{name}_epi_lambda"] = jnp.ones((S, P)) * 0.5
        g[f"{name}_epi_offset"] = jnp.zeros((S, P))
    return g


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    geno_dim     = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    titrant_dim  = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    num_mut = len(ctx.mut_labels)

    # Scalar per-genotype K values (includes ln_K_u)
    geno_df = (ctx.growth_tm.df[["genotype", "genotype_idx"]]
               .drop_duplicates().copy())
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(
        input_df=geno_df,
        params_to_get=["ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u"],
        map_column="map_geno",
        get_columns=["genotype"],
        in_run_prefix="theta_",
    )]

    # T-dimensional per-genotype K values (K_h_e, K_l_e)
    theta_KE_df = (ctx.growth_tm.df[["genotype", "titrant_name",
                                     "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates().copy())
    theta_KE_df["map_theta_KE"] = (theta_KE_df["titrant_name_idx"] * num_genotype
                                   + theta_KE_df["genotype_idx"])
    specs.append(dict(
        input_df=theta_KE_df,
        params_to_get=["ln_K_h_e", "ln_K_l_e"],
        map_column="map_theta_KE",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    ))

    # Per-mutation projected Δln_K values
    mut_df = pd.DataFrame({
        "mutation": ctx.mut_labels,
        "map_mut":  range(num_mut),
    })
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_h_l", "d_ln_K_h_o", "d_ln_K_h_e",
                        "d_ln_K_l_o", "d_ln_K_l_e", "d_ln_K_u"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        pair_df = pd.DataFrame({
            "pair":     ctx.pair_labels,
            "map_pair": range(len(ctx.pair_labels)),
        })
        specs.append(dict(
            input_df=pair_df,
            params_to_get=["epi_ln_K_h_l", "epi_ln_K_h_o", "epi_ln_K_h_e",
                           "epi_ln_K_l_o", "epi_ln_K_l_e", "epi_ln_K_u"],
            map_column="map_pair",
            get_columns=["pair"],
            in_run_prefix="theta_",
        ))

    # Per-structure ddG and its prior mean, in STRUCTURE_NAMES order (the
    # order the model samples in and the simulator labels its truth with).
    specs.extend(ddG_extract_specs(ctx, STRUCTURE_NAMES))

    return specs


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
    Predict theta for unmeasured genotypes using ddG-prior ln-K MWC-dimer-unfolded assembly.

    The d_ln_K_h_e and d_ln_K_l_e deterministic sites have shape (S, M) with no
    T dimension; the scalar per-mutation effect is broadcast uniformly across
    effectors.  ln_K_u is assembled like the folded-state constants, from
    ln_K_u_wt plus the per-mutation ``d_ln_K_u`` (= ΔΔG_H) and per-pair
    ``epi_ln_K_u``.

    Parameters
    ----------
    target_genotypes : list[str]
    titrant_names : list[str]
    manual_titrant_df : pd.DataFrame
    mut_labels : list[str]
    pair_labels : list[str]
    param_posteriors : dict-like
    q_to_get : dict
    tf_total : float — monomer units (M)
    conc_unit_scale : float
        Multiply titrant_conc by this factor before thermodynamic equations.
        Use 1e-3 when concentrations are in mM and K values are in M⁻¹.

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
    from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U1_a.thermo import (
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

    ln_K_h_l_wt = _load("theta_ln_K_h_l_wt")   # (S,)
    ln_K_h_o_wt = _load("theta_ln_K_h_o_wt")   # (S,)
    ln_K_l_o_wt = _load("theta_ln_K_l_o_wt")   # (S,)
    ln_K_h_e_wt = _load("theta_ln_K_h_e_wt")   # (S, T)
    ln_K_l_e_wt = _load("theta_ln_K_l_e_wt")   # (S, T)
    ln_K_u_wt   = _load("theta_ln_K_u_wt")     # (S,)

    d_h_l = _load("theta_d_ln_K_h_l")   # (S, M)
    d_h_o = _load("theta_d_ln_K_h_o")   # (S, M)
    d_l_o = _load("theta_d_ln_K_l_o")   # (S, M)
    d_h_e = _load("theta_d_ln_K_h_e")   # (S, M) — scalar delta, no T dim
    d_l_e = _load("theta_d_ln_K_l_e")   # (S, M) — scalar delta, no T dim
    d_u   = _load("theta_d_ln_K_u")     # (S, M)

    ln_K_h_l_geno = ln_K_h_l_wt[:, None] + np.einsum("sm,nm->sn", d_h_l, mut_mat)
    ln_K_h_o_geno = ln_K_h_o_wt[:, None] + np.einsum("sm,nm->sn", d_h_o, mut_mat)
    ln_K_l_o_geno = ln_K_l_o_wt[:, None] + np.einsum("sm,nm->sn", d_l_o, mut_mat)
    # d_h_e/d_l_e are scalar per mutation (no T dim); broadcast scalar delta across T
    ln_K_h_e_geno = (ln_K_h_e_wt[:, :, None]
                     + np.einsum("sm,nm->sn", d_h_e, mut_mat)[:, None, :])
    ln_K_l_e_geno = (ln_K_l_e_wt[:, :, None]
                     + np.einsum("sm,nm->sn", d_l_e, mut_mat)[:, None, :])
    ln_K_u_geno   = ln_K_u_wt[:, None] + np.einsum("sm,nm->sn", d_u, mut_mat)

    if len(pair_labels) > 0:
        epi_h_l = _load("theta_epi_ln_K_h_l")   # (S, P)
        epi_h_o = _load("theta_epi_ln_K_h_o")   # (S, P)
        epi_l_o = _load("theta_epi_ln_K_l_o")   # (S, P)
        epi_h_e = _load("theta_epi_ln_K_h_e")   # (S, P) — scalar, no T dim
        epi_l_e = _load("theta_epi_ln_K_l_e")   # (S, P) — scalar, no T dim
        epi_u   = _load("theta_epi_ln_K_u")     # (S, P)
        ln_K_h_l_geno += np.einsum("sp,np->sn", epi_h_l, pair_mat)
        ln_K_h_o_geno += np.einsum("sp,np->sn", epi_h_o, pair_mat)
        ln_K_l_o_geno += np.einsum("sp,np->sn", epi_l_o, pair_mat)
        ln_K_h_e_geno += np.einsum("sp,np->sn", epi_h_e, pair_mat)[:, None, :]
        ln_K_l_e_geno += np.einsum("sp,np->sn", epi_l_e, pair_mat)[:, None, :]
        ln_K_u_geno   += np.einsum("sp,np->sn", epi_u,   pair_mat)

    ln_K_h_l_rows = ln_K_h_l_geno[:, geno_idx]
    ln_K_h_o_rows = ln_K_h_o_geno[:, geno_idx]
    ln_K_l_o_rows = ln_K_l_o_geno[:, geno_idx]
    ln_K_h_e_rows = ln_K_h_e_geno[:, titrant_idx, geno_idx]
    ln_K_l_e_rows = ln_K_l_e_geno[:, titrant_idx, geno_idx]
    ln_K_u_rows   = ln_K_u_geno[:, geno_idx]

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_h_l_rows, ln_K_h_o_rows, ln_K_h_e_rows,
        ln_K_l_o_rows, ln_K_l_e_rows, ln_K_u_rows,
        conc, tf_total,
    )

    theta_samples[:, ~is_valid[geno_idx]] = np.nan

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df


from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U1_a.thermo import (
    build_calc_df,
    compute_theta_samples,
)
