"""
PddG — mutation effects on equilibrium constants from user-supplied ΔΔG prior means (lac-dimer-unfolded).

Extends the lac-dimer PddG model with a WT-level unfolding constant (ln_K_U_wt).
Per-mutation effects on K_op, K_HL, K_E, and K_U are derived from user-supplied
structural ΔΔG (CSV: columns mut, H, HD, L, LE2).  Registry name:
``theta_lac_dimer_unfolded_lnK_ddG_prior``.

Contrast with PK (all K shifts inferred from data, including K_U per mutation)
and PnnC (ΔΔG prior means from a per-structure MLP rather than a user CSV).

Identical to lac_dimer/lnK_ddG_prior.py with one additional scalar parameter:

    ln_K_U_wt  — WT H→U unfolding equilibrium constant

and one additional projection row.  A mutation shifts the free energy of each
folded structure X by ΔΔG_X and leaves the unfolded state alone (ΔΔG_U = 0,
the standard assumption).  Because K_U = [U]/[H], the projection convention
Δln K_{A→B} = ΔΔG_A − ΔΔG_B gives

    Δln K_U = ΔΔG_H − ΔΔG_U = ΔΔG_H

so ln_K_U[g] = ln_K_U_wt + Σ_m ΔΔG_H[m] (plus the pairwise ΔΔG_H epistasis).
A mutation that raises every folded state by the same ΔΔG leaves K_op, K_HL
and K_E unchanged and raises K_U by ΔΔG: a pure stability effect.  The
per-mutation shift is exposed as ``d_ln_K_U`` (and ``epi_ln_K_U`` per pair),
the same name the PK model uses for its free per-mutation unfolding shift.

Input CSV format (passed via ``thermo_data``)
------------------------------------------------------
A CSV with a mutation column (``mut``, ``Mutation`` or ``mutation``) and one
column per structure (``H``, ``HD``, ``L``, ``LE2``, any order; other columns are
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
from flax.struct import dataclass
from functools import partial
from typing import Dict, Any, Union

from tfscreen.tfmodel.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix
from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    ddG_extract_specs,
    sample_ddG,
)
from tfscreen.tfmodel.generative.components.theta.thermo.horseshoe import (
    sample_pair_ddG,
    _DEFAULT_D0,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U1_a.thermo import (
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

# Unfolding row. K_U = [U]/[H] and ΔΔG_U = 0, so Δln K_U = ΔΔG_H.
_PROJ_U = jnp.asarray(mwc.PROJ_U_C4)


# ──────────────────────────────────────────────────────────────────────────────
# Priors pytree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the lnK_ddG_prior lac-dimer-unfolded theta model."""

    theta_ln_K_op_wt_loc:   float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc:   float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc:    float
    theta_ln_K_E_wt_scale:  float
    theta_ln_K_U_wt_loc:    float
    theta_ln_K_U_wt_scale:  float

    theta_tf_total_M: float   # M, monomer units; default 1e-5 (thermo/mwc.py)
    # Factor converting data titrant_conc (mM) to M (1e-3); see thermo.py.
    theta_conc_unit_scale: float

    # Regularized horseshoe for pairwise epistasis
    theta_epi_tau_scale:  float
    theta_epi_slab_scale: float
    theta_epi_slab_df:    float
    theta_epi_d0:         float


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_struct_perm(data):
    data_names = tuple(data.struct_names or ())
    missing = [s for s in STRUCTURE_NAMES if s not in data_names]
    if missing:
        raise ValueError(
            f"lnK_ddG_prior (lac_dimer_unfolded) needs structures {STRUCTURE_NAMES}; "
            f"struct_names {data_names} lacks {missing}. Extra "
            f"structures are ignored."
        )
    name_to_idx = {n: i for i, n in enumerate(data_names)}
    return [name_to_idx[s] for s in STRUCTURE_NAMES]


def _project_ddG(ddG):
    return ddG @ _PROJ.T


def _project_unfolding(ddG):
    """Per-mutation Δln K_U = ΔΔG_H (ΔΔG_U = 0); (..., 4) → (...,)."""
    return ddG @ _PROJ_U


def _assemble_K(ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter,
                epi_delta_lnK=None, pair_scatter=None):
    d_op = delta_lnK[:, 0]
    d_HL = delta_lnK[:, 1]
    d_E  = delta_lnK[:, 2]
    ln_K_op = ln_K_op_wt + mut_scatter(d_op)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_HL)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_E)[None, :]
    if epi_delta_lnK is not None and pair_scatter is not None:
        ln_K_op = ln_K_op + pair_scatter(epi_delta_lnK[:, 0])
        ln_K_HL = ln_K_HL + pair_scatter(epi_delta_lnK[:, 1])
        ln_K_E  = ln_K_E  + pair_scatter(epi_delta_lnK[:, 2])[None, :]
    return ln_K_op, ln_K_HL, ln_K_E


def _assemble_K_U(ln_K_U_wt, d_ln_K_U, mut_scatter,
                  epi_ln_K_U=None, pair_scatter=None):
    """ln_K_U (G,) from the wt value plus scattered per-mutation/pair shifts."""
    ln_K_U = ln_K_U_wt + mut_scatter(d_ln_K_U)
    if epi_ln_K_U is not None and pair_scatter is not None:
        ln_K_U = ln_K_U + pair_scatter(epi_ln_K_U)
    return ln_K_U


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_ddG_prior lac-dimer-unfolded theta model.

    Adds ``ln_K_U_wt`` to the standard lnK_ddG_prior model.  Per-mutation
    effects on K_U follow from ΔΔG_H (``d_ln_K_U = ΔΔG_H``; ΔΔG_U = 0).
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
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]

    tf_total = priors.theta_tf_total_M
    conc_unit_scale = priors.theta_conc_unit_scale

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

    ddG = sample_ddG(name, list(STRUCTURE_NAMES), num_mut, ddG_prior_means)
    delta_lnK = _project_ddG(ddG)

    pyro.deterministic(f"{name}_d_ln_K_op", delta_lnK[:, 0])
    pyro.deterministic(f"{name}_d_ln_K_HL", delta_lnK[:, 1])
    pyro.deterministic(f"{name}_d_ln_K_E",  delta_lnK[:, 2])
    d_ln_K_U = _project_unfolding(ddG)
    pyro.deterministic(f"{name}_d_ln_K_U",  d_ln_K_U)

    # ── Optional pairwise epistasis ───────────────────────────────────────────
    epi_delta_lnK = None
    epi_ln_K_U    = None
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
        epi_delta_lnK = _project_ddG(epi_ddG)   # (P, 3)
        pyro.deterministic(f"{name}_epi_ln_K_op", epi_delta_lnK[:, 0])
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_delta_lnK[:, 1])
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_delta_lnK[:, 2])
        epi_ln_K_U = _project_unfolding(epi_ddG)  # (P,)
        pyro.deterministic(f"{name}_epi_ln_K_U",  epi_ln_K_U)

    ln_K_op, ln_K_HL, ln_K_E = _assemble_K(
        ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter,
        epi_delta_lnK=epi_delta_lnK, pair_scatter=pair_scatter,
    )
    ln_K_U = _assemble_K_U(ln_K_U_wt, d_ln_K_U, mut_scatter,
                           epi_ln_K_U=epi_ln_K_U, pair_scatter=pair_scatter)

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)
    pyro.deterministic(f"{name}_ln_K_U",  ln_K_U)

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
                      tf_total=tf_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Guide
# ──────────────────────────────────────────────────────────────────────────────

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the lnK_ddG_prior lac-dimer-unfolded theta model."""

    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    S       = len(STRUCTURE_NAMES)
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]

    tf_total = priors.theta_tf_total_M
    conc_unit_scale = priors.theta_conc_unit_scale
    has_epi  = data.num_pair > 0

    ln_K_op_wt_loc   = pyro.param(f"{name}_ln_K_op_wt_loc",   jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(f"{name}_ln_K_op_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_HL_wt_loc   = pyro.param(f"{name}_ln_K_HL_wt_loc",   jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(f"{name}_ln_K_HL_wt_scale", jnp.array(1.0), constraint=constraints.positive)
    ln_K_U_wt_loc    = pyro.param(f"{name}_ln_K_U_wt_loc",    jnp.array(priors.theta_ln_K_U_wt_loc))
    ln_K_U_wt_scale  = pyro.param(f"{name}_ln_K_U_wt_scale",  jnp.array(1.0), constraint=constraints.positive)
    ln_K_E_wt_locs   = pyro.param(f"{name}_ln_K_E_wt_locs",   jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(f"{name}_ln_K_E_wt_scales", jnp.ones(T), constraint=constraints.positive)

    pyro.param(f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive)

    ddG_offset_locs   = pyro.param(f"{name}_ddG_offset_locs",   jnp.zeros((S, num_mut)))
    ddG_offset_scales = pyro.param(f"{name}_ddG_offset_scales", jnp.ones((S, num_mut)), constraint=constraints.positive)

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

    ln_K_op_wt = pyro.sample(f"{name}_ln_K_op_wt", dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(f"{name}_ln_K_HL_wt", dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    ln_K_U_wt  = pyro.sample(f"{name}_ln_K_U_wt",  dist.Normal(ln_K_U_wt_loc,  ln_K_U_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(f"{name}_ln_K_E_wt", dist.Normal(ln_K_E_wt_locs, ln_K_E_wt_scales))

    if num_mut > 0:
        with pyro.plate(f"{name}_struct_plate", S, dim=-2):
            with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
                offsets = pyro.sample(
                    f"{name}_ddG_offset",
                    dist.Normal(ddG_offset_locs, ddG_offset_scales),
                )
    else:
        # num_mut == 0 (wt-only library): numpyro.plate needs size > 0,
        # so skip the site (prior.sample_ddG does the same).
        offsets = jnp.zeros((S, 0))

    sigma_s = pyro.param(f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive)
    ddG_SM  = ddG_prior_means.T + sigma_s[:, None] * offsets
    ddG     = ddG_SM.T

    # ── Sample epistasis ──────────────────────────────────────────────────────
    epi_delta_lnK = None
    epi_ln_K_U    = None
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
        epi_delta_lnK = _project_ddG(epi_ddG)                 # (P, 3)
        epi_ln_K_U    = _project_unfolding(epi_ddG)           # (P,)

        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)

    delta_lnK = _project_ddG(ddG)
    ln_K_op, ln_K_HL, ln_K_E = _assemble_K(
        ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter,
        epi_delta_lnK=epi_delta_lnK, pair_scatter=pair_scatter,
    )
    ln_K_U = _assemble_K_U(ln_K_U_wt, _project_unfolding(ddG), mut_scatter,
                           epi_ln_K_U=epi_ln_K_U, pair_scatter=pair_scatter)

    theta_vals = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                data.titrant_conc * conc_unit_scale,
                                tf_total)
    mu, sigma  = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
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
    p["theta_ln_K_U_wt_loc"]    = -12.0
    p["theta_ln_K_U_wt_scale"]  = 3.0
    p["theta_tf_total_M"]     = DEFAULT_TF_TOTAL_M   # M, monomer units (5e-6 M dimer)
    p["theta_conc_unit_scale"] = DEFAULT_CONC_UNIT_SCALE   # titrant_conc mM → M
    p["theta_epi_tau_scale"]  = 0.1
    p["theta_epi_slab_scale"] = 2.0
    p["theta_epi_slab_df"]    = 4.0
    p["theta_epi_d0"]         = float(_DEFAULT_D0)
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    S = len(STRUCTURE_NAMES)
    g = {}
    g[f"{name}_ln_K_op_wt"]  = jnp.array(DEFAULT_LN_K_OP_WT_LOC)
    g[f"{name}_ln_K_HL_wt"]  = jnp.array(DEFAULT_LN_K_HL_WT_LOC)
    g[f"{name}_ln_K_E_wt"]   = jnp.full(T, DEFAULT_LN_K_E_WT_LOC)
    g[f"{name}_ln_K_U_wt"]   = jnp.array(-12.0)
    g[f"{name}_ddG_offset"]   = jnp.zeros((S, M))
    if data.num_pair > 0:
        P = data.num_pair
        g[f"{name}_epi_tau"]    = jnp.array(0.05)
        g[f"{name}_epi_c2"]     = jnp.array(4.0)
        g[f"{name}_epi_lambda"] = jnp.ones((S, P)) * 0.5
        g[f"{name}_epi_offset"] = jnp.zeros((S, P))
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
    Predict theta for unmeasured genotypes (lac_dimer_unfolded/lnK_ddG_prior).

    ln_K_U is assembled like the folded-state constants, from ln_K_U_wt plus
    the per-mutation ``d_ln_K_U`` (= ΔΔG_H) and per-pair ``epi_ln_K_U``.
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

    ln_K_op_wt = _load("theta_ln_K_op_wt")
    ln_K_HL_wt = _load("theta_ln_K_HL_wt")
    ln_K_E_wt  = _load("theta_ln_K_E_wt")
    ln_K_U_wt  = _load("theta_ln_K_U_wt")

    d_op = _load("theta_d_ln_K_op")
    d_HL = _load("theta_d_ln_K_HL")
    d_E  = _load("theta_d_ln_K_E")
    d_U  = _load("theta_d_ln_K_U")

    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    ln_K_E_geno  = (ln_K_E_wt[:, :, None]
                    + np.einsum("sm,nm->sn", d_E, mut_mat)[:, None, :])
    ln_K_U_geno  = ln_K_U_wt[:, None] + np.einsum("sm,nm->sn", d_U, mut_mat)

    if len(pair_labels) > 0:
        epi_op = _load("theta_epi_ln_K_op")   # (S, P)
        epi_HL = _load("theta_epi_ln_K_HL")   # (S, P)
        epi_E  = _load("theta_epi_ln_K_E")    # (S, P) — scalar, no T dim
        epi_U  = _load("theta_epi_ln_K_U")    # (S, P)
        ln_K_op_geno += np.einsum("sp,np->sn", epi_op, pair_mat)
        ln_K_HL_geno += np.einsum("sp,np->sn", epi_HL, pair_mat)
        ln_K_E_geno  += np.einsum("sp,np->sn", epi_E,  pair_mat)[:, None, :]
        ln_K_U_geno  += np.einsum("sp,np->sn", epi_U,  pair_mat)

    ln_K_op_rows = ln_K_op_geno[:, geno_idx]
    ln_K_HL_rows = ln_K_HL_geno[:, geno_idx]
    ln_K_E_rows  = ln_K_E_geno[:, titrant_idx, geno_idx]
    ln_K_U_rows  = ln_K_U_geno[:, geno_idx]

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_op_rows, ln_K_HL_rows, ln_K_E_rows, ln_K_U_rows,
        conc, tf_total,
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
        params_to_get=["d_ln_K_op", "d_ln_K_HL", "d_ln_K_E", "d_ln_K_U"],
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
            params_to_get=["epi_ln_K_op", "epi_ln_K_HL", "epi_ln_K_E",
                           "epi_ln_K_U"],
            map_column="map_pair",
            get_columns=["pair"],
            in_run_prefix="theta_",
        ))

    # Per-structure ddG and its prior mean, in STRUCTURE_NAMES order (the
    # order the model samples in and the simulator labels its truth with).
    specs.extend(ddG_extract_specs(ctx, STRUCTURE_NAMES))

    return specs


from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U1_a.thermo import (
    build_calc_df,
    compute_theta_samples,
)
