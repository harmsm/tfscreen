"""
K-assembly via per-mutation per-structure ΔΔG with user-supplied prior means.

Implements the full MWC two-state lac-dimer model (Sochor 2014) with five
equilibrium constants, using six structural states as the free-energy basis:

    Structures: H, HO, L, LO, HE2, LE2

    ΔΔG[m, s]  ~ Normal( ddG_prior[m, s],  σ_s )
    σ_s        — per-structure trust scale (``pyro.param``, init=1.0)

where ``ddG_prior`` is a (M, S) float32 array loaded from a CSV spreadsheet
(columns: ``mut``, ``H``, ``HO``, ``L``, ``LO``, ``HE2``, ``LE2``).

Projection from per-structure ΔΔG to Δln_K is identical to lnK_nn_prior
(rows = K, cols = structures):

    Δln_K_h_l = ΔΔG_H  − ΔΔG_L
    Δln_K_h_o = ΔΔG_H  − ΔΔG_HO
    Δln_K_h_e = (ΔΔG_H − ΔΔG_HE2) / 2
    Δln_K_l_o = ΔΔG_L  − ΔΔG_LO
    Δln_K_l_e = (ΔΔG_L − ΔΔG_LE2) / 2

Data requirements
-----------------
``data`` must have (stored as ``pytree_node=False`` statics):
    struct_names            — tuple containing exactly {'H', 'HO', 'L', 'LO', 'HE2', 'LE2'}
    struct_features         — (M, S) float32  per-mutation per-structure ΔΔG prior means
    mut_geno_matrix         — (M, G) for mutation → genotype scatter
    num_mutation            — M
    num_struct              — S (must equal 6)

Input CSV format (passed via ``struct_ensemble_path``)
------------------------------------------------------
A CSV with columns ``mut`` (mutation label) and one column per structure
(``H``, ``HO``, ``L``, ``LO``, ``HE2``, ``LE2`` in any order).  Mutations
absent from the CSV receive a prior mean of 0.0 (no structural prediction).
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import pandas as pd
from flax.struct import dataclass
from functools import partial
from typing import Dict, Any, Union

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.prior import (
    sample_ddG,
)
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# Expected structure names — column order matches _PROJ below.
STRUCTURE_NAMES = ('H', 'HO', 'L', 'LO', 'HE2', 'LE2')

# Fixed thermodynamic projection: rows = (K_h_l, K_h_o, K_h_e, K_l_o, K_l_e),
# cols = (H, HO, L, LO, HE2, LE2).
_PROJ = jnp.array([
    [ 1.,   0., -1.,   0.,   0.,   0. ],  # K_h_l = ΔΔG_H − ΔΔG_L
    [ 1.,  -1.,  0.,   0.,   0.,   0. ],  # K_h_o = ΔΔG_H − ΔΔG_HO
    [ 0.5,  0.,  0.,   0.,  -0.5,  0. ],  # K_h_e = (ΔΔG_H − ΔΔG_HE2)/2
    [ 0.,   0.,  1.,  -1.,   0.,   0. ],  # K_l_o = ΔΔG_L − ΔΔG_LO
    [ 0.,   0.,  0.5,  0.,   0.,  -0.5],  # K_l_e = (ΔΔG_L − ΔΔG_LE2)/2
])


# ──────────────────────────────────────────────────────────────────────────────
# Priors pytree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the lnK_ddG_prior mwc-dimer theta model."""

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

    # Physical concentrations (M); Sochor, PeerJ 2014
    theta_tf_total_M: float    # ≈ 6.5e-7 M (650 nM)
    theta_op_total_M: float    # ≈ 2.5e-8 M (25 nM)

    # Unit-conversion factor applied to data.titrant_conc before thermodynamics.
    # Default 1e-3 assumes concentrations are in mM and K values are in M⁻¹.
    theta_conc_unit_scale: float


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_struct_perm(data):
    """
    Verify struct_names contains exactly the required structures and return a
    permutation index mapping data's S ordering to STRUCTURE_NAMES order.
    """
    data_names = data.struct_names
    if set(data_names) != set(STRUCTURE_NAMES):
        raise ValueError(
            f"lnK_ddG_prior (mwc_dimer) requires struct_names to contain exactly "
            f"{set(STRUCTURE_NAMES)}; got {data_names}."
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


def _assemble_K(ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
                ln_K_l_o_wt, ln_K_l_e_wt,
                delta_lnK, M_mat):
    """
    Assemble per-genotype K values from WT scalars + per-mutation deltas.

    Parameters
    ----------
    delta_lnK : (M, 5)
    M_mat     : (M, G)

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

    ln_K_h_l = ln_K_h_l_wt + d_h_l @ M_mat
    ln_K_h_o = ln_K_h_o_wt + d_h_o @ M_mat
    ln_K_h_e = ln_K_h_e_wt[:, None] + (d_h_e @ M_mat)[None, :]
    ln_K_l_o = ln_K_l_o_wt + d_l_o @ M_mat
    ln_K_l_e = ln_K_l_e_wt[:, None] + (d_l_e @ M_mat)[None, :]

    return ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_ddG_prior mwc-dimer theta model.

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
    M_mat   = jnp.array(data.mut_geno_matrix)
    num_mut = data.num_mutation
    # Reorder columns to canonical STRUCTURE_NAMES order
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]   # (M, S)

    tf_total        = priors.theta_tf_total_M
    op_total        = priors.theta_op_total_M
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

    # ── Per-mutation per-structure ΔΔG latent variables ──────────────────────
    # ddG_prior_means (M, S) serve as the Normal prior means; σ_s is learned.
    ddG = sample_ddG(name, list(STRUCTURE_NAMES), num_mut, ddG_prior_means)  # (M, S)

    # Project (M, 6) → (M, 5) Δln_K
    delta_lnK  = _project_ddG(ddG)

    pyro.deterministic(f"{name}_d_ln_K_h_l", delta_lnK[:, 0])
    pyro.deterministic(f"{name}_d_ln_K_h_o", delta_lnK[:, 1])
    pyro.deterministic(f"{name}_d_ln_K_h_e", delta_lnK[:, 2])
    pyro.deterministic(f"{name}_d_ln_K_l_o", delta_lnK[:, 3])
    pyro.deterministic(f"{name}_d_ln_K_l_e", delta_lnK[:, 4])

    # ── Assemble per-genotype K values ────────────────────────────────────────
    ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e = _assemble_K(
        ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
        ln_K_l_o_wt, ln_K_l_e_wt,
        delta_lnK, M_mat,
    )

    pyro.deterministic(f"{name}_ln_K_h_l", ln_K_h_l)
    pyro.deterministic(f"{name}_ln_K_h_o", ln_K_h_o)
    pyro.deterministic(f"{name}_ln_K_h_e", ln_K_h_e)
    pyro.deterministic(f"{name}_ln_K_l_o", ln_K_l_o)
    pyro.deterministic(f"{name}_ln_K_l_e", ln_K_l_e)

    # ── Population moments for transformation model ───────────────────────────
    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                data.titrant_conc * conc_unit_scale,
                                tf_total, op_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Guide
# ──────────────────────────────────────────────────────────────────────────────

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the lnK_ddG_prior mwc-dimer theta model."""

    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    M_mat   = jnp.array(data.mut_geno_matrix)
    num_mut = data.num_mutation
    S       = data.num_struct
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]   # (M, S)

    tf_total        = priors.theta_tf_total_M
    op_total        = priors.theta_op_total_M
    conc_unit_scale = priors.theta_conc_unit_scale

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

    # ── Register σ_s so param store includes it before sampling ──────────────
    pyro.param(
        f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive,
    )

    # ── Variational params for ddG offsets, shape (S, M) in plate layout ─────
    ddG_offset_locs   = pyro.param(f"{name}_ddG_offset_locs",
                                   jnp.zeros((S, num_mut)))
    ddG_offset_scales = pyro.param(f"{name}_ddG_offset_scales",
                                   jnp.ones((S, num_mut)),
                                   constraint=constraints.positive)

    # ── Sample WT K values ────────────────────────────────────────────────────
    ln_K_h_l_wt = pyro.sample(f"{name}_ln_K_h_l_wt", dist.Normal(ln_K_h_l_wt_loc, ln_K_h_l_wt_scale))
    ln_K_h_o_wt = pyro.sample(f"{name}_ln_K_h_o_wt", dist.Normal(ln_K_h_o_wt_loc, ln_K_h_o_wt_scale))
    ln_K_l_o_wt = pyro.sample(f"{name}_ln_K_l_o_wt", dist.Normal(ln_K_l_o_wt_loc, ln_K_l_o_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_h_e_wt = pyro.sample(f"{name}_ln_K_h_e_wt", dist.Normal(ln_K_h_e_wt_locs, ln_K_h_e_wt_scales))
        ln_K_l_e_wt = pyro.sample(f"{name}_ln_K_l_e_wt", dist.Normal(ln_K_l_e_wt_locs, ln_K_l_e_wt_scales))

    # ── Sample ddG offsets (matches model's prior.sample_ddG plate layout) ────
    with pyro.plate(f"{name}_struct_plate", S, dim=-2):
        with pyro.plate(f"{name}_mut_plate", num_mut, dim=-1):
            offsets = pyro.sample(
                f"{name}_ddG_offset",
                dist.Normal(ddG_offset_locs, ddG_offset_scales),
            )  # (S, M)

    sigma_s = pyro.param(
        f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive,
    )
    ddG_SM = ddG_prior_means.T + sigma_s[:, None] * offsets   # (S, M)
    ddG    = ddG_SM.T                                          # (M, S)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    delta_lnK = _project_ddG(ddG)
    ln_K_h_l, ln_K_h_o, ln_K_h_e, ln_K_l_o, ln_K_l_e = _assemble_K(
        ln_K_h_l_wt, ln_K_h_o_wt, ln_K_h_e_wt,
        ln_K_l_o_wt, ln_K_l_e_wt,
        delta_lnK, M_mat,
    )

    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                data.titrant_conc * conc_unit_scale,
                                tf_total, op_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # All values calibrated to Sochor et al. 2014 (PeerJ 2:e498), Table 1.
    p["theta_ln_K_h_l_wt_loc"]   = 1.84
    p["theta_ln_K_h_l_wt_scale"] = 2.0
    p["theta_ln_K_h_o_wt_loc"]   = 19.9
    p["theta_ln_K_h_o_wt_scale"] = 2.0
    p["theta_ln_K_h_e_wt_loc"]   = 10.9
    p["theta_ln_K_h_e_wt_scale"] = 2.0
    p["theta_ln_K_l_o_wt_loc"]   = -2.3
    p["theta_ln_K_l_o_wt_scale"] = 3.0
    p["theta_ln_K_l_e_wt_loc"]   = 13.5
    p["theta_ln_K_l_e_wt_scale"] = 2.0
    p["theta_tf_total_M"]        = 6.5e-7
    p["theta_op_total_M"]        = 2.5e-8
    p["theta_conc_unit_scale"]   = 1e-3
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    S = data.num_struct
    g = {}
    g[f"{name}_ln_K_h_l_wt"]  = jnp.array(1.84)
    g[f"{name}_ln_K_h_o_wt"]  = jnp.array(19.9)
    g[f"{name}_ln_K_h_e_wt"]  = jnp.full(T, 10.9)
    g[f"{name}_ln_K_l_o_wt"]  = jnp.array(-2.3)
    g[f"{name}_ln_K_l_e_wt"]  = jnp.full(T, 13.5)
    g[f"{name}_ddG_offset"]    = jnp.zeros((S, M))
    return g


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    geno_dim     = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    titrant_dim  = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    num_mut = len(ctx.mut_labels)

    # Scalar per-genotype K values
    geno_df = (ctx.growth_tm.df[["genotype", "genotype_idx"]]
               .drop_duplicates().copy())
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(
        input_df=geno_df,
        params_to_get=["ln_K_h_l", "ln_K_h_o", "ln_K_l_o"],
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
                        "d_ln_K_l_o", "d_ln_K_l_e"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    return specs


from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer.thermo import (
    build_calc_df,
    compute_theta_samples,
)
