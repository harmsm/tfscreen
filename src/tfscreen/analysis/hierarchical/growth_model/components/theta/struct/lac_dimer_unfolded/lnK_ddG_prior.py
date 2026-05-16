"""
K-assembly via per-mutation per-structure ΔΔG with user-supplied prior means (lac-dimer-unfolded).

Identical to lac_dimer/lnK_ddG_prior.py with one additional scalar parameter:

    ln_K_U_wt  — WT H→U unfolding equilibrium constant (same for all genotypes)

Per-mutation effects on the unfolded state are deliberately absent: structure-
based ΔΔG predictors describe effects on the *folded* conformational equilibria,
not global protein stability.  Setting d_ln_K_U = 0 for all mutations is an
explicit design choice — use lac_dimer_unfolded/lnK_mut if per-mutation
unfolding effects should be inferred from the data.
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
from tfscreen.genetics.build_mut_geno_matrix import apply_mut_matrix
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.prior import (
    sample_ddG,
)
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
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
    """Hyperparameters for the lnK_ddG_prior lac-dimer-unfolded theta model."""

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


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_struct_perm(data):
    data_names = data.struct_names
    if set(data_names) != set(STRUCTURE_NAMES):
        raise ValueError(
            f"lnK_ddG_prior (lac_dimer_unfolded) requires struct_names to contain exactly "
            f"{set(STRUCTURE_NAMES)}; got {data_names}."
        )
    name_to_idx = {n: i for i, n in enumerate(data_names)}
    return [name_to_idx[s] for s in STRUCTURE_NAMES]


def _project_ddG(ddG):
    return ddG @ _PROJ.T


def _assemble_K(ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter):
    d_op = delta_lnK[:, 0]
    d_HL = delta_lnK[:, 1]
    d_E  = delta_lnK[:, 2]
    ln_K_op = ln_K_op_wt + mut_scatter(d_op)
    ln_K_HL = ln_K_HL_wt + mut_scatter(d_HL)
    ln_K_E  = ln_K_E_wt[:, None] + mut_scatter(d_E)[None, :]
    return ln_K_op, ln_K_HL, ln_K_E


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the lnK_ddG_prior lac-dimer-unfolded theta model.

    Adds ``ln_K_U_wt`` to the standard lnK_ddG_prior model.  Per-mutation
    effects on K_U are fixed at zero (no d_ln_K_U parameter).
    """
    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

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

    ln_K_op, ln_K_HL, ln_K_E = _assemble_K(
        ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter,
    )
    # K_U is homogeneous across genotypes — no per-mutation effects
    ln_K_U = jnp.full(data.num_genotype, ln_K_U_wt)

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
    """Variational guide for the lnK_ddG_prior lac-dimer-unfolded theta model."""

    perm = _get_struct_perm(data)
    perm_idx = jnp.array(perm)

    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    S       = data.num_struct
    ddG_prior_means = jnp.array(data.struct_features)[:, perm_idx]

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

    pyro.param(f"{name}_ddG_sigma_s", jnp.ones(S), constraint=constraints.positive)

    ddG_offset_locs   = pyro.param(f"{name}_ddG_offset_locs",   jnp.zeros((S, num_mut)))
    ddG_offset_scales = pyro.param(f"{name}_ddG_offset_scales", jnp.ones((S, num_mut)), constraint=constraints.positive)

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
    ddG_SM  = ddG_prior_means.T + sigma_s[:, None] * offsets
    ddG     = ddG_SM.T

    delta_lnK = _project_ddG(ddG)
    ln_K_op, ln_K_HL, ln_K_E = _assemble_K(
        ln_K_op_wt, ln_K_HL_wt, ln_K_E_wt, delta_lnK, mut_scatter,
    )
    ln_K_U = jnp.full(data.num_genotype, ln_K_U_wt)

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
    p["theta_tf_total_M"]  = 6.5e-7
    p["theta_op_total_M"]  = 2.5e-8
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
    g[f"{name}_ddG_offset"]   = jnp.zeros((S, M))
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
    Predict theta for unmeasured genotypes (lac_dimer_unfolded/lnK_ddG_prior).

    K_U is homogeneous across genotypes: ln_K_U_wt is broadcast to all rows.
    """
    import numpy as np
    from tfscreen.analysis.hierarchical.posteriors import get_posterior_samples
    from tfscreen.analysis.hierarchical.growth_model.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )
    from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
        _solve_theta_np,
        _ZERO_CONC_VALUE,
    )

    target_genotypes = list(target_genotypes)
    mut_mat, _, is_valid = _build_genotype_indicators(
        target_genotypes, mut_labels, []
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

    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    ln_K_E_geno  = (ln_K_E_wt[:, :, None]
                    + np.einsum("sm,nm->sn", d_E, mut_mat)[:, None, :])
    ln_K_U_rows  = ln_K_U_wt[:, None]

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

    return specs


from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
    build_calc_df,
    compute_theta_samples,
)
