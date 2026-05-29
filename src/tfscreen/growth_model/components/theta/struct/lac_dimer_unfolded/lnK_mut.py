"""
K-assembly via additive mutation effects in ln-K space (lac-dimer-unfolded).

Identical to lac_dimer/lnK_mut.py with one additional scalar parameter:

    ln_K_U[g] = ln_K_U_wt + (d_ln_K_U · M)[g]

where K_U = [U]/[H] is the H→U unfolding equilibrium.  In WT K_U << 1.
Mutations that destabilise the protein increase K_U (positive d_ln_K_U).
When K_U dominates the partition function, θ collapses toward zero for all
effector concentrations regardless of the folded-state equilibria.

K_U has no effector or DNA binding, so it carries no T dimension and does
not interact with the epistasis terms (which remain on the folded-state K
values).
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import pandas as pd
from flax.struct import dataclass
from functools import partial
from typing import Dict, Any

from tfscreen.growth_model.data_class import GrowthData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix
from tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)


# ---------------------------------------------------------------------------
# Priors pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the additive ln-K lac-dimer-unfolded theta model."""

    # WT priors (Normal in ln-K space)
    theta_ln_K_op_wt_loc: float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc: float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc: float
    theta_ln_K_E_wt_scale: float
    theta_ln_K_U_wt_loc: float
    theta_ln_K_U_wt_scale: float

    # Physical concentrations (M)
    theta_tf_total_M: float
    theta_op_total_M: float

    # HalfNormal scales for mutation delta distributions
    theta_sigma_d_ln_K_op_scale: float
    theta_sigma_d_ln_K_HL_scale: float
    theta_sigma_d_ln_K_E_scale: float
    theta_sigma_d_ln_K_U_scale: float

    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    theta_epi_tau_scale: float
    theta_epi_slab_scale: float
    theta_epi_slab_df: float


# ---------------------------------------------------------------------------
# Assembly helpers (unchanged from lac_dimer/lnK_mut.py)
# ---------------------------------------------------------------------------

def _assemble_scalar(wt, d_offsets, sigma_d, mut_scatter,
                     epi_offsets=None, sigma_epi=None, pair_scatter=None):
    d = d_offsets * sigma_d
    result = wt + mut_scatter(d)
    if epi_offsets is not None:
        result = result + pair_scatter(epi_offsets * sigma_epi)
    return result


def _assemble_titrant(wt, d_offsets, sigma_d, mut_scatter,
                      epi_offsets=None, sigma_epi=None, pair_scatter=None):
    d = d_offsets * sigma_d[:, None]
    result = wt[:, None] + mut_scatter(d)
    if epi_offsets is not None:
        result = result + pair_scatter(epi_offsets * sigma_epi)
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the additive ln-K lac-dimer-unfolded hierarchical model.

    Identical to lac_dimer/lnK_mut.py with an additional scalar K_U parameter
    representing global protein stability (H→U equilibrium).
    """
    T = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

    # ── WT parameters ────────────────────────────────────────────────────────
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt",
        dist.Normal(priors.theta_ln_K_op_wt_loc, priors.theta_ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt",
        dist.Normal(priors.theta_ln_K_HL_wt_loc, priors.theta_ln_K_HL_wt_scale))
    ln_K_U_wt = pyro.sample(
        f"{name}_ln_K_U_wt",
        dist.Normal(priors.theta_ln_K_U_wt_loc, priors.theta_ln_K_U_wt_scale))

    sigma_d_K_op = pyro.sample(
        f"{name}_sigma_d_ln_K_op",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_op_scale))
    sigma_d_K_HL = pyro.sample(
        f"{name}_sigma_d_ln_K_HL",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_HL_scale))
    sigma_d_K_U = pyro.sample(
        f"{name}_sigma_d_ln_K_U",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_U_scale))

    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(priors.theta_ln_K_E_wt_loc, priors.theta_ln_K_E_wt_scale))
        sigma_d_K_E = pyro.sample(
            f"{name}_sigma_d_ln_K_E",
            dist.HalfNormal(priors.theta_sigma_d_ln_K_E_scale))

    # ── Mutation delta offsets ────────────────────────────────────────────────
    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_op_off = pyro.sample(f"{name}_d_ln_K_op_offset", dist.Normal(0.0, 1.0))
        d_K_HL_off = pyro.sample(f"{name}_d_ln_K_HL_offset", dist.Normal(0.0, 1.0))
        d_K_U_off  = pyro.sample(f"{name}_d_ln_K_U_offset",  dist.Normal(0.0, 1.0))

    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_E_off = pyro.sample(f"{name}_d_ln_K_E_offset", dist.Normal(0.0, 1.0))

    # ── Optional epistasis (folded-state K values only) ───────────────────────
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        num_pair = data.num_pair

        tau_epi = pyro.sample(
            f"{name}_epi_tau",
            dist.HalfCauchy(priors.theta_epi_tau_scale))
        c2_epi = pyro.sample(
            f"{name}_epi_c2",
            dist.InverseGamma(priors.theta_epi_slab_df / 2.0,
                              priors.theta_epi_slab_df * priors.theta_epi_slab_scale ** 2 / 2.0))

        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_lam = pyro.sample(f"{name}_epi_ln_K_op_lambda", dist.HalfCauchy(1.0))
            epi_K_op_off = pyro.sample(f"{name}_epi_ln_K_op_offset", dist.Normal(0.0, 1.0))
            epi_K_HL_lam = pyro.sample(f"{name}_epi_ln_K_HL_lambda", dist.HalfCauchy(1.0))
            epi_K_HL_off = pyro.sample(f"{name}_epi_ln_K_HL_offset", dist.Normal(0.0, 1.0))

        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_lam = pyro.sample(f"{name}_epi_ln_K_E_lambda", dist.HalfCauchy(1.0))
                epi_K_E_off = pyro.sample(f"{name}_epi_ln_K_E_offset", dist.Normal(0.0, 1.0))
    else:
        pair_scatter = None

    # ── Assembled mutation deltas ─────────────────────────────────────────────
    d_ln_K_op = d_K_op_off * sigma_d_K_op
    d_ln_K_HL = d_K_HL_off * sigma_d_K_HL
    d_ln_K_E  = d_K_E_off * sigma_d_K_E[:, None]
    d_ln_K_U  = d_K_U_off * sigma_d_K_U

    pyro.deterministic(f"{name}_d_ln_K_op", d_ln_K_op)
    pyro.deterministic(f"{name}_d_ln_K_HL", d_ln_K_HL)
    pyro.deterministic(f"{name}_d_ln_K_E",  d_ln_K_E)
    pyro.deterministic(f"{name}_d_ln_K_U",  d_ln_K_U)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_ln_K_op = epi_K_op_off * tau_epi * _lam_tilde(epi_K_op_lam)
        epi_ln_K_HL = epi_K_HL_off * tau_epi * _lam_tilde(epi_K_HL_lam)
        epi_ln_K_E  = epi_K_E_off  * tau_epi * _lam_tilde(epi_K_E_lam)

        pyro.deterministic(f"{name}_epi_ln_K_op", epi_ln_K_op)
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_ln_K_HL)
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_ln_K_E)

    # ── Assemble per-genotype K values ────────────────────────────────────────
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, mut_scatter)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, mut_scatter)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, mut_scatter)
    ln_K_U  = ln_K_U_wt + mut_scatter(d_ln_K_U)   # (G,)

    if has_epi:
        ln_K_op = ln_K_op + pair_scatter(epi_ln_K_op)
        ln_K_HL = ln_K_HL + pair_scatter(epi_ln_K_HL)
        ln_K_E  = ln_K_E  + pair_scatter(epi_ln_K_E)

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)
    pyro.deterministic(f"{name}_ln_K_U",  ln_K_U)

    # ── Population moments ────────────────────────────────────────────────────
    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                       data.titrant_conc, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the additive ln-K lac-dimer-unfolded model."""

    T = data.num_titrant_name
    num_mut = data.num_mutation
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    has_epi = data.num_pair > 0

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

    # ── Variational params for WT K values ────────────────────────────────────
    ln_K_op_wt_loc   = pyro.param(f"{name}_ln_K_op_wt_loc",   jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(f"{name}_ln_K_op_wt_scale", jnp.array(1.0), constraint=dist.constraints.positive)
    ln_K_HL_wt_loc   = pyro.param(f"{name}_ln_K_HL_wt_loc",   jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(f"{name}_ln_K_HL_wt_scale", jnp.array(1.0), constraint=dist.constraints.positive)
    ln_K_U_wt_loc    = pyro.param(f"{name}_ln_K_U_wt_loc",    jnp.array(priors.theta_ln_K_U_wt_loc))
    ln_K_U_wt_scale  = pyro.param(f"{name}_ln_K_U_wt_scale",  jnp.array(1.0), constraint=dist.constraints.positive)

    sigma_d_K_op_loc   = pyro.param(f"{name}_sigma_d_ln_K_op_loc",   jnp.array(-1.0))
    sigma_d_K_op_scale = pyro.param(f"{name}_sigma_d_ln_K_op_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    sigma_d_K_HL_loc   = pyro.param(f"{name}_sigma_d_ln_K_HL_loc",   jnp.array(-1.0))
    sigma_d_K_HL_scale = pyro.param(f"{name}_sigma_d_ln_K_HL_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    sigma_d_K_U_loc    = pyro.param(f"{name}_sigma_d_ln_K_U_loc",    jnp.array(-1.0))
    sigma_d_K_U_scale  = pyro.param(f"{name}_sigma_d_ln_K_U_scale",  jnp.array(0.1), constraint=dist.constraints.positive)

    ln_K_E_wt_locs   = pyro.param(f"{name}_ln_K_E_wt_locs",   jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(f"{name}_ln_K_E_wt_scales", jnp.ones(T), constraint=dist.constraints.positive)
    sigma_d_K_E_locs  = pyro.param(f"{name}_sigma_d_ln_K_E_locs",  jnp.full(T, -1.0))
    sigma_d_K_E_scales= pyro.param(f"{name}_sigma_d_ln_K_E_scales", jnp.full(T, 0.1), constraint=dist.constraints.positive)

    # ── Variational params for mutation delta offsets ─────────────────────────
    d_K_op_locs   = pyro.param(f"{name}_d_ln_K_op_offset_locs",   jnp.zeros(num_mut))
    d_K_op_scales = pyro.param(f"{name}_d_ln_K_op_offset_scales", jnp.ones(num_mut), constraint=dist.constraints.positive)
    d_K_HL_locs   = pyro.param(f"{name}_d_ln_K_HL_offset_locs",   jnp.zeros(num_mut))
    d_K_HL_scales = pyro.param(f"{name}_d_ln_K_HL_offset_scales", jnp.ones(num_mut), constraint=dist.constraints.positive)
    d_K_U_locs    = pyro.param(f"{name}_d_ln_K_U_offset_locs",    jnp.zeros(num_mut))
    d_K_U_scales  = pyro.param(f"{name}_d_ln_K_U_offset_scales",  jnp.ones(num_mut), constraint=dist.constraints.positive)
    d_K_E_locs    = pyro.param(f"{name}_d_ln_K_E_offset_locs",    jnp.zeros((T, num_mut)))
    d_K_E_scales  = pyro.param(f"{name}_d_ln_K_E_offset_scales",  jnp.ones((T, num_mut)), constraint=dist.constraints.positive)

    # ── Optional epistasis variational params ─────────────────────────────────
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        num_pair = data.num_pair

        tau_epi_loc   = pyro.param(f"{name}_epi_tau_loc",   jnp.array(-1.0))
        tau_epi_scale = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1), constraint=dist.constraints.positive)
        c2_epi_loc    = pyro.param(f"{name}_epi_c2_loc",    jnp.array(1.4))
        c2_epi_scale  = pyro.param(f"{name}_epi_c2_scale",  jnp.array(0.5), constraint=dist.constraints.positive)

        epi_K_op_lam_locs   = pyro.param(f"{name}_epi_ln_K_op_lambda_locs",   jnp.zeros(num_pair))
        epi_K_op_lam_scales = pyro.param(f"{name}_epi_ln_K_op_lambda_scales", jnp.ones(num_pair), constraint=dist.constraints.positive)
        epi_K_op_locs       = pyro.param(f"{name}_epi_ln_K_op_offset_locs",   jnp.zeros(num_pair))
        epi_K_op_scales     = pyro.param(f"{name}_epi_ln_K_op_offset_scales", jnp.ones(num_pair), constraint=dist.constraints.positive)
        epi_K_HL_lam_locs   = pyro.param(f"{name}_epi_ln_K_HL_lambda_locs",   jnp.zeros(num_pair))
        epi_K_HL_lam_scales = pyro.param(f"{name}_epi_ln_K_HL_lambda_scales", jnp.ones(num_pair), constraint=dist.constraints.positive)
        epi_K_HL_locs       = pyro.param(f"{name}_epi_ln_K_HL_offset_locs",   jnp.zeros(num_pair))
        epi_K_HL_scales     = pyro.param(f"{name}_epi_ln_K_HL_offset_scales", jnp.ones(num_pair), constraint=dist.constraints.positive)
        epi_K_E_lam_locs    = pyro.param(f"{name}_epi_ln_K_E_lambda_locs",    jnp.zeros((T, num_pair)))
        epi_K_E_lam_scales  = pyro.param(f"{name}_epi_ln_K_E_lambda_scales",  jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
        epi_K_E_locs        = pyro.param(f"{name}_epi_ln_K_E_offset_locs",    jnp.zeros((T, num_pair)))
        epi_K_E_scales      = pyro.param(f"{name}_epi_ln_K_E_offset_scales",  jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
    else:
        pair_scatter = None

    # ── Sample ───────────────────────────────────────────────────────────────
    ln_K_op_wt = pyro.sample(f"{name}_ln_K_op_wt", dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(f"{name}_ln_K_HL_wt", dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    ln_K_U_wt  = pyro.sample(f"{name}_ln_K_U_wt",  dist.Normal(ln_K_U_wt_loc,  ln_K_U_wt_scale))
    sigma_d_K_op = pyro.sample(f"{name}_sigma_d_ln_K_op", dist.LogNormal(sigma_d_K_op_loc, sigma_d_K_op_scale))
    sigma_d_K_HL = pyro.sample(f"{name}_sigma_d_ln_K_HL", dist.LogNormal(sigma_d_K_HL_loc, sigma_d_K_HL_scale))
    sigma_d_K_U  = pyro.sample(f"{name}_sigma_d_ln_K_U",  dist.LogNormal(sigma_d_K_U_loc,  sigma_d_K_U_scale))

    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt   = pyro.sample(f"{name}_ln_K_E_wt",   dist.Normal(ln_K_E_wt_locs,   ln_K_E_wt_scales))
        sigma_d_K_E = pyro.sample(f"{name}_sigma_d_ln_K_E", dist.LogNormal(sigma_d_K_E_locs, sigma_d_K_E_scales))

    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_op_off = pyro.sample(f"{name}_d_ln_K_op_offset", dist.Normal(d_K_op_locs, d_K_op_scales))
        d_K_HL_off = pyro.sample(f"{name}_d_ln_K_HL_offset", dist.Normal(d_K_HL_locs, d_K_HL_scales))
        d_K_U_off  = pyro.sample(f"{name}_d_ln_K_U_offset",  dist.Normal(d_K_U_locs,  d_K_U_scales))

    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_E_off = pyro.sample(f"{name}_d_ln_K_E_offset", dist.Normal(d_K_E_locs, d_K_E_scales))

    if has_epi:
        tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_epi_loc, tau_epi_scale))
        c2_epi  = pyro.sample(f"{name}_epi_c2",  dist.LogNormal(c2_epi_loc,  c2_epi_scale))

        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_lam = pyro.sample(f"{name}_epi_ln_K_op_lambda", dist.LogNormal(epi_K_op_lam_locs, epi_K_op_lam_scales))
            epi_K_op_off = pyro.sample(f"{name}_epi_ln_K_op_offset", dist.Normal(epi_K_op_locs, epi_K_op_scales))
            epi_K_HL_lam = pyro.sample(f"{name}_epi_ln_K_HL_lambda", dist.LogNormal(epi_K_HL_lam_locs, epi_K_HL_lam_scales))
            epi_K_HL_off = pyro.sample(f"{name}_epi_ln_K_HL_offset", dist.Normal(epi_K_HL_locs, epi_K_HL_scales))

        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_lam = pyro.sample(f"{name}_epi_ln_K_E_lambda", dist.LogNormal(epi_K_E_lam_locs, epi_K_E_lam_scales))
                epi_K_E_off = pyro.sample(f"{name}_epi_ln_K_E_offset", dist.Normal(epi_K_E_locs, epi_K_E_scales))

    # ── Assemble ──────────────────────────────────────────────────────────────
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, mut_scatter)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, mut_scatter)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, mut_scatter)
    ln_K_U  = ln_K_U_wt + mut_scatter(d_K_U_off * sigma_d_K_U)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        ln_K_op = ln_K_op + pair_scatter(epi_K_op_off * tau_epi * _lam_tilde(epi_K_op_lam))
        ln_K_HL = ln_K_HL + pair_scatter(epi_K_HL_off * tau_epi * _lam_tilde(epi_K_HL_lam))
        ln_K_E  = ln_K_E  + pair_scatter(epi_K_E_off  * tau_epi * _lam_tilde(epi_K_E_lam))

    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, ln_K_U,
                                       data.titrant_conc, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E, ln_K_U=ln_K_U,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    p["theta_ln_K_op_wt_loc"]   = 23.0
    p["theta_ln_K_op_wt_scale"] = 2.0
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    p["theta_ln_K_E_wt_loc"]    = 33.4
    p["theta_ln_K_E_wt_scale"]  = 3.0
    # WT unfolded state: K_U << 1 in WT (ln_K_U ≈ -12 → K_U ≈ 6e-6)
    p["theta_ln_K_U_wt_loc"]    = -12.0
    p["theta_ln_K_U_wt_scale"]  = 3.0
    p["theta_tf_total_M"]  = 6.5e-7
    p["theta_op_total_M"]  = 2.5e-8
    p["theta_sigma_d_ln_K_op_scale"]  = 1.0
    p["theta_sigma_d_ln_K_HL_scale"]  = 1.0
    p["theta_sigma_d_ln_K_E_scale"]   = 1.0
    p["theta_sigma_d_ln_K_U_scale"]   = 1.0
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    return p


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    g = {}
    g[f"{name}_ln_K_op_wt"]       = jnp.array(23.0)
    g[f"{name}_ln_K_HL_wt"]       = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]        = jnp.full(T, 33.4)
    g[f"{name}_ln_K_U_wt"]        = jnp.array(-12.0)
    g[f"{name}_sigma_d_ln_K_op"]  = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_HL"]  = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_E"]   = jnp.full(T, 0.5)
    g[f"{name}_sigma_d_ln_K_U"]   = jnp.array(0.5)
    g[f"{name}_d_ln_K_op_offset"] = jnp.zeros(M)
    g[f"{name}_d_ln_K_HL_offset"] = jnp.zeros(M)
    g[f"{name}_d_ln_K_E_offset"]  = jnp.zeros((T, M))
    g[f"{name}_d_ln_K_U_offset"]  = jnp.zeros(M)
    if data.num_pair > 0:
        P = data.num_pair
        g[f"{name}_epi_tau"]              = jnp.array(0.05)
        g[f"{name}_epi_c2"]              = jnp.array(4.0)
        g[f"{name}_epi_ln_K_op_lambda"]  = jnp.ones(P) * 0.5
        g[f"{name}_epi_ln_K_op_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_HL_lambda"]  = jnp.ones(P) * 0.5
        g[f"{name}_epi_ln_K_HL_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_E_lambda"]   = jnp.ones((T, P)) * 0.5
        g[f"{name}_epi_ln_K_E_offset"]   = jnp.zeros((T, P))
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
    """Predict theta for unmeasured genotypes (lac_dimer_unfolded/lnK_mut)."""
    import numpy as np
    from tfscreen.growth_model.posteriors import get_posterior_samples
    from tfscreen.growth_model.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )
    from tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
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
    d_E  = _load("theta_d_ln_K_E")    # (S, T, M)
    d_U  = _load("theta_d_ln_K_U")    # (S, M)

    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    ln_K_E_geno  = ln_K_E_wt[:, :, None] + np.einsum("stm,nm->stn", d_E, mut_mat)
    ln_K_U_geno  = ln_K_U_wt[:, None] + np.einsum("sm,nm->sn", d_U, mut_mat)

    if len(pair_labels) > 0:
        epi_op = _load("theta_epi_ln_K_op")
        epi_HL = _load("theta_epi_ln_K_HL")
        epi_E  = _load("theta_epi_ln_K_E")
        ln_K_op_geno += np.einsum("sp,np->sn", epi_op, pair_mat)
        ln_K_HL_geno += np.einsum("sp,np->sn", epi_HL, pair_mat)
        ln_K_E_geno  += np.einsum("stp,np->stn", epi_E, pair_mat)

    ln_K_op_rows = ln_K_op_geno[:, geno_idx]
    ln_K_HL_rows = ln_K_HL_geno[:, geno_idx]
    ln_K_E_rows  = ln_K_E_geno[:, titrant_idx, geno_idx]
    ln_K_U_rows  = ln_K_U_geno[:, geno_idx]

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
    geno_dim = ctx.growth_tm.tensor_dim_names.index("genotype")
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

    mut_df = pd.DataFrame({
        "mutation": ctx.mut_labels,
        "map_mut": range(num_mut),
    })
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_op", "d_ln_K_HL", "d_ln_K_U"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    theta_d_KE_rows = [
        {"titrant_name": t, "mutation": m,
         "map_theta_d_KE": ti * num_mut + mi}
        for ti, t in enumerate(titrant_names)
        for mi, m in enumerate(ctx.mut_labels)
    ]
    specs.append(dict(
        input_df=pd.DataFrame(theta_d_KE_rows),
        params_to_get=["d_ln_K_E"],
        map_column="map_theta_d_KE",
        get_columns=["titrant_name", "mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        num_pair = len(ctx.pair_labels)
        pair_df = pd.DataFrame({
            "pair": ctx.pair_labels,
            "map_pair": range(num_pair),
        })
        specs.append(dict(
            input_df=pair_df,
            params_to_get=["epi_ln_K_op", "epi_ln_K_HL"],
            map_column="map_pair",
            get_columns=["pair"],
            in_run_prefix="theta_",
        ))
        theta_epi_KE_rows = [
            {"titrant_name": t, "pair": p,
             "map_theta_epi_KE": ti * num_pair + pi}
            for ti, t in enumerate(titrant_names)
            for pi, p in enumerate(ctx.pair_labels)
        ]
        specs.append(dict(
            input_df=pd.DataFrame(theta_epi_KE_rows),
            params_to_get=["epi_ln_K_E"],
            map_column="map_theta_epi_KE",
            get_columns=["titrant_name", "pair"],
            in_run_prefix="theta_",
        ))

    return specs


from tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
    build_calc_df,
    compute_theta_samples,
)
