"""
K-assembly via additive mutation effects in ln-K space.

Mutation effects are additive in ln(K) space, with optional pairwise
epistasis terms:

    ln_K_op[g]   = ln_K_op_wt   + (d_ln_K_op   · M)[g]   + (epi_ln_K_op   · P)[g]
    ln_K_HL[g]   = ln_K_HL_wt   + (d_ln_K_HL   · M)[g]   + (epi_ln_K_HL   · P)[g]
    ln_K_E[t, g] = ln_K_E_wt[t] + (d_ln_K_E[t] · M)[g]   + (epi_ln_K_E[t] · P)[g]

where M[m, g] = 1 if mutation m is in genotype g  (shape: num_mutation × G)
and   P[p, g] = 1 if pair   p is in genotype g    (shape: num_pair    × G).

K_op and K_HL are protein properties with no dependence on effector identity
(no T dimension). K_E is per effector species (T dimension).

Epistasis uses a regularised horseshoe prior (shared τ, slab c², per-pair λ)
and is only sampled when data.num_pair > 0.

Thermodynamic functions (partition function, Newton solve for free effector,
population moments) are imported from thermo.py.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import pandas as pd
from flax.struct import dataclass
from functools import partial
from typing import Dict, Any

from tfscreen.tfmodel.data_class import GrowthData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix
from tfscreen.tfmodel.generative.components.theta.struct.lac_dimer.thermo import (
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
    """Hyperparameters for the additive ln-K lac-dimer theta model."""

    # WT priors (Normal in ln-K space)
    theta_ln_K_op_wt_loc: float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc: float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc: float
    theta_ln_K_E_wt_scale: float

    # Total TF and operator concentrations (M); treated as fixed physical constants.
    # Default values from Sochor, PeerJ 2014, https://doi.org/10.7717/peerj.498
    theta_tf_total_M: float   # ≈ 6.5e-7 M (650 nM)
    theta_op_total_M: float   # ≈ 2.5e-8 M (25 nM)

    # HalfNormal scales for mutation delta distributions
    theta_sigma_d_ln_K_op_scale: float
    theta_sigma_d_ln_K_HL_scale: float
    theta_sigma_d_ln_K_E_scale: float

    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    theta_epi_tau_scale: float     # HalfCauchy scale for global τ (shared across K_op, K_HL, K_E)
    theta_epi_slab_scale: float    # typical size of a large epistasis effect
    theta_epi_slab_df: float       # InvGamma shape ν (usually 4)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_scalar(wt, d_offsets, sigma_d, mut_scatter,
                     epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype scalar parameter from WT + mutation deltas.

    Parameters
    ----------
    wt : scalar
    d_offsets : (num_mutation,)
    sigma_d : scalar
    mut_scatter : callable (M,) -> (G,)
        Scatters per-mutation values to genotype space via COO index arrays.
    epi_offsets : (num_pair,) or None
    sigma_epi : scalar or None
    pair_scatter : callable or None
        ``pair_scatter(epi) -> (G,)``.  Scatters pair epistasis values to
        genotype-space via COO scatter-add.

    Returns
    -------
    (G,)
    """
    d = d_offsets * sigma_d         # (M,)
    result = wt + mut_scatter(d)    # (G,)
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi      # (P,)
        result = result + pair_scatter(epi)  # (G,)
    return result


def _assemble_titrant(wt, d_offsets, sigma_d, mut_scatter,
                      epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype parameter from T-shaped WT + mutation deltas.

    Parameters
    ----------
    wt : (T,)
    d_offsets : (T, num_mutation)
    sigma_d : (T,)
    mut_scatter : callable (..., M) -> (..., G)
        Scatters per-mutation values to genotype space; handles leading dims.
    epi_offsets : (T, num_pair) or None
    sigma_epi : (T,) or None
    pair_scatter : callable or None
        ``pair_scatter(epi) -> (T, G)``.  Scatters pair epistasis values to
        genotype-space via COO scatter-add.

    Returns
    -------
    (T, G)
    """
    d = d_offsets * sigma_d[:, None]              # (T, M)
    result = wt[:, None] + mut_scatter(d)         # (T, G)
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi[:, None]   # (T, P)
        result = result + pair_scatter(epi)       # (T, G)
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the additive ln-K lac-dimer hierarchical model.

    Samples WT equilibrium constants and per-mutation delta parameters, then
    assembles per-genotype constants via matrix multiplication.  Epistasis uses
    a regularised horseshoe prior (shared τ and slab c², per-pair local λ).

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : GrowthData
        Must have ``mut_geno_matrix`` (num_mutation × G) and ``num_mutation``.
        If ``num_pair > 0``, must also have ``pair_geno_matrix`` (num_pair × G).
    priors : ModelPriors

    Returns
    -------
    ThetaParam
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

    # ------------------------------------------------------------------
    # WT parameters for K_op and K_HL (scalars — no T dimension)
    # ------------------------------------------------------------------
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt",
        dist.Normal(priors.theta_ln_K_op_wt_loc, priors.theta_ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt",
        dist.Normal(priors.theta_ln_K_HL_wt_loc, priors.theta_ln_K_HL_wt_scale))

    sigma_d_K_op = pyro.sample(
        f"{name}_sigma_d_ln_K_op",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_op_scale))
    sigma_d_K_HL = pyro.sample(
        f"{name}_sigma_d_ln_K_HL",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_HL_scale))

    # ------------------------------------------------------------------
    # WT parameters for K_E (T-dim — one per effector species)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(priors.theta_ln_K_E_wt_loc, priors.theta_ln_K_E_wt_scale))
        sigma_d_K_E = pyro.sample(
            f"{name}_sigma_d_ln_K_E",
            dist.HalfNormal(priors.theta_sigma_d_ln_K_E_scale))

    # ------------------------------------------------------------------
    # Mutation delta offsets for K_op and K_HL: shape (num_mutation,)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_op_off = pyro.sample(
            f"{name}_d_ln_K_op_offset", dist.Normal(0.0, 1.0))
        d_K_HL_off = pyro.sample(
            f"{name}_d_ln_K_HL_offset", dist.Normal(0.0, 1.0))

    # ------------------------------------------------------------------
    # Mutation delta offsets for K_E: shape (T, num_mutation)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_E_off = pyro.sample(
                f"{name}_d_ln_K_E_offset", dist.Normal(0.0, 1.0))

    # ------------------------------------------------------------------
    # Optional epistasis — regularised horseshoe prior
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                              pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                              pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                              num_genotype=data.num_genotype)
        num_pair = data.num_pair

        # Shared global scale and slab variance across all equilibrium constants
        tau_epi = pyro.sample(
            f"{name}_epi_tau",
            dist.HalfCauchy(priors.theta_epi_tau_scale))
        c2_epi = pyro.sample(
            f"{name}_epi_c2",
            dist.InverseGamma(priors.theta_epi_slab_df / 2.0,
                              priors.theta_epi_slab_df * priors.theta_epi_slab_scale ** 2 / 2.0))

        # Scalar K_op and K_HL: local scales and offsets shape (num_pair,)
        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_lam = pyro.sample(
                f"{name}_epi_ln_K_op_lambda", dist.HalfCauchy(1.0))
            epi_K_op_off = pyro.sample(
                f"{name}_epi_ln_K_op_offset", dist.Normal(0.0, 1.0))
            epi_K_HL_lam = pyro.sample(
                f"{name}_epi_ln_K_HL_lambda", dist.HalfCauchy(1.0))
            epi_K_HL_off = pyro.sample(
                f"{name}_epi_ln_K_HL_offset", dist.Normal(0.0, 1.0))

        # T-dim K_E: local scales and offsets shape (T, num_pair)
        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_lam = pyro.sample(
                    f"{name}_epi_ln_K_E_lambda", dist.HalfCauchy(1.0))
                epi_K_E_off = pyro.sample(
                    f"{name}_epi_ln_K_E_offset", dist.Normal(0.0, 1.0))
    else:
        pair_scatter = None

    # ------------------------------------------------------------------
    # Assembled mutation deltas (registered for posterior extraction)
    # ------------------------------------------------------------------
    d_ln_K_op = d_K_op_off * sigma_d_K_op           # (M,)
    d_ln_K_HL = d_K_HL_off * sigma_d_K_HL           # (M,)
    d_ln_K_E  = d_K_E_off * sigma_d_K_E[:, None]    # (T, M)

    pyro.deterministic(f"{name}_d_ln_K_op", d_ln_K_op)
    pyro.deterministic(f"{name}_d_ln_K_HL", d_ln_K_HL)
    pyro.deterministic(f"{name}_d_ln_K_E",  d_ln_K_E)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_ln_K_op = epi_K_op_off * tau_epi * _lam_tilde(epi_K_op_lam)          # (P,)
        epi_ln_K_HL = epi_K_HL_off * tau_epi * _lam_tilde(epi_K_HL_lam)          # (P,)
        epi_ln_K_E  = epi_K_E_off  * tau_epi * _lam_tilde(epi_K_E_lam)           # (T, P)

        pyro.deterministic(f"{name}_epi_ln_K_op", epi_ln_K_op)
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_ln_K_HL)
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_ln_K_E)

    # ------------------------------------------------------------------
    # Assemble per-genotype equilibrium constants
    # ------------------------------------------------------------------
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, mut_scatter)    # (G,)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, mut_scatter)    # (G,)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, mut_scatter)      # (T, G)

    if has_epi:
        ln_K_op = ln_K_op + pair_scatter(epi_ln_K_op)
        ln_K_HL = ln_K_HL + pair_scatter(epi_ln_K_HL)
        ln_K_E  = ln_K_E  + pair_scatter(epi_ln_K_E)

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)

    # ------------------------------------------------------------------
    # Population moments over growth concentrations
    # ------------------------------------------------------------------
    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                       data.titrant_conc, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op,
                      ln_K_HL=ln_K_HL,
                      ln_K_E=ln_K_E,
                      tf_total=tf_total,
                      op_total=op_total,
                      mu=mu,
                      sigma=sigma)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the additive ln-K lac-dimer model."""

    T = data.num_titrant_name
    num_mut = data.num_mutation
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    has_epi = data.num_pair > 0

    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M

    # ------------------------------------------------------------------
    # Scalar variational parameters for K_op_wt, K_HL_wt, sigma_d_K_op/HL
    # ------------------------------------------------------------------
    ln_K_op_wt_loc = pyro.param(
        f"{name}_ln_K_op_wt_loc", jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(
        f"{name}_ln_K_op_wt_scale", jnp.array(1.0),
        constraint=dist.constraints.positive)

    ln_K_HL_wt_loc = pyro.param(
        f"{name}_ln_K_HL_wt_loc", jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(
        f"{name}_ln_K_HL_wt_scale", jnp.array(1.0),
        constraint=dist.constraints.positive)

    sigma_d_K_op_loc = pyro.param(
        f"{name}_sigma_d_ln_K_op_loc", jnp.array(-1.0))
    sigma_d_K_op_scale = pyro.param(
        f"{name}_sigma_d_ln_K_op_scale", jnp.array(0.1),
        constraint=dist.constraints.positive)

    sigma_d_K_HL_loc = pyro.param(
        f"{name}_sigma_d_ln_K_HL_loc", jnp.array(-1.0))
    sigma_d_K_HL_scale = pyro.param(
        f"{name}_sigma_d_ln_K_HL_scale", jnp.array(0.1),
        constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # T-dim variational parameters for K_E_wt and sigma_d_K_E
    # ------------------------------------------------------------------
    ln_K_E_wt_locs = pyro.param(
        f"{name}_ln_K_E_wt_locs", jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(
        f"{name}_ln_K_E_wt_scales", jnp.ones(T),
        constraint=dist.constraints.positive)

    sigma_d_K_E_locs = pyro.param(
        f"{name}_sigma_d_ln_K_E_locs", jnp.full(T, -1.0))
    sigma_d_K_E_scales = pyro.param(
        f"{name}_sigma_d_ln_K_E_scales", jnp.full(T, 0.1),
        constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for mutation delta offsets
    # ------------------------------------------------------------------
    d_K_op_locs = pyro.param(
        f"{name}_d_ln_K_op_offset_locs", jnp.zeros(num_mut))
    d_K_op_scales = pyro.param(
        f"{name}_d_ln_K_op_offset_scales", jnp.ones(num_mut),
        constraint=dist.constraints.positive)

    d_K_HL_locs = pyro.param(
        f"{name}_d_ln_K_HL_offset_locs", jnp.zeros(num_mut))
    d_K_HL_scales = pyro.param(
        f"{name}_d_ln_K_HL_offset_scales", jnp.ones(num_mut),
        constraint=dist.constraints.positive)

    d_K_E_locs = pyro.param(
        f"{name}_d_ln_K_E_offset_locs", jnp.zeros((T, num_mut)))
    d_K_E_scales = pyro.param(
        f"{name}_d_ln_K_E_offset_scales", jnp.ones((T, num_mut)),
        constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Optional epistasis — regularised horseshoe variational parameters
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        num_pair = data.num_pair

        # Shared τ and c²
        tau_epi_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-1.0))
        tau_epi_scale_p = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1),
                                     constraint=dist.constraints.positive)
        c2_epi_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
        c2_epi_scale_p = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                                    constraint=dist.constraints.positive)

        # Per-type local scales and offsets
        epi_K_op_lam_locs = pyro.param(
            f"{name}_epi_ln_K_op_lambda_locs", jnp.zeros(num_pair))
        epi_K_op_lam_scales = pyro.param(
            f"{name}_epi_ln_K_op_lambda_scales", jnp.ones(num_pair),
            constraint=dist.constraints.positive)
        epi_K_op_locs = pyro.param(
            f"{name}_epi_ln_K_op_offset_locs", jnp.zeros(num_pair))
        epi_K_op_scales = pyro.param(
            f"{name}_epi_ln_K_op_offset_scales", jnp.ones(num_pair),
            constraint=dist.constraints.positive)

        epi_K_HL_lam_locs = pyro.param(
            f"{name}_epi_ln_K_HL_lambda_locs", jnp.zeros(num_pair))
        epi_K_HL_lam_scales = pyro.param(
            f"{name}_epi_ln_K_HL_lambda_scales", jnp.ones(num_pair),
            constraint=dist.constraints.positive)
        epi_K_HL_locs = pyro.param(
            f"{name}_epi_ln_K_HL_offset_locs", jnp.zeros(num_pair))
        epi_K_HL_scales = pyro.param(
            f"{name}_epi_ln_K_HL_offset_scales", jnp.ones(num_pair),
            constraint=dist.constraints.positive)

        epi_K_E_lam_locs = pyro.param(
            f"{name}_epi_ln_K_E_lambda_locs", jnp.zeros((T, num_pair)))
        epi_K_E_lam_scales = pyro.param(
            f"{name}_epi_ln_K_E_lambda_scales", jnp.ones((T, num_pair)),
            constraint=dist.constraints.positive)
        epi_K_E_locs = pyro.param(
            f"{name}_epi_ln_K_E_offset_locs", jnp.zeros((T, num_pair)))
        epi_K_E_scales = pyro.param(
            f"{name}_epi_ln_K_E_offset_scales", jnp.ones((T, num_pair)),
            constraint=dist.constraints.positive)
    else:
        pair_scatter = None

    # ------------------------------------------------------------------
    # Sample within plates
    # ------------------------------------------------------------------
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt",
        dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt",
        dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    sigma_d_K_op = pyro.sample(
        f"{name}_sigma_d_ln_K_op",
        dist.LogNormal(sigma_d_K_op_loc, sigma_d_K_op_scale))
    sigma_d_K_HL = pyro.sample(
        f"{name}_sigma_d_ln_K_HL",
        dist.LogNormal(sigma_d_K_HL_loc, sigma_d_K_HL_scale))

    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(ln_K_E_wt_locs, ln_K_E_wt_scales))
        sigma_d_K_E = pyro.sample(
            f"{name}_sigma_d_ln_K_E",
            dist.LogNormal(sigma_d_K_E_locs, sigma_d_K_E_scales))

    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_op_off = pyro.sample(
            f"{name}_d_ln_K_op_offset",
            dist.Normal(d_K_op_locs, d_K_op_scales))
        d_K_HL_off = pyro.sample(
            f"{name}_d_ln_K_HL_offset",
            dist.Normal(d_K_HL_locs, d_K_HL_scales))

    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_E_off = pyro.sample(
                f"{name}_d_ln_K_E_offset",
                dist.Normal(d_K_E_locs, d_K_E_scales))

    if has_epi:
        tau_epi = pyro.sample(f"{name}_epi_tau",
                              dist.LogNormal(tau_epi_loc, tau_epi_scale_p))
        c2_epi = pyro.sample(f"{name}_epi_c2",
                             dist.LogNormal(c2_epi_loc, c2_epi_scale_p))

        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_lam = pyro.sample(
                f"{name}_epi_ln_K_op_lambda",
                dist.LogNormal(epi_K_op_lam_locs, epi_K_op_lam_scales))
            epi_K_op_off = pyro.sample(
                f"{name}_epi_ln_K_op_offset",
                dist.Normal(epi_K_op_locs, epi_K_op_scales))
            epi_K_HL_lam = pyro.sample(
                f"{name}_epi_ln_K_HL_lambda",
                dist.LogNormal(epi_K_HL_lam_locs, epi_K_HL_lam_scales))
            epi_K_HL_off = pyro.sample(
                f"{name}_epi_ln_K_HL_offset",
                dist.Normal(epi_K_HL_locs, epi_K_HL_scales))

        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_lam = pyro.sample(
                    f"{name}_epi_ln_K_E_lambda",
                    dist.LogNormal(epi_K_E_lam_locs, epi_K_E_lam_scales))
                epi_K_E_off = pyro.sample(
                    f"{name}_epi_ln_K_E_offset",
                    dist.Normal(epi_K_E_locs, epi_K_E_scales))

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, mut_scatter)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, mut_scatter)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, mut_scatter)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        ln_K_op = ln_K_op + pair_scatter(epi_K_op_off * tau_epi * _lam_tilde(epi_K_op_lam))
        ln_K_HL = ln_K_HL + pair_scatter(epi_K_HL_off * tau_epi * _lam_tilde(epi_K_HL_lam))
        ln_K_E  = ln_K_E  + pair_scatter(epi_K_E_off  * tau_epi * _lam_tilde(epi_K_E_lam))

    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                       data.titrant_conc, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op,
                      ln_K_HL=ln_K_HL,
                      ln_K_E=ln_K_E,
                      tf_total=tf_total,
                      op_total=op_total,
                      mu=mu,
                      sigma=sigma)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # K_op ≈ 10 nM⁻¹ = 1e10 M⁻¹  → ln_K_op ≈ 23.0  (protein-DNA association)
    p["theta_ln_K_op_wt_loc"]   = 23.0
    p["theta_ln_K_op_wt_scale"] = 2.0
    # K_HL ≈ 1e-4 → ln_K_HL ≈ -9.0  (dimensionless conformational equilibrium)
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    # K_E ≈ 1e-4 nM⁻² = 1e14 M⁻²  → ln_K_E ≈ 33.4  (effector-binding constant)
    p["theta_ln_K_E_wt_loc"]    = 33.4
    p["theta_ln_K_E_wt_scale"]  = 3.0
    # Physical concentrations (M); Sochor, PeerJ 2014
    p["theta_tf_total_M"]  = 6.5e-7   # 650 nM
    p["theta_op_total_M"]  = 2.5e-8   # 25 nM
    # Mutation effect scales
    p["theta_sigma_d_ln_K_op_scale"]  = 1.0
    p["theta_sigma_d_ln_K_HL_scale"]  = 1.0
    p["theta_sigma_d_ln_K_E_scale"]   = 1.0
    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    return p


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    g = {}
    g[f"{name}_ln_K_op_wt"]         = jnp.array(23.0)
    g[f"{name}_ln_K_HL_wt"]         = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]          = jnp.full(T, 33.4)
    g[f"{name}_sigma_d_ln_K_op"]    = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_HL"]    = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_E"]     = jnp.full(T, 0.5)
    g[f"{name}_d_ln_K_op_offset"]   = jnp.zeros(M)
    g[f"{name}_d_ln_K_HL_offset"]   = jnp.zeros(M)
    g[f"{name}_d_ln_K_E_offset"]    = jnp.zeros((T, M))
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
    """
    Predict theta for unmeasured genotypes using additive ln-K assembly.

    Parameters
    ----------
    target_genotypes : list[str]
    titrant_names : list[str]
    manual_titrant_df : pd.DataFrame
        Must have 'titrant_name' and 'titrant_conc' columns.
    mut_labels : list[str]
    pair_labels : list[str]
    param_posteriors : dict-like
    q_to_get : dict  mapping column name -> quantile value
    tf_total : float  — total TF concentration in M
    op_total : float  — total operator concentration in M

    Returns
    -------
    pd.DataFrame
        Columns: 'genotype', 'titrant_name', 'titrant_conc', then one per q_to_get key.
        Rows with unrecognised mutations have NaN quantiles.
    """
    import numpy as np
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )
    from tfscreen.tfmodel.generative.components.theta.struct.lac_dimer.thermo import (
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

    # WT values
    ln_K_op_wt = _load("theta_ln_K_op_wt")   # (S,)
    ln_K_HL_wt = _load("theta_ln_K_HL_wt")   # (S,)
    ln_K_E_wt  = _load("theta_ln_K_E_wt")    # (S, T)

    # Per-mutation deltas (deterministic sites)
    d_op = _load("theta_d_ln_K_op")   # (S, M)
    d_HL = _load("theta_d_ln_K_HL")   # (S, M)
    d_E  = _load("theta_d_ln_K_E")    # (S, T, M)

    # Assembly over N target genotypes
    # scalar Ks: (S, N)
    ln_K_op_geno = ln_K_op_wt[:, None] + np.einsum("sm,nm->sn", d_op, mut_mat)
    ln_K_HL_geno = ln_K_HL_wt[:, None] + np.einsum("sm,nm->sn", d_HL, mut_mat)
    # T-dim K: (S, T, N)
    ln_K_E_geno = ln_K_E_wt[:, :, None] + np.einsum("stm,nm->stn", d_E, mut_mat)

    if len(pair_labels) > 0:
        epi_op = _load("theta_epi_ln_K_op")   # (S, P)
        epi_HL = _load("theta_epi_ln_K_HL")   # (S, P)
        epi_E  = _load("theta_epi_ln_K_E")    # (S, T, P)
        ln_K_op_geno += np.einsum("sp,np->sn", epi_op, pair_mat)
        ln_K_HL_geno += np.einsum("sp,np->sn", epi_HL, pair_mat)
        ln_K_E_geno  += np.einsum("stp,np->stn", epi_E, pair_mat)

    # Index rows from the (S, T/G) genotype grid to (S, N_rows)
    ln_K_op_rows = ln_K_op_geno[:, geno_idx]                    # (S, N_rows)
    ln_K_HL_rows = ln_K_HL_geno[:, geno_idx]
    ln_K_E_rows  = ln_K_E_geno[:, titrant_idx, geno_idx]        # (S, N_rows)

    conc = calc_df["titrant_conc"].values.copy().astype(float)
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_op_rows, ln_K_HL_rows, ln_K_E_rows, conc, tf_total, op_total
    )   # (S, N_rows)

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

    mut_df = pd.DataFrame({
        "mutation": ctx.mut_labels,
        "map_mut": range(num_mut),
    })
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_op", "d_ln_K_HL"],
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


from tfscreen.tfmodel.generative.components.theta.struct.lac_dimer.thermo import (
    build_calc_df,
    compute_theta_samples,
)
