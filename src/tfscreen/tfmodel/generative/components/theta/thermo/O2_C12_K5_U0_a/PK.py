"""
K-assembly via additive mutation effects in ln-K space (MWC dimer).

Mutation effects are additive in ln(K) space, with optional pairwise
epistasis terms:

    ln_K_h_l[g]    = ln_K_h_l_wt    + (d_ln_K_h_l    · M)[g]  + (epi_ln_K_h_l    · P)[g]
    ln_K_h_o[g]    = ln_K_h_o_wt    + (d_ln_K_h_o    · M)[g]  + (epi_ln_K_h_o    · P)[g]
    ln_K_l_o[g]    = ln_K_l_o_wt    + (d_ln_K_l_o    · M)[g]  + (epi_ln_K_l_o    · P)[g]
    ln_K_h_e[t, g] = ln_K_h_e_wt[t] + (d_ln_K_h_e[t] · M)[g]  + (epi_ln_K_h_e[t] · P)[g]
    ln_K_l_e[t, g] = ln_K_l_e_wt[t] + (d_ln_K_l_e[t] · M)[g]  + (epi_ln_K_l_e[t] · P)[g]

where M[m, g] = 1 if mutation m is in genotype g  (shape: num_mutation × G)
and   P[p, g] = 1 if pair   p is in genotype g    (shape: num_pair    × G).

K_h_l, K_h_o, and K_l_o are protein properties with no dependence on effector
identity (no T dimension).  K_h_e and K_l_e carry a T dimension to allow
different effector species to have different WT affinities; the per-mutation
delta is shared across effector types.

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
from typing import Dict, Any, Union

from tfscreen.tfmodel.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix
from tfscreen.tfmodel.generative.components.theta.struct.mwc_dimer.thermo import (  # noqa: F401
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
    """Hyperparameters for the additive ln-K MWC-dimer theta model."""

    # WT equilibrium constants (Normal in ln-K space)
    theta_ln_K_h_l_wt_loc:   float
    theta_ln_K_h_l_wt_scale: float
    theta_ln_K_h_o_wt_loc:   float
    theta_ln_K_h_o_wt_scale: float
    theta_ln_K_l_o_wt_loc:   float
    theta_ln_K_l_o_wt_scale: float
    theta_ln_K_h_e_wt_loc:   float
    theta_ln_K_h_e_wt_scale: float
    theta_ln_K_l_e_wt_loc:   float
    theta_ln_K_l_e_wt_scale: float

    # Physical concentrations (M).
    # theta_tf_total_M is the total TF in MONOMER units (same convention as
    # MWCDimerModel.r_total in src/tfscreen/models/lac_model/mwc_dimer.py).
    # The dimer concentration used in the mass-balance is tf_total_M / 2.
    theta_tf_total_M: float   # ≈ 6.5e-7 M monomer (= 325 nM dimer)
    theta_op_total_M: float   # ≈ 2.5e-8 M (25 nM)

    # Unit-conversion factor applied to data.titrant_conc before it enters
    # the thermodynamic equations.  Set to 1e-3 when concentrations in the
    # data are in mM and K values are in M⁻¹ (recommended convention).
    theta_conc_unit_scale: float

    # HalfNormal scales for mutation delta distributions
    theta_sigma_d_ln_K_h_l_scale: float
    theta_sigma_d_ln_K_h_o_scale: float
    theta_sigma_d_ln_K_l_o_scale: float
    theta_sigma_d_ln_K_h_e_scale: float
    theta_sigma_d_ln_K_l_e_scale: float

    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    theta_epi_tau_scale:  float   # HalfCauchy scale for global τ
    theta_epi_slab_scale: float   # typical size of a large epistasis effect
    theta_epi_slab_df:    float   # InvGamma shape ν (usually 4)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_scalar(wt, d_offsets, sigma_d, mut_scatter,
                     epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype scalar parameter from WT + mutation deltas.

    Parameters
    ----------
    wt          : scalar
    d_offsets   : (M,)
    sigma_d     : scalar
    mut_scatter : callable (M,) -> (G,)
        Scatters per-mutation values to genotype space via COO index arrays.

    Returns
    -------
    (G,)
    """
    result = wt + mut_scatter(d_offsets * sigma_d)     # (G,)
    if epi_offsets is not None:
        result = result + pair_scatter(epi_offsets * sigma_epi)
    return result


def _assemble_titrant(wt, d_offsets, sigma_d, mut_scatter,
                      epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype parameter from T-shaped WT + mutation deltas.

    Parameters
    ----------
    wt          : (T,)
    d_offsets   : (T, num_mutation)
    sigma_d     : (T,)
    mut_scatter : callable (T, M) -> (T, G)
        Scatters per-mutation values to genotype space; handles leading dims.

    Returns
    -------
    (T, G)
    """
    result = wt[:, None] + mut_scatter(d_offsets * sigma_d[:, None])   # (T, G)
    if epi_offsets is not None:
        # epi_offsets and sigma_epi are both (T, P); no extra dim needed.
        result = result + pair_scatter(epi_offsets * sigma_epi)
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the additive ln-K MWC-dimer hierarchical model.

    Samples WT equilibrium constants and per-mutation delta parameters, then
    assembles per-genotype constants via matrix multiplication.  Epistasis uses
    a regularised horseshoe prior (shared τ and slab c², per-pair local λ).

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : GrowthData or BindingData
        Must have ``mut_geno_matrix`` (num_mutation × G) and ``num_mutation``.
        If ``num_pair > 0``, must also have ``pair_nnz_pair_idx``,
        ``pair_nnz_geno_idx``, and ``num_genotype``.
    priors : ModelPriors

    Returns
    -------
    ThetaParam
    """
    T       = data.num_titrant_name
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    tf_total         = priors.theta_tf_total_M
    op_total         = priors.theta_op_total_M
    conc_unit_scale  = priors.theta_conc_unit_scale

    # ------------------------------------------------------------------
    # WT equilibrium constants
    # K_h_l, K_h_o, K_l_o: scalar (no effector-type dimension)
    # ------------------------------------------------------------------
    ln_K_h_l_wt = pyro.sample(
        f"{name}_ln_K_h_l_wt",
        dist.Normal(priors.theta_ln_K_h_l_wt_loc, priors.theta_ln_K_h_l_wt_scale))
    ln_K_h_o_wt = pyro.sample(
        f"{name}_ln_K_h_o_wt",
        dist.Normal(priors.theta_ln_K_h_o_wt_loc, priors.theta_ln_K_h_o_wt_scale))
    ln_K_l_o_wt = pyro.sample(
        f"{name}_ln_K_l_o_wt",
        dist.Normal(priors.theta_ln_K_l_o_wt_loc, priors.theta_ln_K_l_o_wt_scale))

    sigma_d_K_h_l = pyro.sample(
        f"{name}_sigma_d_ln_K_h_l",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_h_l_scale))
    sigma_d_K_h_o = pyro.sample(
        f"{name}_sigma_d_ln_K_h_o",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_h_o_scale))
    sigma_d_K_l_o = pyro.sample(
        f"{name}_sigma_d_ln_K_l_o",
        dist.HalfNormal(priors.theta_sigma_d_ln_K_l_o_scale))

    # K_h_e, K_l_e: T-dimensional (one per effector species)
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_h_e_wt = pyro.sample(
            f"{name}_ln_K_h_e_wt",
            dist.Normal(priors.theta_ln_K_h_e_wt_loc, priors.theta_ln_K_h_e_wt_scale))
        ln_K_l_e_wt = pyro.sample(
            f"{name}_ln_K_l_e_wt",
            dist.Normal(priors.theta_ln_K_l_e_wt_loc, priors.theta_ln_K_l_e_wt_scale))
        sigma_d_K_h_e = pyro.sample(
            f"{name}_sigma_d_ln_K_h_e",
            dist.HalfNormal(priors.theta_sigma_d_ln_K_h_e_scale))
        sigma_d_K_l_e = pyro.sample(
            f"{name}_sigma_d_ln_K_l_e",
            dist.HalfNormal(priors.theta_sigma_d_ln_K_l_e_scale))

    # ------------------------------------------------------------------
    # Per-mutation delta offsets
    # Scalar K_h_l, K_h_o, K_l_o: shape (num_mutation,)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_h_l_off = pyro.sample(f"{name}_d_ln_K_h_l_offset", dist.Normal(0.0, 1.0))
        d_K_h_o_off = pyro.sample(f"{name}_d_ln_K_h_o_offset", dist.Normal(0.0, 1.0))
        d_K_l_o_off = pyro.sample(f"{name}_d_ln_K_l_o_offset", dist.Normal(0.0, 1.0))

    # T-dimensional K_h_e, K_l_e: shape (T, num_mutation)
    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_h_e_off = pyro.sample(f"{name}_d_ln_K_h_e_offset", dist.Normal(0.0, 1.0))
            d_K_l_e_off = pyro.sample(f"{name}_d_ln_K_l_e_offset", dist.Normal(0.0, 1.0))

    # ------------------------------------------------------------------
    # Optional epistasis — regularised horseshoe prior
    # ------------------------------------------------------------------
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

        # Scalar K_h_l, K_h_o, K_l_o: (num_pair,) each
        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_h_l_lam = pyro.sample(f"{name}_epi_ln_K_h_l_lambda", dist.HalfCauchy(1.0))
            epi_K_h_l_off = pyro.sample(f"{name}_epi_ln_K_h_l_offset", dist.Normal(0.0, 1.0))
            epi_K_h_o_lam = pyro.sample(f"{name}_epi_ln_K_h_o_lambda", dist.HalfCauchy(1.0))
            epi_K_h_o_off = pyro.sample(f"{name}_epi_ln_K_h_o_offset", dist.Normal(0.0, 1.0))
            epi_K_l_o_lam = pyro.sample(f"{name}_epi_ln_K_l_o_lambda", dist.HalfCauchy(1.0))
            epi_K_l_o_off = pyro.sample(f"{name}_epi_ln_K_l_o_offset", dist.Normal(0.0, 1.0))

        # T-dimensional K_h_e, K_l_e: (T, num_pair) each
        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_h_e_lam = pyro.sample(f"{name}_epi_ln_K_h_e_lambda", dist.HalfCauchy(1.0))
                epi_K_h_e_off = pyro.sample(f"{name}_epi_ln_K_h_e_offset", dist.Normal(0.0, 1.0))
                epi_K_l_e_lam = pyro.sample(f"{name}_epi_ln_K_l_e_lambda", dist.HalfCauchy(1.0))
                epi_K_l_e_off = pyro.sample(f"{name}_epi_ln_K_l_e_offset", dist.Normal(0.0, 1.0))
    else:
        pair_scatter = None

    # ------------------------------------------------------------------
    # Assembled mutation deltas (registered for posterior extraction)
    # ------------------------------------------------------------------
    d_ln_K_h_l = d_K_h_l_off * sigma_d_K_h_l           # (M,)
    d_ln_K_h_o = d_K_h_o_off * sigma_d_K_h_o           # (M,)
    d_ln_K_l_o = d_K_l_o_off * sigma_d_K_l_o           # (M,)
    d_ln_K_h_e = d_K_h_e_off * sigma_d_K_h_e[:, None]  # (T, M)
    d_ln_K_l_e = d_K_l_e_off * sigma_d_K_l_e[:, None]  # (T, M)

    pyro.deterministic(f"{name}_d_ln_K_h_l", d_ln_K_h_l)
    pyro.deterministic(f"{name}_d_ln_K_h_o", d_ln_K_h_o)
    pyro.deterministic(f"{name}_d_ln_K_l_o", d_ln_K_l_o)
    pyro.deterministic(f"{name}_d_ln_K_h_e", d_ln_K_h_e)
    pyro.deterministic(f"{name}_d_ln_K_l_e", d_ln_K_l_e)

    if has_epi:
        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        epi_ln_K_h_l = epi_K_h_l_off * tau_epi * _lam_tilde(epi_K_h_l_lam)   # (P,)
        epi_ln_K_h_o = epi_K_h_o_off * tau_epi * _lam_tilde(epi_K_h_o_lam)   # (P,)
        epi_ln_K_l_o = epi_K_l_o_off * tau_epi * _lam_tilde(epi_K_l_o_lam)   # (P,)
        epi_ln_K_h_e = epi_K_h_e_off * tau_epi * _lam_tilde(epi_K_h_e_lam)   # (T, P)
        epi_ln_K_l_e = epi_K_l_e_off * tau_epi * _lam_tilde(epi_K_l_e_lam)   # (T, P)

        pyro.deterministic(f"{name}_epi_ln_K_h_l", epi_ln_K_h_l)
        pyro.deterministic(f"{name}_epi_ln_K_h_o", epi_ln_K_h_o)
        pyro.deterministic(f"{name}_epi_ln_K_l_o", epi_ln_K_l_o)
        pyro.deterministic(f"{name}_epi_ln_K_h_e", epi_ln_K_h_e)
        pyro.deterministic(f"{name}_epi_ln_K_l_e", epi_ln_K_l_e)

    # ------------------------------------------------------------------
    # Assemble per-genotype equilibrium constants
    # ------------------------------------------------------------------
    ln_K_h_l = _assemble_scalar(
        ln_K_h_l_wt, d_K_h_l_off, sigma_d_K_h_l, mut_scatter,
        epi_K_h_l_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_l_lam) if has_epi else None,
        pair_scatter)                                   # (G,)

    ln_K_h_o = _assemble_scalar(
        ln_K_h_o_wt, d_K_h_o_off, sigma_d_K_h_o, mut_scatter,
        epi_K_h_o_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_o_lam) if has_epi else None,
        pair_scatter)                                   # (G,)

    ln_K_l_o = _assemble_scalar(
        ln_K_l_o_wt, d_K_l_o_off, sigma_d_K_l_o, mut_scatter,
        epi_K_l_o_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_l_o_lam) if has_epi else None,
        pair_scatter)                                   # (G,)

    ln_K_h_e = _assemble_titrant(
        ln_K_h_e_wt, d_K_h_e_off, sigma_d_K_h_e, mut_scatter,
        epi_K_h_e_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_e_lam) if has_epi else None,
        pair_scatter)                                   # (T, G)

    ln_K_l_e = _assemble_titrant(
        ln_K_l_e_wt, d_K_l_e_off, sigma_d_K_l_e, mut_scatter,
        epi_K_l_e_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_l_e_lam) if has_epi else None,
        pair_scatter)                                   # (T, G)

    pyro.deterministic(f"{name}_ln_K_h_l", ln_K_h_l)
    pyro.deterministic(f"{name}_ln_K_h_o", ln_K_h_o)
    pyro.deterministic(f"{name}_ln_K_l_o", ln_K_l_o)
    pyro.deterministic(f"{name}_ln_K_h_e", ln_K_h_e)
    pyro.deterministic(f"{name}_ln_K_l_e", ln_K_l_e)

    # ------------------------------------------------------------------
    # Population moments over genotypes (for transformation model)
    # ------------------------------------------------------------------
    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                data.titrant_conc * conc_unit_scale,
                                tf_total, op_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the additive ln-K MWC-dimer model."""

    T       = data.num_titrant_name
    num_mut = data.num_mutation
    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)
    has_epi = data.num_pair > 0

    tf_total         = priors.theta_tf_total_M
    op_total         = priors.theta_op_total_M
    conc_unit_scale  = priors.theta_conc_unit_scale

    # ------------------------------------------------------------------
    # Variational parameters for scalar WT K values and sigma_d
    # ------------------------------------------------------------------
    ln_K_h_l_wt_loc   = pyro.param(f"{name}_ln_K_h_l_wt_loc",   jnp.array(priors.theta_ln_K_h_l_wt_loc))
    ln_K_h_l_wt_scale = pyro.param(f"{name}_ln_K_h_l_wt_scale", jnp.array(1.0), constraint=dist.constraints.positive)
    ln_K_h_o_wt_loc   = pyro.param(f"{name}_ln_K_h_o_wt_loc",   jnp.array(priors.theta_ln_K_h_o_wt_loc))
    ln_K_h_o_wt_scale = pyro.param(f"{name}_ln_K_h_o_wt_scale", jnp.array(1.0), constraint=dist.constraints.positive)
    ln_K_l_o_wt_loc   = pyro.param(f"{name}_ln_K_l_o_wt_loc",   jnp.array(priors.theta_ln_K_l_o_wt_loc))
    ln_K_l_o_wt_scale = pyro.param(f"{name}_ln_K_l_o_wt_scale", jnp.array(1.0), constraint=dist.constraints.positive)

    sigma_d_K_h_l_loc   = pyro.param(f"{name}_sigma_d_ln_K_h_l_loc",   jnp.array(-1.0))
    sigma_d_K_h_l_scale = pyro.param(f"{name}_sigma_d_ln_K_h_l_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    sigma_d_K_h_o_loc   = pyro.param(f"{name}_sigma_d_ln_K_h_o_loc",   jnp.array(-1.0))
    sigma_d_K_h_o_scale = pyro.param(f"{name}_sigma_d_ln_K_h_o_scale", jnp.array(0.1), constraint=dist.constraints.positive)
    sigma_d_K_l_o_loc   = pyro.param(f"{name}_sigma_d_ln_K_l_o_loc",   jnp.array(-1.0))
    sigma_d_K_l_o_scale = pyro.param(f"{name}_sigma_d_ln_K_l_o_scale", jnp.array(0.1), constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for T-dimensional WT K values and sigma_d
    # ------------------------------------------------------------------
    ln_K_h_e_wt_locs   = pyro.param(f"{name}_ln_K_h_e_wt_locs",   jnp.full(T, priors.theta_ln_K_h_e_wt_loc))
    ln_K_h_e_wt_scales = pyro.param(f"{name}_ln_K_h_e_wt_scales", jnp.ones(T), constraint=dist.constraints.positive)
    ln_K_l_e_wt_locs   = pyro.param(f"{name}_ln_K_l_e_wt_locs",   jnp.full(T, priors.theta_ln_K_l_e_wt_loc))
    ln_K_l_e_wt_scales = pyro.param(f"{name}_ln_K_l_e_wt_scales", jnp.ones(T), constraint=dist.constraints.positive)

    sigma_d_K_h_e_locs   = pyro.param(f"{name}_sigma_d_ln_K_h_e_locs",   jnp.full(T, -1.0))
    sigma_d_K_h_e_scales = pyro.param(f"{name}_sigma_d_ln_K_h_e_scales", jnp.full(T, 0.1), constraint=dist.constraints.positive)
    sigma_d_K_l_e_locs   = pyro.param(f"{name}_sigma_d_ln_K_l_e_locs",   jnp.full(T, -1.0))
    sigma_d_K_l_e_scales = pyro.param(f"{name}_sigma_d_ln_K_l_e_scales", jnp.full(T, 0.1), constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for mutation delta offsets
    # ------------------------------------------------------------------
    d_K_h_l_locs   = pyro.param(f"{name}_d_ln_K_h_l_offset_locs",   jnp.zeros(num_mut))
    d_K_h_l_scales = pyro.param(f"{name}_d_ln_K_h_l_offset_scales", jnp.ones(num_mut),  constraint=dist.constraints.positive)
    d_K_h_o_locs   = pyro.param(f"{name}_d_ln_K_h_o_offset_locs",   jnp.zeros(num_mut))
    d_K_h_o_scales = pyro.param(f"{name}_d_ln_K_h_o_offset_scales", jnp.ones(num_mut),  constraint=dist.constraints.positive)
    d_K_l_o_locs   = pyro.param(f"{name}_d_ln_K_l_o_offset_locs",   jnp.zeros(num_mut))
    d_K_l_o_scales = pyro.param(f"{name}_d_ln_K_l_o_offset_scales", jnp.ones(num_mut),  constraint=dist.constraints.positive)

    d_K_h_e_locs   = pyro.param(f"{name}_d_ln_K_h_e_offset_locs",   jnp.zeros((T, num_mut)))
    d_K_h_e_scales = pyro.param(f"{name}_d_ln_K_h_e_offset_scales", jnp.ones((T, num_mut)), constraint=dist.constraints.positive)
    d_K_l_e_locs   = pyro.param(f"{name}_d_ln_K_l_e_offset_locs",   jnp.zeros((T, num_mut)))
    d_K_l_e_scales = pyro.param(f"{name}_d_ln_K_l_e_offset_scales", jnp.ones((T, num_mut)), constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Optional epistasis variational parameters
    # ------------------------------------------------------------------
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

        for k in ("K_h_l", "K_h_o", "K_l_o"):
            pyro.param(f"{name}_epi_ln_{k}_lambda_locs",   jnp.zeros(num_pair))
            pyro.param(f"{name}_epi_ln_{k}_lambda_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
            pyro.param(f"{name}_epi_ln_{k}_offset_locs",   jnp.zeros(num_pair))
            pyro.param(f"{name}_epi_ln_{k}_offset_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        for k in ("K_h_e", "K_l_e"):
            pyro.param(f"{name}_epi_ln_{k}_lambda_locs",   jnp.zeros((T, num_pair)))
            pyro.param(f"{name}_epi_ln_{k}_lambda_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
            pyro.param(f"{name}_epi_ln_{k}_offset_locs",   jnp.zeros((T, num_pair)))
            pyro.param(f"{name}_epi_ln_{k}_offset_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
    else:
        pair_scatter = None

    # ------------------------------------------------------------------
    # Sample sites — must match plate structure of define_model exactly
    # ------------------------------------------------------------------
    ln_K_h_l_wt = pyro.sample(f"{name}_ln_K_h_l_wt", dist.Normal(ln_K_h_l_wt_loc, ln_K_h_l_wt_scale))
    ln_K_h_o_wt = pyro.sample(f"{name}_ln_K_h_o_wt", dist.Normal(ln_K_h_o_wt_loc, ln_K_h_o_wt_scale))
    ln_K_l_o_wt = pyro.sample(f"{name}_ln_K_l_o_wt", dist.Normal(ln_K_l_o_wt_loc, ln_K_l_o_wt_scale))
    sigma_d_K_h_l = pyro.sample(f"{name}_sigma_d_ln_K_h_l", dist.LogNormal(sigma_d_K_h_l_loc, sigma_d_K_h_l_scale))
    sigma_d_K_h_o = pyro.sample(f"{name}_sigma_d_ln_K_h_o", dist.LogNormal(sigma_d_K_h_o_loc, sigma_d_K_h_o_scale))
    sigma_d_K_l_o = pyro.sample(f"{name}_sigma_d_ln_K_l_o", dist.LogNormal(sigma_d_K_l_o_loc, sigma_d_K_l_o_scale))

    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_h_e_wt   = pyro.sample(f"{name}_ln_K_h_e_wt",   dist.Normal(ln_K_h_e_wt_locs,   ln_K_h_e_wt_scales))
        ln_K_l_e_wt   = pyro.sample(f"{name}_ln_K_l_e_wt",   dist.Normal(ln_K_l_e_wt_locs,   ln_K_l_e_wt_scales))
        sigma_d_K_h_e = pyro.sample(f"{name}_sigma_d_ln_K_h_e", dist.LogNormal(sigma_d_K_h_e_locs, sigma_d_K_h_e_scales))
        sigma_d_K_l_e = pyro.sample(f"{name}_sigma_d_ln_K_l_e", dist.LogNormal(sigma_d_K_l_e_locs, sigma_d_K_l_e_scales))

    with pyro.plate(f"{name}_mutation_scalar_plate", num_mut, dim=-1):
        d_K_h_l_off = pyro.sample(f"{name}_d_ln_K_h_l_offset", dist.Normal(d_K_h_l_locs, d_K_h_l_scales))
        d_K_h_o_off = pyro.sample(f"{name}_d_ln_K_h_o_offset", dist.Normal(d_K_h_o_locs, d_K_h_o_scales))
        d_K_l_o_off = pyro.sample(f"{name}_d_ln_K_l_o_offset", dist.Normal(d_K_l_o_locs, d_K_l_o_scales))

    with pyro.plate(f"{name}_titrant_mut_outer_plate", T, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            d_K_h_e_off = pyro.sample(f"{name}_d_ln_K_h_e_offset", dist.Normal(d_K_h_e_locs, d_K_h_e_scales))
            d_K_l_e_off = pyro.sample(f"{name}_d_ln_K_l_e_offset", dist.Normal(d_K_l_e_locs, d_K_l_e_scales))

    if has_epi:
        tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_epi_loc, tau_epi_scale))
        c2_epi  = pyro.sample(f"{name}_epi_c2",  dist.LogNormal(c2_epi_loc,  c2_epi_scale))

        def _lam_tilde(lam):
            return jnp.sqrt(c2_epi * lam ** 2 / (c2_epi + tau_epi ** 2 * lam ** 2))

        # Scalar K_h_l epistasis: (num_pair,)
        epi_K_h_l_lam_locs   = pyro.param(f"{name}_epi_ln_K_h_l_lambda_locs",   jnp.zeros(num_pair))
        epi_K_h_l_lam_scales = pyro.param(f"{name}_epi_ln_K_h_l_lambda_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        epi_K_h_l_off_locs   = pyro.param(f"{name}_epi_ln_K_h_l_offset_locs",   jnp.zeros(num_pair))
        epi_K_h_l_off_scales = pyro.param(f"{name}_epi_ln_K_h_l_offset_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_pair_scalar_K_h_l_plate", num_pair, dim=-1):
            epi_K_h_l_lam = pyro.sample(f"{name}_epi_ln_K_h_l_lambda", dist.LogNormal(epi_K_h_l_lam_locs, epi_K_h_l_lam_scales))
            epi_K_h_l_off = pyro.sample(f"{name}_epi_ln_K_h_l_offset", dist.Normal(epi_K_h_l_off_locs, epi_K_h_l_off_scales))

        # Scalar K_h_o epistasis: (num_pair,)
        epi_K_h_o_lam_locs   = pyro.param(f"{name}_epi_ln_K_h_o_lambda_locs",   jnp.zeros(num_pair))
        epi_K_h_o_lam_scales = pyro.param(f"{name}_epi_ln_K_h_o_lambda_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        epi_K_h_o_off_locs   = pyro.param(f"{name}_epi_ln_K_h_o_offset_locs",   jnp.zeros(num_pair))
        epi_K_h_o_off_scales = pyro.param(f"{name}_epi_ln_K_h_o_offset_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_pair_scalar_K_h_o_plate", num_pair, dim=-1):
            epi_K_h_o_lam = pyro.sample(f"{name}_epi_ln_K_h_o_lambda", dist.LogNormal(epi_K_h_o_lam_locs, epi_K_h_o_lam_scales))
            epi_K_h_o_off = pyro.sample(f"{name}_epi_ln_K_h_o_offset", dist.Normal(epi_K_h_o_off_locs, epi_K_h_o_off_scales))

        # Scalar K_l_o epistasis: (num_pair,)
        epi_K_l_o_lam_locs   = pyro.param(f"{name}_epi_ln_K_l_o_lambda_locs",   jnp.zeros(num_pair))
        epi_K_l_o_lam_scales = pyro.param(f"{name}_epi_ln_K_l_o_lambda_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        epi_K_l_o_off_locs   = pyro.param(f"{name}_epi_ln_K_l_o_offset_locs",   jnp.zeros(num_pair))
        epi_K_l_o_off_scales = pyro.param(f"{name}_epi_ln_K_l_o_offset_scales", jnp.ones(num_pair),  constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_pair_scalar_K_l_o_plate", num_pair, dim=-1):
            epi_K_l_o_lam = pyro.sample(f"{name}_epi_ln_K_l_o_lambda", dist.LogNormal(epi_K_l_o_lam_locs, epi_K_l_o_lam_scales))
            epi_K_l_o_off = pyro.sample(f"{name}_epi_ln_K_l_o_offset", dist.Normal(epi_K_l_o_off_locs, epi_K_l_o_off_scales))

        # T-dimensional K_h_e epistasis: (T, num_pair)
        epi_K_h_e_lam_locs   = pyro.param(f"{name}_epi_ln_K_h_e_lambda_locs",   jnp.zeros((T, num_pair)))
        epi_K_h_e_lam_scales = pyro.param(f"{name}_epi_ln_K_h_e_lambda_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
        epi_K_h_e_off_locs   = pyro.param(f"{name}_epi_ln_K_h_e_offset_locs",   jnp.zeros((T, num_pair)))
        epi_K_h_e_off_scales = pyro.param(f"{name}_epi_ln_K_h_e_offset_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_titrant_epi_K_h_e_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_K_h_e_plate", num_pair, dim=-1):
                epi_K_h_e_lam = pyro.sample(f"{name}_epi_ln_K_h_e_lambda", dist.LogNormal(epi_K_h_e_lam_locs, epi_K_h_e_lam_scales))
                epi_K_h_e_off = pyro.sample(f"{name}_epi_ln_K_h_e_offset", dist.Normal(epi_K_h_e_off_locs, epi_K_h_e_off_scales))

        # T-dimensional K_l_e epistasis: (T, num_pair)
        epi_K_l_e_lam_locs   = pyro.param(f"{name}_epi_ln_K_l_e_lambda_locs",   jnp.zeros((T, num_pair)))
        epi_K_l_e_lam_scales = pyro.param(f"{name}_epi_ln_K_l_e_lambda_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
        epi_K_l_e_off_locs   = pyro.param(f"{name}_epi_ln_K_l_e_offset_locs",   jnp.zeros((T, num_pair)))
        epi_K_l_e_off_scales = pyro.param(f"{name}_epi_ln_K_l_e_offset_scales", jnp.ones((T, num_pair)), constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_titrant_epi_K_l_e_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_K_l_e_plate", num_pair, dim=-1):
                epi_K_l_e_lam = pyro.sample(f"{name}_epi_ln_K_l_e_lambda", dist.LogNormal(epi_K_l_e_lam_locs, epi_K_l_e_lam_scales))
                epi_K_l_e_off = pyro.sample(f"{name}_epi_ln_K_l_e_offset", dist.Normal(epi_K_l_e_off_locs, epi_K_l_e_off_scales))

    # ------------------------------------------------------------------
    # Assemble per-genotype equilibrium constants
    # ------------------------------------------------------------------
    ln_K_h_l = _assemble_scalar(
        ln_K_h_l_wt, d_K_h_l_off, sigma_d_K_h_l, mut_scatter,
        epi_K_h_l_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_l_lam) if has_epi else None,
        pair_scatter)

    ln_K_h_o = _assemble_scalar(
        ln_K_h_o_wt, d_K_h_o_off, sigma_d_K_h_o, mut_scatter,
        epi_K_h_o_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_o_lam) if has_epi else None,
        pair_scatter)

    ln_K_l_o = _assemble_scalar(
        ln_K_l_o_wt, d_K_l_o_off, sigma_d_K_l_o, mut_scatter,
        epi_K_l_o_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_l_o_lam) if has_epi else None,
        pair_scatter)

    ln_K_h_e = _assemble_titrant(
        ln_K_h_e_wt, d_K_h_e_off, sigma_d_K_h_e, mut_scatter,
        epi_K_h_e_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_h_e_lam) if has_epi else None,
        pair_scatter)

    ln_K_l_e = _assemble_titrant(
        ln_K_l_e_wt, d_K_l_e_off, sigma_d_K_l_e, mut_scatter,
        epi_K_l_e_off if has_epi else None,
        tau_epi * _lam_tilde(epi_K_l_e_lam) if has_epi else None,
        pair_scatter)

    theta_vals = _compute_theta(ln_K_h_l, ln_K_h_o, ln_K_h_e,
                                ln_K_l_o, ln_K_l_e,
                                data.titrant_conc * conc_unit_scale,
                                tf_total, op_total)
    mu, sigma = _population_moments(theta_vals, data)

    return ThetaParam(ln_K_h_l=ln_K_h_l, ln_K_h_o=ln_K_h_o, ln_K_h_e=ln_K_h_e,
                      ln_K_l_o=ln_K_l_o, ln_K_l_e=ln_K_l_e,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma,
                      conc_unit_scale=conc_unit_scale)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # All values calibrated to Sochor et al. 2014 (PeerJ 2:e498), Table 1.
    # Concentrations must be in Molar; equilibrium constants in M⁻¹.
    #
    # K_h_l = K_RR* = [L]/[H] ≈ 6.3 → ln ≈ 1.84
    # The L (low-affinity) state is DOMINANT without IPTG (~86% of TF).
    # K_h_l > 1 is required for the effector to produce derepression:
    # when K_h_l << 1, adding effector has no effect on theta (flat curves).
    p["theta_ln_K_h_l_wt_loc"]   = 1.84
    p["theta_ln_K_h_l_wt_scale"] = 2.0
    # K_h_o = K_RO ≈ 0.42 nM⁻¹ = 4.2×10⁸ M⁻¹ → ln ≈ 19.9
    p["theta_ln_K_h_o_wt_loc"]   = 19.9
    p["theta_ln_K_h_o_wt_scale"] = 2.0
    # K_l_o = K_R*O ≈ 0 (L-state does not bind operator DNA).
    # Sochor fixes this at 10⁻¹⁰ nM⁻¹ = 0.1 M⁻¹ → ln ≈ -2.3.
    p["theta_ln_K_l_o_wt_loc"]   = -2.3
    p["theta_ln_K_l_o_wt_scale"] = 3.0
    # K_h_e = K_RE ≈ 5.6×10⁻⁵ nM⁻¹ = 5.6×10⁴ M⁻¹ → ln ≈ 10.9
    p["theta_ln_K_h_e_wt_loc"]   = 10.9
    p["theta_ln_K_h_e_wt_scale"] = 2.0
    # K_l_e = K_R*E ≈ 7.6×10⁻⁴ nM⁻¹ = 7.6×10⁵ M⁻¹ → ln ≈ 13.5
    # L-state binds IPTG ~14× more tightly than H-state (K_l_e/K_h_e ≈ 13.6).
    p["theta_ln_K_l_e_wt_loc"]   = 13.5
    p["theta_ln_K_l_e_wt_scale"] = 2.0
    # Physical concentrations — theta_tf_total_M is in MONOMER units.
    # Sochor in vitro: [TF]≈664 nM dimer = ~1328 nM monomer.
    # 6.5e-7 M monomer = 650 nM monomer = 325 nM dimer is a round-number
    # approximation suitable as the prior centre.
    p["theta_tf_total_M"] = 6.5e-7   # monomer
    p["theta_op_total_M"] = 2.5e-8   # operator (M)
    # Unit-conversion factor: multiply data.titrant_conc by this before
    # entering thermodynamic equations.  Default 1e-3 assumes concentrations
    # are in mM and K values are in M⁻¹.
    p["theta_conc_unit_scale"] = 1e-3
    # Mutation effect scales
    p["theta_sigma_d_ln_K_h_l_scale"] = 1.0
    p["theta_sigma_d_ln_K_h_o_scale"] = 1.0
    p["theta_sigma_d_ln_K_l_o_scale"] = 1.0
    p["theta_sigma_d_ln_K_h_e_scale"] = 1.0
    p["theta_sigma_d_ln_K_l_e_scale"] = 1.0
    # Shared regularised horseshoe hyperparameters
    p["theta_epi_tau_scale"]   = 0.1
    p["theta_epi_slab_scale"]  = 2.0
    p["theta_epi_slab_df"]     = 4.0
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    g = {}
    g[f"{name}_ln_K_h_l_wt"]          = jnp.array(1.84)
    g[f"{name}_ln_K_h_o_wt"]          = jnp.array(19.9)
    g[f"{name}_ln_K_l_o_wt"]          = jnp.array(-2.3)
    g[f"{name}_ln_K_h_e_wt"]          = jnp.full(T, 10.9)
    g[f"{name}_ln_K_l_e_wt"]          = jnp.full(T, 13.5)
    g[f"{name}_sigma_d_ln_K_h_l"]     = jnp.array(0.3)
    g[f"{name}_sigma_d_ln_K_h_o"]     = jnp.array(0.3)
    g[f"{name}_sigma_d_ln_K_l_o"]     = jnp.array(0.3)
    g[f"{name}_sigma_d_ln_K_h_e"]     = jnp.full(T, 0.3)
    g[f"{name}_sigma_d_ln_K_l_e"]     = jnp.full(T, 0.3)
    g[f"{name}_d_ln_K_h_l_offset"]    = jnp.zeros(M)
    g[f"{name}_d_ln_K_h_o_offset"]    = jnp.zeros(M)
    g[f"{name}_d_ln_K_l_o_offset"]    = jnp.zeros(M)
    g[f"{name}_d_ln_K_h_e_offset"]    = jnp.zeros((T, M))
    g[f"{name}_d_ln_K_l_e_offset"]    = jnp.zeros((T, M))
    if data.num_pair > 0:
        P = data.num_pair
        g[f"{name}_epi_tau"]               = jnp.array(0.05)
        g[f"{name}_epi_c2"]                = jnp.array(4.0)
        g[f"{name}_epi_ln_K_h_l_lambda"]  = jnp.ones(P) * 0.5
        g[f"{name}_epi_ln_K_h_l_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_h_o_lambda"]  = jnp.ones(P) * 0.5
        g[f"{name}_epi_ln_K_h_o_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_l_o_lambda"]  = jnp.ones(P) * 0.5
        g[f"{name}_epi_ln_K_l_o_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_h_e_lambda"]  = jnp.ones((T, P)) * 0.5
        g[f"{name}_epi_ln_K_h_e_offset"]  = jnp.zeros((T, P))
        g[f"{name}_epi_ln_K_l_e_lambda"]  = jnp.ones((T, P)) * 0.5
        g[f"{name}_epi_ln_K_l_e_offset"]  = jnp.zeros((T, P))
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
    conc_unit_scale=1.0,
):
    """
    Predict theta for unmeasured genotypes using additive ln-K MWC-dimer assembly.

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
    op_total : float
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
    from tfscreen.tfmodel.generative.components.theta.struct.mwc_dimer.thermo import (
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
    ln_K_h_l_wt = _load("theta_ln_K_h_l_wt")   # (S,)
    ln_K_h_o_wt = _load("theta_ln_K_h_o_wt")   # (S,)
    ln_K_l_o_wt = _load("theta_ln_K_l_o_wt")   # (S,)
    ln_K_h_e_wt = _load("theta_ln_K_h_e_wt")   # (S, T)
    ln_K_l_e_wt = _load("theta_ln_K_l_e_wt")   # (S, T)

    # Per-mutation deltas
    d_h_l = _load("theta_d_ln_K_h_l")   # (S, M)
    d_h_o = _load("theta_d_ln_K_h_o")   # (S, M)
    d_l_o = _load("theta_d_ln_K_l_o")   # (S, M)
    d_h_e = _load("theta_d_ln_K_h_e")   # (S, T, M)
    d_l_e = _load("theta_d_ln_K_l_e")   # (S, T, M)

    # Assembly over N target genotypes
    ln_K_h_l_geno = ln_K_h_l_wt[:, None] + np.einsum("sm,nm->sn", d_h_l, mut_mat)
    ln_K_h_o_geno = ln_K_h_o_wt[:, None] + np.einsum("sm,nm->sn", d_h_o, mut_mat)
    ln_K_l_o_geno = ln_K_l_o_wt[:, None] + np.einsum("sm,nm->sn", d_l_o, mut_mat)
    ln_K_h_e_geno = ln_K_h_e_wt[:, :, None] + np.einsum("stm,nm->stn", d_h_e, mut_mat)
    ln_K_l_e_geno = ln_K_l_e_wt[:, :, None] + np.einsum("stm,nm->stn", d_l_e, mut_mat)

    if len(pair_labels) > 0:
        epi_h_l = _load("theta_epi_ln_K_h_l")   # (S, P)
        epi_h_o = _load("theta_epi_ln_K_h_o")   # (S, P)
        epi_l_o = _load("theta_epi_ln_K_l_o")   # (S, P)
        epi_h_e = _load("theta_epi_ln_K_h_e")   # (S, T, P)
        epi_l_e = _load("theta_epi_ln_K_l_e")   # (S, T, P)
        ln_K_h_l_geno += np.einsum("sp,np->sn", epi_h_l, pair_mat)
        ln_K_h_o_geno += np.einsum("sp,np->sn", epi_h_o, pair_mat)
        ln_K_l_o_geno += np.einsum("sp,np->sn", epi_l_o, pair_mat)
        ln_K_h_e_geno += np.einsum("stp,np->stn", epi_h_e, pair_mat)
        ln_K_l_e_geno += np.einsum("stp,np->stn", epi_l_e, pair_mat)

    # Index rows from genotype grid to (S, N_rows)
    ln_K_h_l_rows = ln_K_h_l_geno[:, geno_idx]
    ln_K_h_o_rows = ln_K_h_o_geno[:, geno_idx]
    ln_K_l_o_rows = ln_K_l_o_geno[:, geno_idx]
    ln_K_h_e_rows = ln_K_h_e_geno[:, titrant_idx, geno_idx]
    ln_K_l_e_rows = ln_K_l_e_geno[:, titrant_idx, geno_idx]

    conc = calc_df["titrant_conc"].values.copy().astype(float) * conc_unit_scale
    conc[conc == 0] = _ZERO_CONC_VALUE

    theta_samples = _solve_theta_np(
        ln_K_h_l_rows, ln_K_h_o_rows, ln_K_h_e_rows,
        ln_K_l_o_rows, ln_K_l_e_rows,
        conc, tf_total, op_total,
    )

    theta_samples[:, ~is_valid[geno_idx]] = np.nan

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df


def get_extract_specs(ctx):
    geno_dim      = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype  = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    titrant_dim   = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    titrant_names = list(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    num_mut       = len(ctx.mut_labels)

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

    # Per-mutation scalar Δln_K values
    mut_df = pd.DataFrame({"mutation": ctx.mut_labels, "map_mut": range(num_mut)})
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_h_l", "d_ln_K_h_o", "d_ln_K_l_o"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    # Per-mutation T-dimensional Δln_K values (K_h_e, K_l_e)
    theta_d_KE_rows = [
        {"titrant_name": t, "mutation": m,
         "map_theta_d_KE": ti * num_mut + mi}
        for ti, t in enumerate(titrant_names)
        for mi, m in enumerate(ctx.mut_labels)
    ]
    specs.append(dict(
        input_df=pd.DataFrame(theta_d_KE_rows),
        params_to_get=["d_ln_K_h_e", "d_ln_K_l_e"],
        map_column="map_theta_d_KE",
        get_columns=["titrant_name", "mutation"],
        in_run_prefix="theta_",
    ))

    if ctx.pair_labels:
        num_pair = len(ctx.pair_labels)
        pair_df = pd.DataFrame({"pair": ctx.pair_labels, "map_pair": range(num_pair)})
        specs.append(dict(
            input_df=pair_df,
            params_to_get=["epi_ln_K_h_l", "epi_ln_K_h_o", "epi_ln_K_l_o"],
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
            params_to_get=["epi_ln_K_h_e", "epi_ln_K_l_e"],
            map_column="map_theta_epi_KE",
            get_columns=["titrant_name", "pair"],
            in_run_prefix="theta_",
        ))

    return specs

from tfscreen.tfmodel.generative.components.theta.struct.mwc_dimer.thermo import (  # noqa: F401  # noqa: F401
    build_calc_df,
    compute_theta_samples,
)
