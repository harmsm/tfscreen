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

Epistasis terms are only sampled when data.num_pair > 0.

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

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData, BindingData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix
from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import (
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

    # Total TF and operator concentrations (nM); treated as fixed physical constants.
    # Default values from Sochor, PeerJ 2014, https://doi.org/10.7717/peerj.498
    theta_tf_total_nM: float   # ≈ 650 nM
    theta_op_total_nM: float   # ≈ 25 nM

    # HalfNormal scales for mutation delta distributions
    theta_sigma_d_ln_K_op_scale: float
    theta_sigma_d_ln_K_HL_scale: float
    theta_sigma_d_ln_K_E_scale: float

    # HalfNormal scales for pairwise epistasis distributions
    theta_sigma_epi_ln_K_op_scale: float
    theta_sigma_epi_ln_K_HL_scale: float
    theta_sigma_epi_ln_K_E_scale: float


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_scalar(wt, d_offsets, sigma_d, M,
                     epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype scalar parameter from WT + mutation deltas.

    Parameters
    ----------
    wt : scalar
    d_offsets : (num_mutation,)
    sigma_d : scalar
    M : (num_mutation, G)
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
    result = wt + d @ M             # (G,)
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi      # (P,)
        result = result + pair_scatter(epi)  # (G,)
    return result


def _assemble_titrant(wt, d_offsets, sigma_d, M,
                      epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype parameter from T-shaped WT + mutation deltas.

    Parameters
    ----------
    wt : (T,)
    d_offsets : (T, num_mutation)
    sigma_d : (T,)
    M : (num_mutation, G)
    epi_offsets : (T, num_pair) or None
    sigma_epi : (T,) or None
    pair_scatter : callable or None
        ``pair_scatter(epi) -> (T, G)``.  Scatters pair epistasis values to
        genotype-space via COO scatter-add.

    Returns
    -------
    (T, G)
    """
    d = d_offsets * sigma_d[:, None]        # (T, M)
    result = wt[:, None] + d @ M            # (T, G)
    if epi_offsets is not None:
        epi = epi_offsets * sigma_epi[:, None]   # (T, P)
        result = result + pair_scatter(epi)       # (T, G)
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the additive ln-K lac-dimer hierarchical model.

    Samples WT equilibrium constants and per-mutation delta parameters, then
    assembles per-genotype constants via matrix multiplication.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : GrowthData or BindingData
        Must have ``mut_geno_matrix`` (num_mutation × G) and ``num_mutation``.
        If ``num_pair > 0``, must also have ``pair_geno_matrix`` (num_pair × G).
    priors : ModelPriors

    Returns
    -------
    ThetaParam
    """
    T = data.num_titrant_name
    M_mat = jnp.array(data.mut_geno_matrix)    # (M, G)
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    tf_total = priors.theta_tf_total_nM
    op_total = priors.theta_op_total_nM

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
    # Optional epistasis
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                              pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                              pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                              num_genotype=data.num_genotype)
        num_pair = data.num_pair

        sigma_epi_K_op = pyro.sample(
            f"{name}_sigma_epi_ln_K_op",
            dist.HalfNormal(priors.theta_sigma_epi_ln_K_op_scale))
        sigma_epi_K_HL = pyro.sample(
            f"{name}_sigma_epi_ln_K_HL",
            dist.HalfNormal(priors.theta_sigma_epi_ln_K_HL_scale))

        with pyro.plate(f"{name}_wt_epi_plate", T, dim=-1):
            sigma_epi_K_E = pyro.sample(
                f"{name}_sigma_epi_ln_K_E",
                dist.HalfNormal(priors.theta_sigma_epi_ln_K_E_scale))

        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_off = pyro.sample(
                f"{name}_epi_ln_K_op_offset", dist.Normal(0.0, 1.0))
            epi_K_HL_off = pyro.sample(
                f"{name}_epi_ln_K_HL_offset", dist.Normal(0.0, 1.0))

        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_off = pyro.sample(
                    f"{name}_epi_ln_K_E_offset", dist.Normal(0.0, 1.0))
    else:
        pair_scatter = None
        sigma_epi_K_op = sigma_epi_K_HL = sigma_epi_K_E = None
        epi_K_op_off = epi_K_HL_off = epi_K_E_off = None

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
        epi_ln_K_op = epi_K_op_off * sigma_epi_K_op            # (P,)
        epi_ln_K_HL = epi_K_HL_off * sigma_epi_K_HL            # (P,)
        epi_ln_K_E  = epi_K_E_off * sigma_epi_K_E[:, None]     # (T, P)

        pyro.deterministic(f"{name}_epi_ln_K_op", epi_ln_K_op)
        pyro.deterministic(f"{name}_epi_ln_K_HL", epi_ln_K_HL)
        pyro.deterministic(f"{name}_epi_ln_K_E",  epi_ln_K_E)

    # ------------------------------------------------------------------
    # Assemble per-genotype equilibrium constants
    # ------------------------------------------------------------------
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, M_mat,
                               epi_K_op_off, sigma_epi_K_op, pair_scatter)    # (G,)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, M_mat,
                               epi_K_HL_off, sigma_epi_K_HL, pair_scatter)    # (G,)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, M_mat,
                                epi_K_E_off, sigma_epi_K_E, pair_scatter)     # (T, G)

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
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the additive ln-K lac-dimer model."""

    T = data.num_titrant_name
    num_mut = data.num_mutation
    M_mat = jnp.array(data.mut_geno_matrix)
    has_epi = data.num_pair > 0

    tf_total = priors.theta_tf_total_nM
    op_total = priors.theta_op_total_nM

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
    # Optional epistasis variational parameters
    # ------------------------------------------------------------------
    if has_epi:
        pair_scatter = partial(apply_pair_matrix,
                               pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
                               pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
                               num_genotype=data.num_genotype)
        num_pair = data.num_pair

        sigma_epi_K_op_loc = pyro.param(
            f"{name}_sigma_epi_ln_K_op_loc", jnp.array(-1.0))
        sigma_epi_K_op_scale = pyro.param(
            f"{name}_sigma_epi_ln_K_op_scale", jnp.array(0.1),
            constraint=dist.constraints.positive)

        sigma_epi_K_HL_loc = pyro.param(
            f"{name}_sigma_epi_ln_K_HL_loc", jnp.array(-1.0))
        sigma_epi_K_HL_scale = pyro.param(
            f"{name}_sigma_epi_ln_K_HL_scale", jnp.array(0.1),
            constraint=dist.constraints.positive)

        sigma_epi_K_E_locs = pyro.param(
            f"{name}_sigma_epi_ln_K_E_locs", jnp.full(T, -1.0))
        sigma_epi_K_E_scales = pyro.param(
            f"{name}_sigma_epi_ln_K_E_scales", jnp.full(T, 0.1),
            constraint=dist.constraints.positive)

        epi_K_op_locs = pyro.param(
            f"{name}_epi_ln_K_op_offset_locs", jnp.zeros(num_pair))
        epi_K_op_scales = pyro.param(
            f"{name}_epi_ln_K_op_offset_scales", jnp.ones(num_pair),
            constraint=dist.constraints.positive)

        epi_K_HL_locs = pyro.param(
            f"{name}_epi_ln_K_HL_offset_locs", jnp.zeros(num_pair))
        epi_K_HL_scales = pyro.param(
            f"{name}_epi_ln_K_HL_offset_scales", jnp.ones(num_pair),
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
        sigma_epi_K_op = pyro.sample(
            f"{name}_sigma_epi_ln_K_op",
            dist.LogNormal(sigma_epi_K_op_loc, sigma_epi_K_op_scale))
        sigma_epi_K_HL = pyro.sample(
            f"{name}_sigma_epi_ln_K_HL",
            dist.LogNormal(sigma_epi_K_HL_loc, sigma_epi_K_HL_scale))

        with pyro.plate(f"{name}_wt_epi_plate", T, dim=-1):
            sigma_epi_K_E = pyro.sample(
                f"{name}_sigma_epi_ln_K_E",
                dist.LogNormal(sigma_epi_K_E_locs, sigma_epi_K_E_scales))

        with pyro.plate(f"{name}_pair_scalar_plate", num_pair, dim=-1):
            epi_K_op_off = pyro.sample(
                f"{name}_epi_ln_K_op_offset",
                dist.Normal(epi_K_op_locs, epi_K_op_scales))
            epi_K_HL_off = pyro.sample(
                f"{name}_epi_ln_K_HL_offset",
                dist.Normal(epi_K_HL_locs, epi_K_HL_scales))

        with pyro.plate(f"{name}_titrant_epi_outer_plate", T, dim=-2):
            with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
                epi_K_E_off = pyro.sample(
                    f"{name}_epi_ln_K_E_offset",
                    dist.Normal(epi_K_E_locs, epi_K_E_scales))
    else:
        sigma_epi_K_op = sigma_epi_K_HL = sigma_epi_K_E = None
        epi_K_op_off = epi_K_HL_off = epi_K_E_off = None

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, M_mat,
                               epi_K_op_off, sigma_epi_K_op, pair_scatter)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, M_mat,
                               epi_K_HL_off, sigma_epi_K_HL, pair_scatter)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, M_mat,
                                epi_K_E_off, sigma_epi_K_E, pair_scatter)

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
    # K_op ≈ 10 nM⁻¹  → ln_K_op ≈ 2.3  (protein-DNA association)
    p["theta_ln_K_op_wt_loc"]   = 2.3
    p["theta_ln_K_op_wt_scale"] = 2.0
    # K_HL ≈ 1e-4 → ln_K_HL ≈ -9.0  (H conformation strongly favored apo)
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    # K_E ≈ 1e-4 nM⁻² → ln_K_E ≈ -8  (effector-binding constant)
    p["theta_ln_K_E_wt_loc"]    = -8.0
    p["theta_ln_K_E_wt_scale"]  = 3.0
    # Physical concentrations (nM); Sochor, PeerJ 2014
    p["theta_tf_total_nM"]  = 650.0
    p["theta_op_total_nM"]  = 25.0
    # Mutation effect scales
    p["theta_sigma_d_ln_K_op_scale"]  = 1.0
    p["theta_sigma_d_ln_K_HL_scale"]  = 1.0
    p["theta_sigma_d_ln_K_E_scale"]   = 1.0
    # Epistasis effect scales
    p["theta_sigma_epi_ln_K_op_scale"] = 0.5
    p["theta_sigma_epi_ln_K_HL_scale"] = 0.5
    p["theta_sigma_epi_ln_K_E_scale"]  = 0.5
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    M = data.num_mutation
    g = {}
    g[f"{name}_ln_K_op_wt"]         = jnp.array(2.3)
    g[f"{name}_ln_K_HL_wt"]         = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]          = jnp.full(T, -8.0)
    g[f"{name}_sigma_d_ln_K_op"]    = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_HL"]    = jnp.array(0.5)
    g[f"{name}_sigma_d_ln_K_E"]     = jnp.full(T, 0.5)
    g[f"{name}_d_ln_K_op_offset"]   = jnp.zeros(M)
    g[f"{name}_d_ln_K_HL_offset"]   = jnp.zeros(M)
    g[f"{name}_d_ln_K_E_offset"]    = jnp.zeros((T, M))
    if data.num_pair > 0:
        P = data.num_pair
        g[f"{name}_sigma_epi_ln_K_op"]   = jnp.array(0.3)
        g[f"{name}_sigma_epi_ln_K_HL"]   = jnp.array(0.3)
        g[f"{name}_sigma_epi_ln_K_E"]    = jnp.full(T, 0.3)
        g[f"{name}_epi_ln_K_op_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_HL_offset"]  = jnp.zeros(P)
        g[f"{name}_epi_ln_K_E_offset"]   = jnp.zeros((T, P))
    return g


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


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
