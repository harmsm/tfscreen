"""
Partition-function theta model for a dimeric lac repressor with cooperative
effector binding. Mutation effects are additive in ln(K) space.

Thermodynamic model
-------------------
Four-state partition function for the TF homodimer:

    State      Weight
    H          1                               (apo, high-affinity conformation)
    H·op       K_op · [op_free]                (operator-bound, repressing state)
    L          K_HL                            (low-affinity conformation)
    L·E²       K_HL · K_E · [E_free]²          (effector-bound, n=2 hardcoded)

    Z     = 1 + K_op·[op_free] + K_HL + K_HL·K_E·[E_free]²
    θ     = K_op·[op_free] / Z

Units: all concentrations in nM; K_op in nM⁻¹; K_E in nM⁻²; K_HL dimensionless.

Operator depletion approximation
---------------------------------
[TF_total] ≈ 650 nM, [op_total] ≈ 25 nM
(Sochor, PeerJ 2014, https://doi.org/10.7717/peerj.498),
giving [TF]/[op] ≈ 26.  Because the TF is in large excess we treat
[op_free] ≈ [op_total] = const.

Free effector (Newton solve)
----------------------------
Free-effector concentration satisfies the cubic mass balance:

    a·x³ + a·(2·[TF_total] − [E_total])·x² + Z₀·x − Z₀·[E_total] = 0

where  a  = K_HL·K_E (shape T×G)  and  Z₀ = 1 + K_op·[op_total] + K_HL (shape G).
Eight Newton iterations from x₀ = [E_total] converge for all
biophysically plausible parameter combinations.

Mutation decomposition
----------------------
Each equilibrium constant is decomposed as:

    ln_K_op[g]   = ln_K_op_wt   + (d_ln_K_op   · M)[g]   + (epi_ln_K_op   · P)[g]
    ln_K_HL[g]   = ln_K_HL_wt   + (d_ln_K_HL   · M)[g]   + (epi_ln_K_HL   · P)[g]
    ln_K_E[t, g] = ln_K_E_wt[t] + (d_ln_K_E[t] · M)[g]   + (epi_ln_K_E[t] · P)[g]

where M[m, g] = 1 if mutation m is in genotype g  (shape: num_mutation × G)
and   P[p, g] = 1 if pair   p is in genotype g    (shape: num_pair    × G).

K_op and K_HL are protein properties with no dependence on effector identity
(no T dimension). K_E is per effector species (T dimension).

Epistasis terms are only sampled when data.num_pair > 0.
"""

import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from functools import partial
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix

# Number of Newton iterations for the free-effector cubic solve.
NEWTON_ITERATIONS = 8


# ---------------------------------------------------------------------------
# Pytrees
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the partition-function lac-dimer theta model."""

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

    # Shared regularised horseshoe hyperparameters for pairwise epistasis
    theta_epi_tau_scale: float     # HalfCauchy scale for global τ (shared across K_op, K_HL, K_E)
    theta_epi_slab_scale: float    # typical size of a large epistasis effect
    theta_epi_slab_df: float       # InvGamma shape ν (usually 4)


@dataclass(frozen=True)
class ThetaParam:
    """Assembled partition-function parameters."""

    ln_K_op:  jnp.ndarray   # (G,)    — per-genotype log protein-DNA affinity
    ln_K_HL:  jnp.ndarray   # (G,)    — per-genotype log conformational equilibrium
    ln_K_E:   jnp.ndarray   # (T, G)  — per-genotype log effector binding constant
    tf_total: float          # nM
    op_total: float          # nM
    mu:       jnp.ndarray   # (T, C, 1) — population mean logit-theta (growth concs)
    sigma:    jnp.ndarray   # (T, C, 1) — population std  logit-theta (growth concs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assemble_scalar(wt, d_offsets, sigma_d, M,
                     epi_offsets=None, sigma_epi=None, pair_scatter=None):
    """
    Assemble per-genotype parameter from scalar WT + mutation deltas.

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


def _solve_free_effector(e_total, tf_total, a, Z0):
    """
    Solve the cubic mass-balance equation for free effector concentration.

        a·x³ + a·(2·tf_total − e_total)·x² + Z₀·x − Z₀·e_total = 0

    Uses 8 Newton iterations starting from x₀ = e_total.

    Parameters
    ----------
    e_total : (1, C, 1)  — total effector concentration grid
    tf_total : scalar    — total TF concentration (nM)
    a : (T, 1, G)        — K_HL · K_E
    Z0 : (1, 1, G)       — 1 + K_op·[op_total] + K_HL

    Returns
    -------
    x_free : (T, C, G)   — free effector concentration, clipped to [0, e_total]
    """
    coeff_a = a                                     # (T, 1, G)
    coeff_b = a * (2.0 * tf_total - e_total)        # (T, C, G)
    coeff_c = Z0                                    # (1, 1, G)
    coeff_d = -Z0 * e_total                         # (1, C, G)

    def f(x):
        return coeff_a * x**3 + coeff_b * x**2 + coeff_c * x + coeff_d

    def df(x):
        return 3.0 * coeff_a * x**2 + 2.0 * coeff_b * x + coeff_c

    def step(i, x):
        dfx = df(x)
        return x - f(x) / jnp.where(jnp.abs(dfx) < 1e-30, 1e-30, dfx)

    # Broadcast e_total to (T, C, G) for the initial guess
    x0 = jnp.broadcast_to(e_total, coeff_b.shape)
    x_free = jax.lax.fori_loop(0, NEWTON_ITERATIONS, step, x0)
    return jnp.clip(x_free, 0.0, e_total)


def _compute_theta(ln_K_op, ln_K_HL, ln_K_E, titrant_conc, tf_total, op_total):
    """
    Evaluate the partition-function theta for all (T, C, G) combinations.

    Parameters
    ----------
    ln_K_op : (G,)
    ln_K_HL : (G,)
    ln_K_E  : (T, G)
    titrant_conc : (C,)  — effector concentrations in nM
    tf_total : scalar
    op_total : scalar

    Returns
    -------
    theta : (T, C, G)
    """
    K_op = jnp.exp(ln_K_op)[None, None, :]     # (1, 1, G)
    K_HL = jnp.exp(ln_K_HL)[None, None, :]     # (1, 1, G)
    K_E  = jnp.exp(ln_K_E)[:, None, :]         # (T, 1, G)
    e_total = titrant_conc[None, :, None]       # (1, C, 1)

    Z0 = 1.0 + K_op * op_total + K_HL          # (1, 1, G)
    a  = K_HL * K_E                             # (T, 1, G)

    e_free = _solve_free_effector(e_total, tf_total, a, Z0)  # (T, C, G)

    w_Hop = K_op * op_total                     # (1, 1, G)
    w_LE  = K_HL * K_E * e_free**2              # (T, C, G)
    Z     = 1.0 + w_Hop + K_HL + w_LE          # (T, C, G)

    return w_Hop / Z                            # (T, C, G)


def _population_moments(theta, data):
    """
    Compute per-concentration population moments of logit(theta) over genotypes.

    Parameters
    ----------
    theta : (T, C, G)
    data  : GrowthData

    Returns
    -------
    mu    : (T, C, 1)
    sigma : (T, C, 1)
    """
    eps = 1e-6
    theta_clipped = jnp.clip(theta, eps, 1.0 - eps)
    logit_theta = jax.scipy.special.logit(theta_clipped)           # (T, C, G)
    mu    = jnp.mean(logit_theta, axis=-1, keepdims=True)          # (T, C, 1)
    sigma = jnp.std(logit_theta,  axis=-1, keepdims=True)          # (T, C, 1)
    return mu, sigma


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the partition-function lac-dimer hierarchical model.

    Samples WT equilibrium constants and per-mutation delta parameters, then
    assembles per-genotype constants via matrix multiplication.

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
        Assembled parameters with population moments computed over
        ``data.titrant_conc``.
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
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, M_mat)    # (G,)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, M_mat)    # (G,)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, M_mat)      # (T, G)

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
    """Variational guide for the partition-function lac-dimer model."""

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
    # K_op and K_HL: shape (num_mutation,)
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

    # K_E: shape (T, num_mutation)
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
    else:
        pair_scatter = None
        epi_K_op_off = epi_K_HL_off = epi_K_E_off = None

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    ln_K_op = _assemble_scalar(ln_K_op_wt, d_K_op_off, sigma_d_K_op, M_mat)
    ln_K_HL = _assemble_scalar(ln_K_HL_wt, d_K_HL_off, sigma_d_K_HL, M_mat)
    ln_K_E  = _assemble_titrant(ln_K_E_wt, d_K_E_off, sigma_d_K_E, M_mat)

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
# run_model
# ---------------------------------------------------------------------------

def run_model(theta_param: ThetaParam, data) -> jnp.ndarray:
    """
    Evaluate theta at the concentrations and genotypes present in ``data``.

    Recomputes theta from the stored equilibrium constants so that this
    function works correctly for both growth and binding data (which may
    have different concentration grids).

    Returns
    -------
    theta_calc : array, shape (T, C, num_obs) or (1,1,1,1,T,C,num_obs)
    """
    theta_all = _compute_theta(
        theta_param.ln_K_op,
        theta_param.ln_K_HL,
        theta_param.ln_K_E,
        data.titrant_conc,
        theta_param.tf_total,
        theta_param.op_total,
    )                                              # (T, C, G)

    theta_calc = theta_all[:, :, data.geno_theta_idx]   # (T, C, num_obs)

    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]

    return theta_calc


def get_population_moments(theta_param: ThetaParam, data) -> tuple:
    return theta_param.mu, theta_param.sigma


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    # WT equilibrium constants in ln-K space
    # K_op ≈ 10 nM⁻¹  → ln_K_op ≈ 2.3  (protein-DNA association)
    p["theta_ln_K_op_wt_loc"]   = 2.3
    p["theta_ln_K_op_wt_scale"] = 2.0
    # K_HL ≈ 1e-4 → ln_K_HL ≈ -9.0  (H conformation strongly favored apo)
    p["theta_ln_K_HL_wt_loc"]   = -9.0
    p["theta_ln_K_HL_wt_scale"] = 3.0
    # K_E ≈ 1e-4 nM⁻² → ln_K_E ≈ -8  (effector-binding constant)
    p["theta_ln_K_E_wt_loc"]    = -8.0
    p["theta_ln_K_E_wt_scale"]  = 3.0
    # Physical concentrations (nM); Sochor, PeerJ 2014, https://doi.org/10.7717/peerj.498
    p["theta_tf_total_nM"]  = 650.0
    p["theta_op_total_nM"]  = 25.0
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
