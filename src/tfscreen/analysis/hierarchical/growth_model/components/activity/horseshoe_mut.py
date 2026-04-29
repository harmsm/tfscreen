"""
Mutation-decomposed activity model with regularised horseshoe priors.

log(activity[g]) = (d_log_activity @ M)[:, g]
                 + (epi_log_activity @ P)[:, g]   (if num_pair > 0)

Both the per-mutation deltas and pairwise epistasis terms use a
regularised horseshoe prior (Piironen & Vehtari 2017):

    τ  ~ HalfCauchy(τ₀)
    c² ~ InvGamma(ν/2,  ν·s²/2)
    λᵢ ~ HalfCauchy(1)
    λ̃ᵢ = λᵢ · √(c² / (c² + τ²λᵢ²))       (regularised local scale)
    βᵢ = offsetᵢ · τ · λ̃ᵢ,   offsetᵢ ~ Normal(0, 1)   (non-centred)

The Normal(0, 1) non-centred parameterisation ensures βᵢ is symmetric
around 0: effects can be positive or negative with equal probability,
and the horseshoe concentrates near 0 while allowing occasional large
escapes.

Individual mutation deltas and epistasis terms use independent τ, c²,
and λ hierarchies controlled by separate hyperparameters.

Wild-type has no mutations so log_activity[wt] = 0 → activity[wt] = 1.0.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix


@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the regularised-horseshoe mutation-decomposed activity model."""

    # Regularised horseshoe for individual mutation deltas
    activity_d_tau_scale: float       # HalfCauchy scale for global τ_d
    activity_d_slab_scale: float      # typical size of a large individual effect (s_d)
    activity_d_slab_df: float         # InvGamma shape ν_d (usually 4)

    # Regularised horseshoe for pairwise epistasis
    activity_epi_tau_scale: float     # HalfCauchy scale for global τ_epi
    activity_epi_slab_scale: float    # typical size of a large epistasis effect (s_epi)
    activity_epi_slab_df: float       # InvGamma shape ν_epi (usually 4)


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Define the regularised-horseshoe mutation-decomposed activity model.

    Parameters
    ----------
    name : str
    data : GrowthData
        Must have ``mut_geno_matrix`` (num_mutation x G) and ``num_mutation``.
        If ``num_pair > 0``, must also have ``pair_geno_matrix``.
    priors : ModelPriors

    Returns
    -------
    jnp.ndarray
        Activity values broadcast to shape ``(1, 1, 1, 1, 1, 1, num_genotype)``.
    """
    M_mat = jnp.array(data.mut_geno_matrix)   # [num_mutation, G]
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    # ------------------------------------------------------------------
    # Individual mutation deltas — regularised horseshoe
    # ------------------------------------------------------------------
    tau_d = pyro.sample(
        f"{name}_d_tau",
        dist.HalfCauchy(priors.activity_d_tau_scale))
    c2_d = pyro.sample(
        f"{name}_d_c2",
        dist.InverseGamma(priors.activity_d_slab_df / 2.0,
                          priors.activity_d_slab_df * priors.activity_d_slab_scale ** 2 / 2.0))

    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        lambda_d = pyro.sample(f"{name}_d_lambda", dist.HalfCauchy(1.0))
        d_offset = pyro.sample(f"{name}_d_offset", dist.Normal(0.0, 1.0))

    # Regularised local scale, then non-centred delta (symmetric about 0)
    lambda_d_tilde = jnp.sqrt(c2_d * lambda_d ** 2 / (c2_d + tau_d ** 2 * lambda_d ** 2))
    d_log_activity = d_offset * tau_d * lambda_d_tilde   # [num_mutation]
    pyro.deterministic(f"{name}_d_log_activity", d_log_activity)

    log_activity = d_log_activity @ M_mat   # [G]

    # ------------------------------------------------------------------
    # Optional pairwise epistasis — regularised horseshoe
    # ------------------------------------------------------------------
    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        tau_epi = pyro.sample(
            f"{name}_epi_tau",
            dist.HalfCauchy(priors.activity_epi_tau_scale))
        c2_epi = pyro.sample(
            f"{name}_epi_c2",
            dist.InverseGamma(priors.activity_epi_slab_df / 2.0,
                              priors.activity_epi_slab_df * priors.activity_epi_slab_scale ** 2 / 2.0))

        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            lambda_epi = pyro.sample(f"{name}_epi_lambda", dist.HalfCauchy(1.0))
            epi_offset = pyro.sample(f"{name}_epi_offset", dist.Normal(0.0, 1.0))

        lambda_epi_tilde = jnp.sqrt(
            c2_epi * lambda_epi ** 2 / (c2_epi + tau_epi ** 2 * lambda_epi ** 2))
        epi_log_activity = epi_offset * tau_epi * lambda_epi_tilde   # [num_pair]
        pyro.deterministic(f"{name}_epi_log_activity", epi_log_activity)
        log_activity = log_activity + apply_pair_matrix(
            epi_log_activity, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)

    activity = jnp.clip(jnp.exp(log_activity), max=1e30)
    pyro.deterministic(name, activity)

    return activity[None, None, None, None, None, None, :]


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """Variational guide for the regularised-horseshoe mutation-decomposed activity model."""

    num_mut = data.num_mutation
    M_mat = jnp.array(data.mut_geno_matrix)
    has_epi = data.num_pair > 0

    # ------------------------------------------------------------------
    # Individual mutation deltas — variational parameters
    # ------------------------------------------------------------------
    # τ_d: LogNormal guide for HalfCauchy prior
    tau_d_loc = pyro.param(f"{name}_d_tau_loc", jnp.array(-1.0))
    tau_d_scale = pyro.param(f"{name}_d_tau_scale", jnp.array(0.1),
                             constraint=dist.constraints.positive)
    tau_d = pyro.sample(f"{name}_d_tau", dist.LogNormal(tau_d_loc, tau_d_scale))

    # c²_d: LogNormal guide for InvGamma prior
    c2_d_loc = pyro.param(f"{name}_d_c2_loc", jnp.array(1.4))   # ≈ log(slab_scale²=4)
    c2_d_scale = pyro.param(f"{name}_d_c2_scale", jnp.array(0.5),
                            constraint=dist.constraints.positive)
    c2_d = pyro.sample(f"{name}_d_c2", dist.LogNormal(c2_d_loc, c2_d_scale))

    # Per-mutation local scales (LogNormal) and offsets (Normal)
    lambda_d_locs = pyro.param(f"{name}_d_lambda_locs", jnp.zeros(num_mut))
    lambda_d_scales = pyro.param(f"{name}_d_lambda_scales", jnp.ones(num_mut),
                                 constraint=dist.constraints.positive)
    d_offset_locs = pyro.param(f"{name}_d_offset_locs", jnp.zeros(num_mut))
    d_offset_scales = pyro.param(f"{name}_d_offset_scales", jnp.ones(num_mut),
                                 constraint=dist.constraints.positive)

    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        lambda_d = pyro.sample(f"{name}_d_lambda",
                               dist.LogNormal(lambda_d_locs, lambda_d_scales))
        d_offset = pyro.sample(f"{name}_d_offset",
                               dist.Normal(d_offset_locs, d_offset_scales))

    lambda_d_tilde = jnp.sqrt(c2_d * lambda_d ** 2 / (c2_d + tau_d ** 2 * lambda_d ** 2))
    d_log_activity = d_offset * tau_d * lambda_d_tilde
    log_activity = d_log_activity @ M_mat

    # ------------------------------------------------------------------
    # Optional epistasis — variational parameters
    # ------------------------------------------------------------------
    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        tau_epi_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-1.0))
        tau_epi_scale = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1),
                                   constraint=dist.constraints.positive)
        tau_epi = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_epi_loc, tau_epi_scale))

        c2_epi_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
        c2_epi_scale = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                                  constraint=dist.constraints.positive)
        c2_epi = pyro.sample(f"{name}_epi_c2", dist.LogNormal(c2_epi_loc, c2_epi_scale))

        lambda_epi_locs = pyro.param(f"{name}_epi_lambda_locs", jnp.zeros(num_pair))
        lambda_epi_scales = pyro.param(f"{name}_epi_lambda_scales", jnp.ones(num_pair),
                                       constraint=dist.constraints.positive)
        epi_offset_locs = pyro.param(f"{name}_epi_offset_locs", jnp.zeros(num_pair))
        epi_offset_scales = pyro.param(f"{name}_epi_offset_scales", jnp.ones(num_pair),
                                       constraint=dist.constraints.positive)

        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            lambda_epi = pyro.sample(f"{name}_epi_lambda",
                                     dist.LogNormal(lambda_epi_locs, lambda_epi_scales))
            epi_offset = pyro.sample(f"{name}_epi_offset",
                                     dist.Normal(epi_offset_locs, epi_offset_scales))

        lambda_epi_tilde = jnp.sqrt(
            c2_epi * lambda_epi ** 2 / (c2_epi + tau_epi ** 2 * lambda_epi ** 2))
        epi_log_activity = epi_offset * tau_epi * lambda_epi_tilde
        log_activity = log_activity + apply_pair_matrix(
            epi_log_activity, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)

    activity = jnp.clip(jnp.exp(log_activity), max=1e30)

    return activity[None, None, None, None, None, None, :]


def get_hyperparameters() -> Dict[str, Any]:
    return {
        "activity_d_tau_scale": 0.3,
        "activity_d_slab_scale": 2.0,
        "activity_d_slab_df": 4.0,
        "activity_epi_tau_scale": 0.1,
        "activity_epi_slab_scale": 2.0,
        "activity_epi_slab_df": 4.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    guesses = {}
    guesses[f"{name}_d_tau"] = 0.1
    guesses[f"{name}_d_c2"] = 4.0
    guesses[f"{name}_d_lambda"] = jnp.ones(data.num_mutation) * 0.5
    guesses[f"{name}_d_offset"] = jnp.zeros(data.num_mutation)
    if data.num_pair > 0:
        guesses[f"{name}_epi_tau"] = 0.05
        guesses[f"{name}_epi_c2"] = 4.0
        guesses[f"{name}_epi_lambda"] = jnp.ones(data.num_pair) * 0.5
        guesses[f"{name}_epi_offset"] = jnp.zeros(data.num_pair)
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())
