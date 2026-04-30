"""
Mutation-decomposed activity model.

log(activity[g]) = (d_log_activity @ M)[:, g]
                 + (epi_log_activity @ P)[:, g]   (if num_pair > 0)

where M[m, g] = 1 if mutation m is in genotype g, P[p, g] = 1 if pair p is
present in genotype g.

Wild-type has no mutations (M[:, wt] = 0), so log_activity[wt] = 0 and
activity[wt] = 1.0, consistent with the other activity models.

Each d_log_activity[m] ~ Normal(0, sigma_d) where sigma_d ~ HalfNormal(scale).
Epistasis terms ~ Normal(0, sigma_epi) similarly.
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
    """Hyperparameters for the mutation-decomposed activity model."""
    activity_sigma_d_scale: float       # HalfNormal scale for mutation delta spread
    activity_sigma_epi_scale: float     # HalfNormal scale for pairwise epistasis spread


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Define the mutation-decomposed activity model.

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
    # Mutation delta scale and per-mutation offsets
    # ------------------------------------------------------------------
    sigma_d = pyro.sample(
        f"{name}_sigma_d",
        dist.HalfNormal(priors.activity_sigma_d_scale))

    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        d_offset = pyro.sample(f"{name}_d_offset", dist.Normal(0.0, 1.0))

    # Non-centered: d_log_activity[m] ~ Normal(0, sigma_d)
    d_log_activity = d_offset * sigma_d    # [num_mutation]
    pyro.deterministic(f"{name}_d_log_activity", d_log_activity)

    # Assembly: log_activity[g] = d_log_activity @ M[:, g]
    log_activity = d_log_activity @ M_mat  # [G]

    # ------------------------------------------------------------------
    # Optional epistasis
    # ------------------------------------------------------------------
    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        sigma_epi = pyro.sample(
            f"{name}_sigma_epi",
            dist.HalfNormal(priors.activity_sigma_epi_scale))

        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            epi_offset = pyro.sample(f"{name}_epi_offset", dist.Normal(0.0, 1.0))

        epi_log_activity = epi_offset * sigma_epi   # [num_pair]
        pyro.deterministic(f"{name}_epi_log_activity", epi_log_activity)
        log_activity = log_activity + apply_pair_matrix(
            epi_log_activity, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)  # [G]

    activity = jnp.clip(jnp.exp(log_activity), max=1e30)

    pyro.deterministic(name, activity)

    if data.batch_size < data.num_genotype:
        activity = activity[data.batch_idx]
    return activity[None, None, None, None, None, None, :]


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """Variational guide for the mutation-decomposed activity model."""

    num_mut = data.num_mutation
    M_mat = jnp.array(data.mut_geno_matrix)
    has_epi = data.num_pair > 0

    # sigma_d variational params (LogNormal guide for HalfNormal prior)
    sigma_d_loc = pyro.param(f"{name}_sigma_d_loc", jnp.array(-1.0))
    sigma_d_scale = pyro.param(f"{name}_sigma_d_scale", jnp.array(0.1),
                               constraint=dist.constraints.positive)
    sigma_d = pyro.sample(f"{name}_sigma_d",
                          dist.LogNormal(sigma_d_loc, sigma_d_scale))

    # Per-mutation offset variational params
    d_offset_locs = pyro.param(f"{name}_d_offset_locs",
                               jnp.zeros(num_mut))
    d_offset_scales = pyro.param(f"{name}_d_offset_scales",
                                 jnp.ones(num_mut),
                                 constraint=dist.constraints.positive)
    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        d_offset = pyro.sample(f"{name}_d_offset",
                               dist.Normal(d_offset_locs, d_offset_scales))

    d_log_activity = d_offset * sigma_d
    log_activity = d_log_activity @ M_mat

    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        sigma_epi_loc = pyro.param(f"{name}_sigma_epi_loc", jnp.array(-1.0))
        sigma_epi_scale = pyro.param(f"{name}_sigma_epi_scale", jnp.array(0.1),
                                     constraint=dist.constraints.positive)
        sigma_epi = pyro.sample(f"{name}_sigma_epi",
                                dist.LogNormal(sigma_epi_loc, sigma_epi_scale))

        epi_offset_locs = pyro.param(f"{name}_epi_offset_locs",
                                     jnp.zeros(num_pair))
        epi_offset_scales = pyro.param(f"{name}_epi_offset_scales",
                                       jnp.ones(num_pair),
                                       constraint=dist.constraints.positive)
        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            epi_offset = pyro.sample(f"{name}_epi_offset",
                                     dist.Normal(epi_offset_locs, epi_offset_scales))

        epi_log_activity = epi_offset * sigma_epi
        log_activity = log_activity + apply_pair_matrix(
            epi_log_activity, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)

    activity = jnp.clip(jnp.exp(log_activity), max=1e30)

    if data.batch_size < data.num_genotype:
        activity = activity[data.batch_idx]
    return activity[None, None, None, None, None, None, :]


def get_hyperparameters() -> Dict[str, Any]:
    return {
        "activity_sigma_d_scale": 0.5,
        "activity_sigma_epi_scale": 0.25,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    guesses = {}
    guesses[f"{name}_sigma_d"] = 0.3
    guesses[f"{name}_d_offset"] = jnp.zeros(data.num_mutation)
    if data.num_pair > 0:
        guesses[f"{name}_sigma_epi"] = 0.2
        guesses[f"{name}_epi_offset"] = jnp.zeros(data.num_pair)
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())
