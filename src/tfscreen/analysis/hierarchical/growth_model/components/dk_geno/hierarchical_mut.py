"""
Mutation-decomposed dk_geno model.

dk_geno[g] = (d_dk_geno @ M)[:, g]
           + (epi_dk_geno @ P)[:, g]   (if num_pair > 0)

where M[m, g] = 1 if mutation m is in genotype g, P[p, g] = 1 if pair p is
present in genotype g.

Wild-type has no mutations (M[:, wt] = 0) so dk_geno[wt] = 0, consistent
with the other dk_geno models.

Per-mutation pleiotropic effects use the same shifted-lognormal distribution
as dk_geno/hierarchical.py:

    d_dk_geno[m] = shift - exp(hyper_loc + offset[m] * hyper_scale)

This creates a distribution that is mostly negative (deleterious mutations)
with a few slightly positive values, which is the expected biological prior.

Pairwise epistasis terms use a regularised horseshoe prior
(Piironen & Vehtari 2017):

    τ  ~ HalfCauchy(τ₀)
    c² ~ InvGamma(ν/2, ν·s²/2)
    λₚ ~ HalfCauchy(1)
    λ̃ₚ = λₚ · √(c² / (c² + τ²λₚ²))
    epi_dk_geno[p] = offsetₚ · τ · λ̃ₚ,   offsetₚ ~ Normal(0, 1)

This concentrates most epistasis terms near 0 while allowing a sparse
subset to escape shrinkage.  Effects are symmetric around 0: positive
and negative epistasis are equally probable a priori.
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
    """Hyperparameters for the mutation-decomposed dk_geno model."""
    hyper_loc_loc: float
    hyper_loc_scale: float
    hyper_scale_loc: float
    hyper_shift_loc: float
    hyper_shift_scale: float
    sigma_epi_tau_scale: float    # HalfCauchy scale for epistasis global τ
    sigma_epi_slab_scale: float   # typical size of a large epistasis effect (s)
    sigma_epi_slab_df: float      # InvGamma shape ν (usually 4)


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    Define the mutation-decomposed dk_geno model.

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
        dk_geno values broadcast to shape ``(1, 1, 1, 1, 1, 1, num_genotype)``.
    """
    M_mat = jnp.array(data.mut_geno_matrix)   # [num_mutation, G]
    num_mut = data.num_mutation
    has_epi = data.num_pair > 0

    # ------------------------------------------------------------------
    # Shared hyperpriors for the shifted-lognormal mutation effect
    # distribution (same structure as dk_geno/hierarchical.py)
    # ------------------------------------------------------------------
    dk_geno_hyper_loc = pyro.sample(
        f"{name}_hyper_loc",
        dist.Normal(priors.hyper_loc_loc,
                    priors.hyper_loc_scale))
    dk_geno_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.hyper_scale_loc))
    dk_geno_hyper_shift = pyro.sample(
        f"{name}_hyper_shift",
        dist.Normal(priors.hyper_shift_loc,
                    priors.hyper_shift_scale))

    # ------------------------------------------------------------------
    # Per-mutation offsets: shape (num_mutation,)
    # ------------------------------------------------------------------
    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        dk_geno_offset = pyro.sample(f"{name}_offset", dist.Normal(0.0, 1.0))

    # Shifted lognormal per mutation (non-centered)
    dk_geno_lognormal = jnp.clip(
        jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale),
        max=1e30)
    d_dk_geno = dk_geno_hyper_shift - dk_geno_lognormal  # [num_mutation]
    pyro.deterministic(f"{name}_d_dk_geno", d_dk_geno)

    # Assembly: dk_geno[g] = d_dk_geno @ M[:, g]
    dk_geno_per_genotype = d_dk_geno @ M_mat               # [G]

    # ------------------------------------------------------------------
    # Optional epistasis terms: Normal(0, sigma_epi)
    # ------------------------------------------------------------------
    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        tau_epi = pyro.sample(
            f"{name}_epi_tau",
            dist.HalfCauchy(priors.sigma_epi_tau_scale))
        c2_epi = pyro.sample(
            f"{name}_epi_c2",
            dist.InverseGamma(priors.sigma_epi_slab_df / 2.0,
                              priors.sigma_epi_slab_df * priors.sigma_epi_slab_scale ** 2 / 2.0))

        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            lambda_epi = pyro.sample(f"{name}_epi_lambda", dist.HalfCauchy(1.0))
            epi_offset = pyro.sample(f"{name}_epi_offset", dist.Normal(0.0, 1.0))

        lambda_epi_tilde = jnp.sqrt(c2_epi * lambda_epi ** 2 / (c2_epi + tau_epi ** 2 * lambda_epi ** 2))
        epi_dk_geno = epi_offset * tau_epi * lambda_epi_tilde
        pyro.deterministic(f"{name}_epi_dk_geno", epi_dk_geno)
        dk_geno_per_genotype = dk_geno_per_genotype + apply_pair_matrix(
            epi_dk_geno, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)  # [G]

    pyro.deterministic(name, dk_geno_per_genotype)

    return dk_geno_per_genotype[None, None, None, None, None, None, :]


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """Variational guide for the mutation-decomposed dk_geno model."""

    num_mut = data.num_mutation
    M_mat = jnp.array(data.mut_geno_matrix)
    has_epi = data.num_pair > 0

    # --- Global hyperpriors ---
    h_loc_loc = pyro.param(f"{name}_hyper_loc_loc",
                           jnp.array(priors.hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale",
                             jnp.array(priors.hyper_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_loc = pyro.sample(f"{name}_hyper_loc",
                                    dist.Normal(h_loc_loc, h_loc_scale))

    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0))
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1),
                               constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_scale = pyro.sample(f"{name}_hyper_scale",
                                      dist.LogNormal(h_scale_loc, h_scale_scale))

    shift_loc = pyro.param(f"{name}_hyper_shift_loc",
                           jnp.array(priors.hyper_shift_loc))
    shift_scale = pyro.param(f"{name}_hyper_shift_scale",
                             jnp.array(priors.hyper_shift_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_shift = pyro.sample(f"{name}_hyper_shift",
                                      dist.Normal(shift_loc, shift_scale))

    # --- Per-mutation offsets ---
    offset_locs = pyro.param(f"{name}_offset_locs", jnp.zeros(num_mut))
    offset_scales = pyro.param(f"{name}_offset_scales", jnp.ones(num_mut),
                               constraint=dist.constraints.positive)
    with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
        dk_geno_offset = pyro.sample(
            f"{name}_offset",
            dist.Normal(offset_locs, offset_scales))

    dk_geno_lognormal = jnp.clip(
        jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale),
        max=1e30)
    d_dk_geno = dk_geno_hyper_shift - dk_geno_lognormal
    dk_geno_per_genotype = d_dk_geno @ M_mat

    if has_epi:
        pair_nnz_pair_idx = jnp.array(data.pair_nnz_pair_idx)
        pair_nnz_geno_idx = jnp.array(data.pair_nnz_geno_idx)
        num_pair = data.num_pair

        tau_epi_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-1.0))
        tau_epi_scale_p = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.1),
                                     constraint=dist.constraints.positive)
        tau_epi = pyro.sample(f"{name}_epi_tau",
                              dist.LogNormal(tau_epi_loc, tau_epi_scale_p))

        c2_epi_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
        c2_epi_scale_p = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                                    constraint=dist.constraints.positive)
        c2_epi = pyro.sample(f"{name}_epi_c2",
                             dist.LogNormal(c2_epi_loc, c2_epi_scale_p))

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

        lambda_epi_tilde = jnp.sqrt(c2_epi * lambda_epi ** 2 / (c2_epi + tau_epi ** 2 * lambda_epi ** 2))
        epi_dk_geno = epi_offset * tau_epi * lambda_epi_tilde
        dk_geno_per_genotype = dk_geno_per_genotype + apply_pair_matrix(
            epi_dk_geno, pair_nnz_pair_idx, pair_nnz_geno_idx, data.num_genotype)

    return dk_geno_per_genotype[None, None, None, None, None, None, :]


def get_hyperparameters() -> Dict[str, Any]:
    return {
        "hyper_loc_loc": -3.5,
        "hyper_loc_scale": 1.0,
        "hyper_scale_loc": 1.0,
        "hyper_shift_loc": 0.02,
        "hyper_shift_scale": 0.2,
        "sigma_epi_tau_scale": 0.1,
        "sigma_epi_slab_scale": 0.5,
        "sigma_epi_slab_df": 4.0,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    # Offset value that gives dk_geno ≈ 0 on the shifted lognormal
    # (same magic number as in dk_geno/hierarchical.py)
    neutral_offset = -0.8240460108562919
    guesses = {}
    guesses[f"{name}_hyper_loc"] = -3.5
    guesses[f"{name}_hyper_scale"] = 0.5
    guesses[f"{name}_hyper_shift"] = 0.02
    guesses[f"{name}_offset"] = jnp.full(data.num_mutation, neutral_offset)
    if data.num_pair > 0:
        guesses[f"{name}_epi_tau"] = 0.05
        guesses[f"{name}_epi_c2"] = 4.0
        guesses[f"{name}_epi_lambda"] = jnp.ones(data.num_pair) * 0.5
        guesses[f"{name}_epi_offset"] = jnp.zeros(data.num_pair)
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())
