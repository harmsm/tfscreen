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

Pairwise epistasis terms use a Normal(0, sigma_epi) prior.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass
from typing import Dict, Any

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData


@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the mutation-decomposed dk_geno model."""
    dk_geno_hyper_loc_loc: float
    dk_geno_hyper_loc_scale: float
    dk_geno_hyper_scale_loc: float
    dk_geno_hyper_shift_loc: float
    dk_geno_hyper_shift_scale: float
    dk_geno_sigma_epi_scale: float   # HalfNormal scale for epistasis spread


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
        dist.Normal(priors.dk_geno_hyper_loc_loc,
                    priors.dk_geno_hyper_loc_scale))
    dk_geno_hyper_scale = pyro.sample(
        f"{name}_hyper_scale",
        dist.HalfNormal(priors.dk_geno_hyper_scale_loc))
    dk_geno_hyper_shift = pyro.sample(
        f"{name}_shift",
        dist.Normal(priors.dk_geno_hyper_shift_loc,
                    priors.dk_geno_hyper_shift_scale))

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
        P_mat = jnp.array(data.pair_geno_matrix)
        num_pair = data.num_pair

        sigma_epi = pyro.sample(
            f"{name}_sigma_epi",
            dist.HalfNormal(priors.dk_geno_sigma_epi_scale))

        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            epi_offset = pyro.sample(f"{name}_epi_offset", dist.Normal(0.0, 1.0))

        epi_dk_geno = epi_offset * sigma_epi
        pyro.deterministic(f"{name}_epi_dk_geno", epi_dk_geno)
        dk_geno_per_genotype = dk_geno_per_genotype + epi_dk_geno @ P_mat  # [G]

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
                           jnp.array(priors.dk_geno_hyper_loc_loc))
    h_loc_scale = pyro.param(f"{name}_hyper_loc_scale",
                             jnp.array(priors.dk_geno_hyper_loc_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_loc = pyro.sample(f"{name}_hyper_loc",
                                    dist.Normal(h_loc_loc, h_loc_scale))

    h_scale_loc = pyro.param(f"{name}_hyper_scale_loc", jnp.array(-1.0))
    h_scale_scale = pyro.param(f"{name}_hyper_scale_scale", jnp.array(0.1),
                               constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_scale = pyro.sample(f"{name}_hyper_scale",
                                      dist.LogNormal(h_scale_loc, h_scale_scale))

    shift_loc = pyro.param(f"{name}_shift_loc",
                           jnp.array(priors.dk_geno_hyper_shift_loc))
    shift_scale = pyro.param(f"{name}_shift_scale",
                             jnp.array(priors.dk_geno_hyper_shift_scale),
                             constraint=dist.constraints.greater_than(1e-4))
    dk_geno_hyper_shift = pyro.sample(f"{name}_shift",
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
        P_mat = jnp.array(data.pair_geno_matrix)
        num_pair = data.num_pair

        sigma_epi_loc = pyro.param(f"{name}_sigma_epi_loc", jnp.array(-1.0))
        sigma_epi_scale = pyro.param(f"{name}_sigma_epi_scale", jnp.array(0.1),
                                     constraint=dist.constraints.greater_than(1e-4))
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

        epi_dk_geno = epi_offset * sigma_epi
        dk_geno_per_genotype = dk_geno_per_genotype + epi_dk_geno @ P_mat

    return dk_geno_per_genotype[None, None, None, None, None, None, :]


def get_hyperparameters() -> Dict[str, Any]:
    return {
        "dk_geno_hyper_loc_loc": -3.5,
        "dk_geno_hyper_loc_scale": 1.0,
        "dk_geno_hyper_scale_loc": 1.0,
        "dk_geno_hyper_shift_loc": 0.02,
        "dk_geno_hyper_shift_scale": 0.2,
        "dk_geno_sigma_epi_scale": 0.1,
    }


def get_guesses(name: str, data: GrowthData) -> Dict[str, Any]:
    # Offset value that gives dk_geno ≈ 0 on the shifted lognormal
    # (same magic number as in dk_geno/hierarchical.py)
    neutral_offset = -0.8240460108562919
    guesses = {}
    guesses[f"{name}_hyper_loc"] = -3.5
    guesses[f"{name}_hyper_scale"] = 0.5
    guesses[f"{name}_shift"] = 0.02
    guesses[f"{name}_offset"] = jnp.full(data.num_mutation, neutral_offset)
    if data.num_pair > 0:
        guesses[f"{name}_sigma_epi"] = 0.05
        guesses[f"{name}_epi_offset"] = jnp.zeros(data.num_pair)
    return guesses


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())
