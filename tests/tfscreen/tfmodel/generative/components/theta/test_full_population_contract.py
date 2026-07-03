"""
Cross-component API-contract test for theta components used with
transformation="empirical".

The congression correction (transformation/_congression.py) needs a theta
reference covering the *full* genotype population, not just whatever
genotypes are active in the current forward pass.  generative/model.py
gets this "for free" for hill_mut, hill_geno, and the thermo.* components by
re-calling run_model with batch_idx = geno_theta_idx = arange(num_genotype)
against the already-full-size ThetaParam these components assemble in
define_model/guide (see model_orchestrator._THETA_MODELS_INCOMPATIBLE_WITH_EMPIRICAL
and model_orchestrator._check_theta_transformation_compatibility for the one
component, categorical_geno, that does not yet satisfy this contract and is
rejected outright rather than silently mishandled).

This test locks in the contract for the components that DO support it:
run_model(theta_param, subset_data) must equal
run_model(theta_param, full_population_data)[..., subset_positions] whenever
subset_data.batch_idx is an arbitrary (non-prefix, reordered) selection of
genotypes and subset_data.geno_theta_idx = arange(len(batch_idx)) — the
convention tensors.batch.get_batch() always produces.

thermo.* components are intentionally NOT covered here — see the separate
follow-up investigating a suspected pre-existing batch_idx-composition bug
in their run_model.
"""
from collections import namedtuple

import jax.numpy as jnp
import pytest
from numpyro.handlers import substitute

from tfscreen.tfmodel.generative.components.theta import hill_mut, hill_geno
from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices


# ---------------------------------------------------------------------------
# hill_mut
# ---------------------------------------------------------------------------

_HillMutData = namedtuple("_HillMutData", [
    "num_titrant_name", "num_titrant_conc", "num_genotype", "log_titrant_conc",
    "geno_theta_idx", "scatter_theta", "num_mutation", "num_pair",
    "mut_geno_matrix", "mut_nnz_mut_idx", "mut_nnz_geno_idx",
    "pair_nnz_pair_idx", "pair_nnz_geno_idx", "batch_idx",
])


def _hill_mut_full_data():
    """6 genotypes: wt(0), 4 single mutants(1-4), 1 double mutant(5)."""
    num_genotype = 6
    mut_geno = jnp.array([
        [0, 1, 0, 0, 0, 1],   # mutation A present in genotypes 1, 5
        [0, 0, 1, 0, 0, 0],   # mutation B present in genotype 2
        [0, 0, 0, 1, 0, 0],   # mutation C present in genotype 3
        [0, 0, 0, 0, 1, 1],   # mutation D present in genotypes 4, 5
    ], dtype=float)
    mut_nnz_mut_idx, mut_nnz_geno_idx = build_mut_sparse_indices(mut_geno)

    return _HillMutData(
        num_titrant_name=1,
        num_titrant_conc=4,
        num_genotype=num_genotype,
        log_titrant_conc=jnp.linspace(-6, 2, 4),
        geno_theta_idx=jnp.arange(num_genotype, dtype=jnp.int32),
        scatter_theta=0,
        num_mutation=4,
        num_pair=0,
        mut_geno_matrix=mut_geno,
        mut_nnz_mut_idx=mut_nnz_mut_idx,
        mut_nnz_geno_idx=mut_nnz_geno_idx,
        pair_nnz_pair_idx=jnp.zeros(0, dtype=jnp.int32),
        pair_nnz_geno_idx=jnp.zeros(0, dtype=jnp.int32),
        batch_idx=jnp.arange(num_genotype, dtype=jnp.int32),
    )


def test_hill_mut_subset_matches_full_population_indexed_down():
    full_data = _hill_mut_full_data()
    name = "theta"
    priors = hill_mut.get_priors()
    guesses = hill_mut.get_guesses(name, full_data)
    # Non-zero, non-symmetric offsets so genotypes are genuinely distinguishable.
    guesses[f"{name}_d_logit_low_offset"] = jnp.array([[0.8, -1.2, 0.3, -0.5]])
    guesses[f"{name}_sigma_d_logit_low"] = jnp.ones(1)

    theta_param = substitute(hill_mut.define_model, data=guesses)(
        name=name, data=full_data, priors=priors)

    # Full population (identity batch/geno_theta indices).
    theta_full = hill_mut.run_model(theta_param, full_data)  # (T, C, 6)

    # A deliberately non-trivial (reordered, non-prefix) subset: genotypes
    # 4, 1, 5 in that order — exactly what get_batch() would produce for
    # idx=[4, 1, 5] (batch_idx=idx, geno_theta_idx=arange(3)).
    subset_global_idx = jnp.array([4, 1, 5], dtype=jnp.int32)
    subset_data = full_data._replace(
        batch_idx=subset_global_idx,
        geno_theta_idx=jnp.arange(len(subset_global_idx), dtype=jnp.int32),
    )
    theta_subset = hill_mut.run_model(theta_param, subset_data)  # (T, C, 3)

    expected = theta_full[:, :, subset_global_idx]
    assert jnp.allclose(theta_subset, expected)


# ---------------------------------------------------------------------------
# hill_geno
# ---------------------------------------------------------------------------

_HillGenoData = namedtuple("_HillGenoData", [
    "num_titrant_name", "num_titrant_conc", "num_genotype", "batch_size",
    "batch_idx", "scale_vector", "log_titrant_conc", "geno_theta_idx",
    "scatter_theta",
])


def _hill_geno_full_data():
    num_genotype = 5
    return _HillGenoData(
        num_titrant_name=1,
        num_titrant_conc=3,
        num_genotype=num_genotype,
        batch_size=num_genotype,
        batch_idx=jnp.arange(num_genotype, dtype=jnp.int32),
        scale_vector=jnp.ones(num_genotype, dtype=float),
        log_titrant_conc=jnp.linspace(-5, 5, 3),
        geno_theta_idx=jnp.arange(num_genotype, dtype=jnp.int32),
        scatter_theta=0,
    )


def test_hill_geno_subset_matches_full_population_indexed_down():
    full_data = _hill_geno_full_data()
    name = "theta"
    priors = hill_geno.get_priors()
    guesses = hill_geno.get_guesses(name, full_data)
    # Non-zero, non-symmetric offsets so genotypes are genuinely distinguishable.
    guesses[f"{name}_logit_low_offset"] = jnp.array([[0.5, -0.3, 1.1, -0.9, 0.2]])

    theta_param = substitute(hill_geno.define_model, data=guesses)(
        name=name, data=full_data, priors=priors)

    theta_full = hill_geno.run_model(theta_param, full_data)  # (T, C, 5)

    subset_global_idx = jnp.array([3, 0, 4], dtype=jnp.int32)
    subset_data = full_data._replace(
        batch_idx=subset_global_idx,
        geno_theta_idx=jnp.arange(len(subset_global_idx), dtype=jnp.int32),
    )
    theta_subset = hill_geno.run_model(theta_param, subset_data)  # (T, C, 3)

    expected = theta_full[:, :, subset_global_idx]
    assert jnp.allclose(theta_subset, expected)
