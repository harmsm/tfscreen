"""
Tests for struct/mwc_dimer/lnK_mut.py.

Covers: _assemble_scalar, _assemble_titrant, get_hyperparameters, get_priors,
get_guesses, define_model (no epi / with epi), guide, get_extract_specs.
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass
from collections import namedtuple
from functools import partial
from numpyro.handlers import trace, seed, substitute

from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U0_a.PK import (
    ModelPriors,
    _assemble_scalar,
    _assemble_titrant,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_extract_specs,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U0_a.thermo import (
    ThetaParam,
    run_model,
    get_population_moments,
)
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix, build_mut_sparse_indices

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Library: wt(0), M1(1), M2(2), M1+M2 double(3)
_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1]], dtype=np.float32)   # (M=2, G=4)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

# COO for one pair (M1+M2) present only in genotype 3
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

_CONC = np.array([0.0, 1e-5, 1e-4])
_LOG_CONC = np.log(np.where(_CONC == 0, 1e-20, _CONC))


def _make_mut_scatter(num_genotype=4):
    return partial(apply_mut_matrix,
                   mut_nnz_mut_idx=jnp.array(_MUT_NNZ_MUT_IDX),
                   mut_nnz_geno_idx=jnp.array(_MUT_NNZ_GENO_IDX),
                   num_genotype=num_genotype)

def _make_pair_scatter(num_genotype=4):
    return partial(apply_pair_matrix,
                   pair_nnz_pair_idx=jnp.array(_PAIR_NNZ_PAIR),
                   pair_nnz_geno_idx=jnp.array(_PAIR_NNZ_GENO),
                   num_genotype=num_genotype)


MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "titrant_conc",
    "log_titrant_conc",
    "geno_theta_idx",
    "scatter_theta",
    "num_mutation",
    "num_pair",
    "mut_geno_matrix",
    "mut_nnz_mut_idx",
    "mut_nnz_geno_idx",
    "pair_nnz_pair_idx",
    "pair_nnz_geno_idx",
])


@pytest.fixture
def mock_data_no_epi():
    return MockData(
        num_titrant_name=2, num_titrant_conc=len(_CONC), num_genotype=4,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        log_titrant_conc=jnp.array(_LOG_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2, num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
    )


@pytest.fixture
def mock_data_epi():
    return MockData(
        num_titrant_name=2, num_titrant_conc=len(_CONC), num_genotype=4,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        log_titrant_conc=jnp.array(_LOG_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2, num_pair=1,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=_PAIR_NNZ_PAIR,
        pair_nnz_geno_idx=_PAIR_NNZ_GENO,
    )


# ---------------------------------------------------------------------------
# _assemble_scalar
# ---------------------------------------------------------------------------

class TestAssembleScalar:

    def test_zero_offsets_gives_wt_everywhere(self):
        G = 4
        result = _assemble_scalar(jnp.array(5.0), jnp.zeros(2), jnp.array(1.0),
                                  _make_mut_scatter())
        assert result.shape == (G,)
        assert jnp.allclose(result, 5.0)

    def test_mut0_shifts_correct_genotypes(self):
        """Offset [1, 0] with sigma=1 → genotypes 1 and 3 shift by +1."""
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.0]), jnp.array(1.0),
                                  _make_mut_scatter())
        assert jnp.allclose(result, jnp.array([0.0, 1.0, 0.0, 1.0]))

    def test_sigma_scales_effect(self):
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.0]), jnp.array(3.0),
                                  _make_mut_scatter())
        assert jnp.allclose(result, jnp.array([0.0, 3.0, 0.0, 3.0]))

    def test_both_mutations_additive(self):
        """Both mutations present → effect = d0 + d1 at double mutant."""
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.5]), jnp.array(1.0),
                                  _make_mut_scatter())
        assert jnp.allclose(result[0], 0.0)
        assert jnp.allclose(result[1], 1.0)
        assert jnp.allclose(result[2], 0.5)
        assert jnp.allclose(result[3], 1.5)

    def test_epistasis_shifts_only_double_mutant(self):
        """Non-zero epistasis on pair 0 → only genotype 3 (double) changes."""
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.zeros(2), jnp.array(1.0),
                                  _make_mut_scatter(),
                                  jnp.array([2.0]), jnp.array(1.0),
                                  _make_pair_scatter())
        assert jnp.allclose(result[:3], 0.0, atol=1e-6)
        assert jnp.allclose(result[3], 2.0, atol=1e-6)

    def test_additive_mut_plus_epistasis(self):
        """Double-mutant gets both additive effects and epistasis."""
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.5]), jnp.array(1.0),
                                  _make_mut_scatter(),
                                  jnp.array([3.0]), jnp.array(1.0),
                                  _make_pair_scatter())
        assert jnp.allclose(result[3], 4.5, atol=1e-5)


# ---------------------------------------------------------------------------
# _assemble_titrant
# ---------------------------------------------------------------------------

class TestAssembleTitrant:

    def test_output_shape(self):
        T, G, M = 2, 4, 2
        result = _assemble_titrant(jnp.zeros(T), jnp.zeros((T, M)), jnp.ones(T),
                                   _make_mut_scatter())
        assert result.shape == (T, G)

    def test_zero_offsets_all_equal_wt(self):
        T = 2
        wt = jnp.array([1.0, 2.0])
        result = _assemble_titrant(wt, jnp.zeros((T, 2)), jnp.ones(T),
                                   _make_mut_scatter())
        assert jnp.allclose(result, wt[:, None])

    def test_sigma_scales_per_titrant(self):
        T, M = 2, 2
        wt = jnp.zeros(T)
        d  = jnp.ones((T, M))                       # both offsets = 1
        sigma = jnp.array([1.0, 2.0])
        result = _assemble_titrant(wt, d, sigma, _make_mut_scatter())
        # double mutant (col 3): T=0 → 1+1=2; T=1 → 2+2=4
        assert jnp.allclose(result[0, 3], 2.0, atol=1e-5)
        assert jnp.allclose(result[1, 3], 4.0, atol=1e-5)

    def test_epistasis_shifts_only_double_mutant(self):
        T = 1
        result = _assemble_titrant(
            jnp.zeros(T), jnp.zeros((T, 2)), jnp.ones(T),
            _make_mut_scatter(),
            jnp.array([[3.0]]), jnp.ones(T),
            _make_pair_scatter(),
        )
        assert jnp.allclose(result[0, :3], 0.0, atol=1e-6)
        assert jnp.allclose(result[0, 3],  3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters_has_required_keys():
    hp = get_hyperparameters()
    required = [
        "theta_ln_K_h_l_wt_loc", "theta_ln_K_h_l_wt_scale",
        "theta_ln_K_h_o_wt_loc", "theta_ln_K_h_o_wt_scale",
        "theta_ln_K_l_o_wt_loc", "theta_ln_K_l_o_wt_scale",
        "theta_ln_K_h_e_wt_loc", "theta_ln_K_h_e_wt_scale",
        "theta_ln_K_l_e_wt_loc", "theta_ln_K_l_e_wt_scale",
        "theta_tf_total_M", "theta_op_total_M",
        "theta_sigma_d_ln_K_h_l_scale", "theta_sigma_d_ln_K_h_o_scale",
        "theta_sigma_d_ln_K_l_o_scale",
        "theta_sigma_d_ln_K_h_e_scale", "theta_sigma_d_ln_K_l_e_scale",
        "theta_epi_tau_scale", "theta_epi_slab_scale", "theta_epi_slab_df",
    ]
    for key in required:
        assert key in hp, f"Missing key: {key}"


def test_get_priors_type():
    assert isinstance(get_priors(), ModelPriors)


def test_get_priors_physical_constants():
    p = get_priors()
    assert p.theta_tf_total_M == pytest.approx(6.5e-7)
    assert p.theta_op_total_M == pytest.approx(2.5e-8)


def test_get_priors_matches_hyperparameters():
    hp = get_hyperparameters()
    p  = get_priors()
    assert p.theta_ln_K_h_l_wt_loc == hp["theta_ln_K_h_l_wt_loc"]
    assert p.theta_ln_K_h_o_wt_loc == hp["theta_ln_K_h_o_wt_loc"]


def test_get_guesses_shapes_no_epi(mock_data_no_epi):
    name = "theta"
    g = get_guesses(name, mock_data_no_epi)
    T, M = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_mutation
    assert g[f"{name}_ln_K_h_l_wt"].shape == ()
    assert g[f"{name}_ln_K_h_o_wt"].shape == ()
    assert g[f"{name}_ln_K_l_o_wt"].shape == ()
    assert g[f"{name}_ln_K_h_e_wt"].shape == (T,)
    assert g[f"{name}_ln_K_l_e_wt"].shape == (T,)
    assert g[f"{name}_sigma_d_ln_K_h_l"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_h_o"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_l_o"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_h_e"].shape == (T,)
    assert g[f"{name}_sigma_d_ln_K_l_e"].shape == (T,)
    assert g[f"{name}_d_ln_K_h_l_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_h_o_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_l_o_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_h_e_offset"].shape == (T, M)
    assert g[f"{name}_d_ln_K_l_e_offset"].shape == (T, M)
    assert f"{name}_epi_tau" not in g


def test_get_guesses_shapes_with_epi(mock_data_epi):
    name = "theta"
    g = get_guesses(name, mock_data_epi)
    T, P = mock_data_epi.num_titrant_name, mock_data_epi.num_pair
    assert g[f"{name}_epi_tau"].shape == ()
    assert g[f"{name}_epi_c2"].shape  == ()
    for k in ("K_h_l", "K_h_o", "K_l_o"):
        assert g[f"{name}_epi_ln_{k}_lambda"].shape == (P,)
        assert g[f"{name}_epi_ln_{k}_offset"].shape == (P,)
    for k in ("K_h_e", "K_l_e"):
        assert g[f"{name}_epi_ln_{k}_lambda"].shape == (T, P)
        assert g[f"{name}_epi_ln_{k}_offset"].shape == (T, P)


# ---------------------------------------------------------------------------
# define_model — no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_returns_theta_param(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        assert isinstance(tp, ThetaParam)

    def test_parameter_shapes(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        G = mock_data_no_epi.num_genotype
        T = mock_data_no_epi.num_titrant_name
        C = mock_data_no_epi.num_titrant_conc
        assert tp.ln_K_h_l.shape == (G,)
        assert tp.ln_K_h_o.shape == (G,)
        assert tp.ln_K_l_o.shape == (G,)
        assert tp.ln_K_h_e.shape == (T, G)
        assert tp.ln_K_l_e.shape == (T, G)
        assert tp.mu.shape    == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)

    def test_all_genotypes_equal_when_deltas_zero(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        for attr in ("ln_K_h_l", "ln_K_h_o", "ln_K_l_o"):
            arr = getattr(tp, attr)
            assert jnp.allclose(arr, arr[0], atol=1e-5), f"{attr} not uniform"

    def test_assembly_with_known_deltas(self, mock_data_no_epi):
        """Known offsets → expected per-genotype ln_K_h_l."""
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_no_epi)
        guesses[f"{name}_ln_K_h_l_wt"]       = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_h_l"]  = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_h_l_offset"] = jnp.array([1.0, -0.5])
        tp = substitute(define_model, data=guesses)(
            name, mock_data_no_epi, priors)
        expected = jnp.array([1.0, -0.5]) @ jnp.array(_MUT_GENO)
        assert jnp.allclose(tp.ln_K_h_l, expected, atol=1e-5)

    def test_titrant_dim_assembly(self, mock_data_no_epi):
        """Known K_h_e offsets produce correct (T, G) shape."""
        name = "theta"
        priors = get_priors()
        T = mock_data_no_epi.num_titrant_name
        guesses = get_guesses(name, mock_data_no_epi)
        guesses[f"{name}_ln_K_h_e_wt"]       = jnp.zeros(T)
        guesses[f"{name}_sigma_d_ln_K_h_e"]  = jnp.ones(T)
        guesses[f"{name}_d_ln_K_h_e_offset"] = jnp.ones((T, 2))
        tp = substitute(define_model, data=guesses)(
            name, mock_data_no_epi, priors)
        # double mutant (col 3): both offsets=1 → sum=2 for each T
        assert jnp.allclose(tp.ln_K_h_e[:, 3], 2.0, atol=1e-5)

    def test_sigma_nonneg(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        assert jnp.all(tp.sigma >= 0)

    def test_sample_sites_present(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_no_epi, priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        for site in ("theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                     "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt",
                     "theta_d_ln_K_h_l_offset", "theta_d_ln_K_h_o_offset",
                     "theta_d_ln_K_l_o_offset",
                     "theta_d_ln_K_h_e_offset", "theta_d_ln_K_l_e_offset"):
            assert site in sample_names, f"Missing sample site: {site}"

    def test_no_epi_sample_sites_absent(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_no_epi, priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_deterministic_sites_registered(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_no_epi, priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        for site in ("theta_ln_K_h_l", "theta_ln_K_h_o", "theta_ln_K_l_o",
                     "theta_ln_K_h_e", "theta_ln_K_l_e",
                     "theta_d_ln_K_h_l", "theta_d_ln_K_h_o", "theta_d_ln_K_l_o",
                     "theta_d_ln_K_h_e", "theta_d_ln_K_l_e"):
            assert site in det_names, f"Missing deterministic site: {site}"


# ---------------------------------------------------------------------------
# define_model — with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_returns_theta_param(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_epi, priors)
        assert isinstance(tp, ThetaParam)

    def test_parameter_shapes(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_epi, priors)
        G = mock_data_epi.num_genotype
        T = mock_data_epi.num_titrant_name
        assert tp.ln_K_h_l.shape == (G,)
        assert tp.ln_K_h_e.shape == (T, G)

    def test_epistasis_shifts_only_double_mutant_ln_K_h_l(self, mock_data_epi):
        """Zero mut effects + unit epistasis → only genotype 3 shifts.
        With c2→∞, lambda_tilde → lambda, so epi = offset * tau * lambda."""
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_epi)
        guesses[f"{name}_ln_K_h_l_wt"]          = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_h_l"]     = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_h_l_offset"]    = jnp.zeros(2)
        guesses[f"{name}_epi_tau"]               = jnp.array(1.0)
        guesses[f"{name}_epi_c2"]                = jnp.array(1e12)
        guesses[f"{name}_epi_ln_K_h_l_lambda"]  = jnp.ones(1)
        guesses[f"{name}_epi_ln_K_h_l_offset"]  = jnp.array([2.0])
        tp = substitute(define_model, data=guesses)(
            name, mock_data_epi, priors)
        assert jnp.allclose(tp.ln_K_h_l[:3], 0.0, atol=1e-5)
        assert jnp.allclose(tp.ln_K_h_l[3],  2.0, atol=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_epi, priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_epi_tau" in sample_names
        assert "theta_epi_c2"  in sample_names
        for k in ("K_h_l", "K_h_o", "K_l_o"):
            assert f"theta_epi_ln_{k}_lambda" in sample_names
            assert f"theta_epi_ln_{k}_offset" in sample_names
        for k in ("K_h_e", "K_l_e"):
            assert f"theta_epi_ln_{k}_lambda" in sample_names
            assert f"theta_epi_ln_{k}_offset" in sample_names

    def test_epistasis_deterministic_sites_registered(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_epi, priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        for k in ("K_h_l", "K_h_o", "K_l_o", "K_h_e", "K_l_e"):
            assert f"theta_epi_ln_{k}" in det_names


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_no_epi_returns_theta_param(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide("theta", mock_data_no_epi, priors)
        assert isinstance(tp, ThetaParam)

    def test_no_epi_shapes(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide("theta", mock_data_no_epi, priors)
        G = mock_data_no_epi.num_genotype
        T = mock_data_no_epi.num_titrant_name
        C = mock_data_no_epi.num_titrant_conc
        assert tp.ln_K_h_l.shape == (G,)
        assert tp.ln_K_h_o.shape == (G,)
        assert tp.ln_K_l_o.shape == (G,)
        assert tp.ln_K_h_e.shape == (T, G)
        assert tp.ln_K_l_e.shape == (T, G)
        assert tp.mu.shape    == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)

    def test_with_epi_runs(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tp = guide("theta", mock_data_epi, priors)
        assert tp.ln_K_h_l.shape == (mock_data_epi.num_genotype,)
        assert tp.ln_K_h_e.shape == (mock_data_epi.num_titrant_name,
                                     mock_data_epi.num_genotype)

    def test_sample_sites_match_model_no_epi(self, mock_data_no_epi):
        """Guide sample sites must exactly match model sample sites."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        model_tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_no_epi, priors)
        with seed(rng_seed=0):
            guide_tr = trace(guide).get_trace("theta", mock_data_no_epi, priors)

        model_samples = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_samples = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_samples == guide_samples

    def test_sample_sites_match_model_with_epi(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        model_tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_epi, priors)
        with seed(rng_seed=0):
            guide_tr = trace(guide).get_trace("theta", mock_data_epi, priors)

        model_samples = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_samples = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_samples == guide_samples

    def test_no_epi_no_epistasis_sample_sites_in_guide(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace("theta", mock_data_no_epi, priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_tf_op_totals_preserved(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide("theta", mock_data_no_epi, priors)
        assert tp.tf_total == pytest.approx(6.5e-7)
        assert tp.op_total == pytest.approx(2.5e-8)

    def test_variational_params_registered_no_epi(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace("theta", mock_data_no_epi, priors)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for p in ("theta_ln_K_h_l_wt_loc", "theta_ln_K_h_l_wt_scale",
                  "theta_ln_K_h_o_wt_loc", "theta_ln_K_l_o_wt_loc",
                  "theta_d_ln_K_h_l_offset_locs", "theta_d_ln_K_h_l_offset_scales",
                  "theta_d_ln_K_h_e_offset_locs", "theta_d_ln_K_h_e_offset_scales"):
            assert p in param_names, f"Missing param: {p}"


# ---------------------------------------------------------------------------
# Integration: run_model and get_population_moments
# ---------------------------------------------------------------------------

def test_run_model_produces_valid_theta(mock_data_no_epi):
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    tp = substitute(define_model, data=guesses)(
        "theta", mock_data_no_epi, priors)
    data = mock_data_no_epi._replace(scatter_theta=0)
    result = run_model(tp, data)
    T, C, G = (mock_data_no_epi.num_titrant_name,
               mock_data_no_epi.num_titrant_conc,
               mock_data_no_epi.num_genotype)
    assert result.shape == (T, C, G)
    assert jnp.all(result >= 0)
    assert jnp.all(result <= 1)
    assert jnp.all(jnp.isfinite(result))


def test_get_population_moments_shape(mock_data_no_epi):
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    tp = substitute(define_model, data=guesses)(
        "theta", mock_data_no_epi, priors)
    mu, sigma = get_population_moments(tp, mock_data_no_epi)
    T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
    assert mu.shape == (T, C, 1)
    assert sigma.shape == (T, C, 1)
    assert jnp.all(sigma >= 0)


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

def test_registry_entry():
    from tfscreen.tfmodel.generative.registry import model_registry
    assert "thermo.O2_C12_K5_U0_a.PK" in model_registry["theta"]
    import tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U0_a.PK as mod
    assert model_registry["theta"]["thermo.O2_C12_K5_U0_a.PK"] is mod


# ---------------------------------------------------------------------------
# get_extract_specs
# ---------------------------------------------------------------------------

class _FakeTM:
    def __init__(self, genotypes, titrant_names):
        self.tensor_dim_names  = ["genotype", "titrant_name"]
        self.tensor_dim_labels = [np.array(genotypes), np.array(titrant_names)]
        rows = [
            {"genotype": g, "titrant_name": t,
             "genotype_idx": gi, "titrant_name_idx": ti}
            for gi, g in enumerate(genotypes)
            for ti, t in enumerate(titrant_names)
        ]
        self.df = pd.DataFrame(rows)


@dataclass
class _FakeCtx:
    growth_tm: object
    mut_labels: list
    pair_labels: list


def _make_ctx(genotypes=None, titrant_names=None, mut_labels=None, pair_labels=None):
    if genotypes    is None: genotypes    = ["wt", "M1", "M2", "M1M2"]
    if titrant_names is None: titrant_names = ["iptg"]
    if mut_labels   is None: mut_labels   = ["M1", "M2"]
    if pair_labels  is None: pair_labels  = []
    return _FakeCtx(
        growth_tm=_FakeTM(genotypes, titrant_names),
        mut_labels=mut_labels,
        pair_labels=pair_labels,
    )


class TestGetExtractSpecs:

    def test_returns_list(self):
        assert isinstance(get_extract_specs(_make_ctx()), list)

    def test_four_specs_without_epi(self):
        # scalar K, T-dim K, scalar d_ln_K, T-dim d_ln_K
        assert len(get_extract_specs(_make_ctx())) == 4

    def test_six_specs_with_epi(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        assert len(get_extract_specs(ctx)) == 6

    def test_first_spec_has_scalar_K_values(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[0]["params_to_get"]
        assert "ln_K_h_l" in params
        assert "ln_K_h_o" in params
        assert "ln_K_l_o" in params

    def test_second_spec_has_titrant_K_values(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[1]["params_to_get"]
        assert "ln_K_h_e" in params
        assert "ln_K_l_e" in params

    def test_third_spec_has_scalar_delta_lnK(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[2]["params_to_get"]
        assert "d_ln_K_h_l" in params
        assert "d_ln_K_h_o" in params
        assert "d_ln_K_l_o" in params

    def test_fourth_spec_has_titrant_delta_lnK(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[3]["params_to_get"]
        assert "d_ln_K_h_e" in params
        assert "d_ln_K_l_e" in params

    def test_fifth_spec_has_scalar_epi_terms(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        specs = get_extract_specs(ctx)
        params = specs[4]["params_to_get"]
        assert "epi_ln_K_h_l" in params
        assert "epi_ln_K_h_o" in params
        assert "epi_ln_K_l_o" in params

    def test_sixth_spec_has_titrant_epi_terms(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        specs = get_extract_specs(ctx)
        params = specs[5]["params_to_get"]
        assert "epi_ln_K_h_e" in params
        assert "epi_ln_K_l_e" in params

    def test_all_specs_use_theta_prefix(self):
        for spec in get_extract_specs(_make_ctx(pair_labels=["M1M2"])):
            assert spec["in_run_prefix"] == "theta_"

    def test_second_spec_covers_all_titrant_genotype_combos(self):
        genos    = ["wt", "M1"]
        titrants = ["iptg", "tmg"]
        ctx = _make_ctx(genotypes=genos, titrant_names=titrants)
        specs = get_extract_specs(ctx)
        df = specs[1]["input_df"]
        assert len(df) == len(genos) * len(titrants)

    def test_fourth_spec_covers_all_titrant_mutation_combos(self):
        titrants  = ["iptg", "tmg"]
        mut_labels = ["M1", "M2"]
        ctx = _make_ctx(titrant_names=titrants, mut_labels=mut_labels)
        specs = get_extract_specs(ctx)
        df = specs[3]["input_df"]
        assert len(df) == len(titrants) * len(mut_labels)
