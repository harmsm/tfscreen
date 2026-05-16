"""
Tests for struct/mwc_dimer_unfolded/lnK_mut.py.

Covers: _assemble_scalar, _assemble_titrant, get_hyperparameters, get_priors,
get_guesses, define_model (no epi / with epi), guide, get_extract_specs.

Key differences from mwc_dimer/lnK_mut:
- ModelPriors gains theta_ln_K_u_wt_loc/scale and theta_sigma_d_ln_K_u_scale
- get_guesses returns theta_ln_K_u_wt (scalar), theta_sigma_d_ln_K_u (scalar),
  theta_d_ln_K_u_offset (M,)
- define_model samples and assembles ln_K_u per-genotype (no epistasis on K_u)
- ThetaParam.ln_K_u has shape (G,)
- get_extract_specs first spec includes "ln_K_u", third spec includes "d_ln_K_u"
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass
from collections import namedtuple
from functools import partial
from numpyro.handlers import trace, seed, substitute

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_mut import (
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
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.thermo import (
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
# _assemble_scalar  (shared with mwc_dimer; tests remain equivalent)
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
# _assemble_titrant  (shared with mwc_dimer; tests remain equivalent)
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
        d  = jnp.ones((T, M))
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
        # New: unfolded state
        "theta_ln_K_u_wt_loc", "theta_ln_K_u_wt_scale",
        "theta_tf_total_M", "theta_op_total_M",
        "theta_sigma_d_ln_K_h_l_scale", "theta_sigma_d_ln_K_h_o_scale",
        "theta_sigma_d_ln_K_l_o_scale",
        "theta_sigma_d_ln_K_h_e_scale", "theta_sigma_d_ln_K_l_e_scale",
        # New: unfolded mutation scale
        "theta_sigma_d_ln_K_u_scale",
        "theta_epi_tau_scale", "theta_epi_slab_scale", "theta_epi_slab_df",
    ]
    for key in required:
        assert key in hp, f"Missing key: {key}"


def test_get_hyperparameters_K_u_wt_defaults():
    """WT K_u should be << 1 (ln_K_u_wt_loc = -12)."""
    hp = get_hyperparameters()
    assert hp["theta_ln_K_u_wt_loc"] == pytest.approx(-12.0)
    assert hp["theta_ln_K_u_wt_scale"] > 0
    assert hp["theta_sigma_d_ln_K_u_scale"] > 0


def test_get_priors_type():
    assert isinstance(get_priors(), ModelPriors)


def test_get_priors_physical_constants():
    p = get_priors()
    assert p.theta_tf_total_M == pytest.approx(6.5e-7)
    assert p.theta_op_total_M == pytest.approx(2.5e-8)


def test_get_priors_K_u_fields():
    p = get_priors()
    assert p.theta_ln_K_u_wt_loc == pytest.approx(-12.0)
    assert p.theta_ln_K_u_wt_scale > 0
    assert p.theta_sigma_d_ln_K_u_scale > 0


def test_get_priors_matches_hyperparameters():
    hp = get_hyperparameters()
    p  = get_priors()
    assert p.theta_ln_K_h_l_wt_loc  == hp["theta_ln_K_h_l_wt_loc"]
    assert p.theta_ln_K_h_o_wt_loc  == hp["theta_ln_K_h_o_wt_loc"]
    assert p.theta_ln_K_u_wt_loc    == hp["theta_ln_K_u_wt_loc"]
    assert p.theta_sigma_d_ln_K_u_scale == hp["theta_sigma_d_ln_K_u_scale"]


def test_get_guesses_shapes_no_epi(mock_data_no_epi):
    name = "theta"
    g = get_guesses(name, mock_data_no_epi)
    T, M = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_mutation
    assert g[f"{name}_ln_K_h_l_wt"].shape == ()
    assert g[f"{name}_ln_K_h_o_wt"].shape == ()
    assert g[f"{name}_ln_K_l_o_wt"].shape == ()
    assert g[f"{name}_ln_K_h_e_wt"].shape == (T,)
    assert g[f"{name}_ln_K_l_e_wt"].shape == (T,)
    # New: unfolded state
    assert g[f"{name}_ln_K_u_wt"].shape   == ()
    assert g[f"{name}_sigma_d_ln_K_h_l"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_h_o"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_l_o"].shape == ()
    assert g[f"{name}_sigma_d_ln_K_h_e"].shape == (T,)
    assert g[f"{name}_sigma_d_ln_K_l_e"].shape == (T,)
    # New: unfolded sigma
    assert g[f"{name}_sigma_d_ln_K_u"].shape   == ()
    assert g[f"{name}_d_ln_K_h_l_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_h_o_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_l_o_offset"].shape == (M,)
    assert g[f"{name}_d_ln_K_h_e_offset"].shape == (T, M)
    assert g[f"{name}_d_ln_K_l_e_offset"].shape == (T, M)
    # New: per-mutation unfolded offset
    assert g[f"{name}_d_ln_K_u_offset"].shape   == (M,)
    assert f"{name}_epi_tau" not in g


def test_get_guesses_K_u_wt_value(mock_data_no_epi):
    """Default guess for ln_K_u_wt should be -12.0 (WT small)."""
    g = get_guesses("theta", mock_data_no_epi)
    assert float(g["theta_ln_K_u_wt"]) == pytest.approx(-12.0)


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
    # K_u has no epistasis — should not appear in epi guesses
    assert f"{name}_epi_ln_K_u_lambda" not in g
    assert f"{name}_epi_ln_K_u_offset" not in g


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
        # New: unfolded state — scalar per genotype, no T dimension
        assert tp.ln_K_u.shape   == (G,)
        assert tp.mu.shape    == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)

    def test_all_genotypes_equal_when_deltas_zero(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        for attr in ("ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u"):
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

    def test_K_u_assembly_with_known_deltas(self, mock_data_no_epi):
        """Known K_u offsets → expected per-genotype ln_K_u.

        K_u has no epistasis, so assembly is: wt + mut_scatter(d_offset * sigma).
        """
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_no_epi)
        guesses[f"{name}_ln_K_u_wt"]       = jnp.array(-12.0)
        guesses[f"{name}_sigma_d_ln_K_u"]  = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_u_offset"] = jnp.array([2.0, 0.5])
        tp = substitute(define_model, data=guesses)(
            name, mock_data_no_epi, priors)
        # Expect: wt + sum(d_offset * indicator) per genotype
        # wt=0, M1 delta=2, M2 delta=0.5 → [0: -12, 1: -10, 2: -11.5, 3: -9.5]
        expected = jnp.array([-12.0, -12.0 + 2.0, -12.0 + 0.5, -12.0 + 2.5])
        assert jnp.allclose(tp.ln_K_u, expected, atol=1e-5)

    def test_K_u_no_T_dimension(self, mock_data_no_epi):
        """ln_K_u should be (G,) not (T, G) — K_u has no effector-type dimension."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            "theta", mock_data_no_epi, priors)
        G = mock_data_no_epi.num_genotype
        assert tp.ln_K_u.shape == (G,)
        # Confirm it is NOT titrant-dimensional
        assert len(tp.ln_K_u.shape) == 1

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
                     # New: unfolded state
                     "theta_ln_K_u_wt",
                     "theta_d_ln_K_h_l_offset", "theta_d_ln_K_h_o_offset",
                     "theta_d_ln_K_l_o_offset",
                     "theta_d_ln_K_h_e_offset", "theta_d_ln_K_l_e_offset",
                     # New: unfolded offset
                     "theta_d_ln_K_u_offset"):
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
                     # New: assembled ln_K_u and delta
                     "theta_ln_K_u",
                     "theta_d_ln_K_h_l", "theta_d_ln_K_h_o", "theta_d_ln_K_l_o",
                     "theta_d_ln_K_h_e", "theta_d_ln_K_l_e",
                     # New: delta for K_u
                     "theta_d_ln_K_u"):
            assert site in det_names, f"Missing deterministic site: {site}"

    def test_K_u_no_epi_deterministic_site(self, mock_data_no_epi):
        """No epistasis deterministic site for K_u should exist even in no-epi mode."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_no_epi, priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        assert "theta_epi_ln_K_u" not in det_names


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
        assert tp.ln_K_u.shape   == (G,)

    def test_K_u_no_epistasis_with_epi_active(self, mock_data_epi):
        """K_u should NOT be affected by epistasis even when has_epi=True.

        Set all folded-state deltas to 0 and large K_h_l epistasis → only
        ln_K_h_l[3] changes; ln_K_u[3] should equal WT + additive (=WT here).
        """
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_epi)
        guesses[f"{name}_ln_K_u_wt"]       = jnp.array(-12.0)
        guesses[f"{name}_sigma_d_ln_K_u"]  = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_u_offset"] = jnp.zeros(2)   # no mut effects
        guesses[f"{name}_epi_tau"]               = jnp.array(1.0)
        guesses[f"{name}_epi_c2"]                = jnp.array(1e12)
        guesses[f"{name}_epi_ln_K_h_l_lambda"]  = jnp.ones(1)
        guesses[f"{name}_epi_ln_K_h_l_offset"]  = jnp.array([5.0])
        tp = substitute(define_model, data=guesses)(
            name, mock_data_epi, priors)
        # K_h_l[3] should shift; K_u should remain uniform at WT
        assert not jnp.allclose(tp.ln_K_h_l[3], tp.ln_K_h_l[0], atol=1e-3)
        assert jnp.allclose(tp.ln_K_u, -12.0, atol=1e-5), (
            "K_u should not be affected by epistasis"
        )

    def test_epistasis_shifts_only_double_mutant_ln_K_h_l(self, mock_data_epi):
        """Zero mut effects + unit epistasis → only genotype 3 shifts."""
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
        # K_u has no epistasis
        assert "theta_epi_ln_K_u_lambda" not in sample_names
        assert "theta_epi_ln_K_u_offset" not in sample_names

    def test_epistasis_deterministic_sites_registered(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            "theta", mock_data_epi, priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        for k in ("K_h_l", "K_h_o", "K_l_o", "K_h_e", "K_l_e"):
            assert f"theta_epi_ln_{k}" in det_names
        # K_u should NOT have an epistasis deterministic site
        assert "theta_epi_ln_K_u" not in det_names


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
        assert tp.ln_K_u.shape   == (G,)
        assert tp.mu.shape    == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)

    def test_with_epi_runs(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tp = guide("theta", mock_data_epi, priors)
        assert tp.ln_K_h_l.shape == (mock_data_epi.num_genotype,)
        assert tp.ln_K_h_e.shape == (mock_data_epi.num_titrant_name,
                                     mock_data_epi.num_genotype)
        assert tp.ln_K_u.shape   == (mock_data_epi.num_genotype,)

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
                  # New: unfolded variational params
                  "theta_ln_K_u_wt_loc", "theta_ln_K_u_wt_scale",
                  "theta_sigma_d_ln_K_u_loc", "theta_sigma_d_ln_K_u_scale",
                  "theta_d_ln_K_u_offset_locs", "theta_d_ln_K_u_offset_scales",
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


def test_large_K_u_collapses_theta_in_model(mock_data_no_epi):
    """define_model with large ln_K_u_wt (20) → near-zero theta everywhere."""
    name = "theta"
    priors = get_priors()
    guesses_wt = get_guesses(name, mock_data_no_epi)

    guesses_unfolded = get_guesses(name, mock_data_no_epi)
    guesses_unfolded[f"{name}_ln_K_u_wt"] = jnp.array(20.0)  # K_u >> 1

    tp_wt = substitute(define_model, data=guesses_wt)(
        name, mock_data_no_epi, priors)
    tp_unfolded = substitute(define_model, data=guesses_unfolded)(
        name, mock_data_no_epi, priors)

    data = mock_data_no_epi._replace(scatter_theta=0)
    result_wt       = run_model(tp_wt,       data)
    result_unfolded = run_model(tp_unfolded, data)

    assert jnp.all(result_wt > result_unfolded)
    assert jnp.all(result_unfolded < 0.01)


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
    from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
    assert "mwc_dimer_unfolded_lnK_mut" in model_registry["theta"]
    import tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_mut as mod
    assert model_registry["theta"]["mwc_dimer_unfolded_lnK_mut"] is mod


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
    if genotypes     is None: genotypes     = ["wt", "M1", "M2", "M1M2"]
    if titrant_names is None: titrant_names = ["iptg"]
    if mut_labels    is None: mut_labels    = ["M1", "M2"]
    if pair_labels   is None: pair_labels   = []
    return _FakeCtx(
        growth_tm=_FakeTM(genotypes, titrant_names),
        mut_labels=mut_labels,
        pair_labels=pair_labels,
    )


class TestGetExtractSpecs:

    def test_returns_list(self):
        assert isinstance(get_extract_specs(_make_ctx()), list)

    def test_four_specs_without_epi(self):
        # scalar K (incl K_u), T-dim K, scalar d_ln_K (incl d_ln_K_u), T-dim d_ln_K
        assert len(get_extract_specs(_make_ctx())) == 4

    def test_six_specs_with_epi(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        assert len(get_extract_specs(ctx)) == 6

    def test_first_spec_has_scalar_K_values_including_K_u(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[0]["params_to_get"]
        assert "ln_K_h_l" in params
        assert "ln_K_h_o" in params
        assert "ln_K_l_o" in params
        # New: K_u in first spec (scalar, no T dim)
        assert "ln_K_u"   in params

    def test_second_spec_has_titrant_K_values_no_K_u(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[1]["params_to_get"]
        assert "ln_K_h_e" in params
        assert "ln_K_l_e" in params
        # K_u is scalar — should NOT appear in T-dimensional spec
        assert "ln_K_u"   not in params

    def test_third_spec_has_scalar_delta_lnK_including_d_ln_K_u(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[2]["params_to_get"]
        assert "d_ln_K_h_l" in params
        assert "d_ln_K_h_o" in params
        assert "d_ln_K_l_o" in params
        # New: d_ln_K_u in third spec (scalar, no T dim)
        assert "d_ln_K_u"   in params

    def test_fourth_spec_has_titrant_delta_lnK_no_d_K_u(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[3]["params_to_get"]
        assert "d_ln_K_h_e" in params
        assert "d_ln_K_l_e" in params
        # d_ln_K_u is scalar — should NOT appear in T-dimensional spec
        assert "d_ln_K_u"   not in params

    def test_fifth_spec_has_scalar_epi_terms(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        specs = get_extract_specs(ctx)
        params = specs[4]["params_to_get"]
        assert "epi_ln_K_h_l" in params
        assert "epi_ln_K_h_o" in params
        assert "epi_ln_K_l_o" in params
        # K_u has no epistasis
        assert "epi_ln_K_u"   not in params

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
        titrants   = ["iptg", "tmg"]
        mut_labels = ["M1", "M2"]
        ctx = _make_ctx(titrant_names=titrants, mut_labels=mut_labels)
        specs = get_extract_specs(ctx)
        df = specs[3]["input_df"]
        assert len(df) == len(titrants) * len(mut_labels)
