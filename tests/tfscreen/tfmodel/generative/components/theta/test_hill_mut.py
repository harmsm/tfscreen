import warnings

import pytest
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

import jax
from functools import partial
from tfscreen.tfmodel.generative.components.theta.hill_mut import (
    ModelPriors,
    ThetaParam,
    SimPriors,
    define_model,
    guide,
    simulate,
    run_model,
    get_population_moments,
    get_hyperparameters,
    get_sim_hyperparameters,
    get_guesses,
    get_priors,
    _assemble,
    _population_moments,
)
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix, apply_mut_matrix, build_mut_sparse_indices

# ---------------------------------------------------------------------------
# Mock data namedtuples
# ---------------------------------------------------------------------------

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
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
    "batch_idx",
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
# Mutations: M42I(0), K84L(1)
# Pairs:     K84L/M42I(0) ↔ column 3 only
_MUT_GENO = np.array([[0, 1, 0, 1],   # M42I
                       [0, 0, 1, 1]], dtype=np.float32)   # K84L
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)
# COO representation of [[0, 0, 0, 1]]: one nonzero at (pair=0, geno=3)
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

def _make_mut_scatter(num_genotype=4):
    """Build a mut_scatter callable matching the test library."""
    return partial(apply_mut_matrix,
                   mut_nnz_mut_idx=jnp.array(_MUT_NNZ_MUT_IDX),
                   mut_nnz_geno_idx=jnp.array(_MUT_NNZ_GENO_IDX),
                   num_genotype=num_genotype)

def _make_pair_scatter(num_genotype=4):
    """Build a pair_scatter callable matching the test library (1 pair, geno 3)."""
    return partial(apply_pair_matrix,
                   pair_nnz_pair_idx=jnp.array(_PAIR_NNZ_PAIR),
                   pair_nnz_geno_idx=jnp.array(_PAIR_NNZ_GENO),
                   num_genotype=num_genotype)


@pytest.fixture
def mock_data_epi():
    """4 genotypes, 2 mutations, 1 pair (epistasis enabled)."""
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=4,
        log_titrant_conc=jnp.linspace(-5, 5, 3),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2,
        num_pair=1,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=_PAIR_NNZ_PAIR,
        pair_nnz_geno_idx=_PAIR_NNZ_GENO,
        batch_idx=jnp.arange(4, dtype=jnp.int32),
    )


@pytest.fixture
def mock_data_no_epi():
    """4 genotypes, 2 mutations, 0 pairs (no epistasis)."""
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=4,
        log_titrant_conc=jnp.linspace(-5, 5, 3),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2,
        num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
        batch_idx=jnp.arange(4, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# _assemble helper
# ---------------------------------------------------------------------------

class TestAssemble:

    def test_no_epistasis(self):
        T, M, G = 2, 2, 4
        wt = jnp.array([1.0, 2.0])
        d_offsets = jnp.ones((T, M))
        sigma_d = jnp.array([0.5, 1.0])

        result = _assemble(wt, d_offsets, sigma_d, _make_mut_scatter())
        # d = [[0.5, 0.5], [1.0, 1.0]]
        # d @ M = [[0,0.5,0.5,1.0], [0,1.0,1.0,2.0]]
        # wt[:, None] = [[1.0], [2.0]]
        # result = [[1.0, 1.5, 1.5, 2.0], [2.0, 3.0, 3.0, 4.0]]
        expected = jnp.array([[1.0, 1.5, 1.5, 2.0],
                               [2.0, 3.0, 3.0, 4.0]])
        assert jnp.allclose(result, expected)

    def test_with_epistasis(self):
        T, M, G, P = 1, 2, 4, 1
        wt = jnp.array([0.0])
        d_offsets = jnp.zeros((T, M))
        sigma_d = jnp.array([1.0])
        epi_offsets = jnp.array([[3.0]])   # shape (T=1, P=1)
        sigma_epi = jnp.array([1.0])
        pair_scatter = _make_pair_scatter()

        result = _assemble(wt, d_offsets, sigma_d, _make_mut_scatter(),
                           epi_offsets, sigma_epi, pair_scatter)
        # Only the double-mutant column (col 3) should be shifted by 3.0
        assert result[0, 0] == pytest.approx(0.0)   # wt
        assert result[0, 1] == pytest.approx(0.0)   # M42I
        assert result[0, 2] == pytest.approx(0.0)   # K84L
        assert result[0, 3] == pytest.approx(3.0)   # M42I/K84L


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    # Check a few expected keys
    assert "theta_logit_low_wt_loc" in params
    assert "theta_sigma_d_log_hill_K_scale" in params
    assert "theta_epi_tau_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.theta_logit_low_wt_loc == get_hyperparameters()["theta_logit_low_wt_loc"]


def test_get_guesses_no_epi(mock_data_no_epi):
    name = "theta"
    guesses = get_guesses(name, mock_data_no_epi)
    T, M = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_mutation
    assert guesses[f"{name}_logit_low_wt"].shape == (T,)
    assert guesses[f"{name}_d_logit_low_offset"].shape == (T, M)
    assert guesses[f"{name}_sigma_d_logit_low"].shape == (T,)
    # No epistasis keys when num_pair == 0
    assert f"{name}_epi_logit_low_offset" not in guesses


def test_get_guesses_with_epi(mock_data_epi):
    name = "theta"
    guesses = get_guesses(name, mock_data_epi)
    T, P = mock_data_epi.num_titrant_name, mock_data_epi.num_pair
    assert guesses[f"{name}_epi_logit_low_offset"].shape == (T, P)
    assert guesses[f"{name}_epi_logit_low_lambda"].shape == (T, P)
    assert f"{name}_epi_tau" in guesses
    assert f"{name}_epi_c2" in guesses


# ---------------------------------------------------------------------------
# define_model – no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_output_is_theta_param(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        assert isinstance(tp, ThetaParam)

    def test_theta_low_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        T, G = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_genotype
        assert tp.theta_low.shape == (T, G)
        assert tp.theta_high.shape == (T, G)
        assert tp.log_hill_K.shape == (T, G)
        assert tp.hill_n.shape == (T, G)

    def test_population_moments_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
        assert tp.mu.shape == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)
        # sigma >= 0 always; it is 0 when all genotypes are identical (zero offsets)
        assert jnp.all(tp.sigma >= 0)

    def test_wt_column_equals_wt_param_when_deltas_zero(self, mock_data_no_epi):
        """With zero offsets, all genotypes should have the same theta_low = f(wt param)."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        # With zero offsets, d_logit_low = 0, so every genotype column == wt value
        wt_val = tp.theta_low[:, 0]   # WT column
        for g in range(mock_data_no_epi.num_genotype):
            assert jnp.allclose(tp.theta_low[:, g], wt_val, atol=1e-5)

    def test_assembly_with_nonzero_deltas(self, mock_data_no_epi):
        """Verify matrix-multiply assembly produces correct per-genotype values."""
        T = mock_data_no_epi.num_titrant_name
        M = mock_data_no_epi.num_mutation
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_no_epi)

        # Fix sigma=1 and set known offsets: d = [[1.0, -0.5], [0.5, -1.0]]
        guesses[f"{name}_sigma_d_logit_low"] = jnp.ones(T)
        guesses[f"{name}_d_logit_low_offset"] = jnp.array([[1.0, -0.5],
                                                             [0.5, -1.0]])
        guesses[f"{name}_logit_low_wt"] = jnp.zeros(T)

        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_no_epi, priors=priors)

        M_mat = jnp.array(_MUT_GENO)
        d = guesses[f"{name}_d_logit_low_offset"]
        expected_logit_low = d @ M_mat    # [T, G] (wt=0 so just d@M)
        expected_theta_low = dist.transforms.SigmoidTransform()(expected_logit_low)
        assert jnp.allclose(tp.theta_low, expected_theta_low, atol=1e-5)

    def test_theta_values_in_valid_range(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        # Both theta_low and theta_high must be probabilities in [0, 1].
        # Note: the repressor convention allows theta_low > theta_high
        # (logit_delta_wt is negative by default), so no ordering assertion here.
        assert jnp.all(tp.theta_low >= 0) and jnp.all(tp.theta_low <= 1)
        assert jnp.all(tp.theta_high >= 0) and jnp.all(tp.theta_high <= 1)

    def test_sample_sites_in_trace(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_logit_low_wt" in sample_names
        assert "theta_d_logit_low_offset" in sample_names
        # No epistasis sites expected
        assert not any("epi" in k for k in sample_names)


# ---------------------------------------------------------------------------
# define_model – with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_output_shapes(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_epi, priors=priors)
        T, G = mock_data_epi.num_titrant_name, mock_data_epi.num_genotype
        assert tp.theta_low.shape == (T, G)

    def test_epistasis_shifts_only_double_column(self, mock_data_epi):
        """With zero mut deltas but non-zero epistasis, only column 3 differs.
        c2→∞ makes lambda_tilde ≈ lambda = 1, so epi = offset * tau * 1 = 1.0."""
        T = mock_data_epi.num_titrant_name
        P = mock_data_epi.num_pair
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_epi)

        guesses[f"{name}_logit_low_wt"] = jnp.zeros(T)
        guesses[f"{name}_sigma_d_logit_low"] = jnp.ones(T)
        guesses[f"{name}_d_logit_low_offset"] = jnp.zeros((T, 2))
        guesses[f"{name}_epi_tau"] = 1.0
        guesses[f"{name}_epi_c2"] = 1e12
        guesses[f"{name}_epi_logit_low_lambda"] = jnp.ones((T, P))
        guesses[f"{name}_epi_logit_low_offset"] = jnp.ones((T, P))

        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_epi, priors=priors)

        logit_low = dist.transforms.SigmoidTransform().inv(tp.theta_low)
        # wt(0), M42I(1), K84L(2) → logit_low should be 0
        assert jnp.allclose(logit_low[:, :3], 0.0, atol=1e-5)
        # M42I/K84L(3) → logit_low should be offset * tau * lambda_tilde = 1.0
        assert jnp.allclose(logit_low[:, 3], 1.0, atol=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_epi_logit_low_offset" in sample_names
        assert "theta_epi_logit_low_lambda" in sample_names
        assert "theta_epi_tau" in sample_names
        assert "theta_epi_c2" in sample_names


# ---------------------------------------------------------------------------
# run_model
# ---------------------------------------------------------------------------

class TestRunModel:

    @pytest.fixture
    def theta_param(self, mock_data_no_epi):
        T, G = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_genotype
        C = mock_data_no_epi.num_titrant_conc
        return ThetaParam(
            theta_low=jnp.zeros((T, G)),
            theta_high=jnp.ones((T, G)),
            log_hill_K=jnp.zeros((T, G)),
            hill_n=jnp.ones((T, G)),
            mu=jnp.zeros((T, C, 1)),
            sigma=jnp.ones((T, C, 1)),
        )

    def test_scatter_theta_1_shape(self, theta_param, mock_data_no_epi):
        data = mock_data_no_epi._replace(scatter_theta=1)
        result = run_model(theta_param, data)
        T, C, G = (mock_data_no_epi.num_titrant_name,
                   mock_data_no_epi.num_titrant_conc,
                   mock_data_no_epi.num_genotype)
        assert result.shape == (1, 1, 1, 1, T, C, G)

    def test_scatter_theta_0_shape(self, theta_param, mock_data_no_epi):
        data = mock_data_no_epi._replace(scatter_theta=0)
        result = run_model(theta_param, data)
        T, C, G = (mock_data_no_epi.num_titrant_name,
                   mock_data_no_epi.num_titrant_conc,
                   mock_data_no_epi.num_genotype)
        assert result.shape == (T, C, G)

    def test_hill_equation_midpoint(self, mock_data_no_epi):
        """At log_conc == log_K (conc=0), occupancy = 0.5; theta = 0 + 0.5*1 = 0.5."""
        T, G = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_genotype
        # three concentrations: -10, 0.0, +10  → occupancy ~ 0, 0.5, 1
        data = mock_data_no_epi._replace(
            log_titrant_conc=jnp.array([-10.0, 0.0, 10.0]),
            scatter_theta=0)
        tp = ThetaParam(
            theta_low=jnp.zeros((T, G)),
            theta_high=jnp.ones((T, G)),
            log_hill_K=jnp.zeros((T, G)),
            hill_n=jnp.ones((T, G)),
            mu=jnp.zeros((T, 3, 1)),
            sigma=jnp.ones((T, 3, 1)),
        )
        result = run_model(tp, data)
        assert jnp.allclose(result[:, 0, :], 0.0, atol=1e-3)
        assert jnp.allclose(result[:, 1, :], 0.5, atol=1e-5)
        assert jnp.allclose(result[:, 2, :], 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_guide_no_epi_returns_theta_param(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        assert isinstance(tp, ThetaParam)

    def test_guide_no_epi_shapes(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        T = mock_data_no_epi.num_titrant_name
        G = mock_data_no_epi.num_genotype
        C = mock_data_no_epi.num_titrant_conc
        assert tp.theta_low.shape == (T, G)
        assert tp.mu.shape == (T, C, 1)

    def test_guide_with_epi_runs(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tp = guide(name="theta", data=mock_data_epi, priors=priors)
        T, G = mock_data_epi.num_titrant_name, mock_data_epi.num_genotype
        assert tp.theta_low.shape == (T, G)

    def test_guide_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_guide_values_in_valid_range(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=42):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        assert jnp.all(tp.theta_low >= 0) and jnp.all(tp.theta_low <= 1)
        assert jnp.all(tp.theta_high >= 0) and jnp.all(tp.theta_high <= 1)


# ---------------------------------------------------------------------------
# get_population_moments
# ---------------------------------------------------------------------------

def test_get_population_moments_shape(mock_data_no_epi):
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    # Use non-zero offsets so genotypes differ and sigma > 0
    guesses["theta_d_logit_low_offset"] = jnp.array([[1.0, -0.5],
                                                       [0.5, -1.0]])
    tp = substitute(define_model, data=guesses)(
        name="theta", data=mock_data_no_epi, priors=priors)
    mu, sigma = get_population_moments(tp, mock_data_no_epi)
    T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
    assert mu.shape == (T, C, 1)
    assert sigma.shape == (T, C, 1)
    assert jnp.all(sigma > 0)


# ---------------------------------------------------------------------------
# SimPriors / get_sim_hyperparameters
# ---------------------------------------------------------------------------

def test_get_sim_hyperparameters_returns_dict():
    params = get_sim_hyperparameters()
    assert isinstance(params, dict)
    for key in ["wt_theta_low", "wt_theta_high", "wt_log_K", "wt_hill_n",
                "sigma_d_logit_low", "sigma_d_logit_delta",
                "sigma_d_log_K", "sigma_d_log_n",
                "epi_tau_scale", "epi_slab_scale", "epi_slab_df"]:
        assert key in params, f"Missing key: {key}"


def test_get_sim_hyperparameters_horseshoe_defaults_positive():
    """Horseshoe defaults must all be strictly positive."""
    params = get_sim_hyperparameters()
    assert params["epi_tau_scale"]  > 0.0
    assert params["epi_slab_scale"] > 0.0
    assert params["epi_slab_df"]    > 0.0


def test_sim_priors_constructs():
    sp = SimPriors(**get_sim_hyperparameters())
    assert isinstance(sp, SimPriors)
    assert 0.0 < sp.wt_theta_low < 1.0
    assert 0.0 < sp.wt_theta_high < 1.0
    assert sp.wt_hill_n > 0.0
    # Horseshoe attributes must exist and be non-negative
    assert sp.epi_tau_scale  >= 0.0
    assert sp.epi_slab_scale >= 0.0
    assert sp.epi_slab_df    >= 0.0


# ---------------------------------------------------------------------------
# simulate – no epistasis
# ---------------------------------------------------------------------------

class TestSimulateNoEpi:

    def test_output_shapes(self, mock_data_no_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        theta_gc, theta_param = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(0))
        G = mock_data_no_epi.num_genotype
        C = mock_data_no_epi.num_titrant_conc
        assert theta_gc.shape == (G, C)
        assert theta_param.theta_low.shape  == (1, G)
        assert theta_param.theta_high.shape == (1, G)
        assert theta_param.log_hill_K.shape == (1, G)
        assert theta_param.hill_n.shape     == (1, G)
        assert theta_param.mu    is None
        assert theta_param.sigma is None

    def test_theta_in_unit_interval(self, mock_data_no_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        theta_gc, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(1))
        assert np.all(theta_gc >= 0.0)
        assert np.all(theta_gc <= 1.0)

    def test_wt_genotype_exact_reference(self, mock_data_no_epi):
        """Wildtype (column 0, no mutations) receives zero additive deltas, so
        its assembled parameters must exactly match the WT SimPriors reference."""
        sp = SimPriors(**get_sim_hyperparameters())
        _, theta_param = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(2))
        # Column 0 is WT: no mutation indicator → zero additive delta on every param
        wt_theta_low  = float(theta_param.theta_low[0, 0])
        wt_theta_high = float(theta_param.theta_high[0, 0])
        assert abs(wt_theta_low  - sp.wt_theta_low)  < 1e-5
        assert abs(wt_theta_high - sp.wt_theta_high) < 1e-5
        # WT curve must be decreasing: theta goes from ~1 at low conc to ~0 at high conc
        assert wt_theta_low > wt_theta_high

    def test_mutant_differs_from_wt(self, mock_data_no_epi):
        """With non-zero sigma_d_log_K, mutant genotypes should have a shifted K."""
        params = get_sim_hyperparameters()
        params["sigma_d_log_K"] = 2.0   # large effect to ensure visible shift
        sp = SimPriors(**params)
        theta_gc, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(3))
        # Single mutants (columns 1 and 2) should differ from wt (column 0) at mid-conc
        assert not np.allclose(theta_gc[0], theta_gc[1], atol=1e-3) or \
               not np.allclose(theta_gc[0], theta_gc[2], atol=1e-3)

    def test_deterministic_same_key(self, mock_data_no_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        t1, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(99))
        t2, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(99))
        np.testing.assert_array_equal(t1, t2)

    def test_different_keys_differ(self, mock_data_no_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        t1, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(0))
        t2, _ = simulate("theta", mock_data_no_epi, sp, jax.random.PRNGKey(1))
        assert not np.allclose(t1, t2)

    def test_compatible_with_run_model(self, mock_data_no_epi):
        """ThetaParam from simulate() should be usable by run_model."""
        data = mock_data_no_epi._replace(scatter_theta=0)
        sp = SimPriors(**get_sim_hyperparameters())
        _, theta_param = simulate("theta", data, sp, jax.random.PRNGKey(5))
        result = run_model(theta_param, data)
        G = data.num_genotype
        C = data.num_titrant_conc
        # simulate() always produces T=1
        assert result.shape == (1, C, G)


# ---------------------------------------------------------------------------
# simulate – with epistasis
# ---------------------------------------------------------------------------

class TestSimulateEpi:

    def test_output_shapes(self, mock_data_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        theta_gc, theta_param = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(0))
        G = mock_data_epi.num_genotype
        C = mock_data_epi.num_titrant_conc
        assert theta_gc.shape == (G, C)
        assert theta_param.theta_low.shape == (1, G)

    def test_theta_in_unit_interval(self, mock_data_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        theta_gc, _ = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(7))
        assert np.all(theta_gc >= 0.0)
        assert np.all(theta_gc <= 1.0)

    def test_epistasis_only_shifts_double_mutant(self, mock_data_epi):
        """
        With zero mutation deltas but non-zero epistasis, only the double-mutant
        column (index 3) can differ from the wildtype column (index 0).
        Single mutants (indices 1 and 2) have no pair membership, so they must
        equal the wildtype reference regardless of the epistasis draw.
        """
        params = get_sim_hyperparameters()
        params.update({
            "sigma_d_logit_low":   0.0,
            "sigma_d_logit_delta": 0.0,
            "sigma_d_log_K":       0.0,
            "sigma_d_log_n":       0.0,
            "epi_tau_scale":  2.0,    # large τ → large horseshoe effects
            "epi_slab_scale": 2.0,
            "epi_slab_df":    4.0,
        })
        sp = SimPriors(**params)
        theta_gc, _ = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(42))
        # Columns 0, 1, 2 have no pair membership → must equal WT reference
        np.testing.assert_allclose(theta_gc[0], theta_gc[1], atol=1e-6)
        np.testing.assert_allclose(theta_gc[0], theta_gc[2], atol=1e-6)
        # Column 3 (double mutant M42I/K84L) may differ due to epistasis

    def test_epi_tau_zero_produces_no_epistasis(self, mock_data_epi):
        """
        epi_tau_scale=0.0 makes τ=0 exactly, so all epistasis effects are zero.
        With zero mutation deltas too, all four genotypes must be identical.
        """
        params = get_sim_hyperparameters()
        params.update({
            "sigma_d_logit_low":   0.0,
            "sigma_d_logit_delta": 0.0,
            "sigma_d_log_K":       0.0,
            "sigma_d_log_n":       0.0,
            "epi_tau_scale":       0.0,   # hard off-switch: τ = 0 exactly
        })
        sp = SimPriors(**params)
        theta_gc, _ = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(42))
        # No mutation delta, no epistasis → all columns are the WT reference
        np.testing.assert_allclose(theta_gc[0], theta_gc[1], atol=1e-10)
        np.testing.assert_allclose(theta_gc[0], theta_gc[2], atol=1e-10)
        np.testing.assert_allclose(theta_gc[0], theta_gc[3], atol=1e-10)

    def test_horseshoe_larger_tau_produces_larger_effects(self, mock_data_epi):
        """
        Over many random seeds, a larger epi_tau_scale should produce larger
        median absolute epistasis effects on the double-mutant log_K.
        """
        base = get_sim_hyperparameters()
        base.update({
            "sigma_d_logit_low":   0.0,
            "sigma_d_logit_delta": 0.0,
            "sigma_d_log_K":       0.0,
            "sigma_d_log_n":       0.0,
        })

        def median_abs_epi(tau_scale, n=200):
            params = dict(base)
            params["epi_tau_scale"] = tau_scale
            sp = SimPriors(**params)
            effects = []
            for seed_val in range(n):
                _, tp = simulate("theta", mock_data_epi, sp,
                                 jax.random.PRNGKey(seed_val))
                # Epistasis affects col 3 (double mutant); col 0 is pure WT
                effects.append(float(tp.log_hill_K[0, 3] - tp.log_hill_K[0, 0]))
            return float(np.median(np.abs(effects)))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in exp",
                                    category=RuntimeWarning)
            small_epi = median_abs_epi(0.1)
            large_epi = median_abs_epi(1.0)
        assert large_epi > small_epi, (
            f"Expected larger tau to produce larger effects, but "
            f"small={small_epi:.4f}, large={large_epi:.4f}")

    def test_deterministic_same_key(self, mock_data_epi):
        sp = SimPriors(**get_sim_hyperparameters())
        t1, _ = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(11))
        t2, _ = simulate("theta", mock_data_epi, sp, jax.random.PRNGKey(11))
        np.testing.assert_array_equal(t1, t2)
