import pytest
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from functools import partial
from tfscreen.analysis.hierarchical.growth_model.components.theta.hill_mut import (
    ModelPriors,
    ThetaParam,
    define_model,
    guide,
    run_model,
    get_population_moments,
    get_hyperparameters,
    get_guesses,
    get_priors,
    _assemble,
    _population_moments,
)
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix

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
    "pair_nnz_pair_idx",
    "pair_nnz_geno_idx",
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
# Mutations: M42I(0), K84L(1)
# Pairs:     K84L/M42I(0) ↔ column 3 only
_MUT_GENO = np.array([[0, 1, 0, 1],   # M42I
                       [0, 0, 1, 1]], dtype=np.float32)   # K84L
# COO representation of [[0, 0, 0, 1]]: one nonzero at (pair=0, geno=3)
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

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
        pair_nnz_pair_idx=_PAIR_NNZ_PAIR,
        pair_nnz_geno_idx=_PAIR_NNZ_GENO,
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
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
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
        M_mat = jnp.array(_MUT_GENO)

        result = _assemble(wt, d_offsets, sigma_d, M_mat)
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
        M_mat = jnp.array(_MUT_GENO)
        epi_offsets = jnp.array([[3.0]])   # shape (T=1, P=1)
        sigma_epi = jnp.array([1.0])
        pair_scatter = _make_pair_scatter()

        result = _assemble(wt, d_offsets, sigma_d, M_mat,
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
