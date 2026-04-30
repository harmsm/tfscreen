import pytest
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from functools import partial
from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer_mut import (
    ModelPriors,
    ThetaParam,
    define_model,
    guide,
    run_model,
    get_population_moments,
    get_hyperparameters,
    get_guesses,
    get_priors,
    _assemble_scalar,
    _assemble_titrant,
    _compute_theta,
    _solve_free_effector,
    _population_moments,
)
from tfscreen.genetics.build_mut_geno_matrix import apply_pair_matrix

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

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
    "pair_nnz_pair_idx",
    "pair_nnz_geno_idx",
    "batch_idx",
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
# Mutations: M42I(0), K84L(1)
# Pairs:     M42I/K84L(0) — present only in column 3
_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1]], dtype=np.float32)
# COO representation of [[0, 0, 0, 1]]: one nonzero at (pair=0, geno=3)
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

def _make_pair_scatter(num_genotype=4):
    """Build a pair_scatter callable matching the test library (1 pair, geno 3)."""
    return partial(apply_pair_matrix,
                   pair_nnz_pair_idx=jnp.array(_PAIR_NNZ_PAIR),
                   pair_nnz_geno_idx=jnp.array(_PAIR_NNZ_GENO),
                   num_genotype=num_genotype)

_CONC = np.array([0.0, 100.0, 1000.0])
_LOG_CONC = np.log(np.where(_CONC == 0, 1e-20, _CONC))


@pytest.fixture
def mock_data_epi():
    """4 genotypes, 2 mutations, 1 pair (epistasis enabled)."""
    return MockData(
        num_titrant_name=2,
        num_titrant_conc=3,
        num_genotype=4,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        log_titrant_conc=jnp.array(_LOG_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2,
        num_pair=1,
        mut_geno_matrix=_MUT_GENO,
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
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        log_titrant_conc=jnp.array(_LOG_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(4, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=2,
        num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
        batch_idx=jnp.arange(4, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# _solve_free_effector
# ---------------------------------------------------------------------------

class TestSolveFreeEffector:

    def test_zero_effector_returns_zero(self):
        """When e_total=0, free effector should be 0."""
        e_total = jnp.zeros((1, 3, 1))
        a  = jnp.ones((2, 1, 4)) * 1e-4
        Z0 = jnp.ones((1, 1, 4)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert jnp.allclose(e_free, 0.0, atol=1e-6)

    def test_e_free_leq_e_total(self):
        """Free effector is always ≤ total effector."""
        e_total = jnp.array([[[0.0]], [[100.0]], [[1000.0]]])  # (3,1,1)
        a  = jnp.ones((1, 1, 1)) * 1e-4
        Z0 = jnp.ones((1, 1, 1)) * 252.0
        e_free = _solve_free_effector(e_total.reshape(1, 3, 1), tf_total=650.0, a=a, Z0=Z0)
        assert jnp.all(e_free >= 0)
        assert jnp.all(e_free <= e_total.reshape(1, 3, 1) + 1e-9)

    def test_output_shape(self):
        """Output shape matches (T, C, G) broadcast."""
        T, C, G = 2, 3, 4
        e_total = jnp.ones((1, C, 1)) * 500.0
        a  = jnp.ones((T, 1, G)) * 1e-5
        Z0 = jnp.ones((1, 1, G)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert e_free.shape == (T, C, G)

    def test_finite_values(self):
        """Solver should produce finite values for typical parameters."""
        e_total = jnp.array([0.0, 100.0, 1000.0])[None, :, None]
        a  = jnp.ones((1, 1, 1)) * 1e-4
        Z0 = jnp.ones((1, 1, 1)) * 252.0
        e_free = _solve_free_effector(e_total, tf_total=650.0, a=a, Z0=Z0)
        assert jnp.all(jnp.isfinite(e_free))


# ---------------------------------------------------------------------------
# _compute_theta
# ---------------------------------------------------------------------------

class TestComputeTheta:

    def test_output_shape(self):
        T, G = 2, 4
        conc = jnp.array([0.0, 100.0, 1000.0])
        ln_K_op = jnp.zeros(G)
        ln_K_HL = jnp.full(G, -9.0)
        ln_K_E  = jnp.full((T, G), -4.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, conc, 650.0, 25.0)
        assert theta.shape == (T, 3, G)

    def test_values_in_0_1(self):
        G = 4
        conc = jnp.array([0.0, 50.0, 500.0, 5000.0])
        ln_K_op = jnp.zeros(G)
        ln_K_HL = jnp.full(G, -9.0)
        ln_K_E  = jnp.full((1, G), -4.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, conc, 650.0, 25.0)
        assert jnp.all(theta >= 0)
        assert jnp.all(theta <= 1)

    def test_all_finite(self):
        G = 4
        conc = jnp.linspace(0, 1000, 6)
        ln_K_op = jnp.linspace(0.0, 4.0, G)
        ln_K_HL = jnp.linspace(-12.0, -6.0, G)
        ln_K_E  = jnp.full((2, G), -5.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, conc, 650.0, 25.0)
        assert jnp.all(jnp.isfinite(theta))

    def test_theta_monotone_in_effector(self):
        """Higher effector → lower theta (effector competes with operator binding)."""
        G = 2
        conc = jnp.array([0.0, 100.0, 1000.0])
        ln_K_op = jnp.full(G, 2.3)
        ln_K_HL = jnp.full(G, -9.0)
        ln_K_E  = jnp.full((1, G), -4.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, conc, 650.0, 25.0)
        # theta[:, 0, :] >= theta[:, 1, :] >= theta[:, 2, :]
        assert jnp.all(theta[:, 0, :] >= theta[:, 1, :])
        assert jnp.all(theta[:, 1, :] >= theta[:, 2, :])

    def test_zero_effector_equals_apo_formula(self):
        """At e_total=0, theta = K_op*[op] / (1 + K_op*[op] + K_HL)."""
        G = 2
        conc = jnp.array([0.0, 1.0])
        ln_K_op = jnp.array([2.3, 2.0])
        ln_K_HL = jnp.array([-9.0, -8.0])
        ln_K_E  = jnp.full((1, G), -4.0)
        theta = _compute_theta(ln_K_op, ln_K_HL, ln_K_E, conc, 650.0, 25.0)

        K_op = np.exp(np.array([2.3, 2.0]))
        K_HL = np.exp(np.array([-9.0, -8.0]))
        op_total = 25.0
        expected = K_op * op_total / (1.0 + K_op * op_total + K_HL)
        assert jnp.allclose(theta[0, 0, :], jnp.array(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# _population_moments
# ---------------------------------------------------------------------------

class TestPopulationMoments:

    def test_output_shape(self, mock_data_no_epi):
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.7)
        mu, sigma = _population_moments(theta, mock_data_no_epi)
        assert mu.shape == (T, C, 1)
        assert sigma.shape == (T, C, 1)

    def test_sigma_zero_when_uniform(self, mock_data_no_epi):
        """All-same theta → sigma = 0."""
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.5)
        _, sigma = _population_moments(theta, mock_data_no_epi)
        assert jnp.allclose(sigma, 0.0, atol=1e-6)

    def test_sigma_positive_when_varying(self, mock_data_no_epi):
        T, C, G = 2, 3, 4
        theta = jnp.linspace(0.1, 0.9, T * C * G).reshape(T, C, G)
        _, sigma = _population_moments(theta, mock_data_no_epi)
        assert jnp.all(sigma > 0)

    def test_mu_finite(self, mock_data_no_epi):
        T, C, G = 2, 3, 4
        theta = jnp.full((T, C, G), 0.8)
        mu, _ = _population_moments(theta, mock_data_no_epi)
        assert jnp.all(jnp.isfinite(mu))


# ---------------------------------------------------------------------------
# _assemble_scalar
# ---------------------------------------------------------------------------

class TestAssembleScalar:

    def test_no_epistasis_wt_only(self):
        """With zero offsets, all genotypes equal wt."""
        G, M = 4, 2
        wt = jnp.array(2.3)
        d_offsets = jnp.zeros(M)
        sigma_d = jnp.array(1.0)
        M_mat = jnp.array(_MUT_GENO)
        result = _assemble_scalar(wt, d_offsets, sigma_d, M_mat)
        assert result.shape == (G,)
        assert jnp.allclose(result, wt)

    def test_no_epistasis_mutation_effect(self):
        """d[m] * sigma_d should shift only genotypes carrying mutation m."""
        G, M = 4, 2
        wt = jnp.array(0.0)
        # mutation 0 has effect 1.0, mutation 1 has effect 0.0
        d_offsets = jnp.array([1.0, 0.0])
        sigma_d = jnp.array(1.0)
        M_mat = jnp.array(_MUT_GENO)
        result = _assemble_scalar(wt, d_offsets, sigma_d, M_mat)
        # Genotypes with M42I (cols 1, 3) get +1; others stay 0
        expected = jnp.array([0.0, 1.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_with_epistasis_only_double_mut(self):
        """Epistasis term should shift only the double-mutant (column 3)."""
        G, M, P = 4, 2, 1
        wt = jnp.array(0.0)
        d_offsets = jnp.zeros(M)
        sigma_d = jnp.array(1.0)
        M_mat = jnp.array(_MUT_GENO)
        epi_offsets = jnp.array([2.0])
        sigma_epi = jnp.array(1.0)
        pair_scatter = _make_pair_scatter()
        result = _assemble_scalar(wt, d_offsets, sigma_d, M_mat,
                                  epi_offsets, sigma_epi, pair_scatter)
        assert jnp.allclose(result[:3], 0.0, atol=1e-6)
        assert jnp.allclose(result[3], 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _assemble_titrant
# ---------------------------------------------------------------------------

class TestAssembleTitrant:

    def test_no_epistasis_shape(self):
        T, G, M = 2, 4, 2
        wt = jnp.zeros(T)
        d_offsets = jnp.ones((T, M))
        sigma_d = jnp.ones(T)
        result = _assemble_titrant(wt, d_offsets, sigma_d, jnp.array(_MUT_GENO))
        assert result.shape == (T, G)

    def test_no_epistasis_wt_only(self):
        T, G, M = 2, 4, 2
        wt = jnp.array([1.0, 2.0])
        d_offsets = jnp.zeros((T, M))
        sigma_d = jnp.ones(T)
        result = _assemble_titrant(wt, d_offsets, sigma_d, jnp.array(_MUT_GENO))
        expected = jnp.array([[1.0] * G, [2.0] * G])
        assert jnp.allclose(result, expected)

    def test_with_epistasis(self):
        T, G, M, P = 1, 4, 2, 1
        wt = jnp.zeros(T)
        d_offsets = jnp.zeros((T, M))
        sigma_d = jnp.ones(T)
        M_mat = jnp.array(_MUT_GENO)
        epi_offsets = jnp.array([[3.0]])   # (T, P)
        sigma_epi = jnp.ones(T)
        pair_scatter = _make_pair_scatter()
        result = _assemble_titrant(wt, d_offsets, sigma_d, M_mat,
                                   epi_offsets, sigma_epi, pair_scatter)
        assert jnp.allclose(result[0, :3], 0.0, atol=1e-6)
        assert jnp.allclose(result[0, 3], 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    hp = get_hyperparameters()
    assert isinstance(hp, dict)
    assert "theta_ln_K_op_wt_loc" in hp
    assert "theta_ln_K_E_wt_scale" in hp
    assert "theta_tf_total_nM" in hp
    assert "theta_op_total_nM" in hp
    assert "theta_sigma_d_ln_K_op_scale" in hp
    assert "theta_epi_tau_scale" in hp
    assert "theta_epi_slab_scale" in hp
    assert "theta_epi_slab_df" in hp


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.theta_tf_total_nM == pytest.approx(650.0)
    assert priors.theta_op_total_nM == pytest.approx(25.0)
    assert priors.theta_ln_K_op_wt_loc == get_hyperparameters()["theta_ln_K_op_wt_loc"]


def test_get_guesses_no_epi(mock_data_no_epi):
    name = "theta"
    guesses = get_guesses(name, mock_data_no_epi)
    T, M = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_mutation
    assert guesses[f"{name}_ln_K_op_wt"].shape == ()
    assert guesses[f"{name}_ln_K_HL_wt"].shape == ()
    assert guesses[f"{name}_ln_K_E_wt"].shape == (T,)
    assert guesses[f"{name}_d_ln_K_op_offset"].shape == (M,)
    assert guesses[f"{name}_d_ln_K_HL_offset"].shape == (M,)
    assert guesses[f"{name}_d_ln_K_E_offset"].shape == (T, M)
    # No epistasis keys
    assert f"{name}_epi_ln_K_op_offset" not in guesses


def test_get_guesses_with_epi(mock_data_epi):
    name = "theta"
    guesses = get_guesses(name, mock_data_epi)
    T, P = mock_data_epi.num_titrant_name, mock_data_epi.num_pair
    assert guesses[f"{name}_epi_ln_K_op_offset"].shape == (P,)
    assert guesses[f"{name}_epi_ln_K_HL_offset"].shape == (P,)
    assert guesses[f"{name}_epi_ln_K_E_offset"].shape == (T, P)
    assert guesses[f"{name}_epi_ln_K_op_lambda"].shape == (P,)
    assert guesses[f"{name}_epi_ln_K_HL_lambda"].shape == (P,)
    assert guesses[f"{name}_epi_ln_K_E_lambda"].shape == (T, P)
    assert guesses[f"{name}_epi_tau"].shape == ()
    assert guesses[f"{name}_epi_c2"].shape == ()


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

    def test_parameter_shapes(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        T = mock_data_no_epi.num_titrant_name
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_HL.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)

    def test_population_moments_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
        assert tp.mu.shape == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)
        assert jnp.all(tp.sigma >= 0)

    def test_wt_column_equals_others_when_deltas_zero(self, mock_data_no_epi):
        """With zero mutation offsets all genotypes share the same ln_K_op."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        wt_val = tp.ln_K_op[0]
        for g in range(mock_data_no_epi.num_genotype):
            assert jnp.allclose(tp.ln_K_op[g], wt_val, atol=1e-5)

    def test_assembly_with_nonzero_deltas(self, mock_data_no_epi):
        """Matrix-multiply assembly produces expected per-genotype ln_K_op."""
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_no_epi)
        guesses[f"{name}_ln_K_op_wt"] = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_op"] = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_op_offset"] = jnp.array([1.0, -0.5])

        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_no_epi, priors=priors)

        M_mat = jnp.array(_MUT_GENO)
        d = guesses[f"{name}_d_ln_K_op_offset"]
        expected = d @ M_mat   # wt=0, sigma=1
        assert jnp.allclose(tp.ln_K_op, expected, atol=1e-5)

    def test_sample_sites_in_trace(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_ln_K_op_wt" in sample_names
        assert "theta_ln_K_HL_wt" in sample_names
        assert "theta_ln_K_E_wt" in sample_names
        assert "theta_d_ln_K_op_offset" in sample_names
        assert "theta_d_ln_K_E_offset" in sample_names
        assert not any("epi" in k for k in sample_names)

    def test_deterministic_sites_registered(self, mock_data_no_epi):
        """ln_K_op, d_ln_K_op etc. should appear as deterministic sites."""
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_no_epi, priors=priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        assert "theta_ln_K_op" in det_names
        assert "theta_ln_K_HL" in det_names
        assert "theta_ln_K_E" in det_names
        assert "theta_d_ln_K_op" in det_names
        assert "theta_d_ln_K_HL" in det_names
        assert "theta_d_ln_K_E" in det_names


# ---------------------------------------------------------------------------
# define_model – with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_output_shapes(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        T = mock_data_epi.num_titrant_name
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)

    def test_epistasis_shifts_only_double_column(self, mock_data_epi):
        """With zero mut deltas but non-zero epistasis, only column 3 differs.
        c2→∞ makes lambda_tilde ≈ lambda = 1, so epi = offset * tau * 1 = 2.0."""
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_epi)
        guesses[f"{name}_ln_K_op_wt"] = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_op"] = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_op_offset"] = jnp.zeros(2)
        guesses[f"{name}_epi_tau"] = jnp.array(1.0)
        guesses[f"{name}_epi_c2"] = jnp.array(1e12)   # effectively no slab
        guesses[f"{name}_epi_ln_K_op_lambda"] = jnp.ones(1)
        guesses[f"{name}_epi_ln_K_op_offset"] = jnp.array([2.0])

        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_epi, priors=priors)

        assert jnp.allclose(tp.ln_K_op[:3], 0.0, atol=1e-5)
        assert jnp.allclose(tp.ln_K_op[3], 2.0, atol=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_epi_tau" in sample_names
        assert "theta_epi_c2" in sample_names
        assert "theta_epi_ln_K_op_lambda" in sample_names
        assert "theta_epi_ln_K_op_offset" in sample_names
        assert "theta_epi_ln_K_HL_offset" in sample_names
        assert "theta_epi_ln_K_E_offset" in sample_names

    def test_epi_deterministic_sites_registered(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="theta", data=mock_data_epi, priors=priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        assert "theta_epi_ln_K_op" in det_names
        assert "theta_epi_ln_K_HL" in det_names
        assert "theta_epi_ln_K_E" in det_names


# ---------------------------------------------------------------------------
# run_model
# ---------------------------------------------------------------------------

class TestRunModel:

    @pytest.fixture
    def theta_param(self, mock_data_no_epi):
        T, G = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_genotype
        return ThetaParam(
            ln_K_op=jnp.full(G, 2.3),
            ln_K_HL=jnp.full(G, -9.0),
            ln_K_E=jnp.full((T, G), -4.0),
            tf_total=650.0,
            op_total=25.0,
            mu=jnp.zeros((T, 3, 1)),
            sigma=jnp.ones((T, 3, 1)),
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

    def test_values_in_0_1(self, theta_param, mock_data_no_epi):
        data = mock_data_no_epi._replace(scatter_theta=0)
        result = run_model(theta_param, data)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_all_finite(self, theta_param, mock_data_no_epi):
        data = mock_data_no_epi._replace(scatter_theta=0)
        result = run_model(theta_param, data)
        assert jnp.all(jnp.isfinite(result))

    def test_zero_effector_matches_apo_formula(self, mock_data_no_epi):
        """At conc=0, theta should match the analytic apo formula."""
        G = mock_data_no_epi.num_genotype
        T = mock_data_no_epi.num_titrant_name
        ln_K_op_val = 2.3
        ln_K_HL_val = -9.0
        tp = ThetaParam(
            ln_K_op=jnp.full(G, ln_K_op_val),
            ln_K_HL=jnp.full(G, ln_K_HL_val),
            ln_K_E=jnp.full((T, G), -4.0),
            tf_total=650.0,
            op_total=25.0,
            mu=jnp.zeros((T, 3, 1)),
            sigma=jnp.ones((T, 3, 1)),
        )
        data = mock_data_no_epi._replace(
            titrant_conc=jnp.array([0.0, 1.0, 2.0]),
            scatter_theta=0)
        result = run_model(tp, data)
        K_op = np.exp(ln_K_op_val)
        K_HL = np.exp(ln_K_HL_val)
        op_total = 25.0
        expected = K_op * op_total / (1.0 + K_op * op_total + K_HL)
        assert jnp.allclose(result[:, 0, :], expected, atol=1e-5)

    def test_uses_data_titrant_conc(self, theta_param):
        """run_model uses data.titrant_conc, so binding data with different
        concentrations produces different (valid) results."""
        MockDataSmall = namedtuple("MockDataSmall", [
            "titrant_conc", "geno_theta_idx", "scatter_theta", "batch_idx"])
        G = 4
        data_a = MockDataSmall(
            titrant_conc=jnp.array([0.0, 100.0]),
            geno_theta_idx=jnp.arange(G, dtype=jnp.int32),
            scatter_theta=0,
            batch_idx=jnp.arange(G, dtype=jnp.int32))
        data_b = MockDataSmall(
            titrant_conc=jnp.array([500.0, 1000.0]),
            geno_theta_idx=jnp.arange(G, dtype=jnp.int32),
            scatter_theta=0,
            batch_idx=jnp.arange(G, dtype=jnp.int32))
        res_a = run_model(theta_param, data_a)
        res_b = run_model(theta_param, data_b)
        # Higher concentrations → lower theta
        assert jnp.all(res_a >= res_b)


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_no_epi_returns_theta_param(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        assert isinstance(tp, ThetaParam)

    def test_no_epi_shapes(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        T = mock_data_no_epi.num_titrant_name
        C = mock_data_no_epi.num_titrant_conc
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)
        assert tp.mu.shape == (T, C, 1)

    def test_with_epi_runs(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tp = guide(name="theta", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        T = mock_data_epi.num_titrant_name
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)

    def test_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_tf_op_totals_preserved(self, mock_data_no_epi):
        """tf_total and op_total from priors should be stored in ThetaParam."""
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        assert tp.tf_total == pytest.approx(650.0)
        assert tp.op_total == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# get_population_moments
# ---------------------------------------------------------------------------

def test_get_population_moments_shape(mock_data_no_epi):
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    # Use non-zero offsets so genotypes differ and sigma > 0
    guesses["theta_d_ln_K_op_offset"] = jnp.array([1.0, -0.5])
    tp = substitute(define_model, data=guesses)(
        name="theta", data=mock_data_no_epi, priors=priors)
    mu, sigma = get_population_moments(tp, mock_data_no_epi)
    T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
    assert mu.shape == (T, C, 1)
    assert sigma.shape == (T, C, 1)
    assert jnp.all(sigma >= 0)
