import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from functools import partial
from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.lnK_mut import (
    ModelPriors,
    _assemble_scalar,
    _assemble_titrant,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
)
from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import (
    ThetaParam,
    run_model,
    get_population_moments,
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
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
_MUT_GENO  = np.array([[0, 1, 0, 1],
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

_CONC     = np.array([0.0, 100.0, 1000.0])
_LOG_CONC = np.log(np.where(_CONC == 0, 1e-20, _CONC))


@pytest.fixture
def mock_data_epi():
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
    )


@pytest.fixture
def mock_data_no_epi():
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
    )


# ---------------------------------------------------------------------------
# _assemble_scalar
# ---------------------------------------------------------------------------

class TestAssembleScalar:

    def test_zero_offsets_all_genotypes_equal_wt(self):
        G, M = 4, 2
        result = _assemble_scalar(jnp.array(2.3), jnp.zeros(M), jnp.array(1.0),
                                  jnp.array(_MUT_GENO))
        assert result.shape == (G,)
        assert jnp.allclose(result, 2.3)

    def test_mutation_effect_shifts_correct_genotypes(self):
        """d[m0]=1.0 should shift only genotypes carrying mutation 0 (cols 1, 3)."""
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.0]), jnp.array(1.0),
                                  jnp.array(_MUT_GENO))
        assert jnp.allclose(result, jnp.array([0.0, 1.0, 0.0, 1.0]))

    def test_sigma_scales_effect(self):
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.0]), jnp.array(2.0),
                                  jnp.array(_MUT_GENO))
        assert jnp.allclose(result, jnp.array([0.0, 2.0, 0.0, 2.0]))

    def test_epistasis_shifts_only_double_mutant(self):
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.zeros(2), jnp.array(1.0),
                                  jnp.array(_MUT_GENO),
                                  jnp.array([2.0]), jnp.array(1.0),
                                  _make_pair_scatter())
        assert jnp.allclose(result[:3], 0.0, atol=1e-6)
        assert jnp.allclose(result[3], 2.0, atol=1e-6)

    def test_additive_mut_plus_epistasis(self):
        result = _assemble_scalar(jnp.array(0.0),
                                  jnp.array([1.0, 0.5]), jnp.array(1.0),
                                  jnp.array(_MUT_GENO),
                                  jnp.array([3.0]), jnp.array(1.0),
                                  _make_pair_scatter())
        # col 3: 1.0 (M42I) + 0.5 (K84L) + 3.0 (epi) = 4.5
        assert jnp.allclose(result[3], 4.5, atol=1e-5)


# ---------------------------------------------------------------------------
# _assemble_titrant
# ---------------------------------------------------------------------------

class TestAssembleTitrant:

    def test_output_shape(self):
        T, G, M = 2, 4, 2
        result = _assemble_titrant(jnp.zeros(T), jnp.ones((T, M)), jnp.ones(T),
                                   jnp.array(_MUT_GENO))
        assert result.shape == (T, G)

    def test_zero_offsets_all_equal_wt(self):
        T = 2
        wt = jnp.array([1.0, 2.0])
        result = _assemble_titrant(wt, jnp.zeros((T, 2)), jnp.ones(T),
                                   jnp.array(_MUT_GENO))
        assert jnp.allclose(result, wt[:, None])

    def test_sigma_scales_per_titrant(self):
        T, M = 2, 2
        wt = jnp.zeros(T)
        d = jnp.ones((T, M))
        sigma = jnp.array([1.0, 2.0])
        result = _assemble_titrant(wt, d, sigma, jnp.array(_MUT_GENO))
        # col 3 (both mutations): T=0 → 1*1+1*1=2; T=1 → 2*1+2*1=4
        assert jnp.allclose(result[0, 3], 2.0)
        assert jnp.allclose(result[1, 3], 4.0)

    def test_epistasis_shifts_only_double_mutant(self):
        T = 1
        result = _assemble_titrant(jnp.zeros(T), jnp.zeros((T, 2)), jnp.ones(T),
                                   jnp.array(_MUT_GENO),
                                   jnp.array([[3.0]]), jnp.ones(T),
                                   _make_pair_scatter())
        assert jnp.allclose(result[0, :3], 0.0, atol=1e-6)
        assert jnp.allclose(result[0, 3], 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters_keys():
    hp = get_hyperparameters()
    assert isinstance(hp, dict)
    for key in ["theta_ln_K_op_wt_loc", "theta_ln_K_HL_wt_loc", "theta_ln_K_E_wt_loc",
                "theta_tf_total_M", "theta_op_total_M",
                "theta_sigma_d_ln_K_op_scale",
                "theta_epi_tau_scale", "theta_epi_slab_scale", "theta_epi_slab_df"]:
        assert key in hp


def test_get_priors_values():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.theta_tf_total_M == pytest.approx(6.5e-7)
    assert priors.theta_op_total_M == pytest.approx(2.5e-8)
    assert priors.theta_ln_K_op_wt_loc == get_hyperparameters()["theta_ln_K_op_wt_loc"]


def test_get_guesses_shapes_no_epi(mock_data_no_epi):
    name = "theta"
    guesses = get_guesses(name, mock_data_no_epi)
    T, M = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_mutation
    assert guesses[f"{name}_ln_K_op_wt"].shape == ()
    assert guesses[f"{name}_ln_K_HL_wt"].shape == ()
    assert guesses[f"{name}_ln_K_E_wt"].shape == (T,)
    assert guesses[f"{name}_d_ln_K_op_offset"].shape == (M,)
    assert guesses[f"{name}_d_ln_K_E_offset"].shape == (T, M)
    assert f"{name}_epi_ln_K_op_offset" not in guesses


def test_get_guesses_shapes_with_epi(mock_data_epi):
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
# define_model — no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_returns_theta_param(self, mock_data_no_epi):
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
        C = mock_data_no_epi.num_titrant_conc
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_HL.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)
        assert tp.mu.shape == (T, C, 1)
        assert tp.sigma.shape == (T, C, 1)

    def test_all_genotypes_equal_when_deltas_zero(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        assert jnp.allclose(tp.ln_K_op, tp.ln_K_op[0], atol=1e-5)

    def test_assembly_with_known_deltas(self, mock_data_no_epi):
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_no_epi)
        guesses[f"{name}_ln_K_op_wt"]        = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_op"]   = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_op_offset"]  = jnp.array([1.0, -0.5])
        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_no_epi, priors=priors)
        expected = jnp.array([1.0, -0.5]) @ jnp.array(_MUT_GENO)
        assert jnp.allclose(tp.ln_K_op, expected, atol=1e-5)

    def test_population_moments_sigma_nonneg(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_no_epi, priors=priors)
        assert jnp.all(tp.sigma >= 0)

    def test_sample_sites_present(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_ln_K_op_wt" in sample_names
        assert "theta_ln_K_HL_wt" in sample_names
        assert "theta_ln_K_E_wt" in sample_names
        assert "theta_d_ln_K_op_offset" in sample_names
        assert "theta_d_ln_K_E_offset" in sample_names
        assert not any("epi" in k for k in sample_names)

    def test_deterministic_sites_registered(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="theta", data=mock_data_no_epi, priors=priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        for site in ["theta_ln_K_op", "theta_ln_K_HL", "theta_ln_K_E",
                     "theta_d_ln_K_op", "theta_d_ln_K_HL", "theta_d_ln_K_E"]:
            assert site in det_names


# ---------------------------------------------------------------------------
# define_model — with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_parameter_shapes(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tp = substitute(define_model, data=guesses)(
            name="theta", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        T = mock_data_epi.num_titrant_name
        assert tp.ln_K_op.shape == (G,)
        assert tp.ln_K_E.shape == (T, G)

    def test_epistasis_shifts_only_double_mutant(self, mock_data_epi):
        """With zero mut deltas but non-zero epistasis, only column 3 differs.
        c2→∞ makes lambda_tilde ≈ lambda = 1, so epi = offset * tau * 1 = 2.0."""
        name = "theta"
        priors = get_priors()
        guesses = get_guesses(name, mock_data_epi)
        guesses[f"{name}_ln_K_op_wt"]          = jnp.array(0.0)
        guesses[f"{name}_sigma_d_ln_K_op"]     = jnp.array(1.0)
        guesses[f"{name}_d_ln_K_op_offset"]    = jnp.zeros(2)
        guesses[f"{name}_epi_tau"]             = jnp.array(1.0)
        guesses[f"{name}_epi_c2"]             = jnp.array(1e12)   # effectively no slab
        guesses[f"{name}_epi_ln_K_op_lambda"]  = jnp.ones(1)
        guesses[f"{name}_epi_ln_K_op_offset"]  = jnp.array([2.0])
        tp = substitute(define_model, data=guesses)(
            name=name, data=mock_data_epi, priors=priors)
        assert jnp.allclose(tp.ln_K_op[:3], 0.0, atol=1e-5)
        assert jnp.allclose(tp.ln_K_op[3], 2.0, atol=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="theta", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "theta_epi_tau"              in sample_names
        assert "theta_epi_c2"              in sample_names
        assert "theta_epi_ln_K_op_lambda"  in sample_names
        assert "theta_epi_ln_K_op_offset"  in sample_names
        assert "theta_epi_ln_K_HL_offset"  in sample_names
        assert "theta_epi_ln_K_E_offset"   in sample_names

    def test_epistasis_deterministic_sites_registered(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("theta", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="theta", data=mock_data_epi, priors=priors)
        det_names = {k for k, v in tr.items() if v["type"] == "deterministic"}
        assert "theta_epi_ln_K_op" in det_names
        assert "theta_epi_ln_K_HL" in det_names
        assert "theta_epi_ln_K_E"  in det_names


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
        assert tp.ln_K_op.shape == (mock_data_epi.num_genotype,)

    def test_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="theta", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_tf_op_totals_preserved(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tp = guide(name="theta", data=mock_data_no_epi, priors=priors)
        assert tp.tf_total == pytest.approx(6.5e-7)
        assert tp.op_total == pytest.approx(2.5e-8)


# ---------------------------------------------------------------------------
# run_model and get_population_moments (imported from thermo, used via lnK_additive)
# ---------------------------------------------------------------------------

def test_run_model_uses_assembled_K(mock_data_no_epi):
    """run_model imported from thermo produces valid theta from lnK_additive output."""
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    tp = substitute(define_model, data=guesses)(
        name="theta", data=mock_data_no_epi, priors=priors)
    data = mock_data_no_epi._replace(scatter_theta=0)
    result = run_model(tp, data)
    assert result.shape == (mock_data_no_epi.num_titrant_name,
                            mock_data_no_epi.num_titrant_conc,
                            mock_data_no_epi.num_genotype)
    assert jnp.all(result >= 0)
    assert jnp.all(result <= 1)


def test_get_population_moments_shape(mock_data_no_epi):
    priors = get_priors()
    guesses = get_guesses("theta", mock_data_no_epi)
    guesses["theta_d_ln_K_op_offset"] = jnp.array([1.0, -0.5])
    tp = substitute(define_model, data=guesses)(
        name="theta", data=mock_data_no_epi, priors=priors)
    mu, sigma = get_population_moments(tp, mock_data_no_epi)
    T, C = mock_data_no_epi.num_titrant_name, mock_data_no_epi.num_titrant_conc
    assert mu.shape == (T, C, 1)
    assert sigma.shape == (T, C, 1)
    assert jnp.all(sigma >= 0)
