import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.dk_geno.hierarchical_mut import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MockGrowthData = namedtuple("MockGrowthData", [
    "num_genotype",
    "num_mutation",
    "num_pair",
    "mut_geno_matrix",
    "pair_nnz_pair_idx",
    "pair_nnz_geno_idx",
    "batch_size",
    "batch_idx",
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
_MUT_GENO = np.array([[0, 1, 0, 1],   # M42I
                       [0, 0, 1, 1]], dtype=np.float32)   # K84L
# COO representation of [[0, 0, 0, 1]]: one nonzero at (pair=0, geno=3)
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)


@pytest.fixture
def mock_data_epi():
    return MockGrowthData(
        num_genotype=4,
        num_mutation=2,
        num_pair=1,
        mut_geno_matrix=_MUT_GENO,
        pair_nnz_pair_idx=_PAIR_NNZ_PAIR,
        pair_nnz_geno_idx=_PAIR_NNZ_GENO,
        batch_size=4,
        batch_idx=np.arange(4, dtype=np.int32),
    )


@pytest.fixture
def mock_data_no_epi():
    return MockGrowthData(
        num_genotype=4,
        num_mutation=2,
        num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
        batch_size=4,
        batch_idx=np.arange(4, dtype=np.int32),
    )


# Neutral offset: the specific value that gives shifted-lognormal ≈ 0.
_NEUTRAL_OFFSET = -0.8240460108562919


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "hyper_loc_loc" in params
    assert "hyper_shift_loc" in params
    assert "sigma_epi_tau_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.hyper_loc_loc == hp["hyper_loc_loc"]
    assert priors.hyper_shift_loc == hp["hyper_shift_loc"]


def test_get_guesses_no_epi(mock_data_no_epi):
    name = "dk_geno"
    guesses = get_guesses(name, mock_data_no_epi)
    assert isinstance(guesses, dict)
    assert f"{name}_hyper_loc" in guesses
    assert f"{name}_hyper_shift" in guesses
    assert f"{name}_offset" in guesses
    assert guesses[f"{name}_offset"].shape == (mock_data_no_epi.num_mutation,)
    # Every offset initialised to the neutral value
    assert jnp.allclose(guesses[f"{name}_offset"],
                        jnp.full(mock_data_no_epi.num_mutation, _NEUTRAL_OFFSET))
    assert f"{name}_epi_offset" not in guesses


def test_get_guesses_with_epi(mock_data_epi):
    name = "dk_geno"
    guesses = get_guesses(name, mock_data_epi)
    assert f"{name}_epi_offset" in guesses
    assert guesses[f"{name}_epi_offset"].shape == (mock_data_epi.num_pair,)
    assert jnp.all(guesses[f"{name}_epi_offset"] == 0.0)
    assert f"{name}_epi_tau" in guesses
    assert f"{name}_epi_c2" in guesses
    assert f"{name}_epi_lambda" in guesses
    assert guesses[f"{name}_epi_lambda"].shape == (mock_data_epi.num_pair,)


# ---------------------------------------------------------------------------
# define_model – no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_output_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_wt_dk_geno_is_zero(self, mock_data_no_epi):
        """WT column of M is all-zero, so dk_geno[wt] = 0 always."""
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert flat[0] == pytest.approx(0.0, abs=1e-6)

    def test_neutral_offset_gives_dk_geno_near_zero(self, mock_data_no_epi):
        """
        The neutral offset is chosen so shift - exp(loc + offset*scale) ≈ 0.
        With the default guesses, all per-mutation dk_geno values should be ~0,
        hence all assembled genotype values should also be ~0.
        """
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert jnp.allclose(flat, 0.0, atol=1e-5)

    def test_assembly_additive(self, mock_data_no_epi):
        """
        With known hyperparams and offsets, verify that the double-mutant
        dk_geno is the sum of its two single-mutant contributions.
        """
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)

        # Fix to simple values so we can compute by hand
        guesses["dk_geno_hyper_loc"] = -3.5
        guesses["dk_geno_hyper_scale"] = 0.5
        guesses["dk_geno_hyper_shift"] = 0.02
        # Use different offsets per mutation to make them distinguishable
        guesses["dk_geno_offset"] = jnp.array([0.0, 1.0])

        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        # Compute expected per-mutation dk values
        def dk(offset):
            return 0.02 - float(jnp.clip(jnp.exp(-3.5 + offset * 0.5), max=1e30))

        expected_M42I = dk(0.0)
        expected_K84L = dk(1.0)

        # wt = 0
        assert flat[0] == pytest.approx(0.0, abs=1e-5)
        # singles match their respective dk
        assert flat[1] == pytest.approx(expected_M42I, rel=1e-5)
        assert flat[2] == pytest.approx(expected_K84L, rel=1e-5)
        # double = sum of singles (additive, no epistasis)
        assert flat[3] == pytest.approx(expected_M42I + expected_K84L, rel=1e-5)

    def test_deterministic_site_in_trace(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        assert "dk_geno" in tr
        assert tr["dk_geno"]["type"] == "deterministic"

    def test_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)


# ---------------------------------------------------------------------------
# define_model – with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_output_shape(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_epistasis_shifts_only_double_column(self, mock_data_epi):
        """
        With neutral mutation offsets (dk ≈ 0) but non-zero epistasis offset,
        only the double-mutant column should be non-zero.
        c2→∞ makes lambda_tilde ≈ lambda = 1, so epi = offset * tau * 1 = 0.5.
        """
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        guesses["dk_geno_epi_tau"] = 1.0
        guesses["dk_geno_epi_c2"] = 1e12   # effectively no slab regularisation
        guesses["dk_geno_epi_lambda"] = jnp.ones(1)
        guesses["dk_geno_epi_offset"] = jnp.array([0.5])

        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        # wt, M42I, K84L should all be ~0 (neutral offset)
        assert jnp.allclose(flat[:3], 0.0, atol=1e-5)
        # M42I/K84L = 0 (mut contribution) + 0.5 * 1 * 1 (epistasis)
        assert flat[3] == pytest.approx(0.5, rel=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        model = substitute(define_model, data=guesses)
        tr = trace(model).get_trace(
            name="dk_geno", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "dk_geno_epi_offset" in sample_names
        assert "dk_geno_epi_tau" in sample_names
        assert "dk_geno_epi_c2" in sample_names
        assert "dk_geno_epi_lambda" in sample_names

    def test_wt_still_zero_with_epistasis(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        guesses["dk_geno_epi_tau"] = 1.0
        guesses["dk_geno_epi_c2"] = 1e12
        guesses["dk_geno_epi_lambda"] = jnp.ones(1)
        guesses["dk_geno_epi_offset"] = jnp.array([1.0])
        result = substitute(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert flat[0] == pytest.approx(0.0, abs=1e-5)   # WT is always 0


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_guide_no_epi_shape(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            result = guide(name="dk_geno", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_guide_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_guide_with_epi_shape(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            result = guide(name="dk_geno", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_guide_with_epi_sample_sites(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tr = trace(guide).get_trace(
                name="dk_geno", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "dk_geno_epi_offset" in sample_names
        assert "dk_geno_epi_tau" in sample_names
        assert "dk_geno_epi_c2" in sample_names
        assert "dk_geno_epi_lambda" in sample_names

    def test_guide_all_hyperprior_sites_sampled(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "dk_geno_hyper_loc" in sample_names
        assert "dk_geno_hyper_scale" in sample_names
        assert "dk_geno_hyper_shift" in sample_names
        assert "dk_geno_offset" in sample_names
