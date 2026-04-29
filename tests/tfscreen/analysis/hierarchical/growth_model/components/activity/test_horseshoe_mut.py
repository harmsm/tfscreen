import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.activity.horseshoe_mut import (
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
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
_MUT_GENO = np.array([[0, 1, 0, 1],   # M42I
                       [0, 0, 1, 1]], dtype=np.float32)   # K84L
# COO: one nonzero at (pair=0, geno=3)
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
    )


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    for key in ("activity_d_tau_scale", "activity_d_slab_scale", "activity_d_slab_df",
                "activity_epi_tau_scale", "activity_epi_slab_scale", "activity_epi_slab_df"):
        assert key in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.activity_d_tau_scale == hp["activity_d_tau_scale"]
    assert priors.activity_epi_tau_scale == hp["activity_epi_tau_scale"]


def test_get_guesses_no_epi(mock_data_no_epi):
    name = "activity"
    guesses = get_guesses(name, mock_data_no_epi)
    assert isinstance(guesses, dict)
    assert f"{name}_d_offset" in guesses
    assert guesses[f"{name}_d_offset"].shape == (mock_data_no_epi.num_mutation,)
    assert jnp.all(guesses[f"{name}_d_offset"] == 0.0)
    assert f"{name}_epi_offset" not in guesses


def test_get_guesses_with_epi(mock_data_epi):
    name = "activity"
    guesses = get_guesses(name, mock_data_epi)
    assert f"{name}_epi_offset" in guesses
    assert guesses[f"{name}_epi_offset"].shape == (mock_data_epi.num_pair,)
    assert f"{name}_epi_lambda" in guesses
    assert f"{name}_epi_tau" in guesses
    assert f"{name}_epi_c2" in guesses


# ---------------------------------------------------------------------------
# define_model – no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_output_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)
        assert result.shape == (1, 1, 1, 1, 1, 1, mock_data_no_epi.num_genotype)

    def test_all_activity_one_with_zero_offsets(self, mock_data_no_epi):
        """Zero d_offset → d_log_activity = 0 → activity = 1 for every genotype."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)
        assert jnp.allclose(result, 1.0, atol=1e-6)

    def test_wt_activity_is_one_with_nonzero_deltas(self, mock_data_no_epi):
        """WT column of M is all-zero, so WT always gets activity = 1 regardless of deltas."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        guesses["activity_d_tau"] = 1.0
        guesses["activity_d_lambda"] = jnp.ones(2)
        guesses["activity_d_offset"] = jnp.array([2.0, -1.5])
        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert flat[0] == pytest.approx(1.0, rel=1e-5)

    def test_assembly_correct_for_known_deltas(self, mock_data_no_epi):
        """
        With tau=1, lambda=[1,1], c2→∞ (lambda_tilde≈lambda=1), d_offset=[1.0, -0.5]:
          d_log_activity = [1.0, -0.5]
          M42I      → log_a =  1.0   → activity = exp(1.0)
          K84L      → log_a = -0.5   → activity = exp(-0.5)
          M42I/K84L → log_a =  0.5   → activity = exp(0.5)
        We set c2 very large so lambda_tilde ≈ lambda.
        """
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        guesses["activity_d_tau"] = 1.0
        guesses["activity_d_c2"] = 1e12          # effectively removes slab regularisation
        guesses["activity_d_lambda"] = jnp.ones(2)
        guesses["activity_d_offset"] = jnp.array([1.0, -0.5])

        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        assert flat[0] == pytest.approx(1.0, rel=1e-5)                       # wt
        assert flat[1] == pytest.approx(float(jnp.exp(1.0)), rel=1e-5)       # M42I
        assert flat[2] == pytest.approx(float(jnp.exp(-0.5)), rel=1e-5)      # K84L
        assert flat[3] == pytest.approx(float(jnp.exp(0.5)), rel=1e-5)       # double

    def test_slab_regularisation_shrinks_large_lambda(self, mock_data_no_epi):
        """With finite c2, lambda_tilde < lambda; effect is smaller than unregularised version."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        guesses["activity_d_tau"] = 1.0
        guesses["activity_d_offset"] = jnp.array([1.0, 0.0])

        # Large c2 ≈ no regularisation
        guesses["activity_d_c2"] = 1e12
        guesses["activity_d_lambda"] = jnp.ones(2) * 10.0
        r_unreg = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)

        # Small c2 ≈ strong regularisation
        guesses["activity_d_c2"] = 0.01
        r_reg = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)

        # Regularised activity for M42I should be closer to 1 than unregularised
        diff_reg = abs(float(r_reg[0, 0, 0, 0, 0, 0, 1]) - 1.0)
        diff_unreg = abs(float(r_unreg[0, 0, 0, 0, 0, 0, 1]) - 1.0)
        assert diff_reg < diff_unreg

    def test_positive_and_negative_effects_both_possible(self, mock_data_no_epi):
        """Positive offset → activity > 1; negative offset → activity < 1 (symmetric)."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        guesses["activity_d_tau"] = 1.0
        guesses["activity_d_c2"] = 1e12
        guesses["activity_d_lambda"] = jnp.ones(2)

        guesses["activity_d_offset"] = jnp.array([1.0, 0.0])
        r_pos = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)

        guesses["activity_d_offset"] = jnp.array([-1.0, 0.0])
        r_neg = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)

        assert float(r_pos[0, 0, 0, 0, 0, 0, 1]) > 1.0
        assert float(r_neg[0, 0, 0, 0, 0, 0, 1]) < 1.0

    def test_activity_non_negative(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        guesses["activity_d_tau"] = 2.0
        guesses["activity_d_c2"] = 1e12
        guesses["activity_d_lambda"] = jnp.ones(2)
        guesses["activity_d_offset"] = jnp.array([5.0, -3.0])
        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_no_epi, priors=priors)
        assert jnp.all(result >= 0.0)

    def test_deterministic_site_in_trace(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="activity", data=mock_data_no_epi, priors=priors)
        assert "activity" in tr
        assert tr["activity"]["type"] == "deterministic"
        assert "activity_d_log_activity" in tr

    def test_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="activity", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_horseshoe_sample_sites_present(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_no_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="activity", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "activity_d_tau" in sample_names
        assert "activity_d_c2" in sample_names
        assert "activity_d_lambda" in sample_names
        assert "activity_d_offset" in sample_names


# ---------------------------------------------------------------------------
# define_model – with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_output_shape(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_epi)
        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_epi, priors=priors)
        assert result.shape == (1, 1, 1, 1, 1, 1, mock_data_epi.num_genotype)

    def test_epistasis_shifts_only_double_column(self, mock_data_epi):
        """Zero mutation deltas but epi=1 (with c2→∞) → only M42I/K84L (col 3) is shifted."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_epi)
        guesses["activity_d_offset"] = jnp.zeros(2)
        guesses["activity_epi_tau"] = 1.0
        guesses["activity_epi_c2"] = 1e12
        guesses["activity_epi_lambda"] = jnp.ones(1)
        guesses["activity_epi_offset"] = jnp.array([1.0])

        result = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        assert jnp.allclose(flat[:3], 1.0, atol=1e-5)
        assert flat[3] == pytest.approx(float(jnp.exp(1.0)), rel=1e-5)

    def test_epistasis_positive_and_negative(self, mock_data_epi):
        """Epistasis can increase or decrease activity (symmetric prior)."""
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_epi)
        guesses["activity_d_offset"] = jnp.zeros(2)
        guesses["activity_epi_tau"] = 1.0
        guesses["activity_epi_c2"] = 1e12
        guesses["activity_epi_lambda"] = jnp.ones(1)

        guesses["activity_epi_offset"] = jnp.array([1.0])
        r_pos = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_epi, priors=priors)

        guesses["activity_epi_offset"] = jnp.array([-1.0])
        r_neg = substitute(define_model, data=guesses)(
            name="activity", data=mock_data_epi, priors=priors)

        assert float(r_pos[0, 0, 0, 0, 0, 0, 3]) > 1.0
        assert float(r_neg[0, 0, 0, 0, 0, 0, 3]) < 1.0

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("activity", mock_data_epi)
        tr = trace(substitute(define_model, data=guesses)).get_trace(
            name="activity", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "activity_epi_tau" in sample_names
        assert "activity_epi_c2" in sample_names
        assert "activity_epi_lambda" in sample_names
        assert "activity_epi_offset" in sample_names


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_guide_no_epi_shape(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            result = guide(name="activity", data=mock_data_no_epi, priors=priors)
        assert result.shape == (1, 1, 1, 1, 1, 1, mock_data_no_epi.num_genotype)

    def test_guide_no_epi_non_negative(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            result = guide(name="activity", data=mock_data_no_epi, priors=priors)
        assert jnp.all(result >= 0.0)

    def test_guide_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        with seed(rng_seed=0):
            tr = trace(guide).get_trace(
                name="activity", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_guide_with_epi_shape(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            result = guide(name="activity", data=mock_data_epi, priors=priors)
        assert result.shape == (1, 1, 1, 1, 1, 1, mock_data_epi.num_genotype)

    def test_guide_with_epi_sample_sites(self, mock_data_epi):
        priors = get_priors()
        with seed(rng_seed=1):
            tr = trace(guide).get_trace(
                name="activity", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
        assert "activity_epi_tau" in sample_names
        assert "activity_epi_c2" in sample_names
        assert "activity_epi_lambda" in sample_names
        assert "activity_epi_offset" in sample_names
