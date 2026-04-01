import pytest
import numpy as np
import torch
import pyro
import pyro.poutine as poutine
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
    "pair_geno_matrix",
])

# Genotypes: wt(0), M42I(1), K84L(2), M42I/K84L(3)
_MUT_GENO = np.array([[0, 1, 0, 1],   # M42I
                       [0, 0, 1, 1]], dtype=np.float32)   # K84L
_PAIR_GENO = np.array([[0, 0, 0, 1]], dtype=np.float32)   # K84L/M42I


@pytest.fixture
def mock_data_epi():
    return MockGrowthData(
        num_genotype=4,
        num_mutation=2,
        num_pair=1,
        mut_geno_matrix=_MUT_GENO,
        pair_geno_matrix=_PAIR_GENO,
    )


@pytest.fixture
def mock_data_no_epi():
    return MockGrowthData(
        num_genotype=4,
        num_mutation=2,
        num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        pair_geno_matrix=np.zeros((0, 4), dtype=np.float32),
    )


# Neutral offset: the specific value that gives shifted-lognormal ≈ 0.
_NEUTRAL_OFFSET = -0.8240460108562919


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "dk_geno_hyper_loc_loc" in params
    assert "dk_geno_hyper_shift_loc" in params
    assert "dk_geno_sigma_epi_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.dk_geno_hyper_loc_loc == hp["dk_geno_hyper_loc_loc"]
    assert priors.dk_geno_hyper_shift_loc == hp["dk_geno_hyper_shift_loc"]


def test_get_guesses_no_epi(mock_data_no_epi):
    name = "dk_geno"
    guesses = get_guesses(name, mock_data_no_epi)
    assert isinstance(guesses, dict)
    assert f"{name}_hyper_loc" in guesses
    assert f"{name}_shift" in guesses
    assert f"{name}_offset" in guesses
    assert guesses[f"{name}_offset"].shape == (mock_data_no_epi.num_mutation,)
    # Every offset initialised to the neutral value
    assert torch.allclose(guesses[f"{name}_offset"],
                          torch.full((mock_data_no_epi.num_mutation,), _NEUTRAL_OFFSET))
    assert f"{name}_epi_offset" not in guesses


def test_get_guesses_with_epi(mock_data_epi):
    name = "dk_geno"
    guesses = get_guesses(name, mock_data_epi)
    assert f"{name}_epi_offset" in guesses
    assert guesses[f"{name}_epi_offset"].shape == (mock_data_epi.num_pair,)
    assert torch.all(guesses[f"{name}_epi_offset"] == 0.0)


# ---------------------------------------------------------------------------
# define_model – no epistasis
# ---------------------------------------------------------------------------

class TestDefineModelNoEpi:

    def test_output_shape(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_wt_dk_geno_is_zero(self, mock_data_no_epi):
        """WT column of M is all-zero, so dk_geno[wt] = 0 always."""
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        result = poutine.condition(define_model, data=guesses)(
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
        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert torch.allclose(flat, torch.tensor(0.0), atol=1e-5)

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
        guesses["dk_geno_shift"] = 0.02
        # Use different offsets per mutation to make them distinguishable
        guesses["dk_geno_offset"] = torch.tensor([0.0, 1.0])

        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        # Compute expected per-mutation dk values
        def dk(offset):
            return 0.02 - float(torch.clamp(torch.exp(torch.tensor(-3.5 + offset * 0.5)), max=1e30))

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
        model = poutine.condition(define_model, data=guesses)
        tr = poutine.trace(model).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        # In Pyro, pyro.deterministic() is implemented as a masked Delta sample,
        # so the site type is "sample" (not "deterministic" as in NumPyro).
        assert "dk_geno" in tr.nodes
        assert tr.nodes["dk_geno"]["type"] in ("sample", "deterministic")

    def test_no_epistasis_sample_sites(self, mock_data_no_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_no_epi)
        model = poutine.condition(define_model, data=guesses)
        tr = poutine.trace(model).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.nodes.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)


# ---------------------------------------------------------------------------
# define_model – with epistasis
# ---------------------------------------------------------------------------

class TestDefineModelWithEpi:

    def test_output_shape(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_epistasis_shifts_only_double_column(self, mock_data_epi):
        """
        With neutral mutation offsets (dk ≈ 0) but non-zero epistasis offset,
        only the double-mutant column should be non-zero.
        """
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        guesses["dk_geno_sigma_epi"] = 1.0
        guesses["dk_geno_epi_offset"] = torch.tensor([0.5])

        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]

        # wt, M42I, K84L should all be ~0 (neutral offset)
        assert torch.allclose(flat[:3], torch.tensor(0.0), atol=1e-5)
        # M42I/K84L = 0 (mut contribution) + 0.5*1.0 (epistasis)
        assert flat[3] == pytest.approx(0.5, rel=1e-5)

    def test_epistasis_sample_sites_present(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        model = poutine.condition(define_model, data=guesses)
        tr = poutine.trace(model).get_trace(
            name="dk_geno", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.nodes.items() if v["type"] == "sample"}
        assert "dk_geno_epi_offset" in sample_names
        assert "dk_geno_sigma_epi" in sample_names

    def test_wt_still_zero_with_epistasis(self, mock_data_epi):
        priors = get_priors()
        guesses = get_guesses("dk_geno", mock_data_epi)
        guesses["dk_geno_sigma_epi"] = 1.0
        guesses["dk_geno_epi_offset"] = torch.tensor([1.0])
        result = poutine.condition(define_model, data=guesses)(
            name="dk_geno", data=mock_data_epi, priors=priors)
        flat = result[0, 0, 0, 0, 0, 0, :]
        assert flat[0] == pytest.approx(0.0, abs=1e-5)   # WT is always 0


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_guide_no_epi_shape(self, mock_data_no_epi):
        pyro.clear_param_store()
        priors = get_priors()
        torch.manual_seed(0)
        result = guide(name="dk_geno", data=mock_data_no_epi, priors=priors)
        G = mock_data_no_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_guide_no_epi_no_epistasis_sample_sites(self, mock_data_no_epi):
        pyro.clear_param_store()
        priors = get_priors()
        torch.manual_seed(0)
        tr = poutine.trace(guide).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.nodes.items() if v["type"] == "sample"}
        assert not any("epi" in k for k in sample_names)

    def test_guide_with_epi_shape(self, mock_data_epi):
        pyro.clear_param_store()
        priors = get_priors()
        torch.manual_seed(1)
        result = guide(name="dk_geno", data=mock_data_epi, priors=priors)
        G = mock_data_epi.num_genotype
        assert result.shape == (1, 1, 1, 1, 1, 1, G)

    def test_guide_with_epi_sample_sites(self, mock_data_epi):
        pyro.clear_param_store()
        priors = get_priors()
        torch.manual_seed(1)
        tr = poutine.trace(guide).get_trace(
            name="dk_geno", data=mock_data_epi, priors=priors)
        sample_names = {k for k, v in tr.nodes.items() if v["type"] == "sample"}
        assert "dk_geno_epi_offset" in sample_names
        assert "dk_geno_sigma_epi" in sample_names

    def test_guide_all_hyperprior_sites_sampled(self, mock_data_no_epi):
        pyro.clear_param_store()
        priors = get_priors()
        torch.manual_seed(0)
        tr = poutine.trace(guide).get_trace(
            name="dk_geno", data=mock_data_no_epi, priors=priors)
        sample_names = {k for k, v in tr.nodes.items() if v["type"] == "sample"}
        assert "dk_geno_hyper_loc" in sample_names
        assert "dk_geno_hyper_scale" in sample_names
        assert "dk_geno_shift" in sample_names
        assert "dk_geno_offset" in sample_names
