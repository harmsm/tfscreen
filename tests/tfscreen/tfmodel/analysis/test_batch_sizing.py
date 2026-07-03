import pytest
import pandas as pd

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.batch_sizing import (
    estimate_bytes_per_genotype,
    get_available_memory_bytes,
    estimate_genotype_batch_size,
    _dtype_itemsize,
)


@pytest.fixture
def dummy_orchestrator():
    """Minimal ModelOrchestrator (2 genotypes) for tensor-shape inspection."""
    growth_df = pd.DataFrame({
        "library": ["lib"] * 4,
        "genotype": ["wt", "wt", "M42V", "M42V"],
        "titrant_name": ["tit1"] * 4,
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "condition_pre": ["pre-1"] * 4,
        "condition_sel": ["sel+1"] * 4,
        "t_pre": [10.0] * 4,
        "t_sel": [0.0, 20.0, 0.0, 20.0],
        "ln_cfu": [0.0, 5.0, 0.0, 3.0],
        "ln_cfu_std": [0.1] * 4,
        "replicate": [1, 1, 1, 1],
    })
    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42V"],
        "titrant_name": ["tit1", "tit1"],
        "titrant_conc": [0.5, 0.5],
        "theta_obs": [0.5, 0.2],
        "theta_std": [0.01, 0.01],
    })
    return ModelOrchestrator(growth_df, binding_df)


class TestEstimateBytesPerGenotype:
    """Unit tests for estimate_bytes_per_genotype."""

    def test_matches_hand_computed_formula(self, dummy_orchestrator):
        tensor_shape = dummy_orchestrator.growth_tm.tensor_shape
        other_axes_product = 1
        for size in tensor_shape[:-1]:
            other_axes_product *= size

        n_samples = 25
        predict_sites = ["growth_pred", "theta_growth_pred"]
        result = estimate_bytes_per_genotype(
            dummy_orchestrator, predict_sites,
            num_marginal_samples=n_samples,
            overhead_multiplier=1.0,
        )
        expected = (other_axes_product * n_samples * len(predict_sites)
                    * _dtype_itemsize())
        assert result == pytest.approx(expected)

    def test_scales_with_overhead_multiplier(self, dummy_orchestrator):
        base = estimate_bytes_per_genotype(
            dummy_orchestrator, ["growth_pred"],
            num_marginal_samples=10, overhead_multiplier=1.0,
        )
        scaled = estimate_bytes_per_genotype(
            dummy_orchestrator, ["growth_pred"],
            num_marginal_samples=10, overhead_multiplier=3.0,
        )
        assert scaled == pytest.approx(base * 3.0)

    def test_defaults_num_marginal_samples_to_one(self, dummy_orchestrator):
        result = estimate_bytes_per_genotype(
            dummy_orchestrator, ["growth_pred"],
            num_marginal_samples=None, overhead_multiplier=1.0,
        )
        expected = estimate_bytes_per_genotype(
            dummy_orchestrator, ["growth_pred"],
            num_marginal_samples=1, overhead_multiplier=1.0,
        )
        assert result == pytest.approx(expected)

    def test_empty_predict_sites_treated_as_one_site(self, dummy_orchestrator):
        result = estimate_bytes_per_genotype(
            dummy_orchestrator, [],
            num_marginal_samples=10, overhead_multiplier=1.0,
        )
        expected = estimate_bytes_per_genotype(
            dummy_orchestrator, ["growth_pred"],
            num_marginal_samples=10, overhead_multiplier=1.0,
        )
        assert result == pytest.approx(expected)


class _FakeDevice:
    """Stand-in for a jax.Device exposing only memory_stats()."""

    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


class TestGetAvailableMemoryBytes:
    """Unit tests for get_available_memory_bytes."""

    def test_uses_bytes_limit_when_available(self, mocker):
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.jax.devices",
            return_value=[_FakeDevice({"bytes_limit": 1_000_000_000})],
        )
        result = get_available_memory_bytes(safety_fraction=0.5)
        assert result == 500_000_000

    def test_falls_back_to_system_memory_when_stats_none(self, mocker):
        """CPU backend returns None from memory_stats(); must fall back."""
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.jax.devices",
            return_value=[_FakeDevice(None)],
        )
        mocker.patch(
            "os.sysconf",
            side_effect=lambda name: {"SC_PAGE_SIZE": 4096,
                                       "SC_PHYS_PAGES": 1000}[name],
        )
        result = get_available_memory_bytes(safety_fraction=0.5)
        assert result == int(4096 * 1000 * 0.5)

    def test_falls_back_when_bytes_limit_key_missing(self, mocker):
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.jax.devices",
            return_value=[_FakeDevice({"peak_bytes_in_use": 123})],
        )
        mocker.patch(
            "os.sysconf",
            side_effect=lambda name: {"SC_PAGE_SIZE": 4096,
                                       "SC_PHYS_PAGES": 2000}[name],
        )
        result = get_available_memory_bytes(safety_fraction=1.0)
        assert result == 4096 * 2000


class TestEstimateGenotypeBatchSize:
    """Unit tests for estimate_genotype_batch_size."""

    def test_batch_size_matches_budget_over_cost(self, dummy_orchestrator, mocker):
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.get_available_memory_bytes",
            return_value=1_000_000,
        )
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.estimate_bytes_per_genotype",
            return_value=10_000,
        )
        result = estimate_genotype_batch_size(
            dummy_orchestrator, ["growth_pred"], num_marginal_samples=10,
        )
        assert result == 100

    def test_clamped_to_at_least_one(self, dummy_orchestrator, mocker):
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.get_available_memory_bytes",
            return_value=10,
        )
        mocker.patch(
            "tfscreen.tfmodel.analysis.batch_sizing.estimate_bytes_per_genotype",
            return_value=10_000_000,
        )
        result = estimate_genotype_batch_size(
            dummy_orchestrator, ["growth_pred"], num_marginal_samples=10,
        )
        assert result == 1
