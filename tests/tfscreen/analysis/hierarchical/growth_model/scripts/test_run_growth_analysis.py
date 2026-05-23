"""
Tests for run_growth_analysis.py.
"""
import pytest
import dill
import jax.numpy as jnp
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.scripts.fit_model_cli import (
    fit_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_common(mocker):
    """Patch read_configuration, os.path.exists, and RunInference."""
    mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".fit_model_cli.read_configuration",
        return_value=(MagicMock(), {}),
    )
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".fit_model_cli.RunInference",
        return_value=MagicMock(_iterations_per_epoch=1),
    )


# ---------------------------------------------------------------------------
# run_growth_analysis — seed/checkpoint validation
# ---------------------------------------------------------------------------

class TestRunGrowthAnalysisValidation:

    def test_raises_if_seed_and_checkpoint_both_none(self, tmp_path, mocker):
        """Non-posterior modes require a seed when no checkpoint is given."""
        _patch_common(mocker)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        with pytest.raises(ValueError, match="seed must be provided"):
            fit_model(
                config_file="dummy.yaml",
                seed=None,
                checkpoint_file=None,
                analysis_method="svi",
            )

    def test_raises_on_unknown_analysis_method(self, tmp_path, mocker):
        """An unrecognised analysis_method raises ValueError."""
        _patch_common(mocker)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        with pytest.raises(ValueError, match="not recognized"):
            fit_model(
                config_file="dummy.yaml",
                seed=1,
                analysis_method="posterior",
            )


# ---------------------------------------------------------------------------
# run_growth_analysis — nuts branch
# ---------------------------------------------------------------------------

class TestRunGrowthAnalysisNuts:
    """analysis_method='nuts' dispatches to _run_nuts with correct arguments."""

    def _setup(self, mocker, tmp_path):
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch("os.path.exists", return_value=False)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        fake_samples = {"param": [1.0]}
        run_nuts_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_nuts",
            return_value=fake_samples,
        )
        return run_nuts_mock, fake_samples

    def test_nuts_dispatches_to_run_nuts(self, tmp_path, mocker):
        """analysis_method='nuts' calls _run_nuts."""
        run_nuts_mock, _ = self._setup(mocker, tmp_path)

        fit_model(
            config_file="dummy.yaml",
            seed=42,
            analysis_method="nuts",
        )

        run_nuts_mock.assert_called_once()

    def test_nuts_passes_params_to_run_nuts(self, tmp_path, mocker):
        """NUTS-specific params are forwarded to _run_nuts."""
        run_nuts_mock, _ = self._setup(mocker, tmp_path)

        fit_model(
            config_file="dummy.yaml",
            seed=42,
            analysis_method="nuts",
            out_prefix="myroot",
            nuts_num_warmup=25,
            nuts_num_samples=50,
            nuts_num_chains=2,
            nuts_target_accept_prob=0.85,
            forward_batch_size=128,
        )

        kwargs = run_nuts_mock.call_args.kwargs
        assert kwargs["nuts_num_warmup"] == 25
        assert kwargs["nuts_num_samples"] == 50
        assert kwargs["nuts_num_chains"] == 2
        assert kwargs["nuts_target_accept_prob"] == 0.85
        assert kwargs["forward_batch_size"] == 128
        assert kwargs["out_prefix"] == "myroot"

    def test_nuts_returns_none_samples_true(self, tmp_path, mocker):
        """analysis_method='nuts' returns (None, mcmc_samples, True)."""
        run_nuts_mock, fake_samples = self._setup(mocker, tmp_path)

        svi_state, params, converged = fit_model(
            config_file="dummy.yaml",
            seed=42,
            analysis_method="nuts",
        )

        assert svi_state is None
        assert params is fake_samples
        assert converged is True


# ---------------------------------------------------------------------------
# epoch_checkpoint_interval passthrough
# ---------------------------------------------------------------------------

class TestEpochCheckpointIntervalPassthrough:
    """epoch_checkpoint_interval is forwarded correctly to internal helpers."""

    def test_svi_epoch_checkpoint_interval_forwarded(self, tmp_path, mocker):
        """analysis_method='svi' passes epoch_checkpoint_interval to _run_svi."""
        _patch_common(mocker)
        run_svi_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_svi",
            return_value=(MagicMock(), {}, True),
        )

        fit_model(
            config_file="dummy.yaml",
            seed=1,
            analysis_method="svi",
            pre_map_num_epoch=0,
            epoch_checkpoint_interval=500,
        )

        kwargs = run_svi_mock.call_args.kwargs
        assert kwargs["epoch_checkpoint_interval"] == 500

    def test_map_epoch_checkpoint_interval_forwarded(self, tmp_path, mocker):
        """analysis_method='map' passes epoch_checkpoint_interval to _run_map."""
        _patch_common(mocker)
        run_map_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_map",
            return_value=(MagicMock(), {}, True),
        )

        fit_model(
            config_file="dummy.yaml",
            seed=1,
            analysis_method="map",
            epoch_checkpoint_interval=250,
        )

        kwargs = run_map_mock.call_args.kwargs
        assert kwargs["epoch_checkpoint_interval"] == 250

    def test_premap_always_disables_epoch_checkpoints(self, tmp_path, mocker):
        """The internal pre-map _run_map call always receives epoch_checkpoint_interval=None."""
        _patch_common(mocker)
        run_map_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_map",
            return_value=(MagicMock(), {}, True),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_svi",
            return_value=(MagicMock(), {}, True),
        )

        fit_model(
            config_file="dummy.yaml",
            seed=1,
            analysis_method="svi",
            pre_map_num_epoch=10,
            epoch_checkpoint_interval=999,
        )

        premap_kwargs = run_map_mock.call_args.kwargs
        assert premap_kwargs["epoch_checkpoint_interval"] is None

    def test_svi_epoch_checkpoint_interval_default_is_1000(self, tmp_path, mocker):
        """Default epoch_checkpoint_interval is 1000."""
        _patch_common(mocker)
        run_svi_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".fit_model_cli._run_svi",
            return_value=(MagicMock(), {}, True),
        )

        fit_model(
            config_file="dummy.yaml",
            seed=1,
            analysis_method="svi",
            pre_map_num_epoch=0,
        )

        kwargs = run_svi_mock.call_args.kwargs
        assert kwargs["epoch_checkpoint_interval"] == 1000
