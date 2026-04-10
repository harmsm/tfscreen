"""
Tests for run_growth_analysis.py:
  - run_growth_analysis() analysis_method="posterior" branch
"""
import os
import pytest
import dill
import jax.numpy as jnp
from unittest.mock import MagicMock, patch, call

from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import (
    run_growth_analysis,
)


class _FakeSVIState:
    """Minimal picklable stand-in for a Numpyro SVIState."""
    def __init__(self):
        self.optim_state = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_checkpoint(path, svi_state=None):
    if svi_state is None:
        svi_state = _FakeSVIState()
    chk = {"svi_state": svi_state}
    with open(path, "wb") as f:
        dill.dump(chk, f)
    return svi_state


def _common_patches(mocker):
    """Patch read_configuration and RunInference for all posterior tests."""
    mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".run_growth_analysis.read_configuration",
        return_value=(MagicMock(), {}),
    )
    # Prevent the pre-flight FileExistsError when checkpoint_file is None.
    mocker.patch("os.path.exists", return_value=False)


# ---------------------------------------------------------------------------
# run_growth_analysis — posterior branch — error cases
# ---------------------------------------------------------------------------

class TestRunGrowthAnalysisPosteriorErrors:

    def test_raises_if_checkpoint_file_is_none(self, tmp_path, mocker):
        _common_patches(mocker)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        with pytest.raises(ValueError, match="requires an existing MAP or SVI checkpoint"):
            run_growth_analysis(
                config_file="dummy.yaml",
                seed=1,
                checkpoint_file=None,
                analysis_method="posterior",
            )

    def test_raises_if_checkpoint_file_does_not_exist(self, tmp_path, mocker):
        _common_patches(mocker)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        missing = str(tmp_path / "nonexistent.pkl")
        with pytest.raises(ValueError, match="requires an existing MAP or SVI checkpoint"):
            run_growth_analysis(
                config_file="dummy.yaml",
                seed=1,
                checkpoint_file=missing,
                analysis_method="posterior",
            )

    def test_raises_if_seed_and_checkpoint_both_none_non_posterior(self, tmp_path, mocker):
        """Non-posterior modes must still require a seed when no checkpoint given."""
        _common_patches(mocker)
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        with pytest.raises(ValueError, match="seed must be provided"):
            run_growth_analysis(
                config_file="dummy.yaml",
                seed=None,
                checkpoint_file=None,
                analysis_method="svi",
            )


# ---------------------------------------------------------------------------
# run_growth_analysis — posterior branch — MAP checkpoint
# ---------------------------------------------------------------------------

class TestRunGrowthAnalysisPosteriorMAP:
    """MAP checkpoint → Laplace approximation path."""

    def _setup(self, mocker, tmp_path, map_params):
        chk_path = str(tmp_path / "map.pkl")
        _write_checkpoint(chk_path)

        svi_obj = MagicMock()
        svi_obj.optim.get_params.return_value = map_params

        ri = MagicMock(_iterations_per_epoch=1)
        ri.setup_svi.return_value = svi_obj

        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=ri,
        )
        summarize_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.summarize_posteriors",
        )
        return chk_path, ri, summarize_mock

    def test_map_detected_by_auto_loc_keys(self, tmp_path, mocker):
        """Params with _auto_loc suffix are recognised as a MAP checkpoint."""
        map_params = {
            "hyper_loc_auto_loc": jnp.array(1.0),
            "hyper_scale_auto_loc": jnp.array(0.5),
        }
        chk_path, ri, _ = self._setup(mocker, tmp_path, map_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        ri.get_laplace_posteriors.assert_called_once()

    def test_get_laplace_posteriors_called_with_correct_params(self, tmp_path, mocker):
        """get_laplace_posteriors receives the map_params and sampling args."""
        map_params = {
            "a_auto_loc": jnp.array(0.3),
            "b_auto_loc": jnp.array(-1.2),
        }
        chk_path, ri, _ = self._setup(mocker, tmp_path, map_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            out_root="myrun",
            num_posterior_samples=500,
            sampling_batch_size=50,
            forward_batch_size=256,
        )

        ri.get_laplace_posteriors.assert_called_once_with(
            map_params=map_params,
            out_root="myrun",
            num_posterior_samples=500,
            sampling_batch_size=50,
            forward_batch_size=256,
        )

    def test_summarize_posteriors_called_after_laplace(self, tmp_path, mocker):
        """summarize_posteriors is called with the correct posterior file path."""
        map_params = {"p_auto_loc": jnp.array(0.0)}
        chk_path, ri, summarize_mock = self._setup(mocker, tmp_path, map_params)

        run_growth_analysis(
            config_file="cfg.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            out_root="myrun",
        )

        summarize_mock.assert_called_once_with(
            posterior_file="myrun_posterior.h5",
            config_file="cfg.yaml",
            out_root="myrun",
        )

    def test_map_posterior_returns_none_params_true(self, tmp_path, mocker):
        """MAP posterior path returns (None, chk_params, True)."""
        map_params = {"p_auto_loc": jnp.array(1.5)}
        chk_path, ri, _ = self._setup(mocker, tmp_path, map_params)

        result = run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        svi_state, params, converged = result
        assert svi_state is None
        assert params is map_params
        assert converged is True

    def test_setup_svi_called_with_delta_guide(self, tmp_path, mocker):
        """setup_svi is called with guide_type='delta' to extract params."""
        map_params = {"p_auto_loc": jnp.array(0.0)}
        chk_path, ri, _ = self._setup(mocker, tmp_path, map_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        ri.setup_svi.assert_called_once_with(guide_type="delta")

    def test_no_seed_required_for_map_posterior(self, tmp_path, mocker):
        """seed=None is allowed for posterior mode when a checkpoint is provided."""
        map_params = {"p_auto_loc": jnp.array(0.0)}
        chk_path, ri, _ = self._setup(mocker, tmp_path, map_params)

        # Should not raise even though seed is None.
        run_growth_analysis(
            config_file="dummy.yaml",
            seed=None,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        ri.get_laplace_posteriors.assert_called_once()

    def test_seed_none_uses_zero_for_run_inference(self, tmp_path, mocker):
        """When seed=None, RunInference is constructed with seed=0."""
        map_params = {"p_auto_loc": jnp.array(0.0)}
        chk_path = str(tmp_path / "map.pkl")
        _write_checkpoint(chk_path)

        svi_obj = MagicMock()
        svi_obj.optim.get_params.return_value = map_params

        ri = MagicMock(_iterations_per_epoch=1)
        ri.setup_svi.return_value = svi_obj

        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        ri_cls = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=ri,
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.summarize_posteriors",
        )

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=None,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        _, call_seed = ri_cls.call_args.args
        assert call_seed == 0


# ---------------------------------------------------------------------------
# run_growth_analysis — posterior branch — SVI checkpoint
# ---------------------------------------------------------------------------

class TestRunGrowthAnalysisPosteriorSVI:
    """SVI checkpoint → load directly, 0 epochs path."""

    def _setup(self, mocker, tmp_path, svi_params):
        chk_path = str(tmp_path / "svi.pkl")
        _write_checkpoint(chk_path)

        svi_obj = MagicMock()
        svi_obj.optim.get_params.return_value = svi_params

        ri = MagicMock(_iterations_per_epoch=1)
        ri.setup_svi.return_value = svi_obj

        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=ri,
        )
        run_svi_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis._run_svi",
            return_value=(MagicMock(), {}, True),
        )
        return chk_path, ri, run_svi_mock

    def test_svi_detected_by_absence_of_auto_loc(self, tmp_path, mocker):
        """Params without _auto_loc are recognised as an SVI checkpoint."""
        svi_params = {
            "hyper_loc_loc": jnp.array(1.0),
            "hyper_scale_scale": jnp.array(0.5),
        }
        chk_path, ri, run_svi_mock = self._setup(mocker, tmp_path, svi_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        run_svi_mock.assert_called_once()
        ri.get_laplace_posteriors.assert_not_called()

    def test_svi_checkpoint_passes_original_path(self, tmp_path, mocker):
        """checkpoint_file is forwarded unchanged to _run_svi."""
        svi_params = {"p_loc": jnp.array(0.0)}
        chk_path, ri, run_svi_mock = self._setup(mocker, tmp_path, svi_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        kwargs = run_svi_mock.call_args.kwargs
        assert kwargs["checkpoint_file"] == chk_path

    def test_svi_checkpoint_zero_epochs(self, tmp_path, mocker):
        """_run_svi is called with max_num_epochs=0 for SVI checkpoints."""
        svi_params = {"p_loc": jnp.array(0.0)}
        chk_path, ri, run_svi_mock = self._setup(mocker, tmp_path, svi_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            max_num_epochs=9999,   # should be overridden to 0
        )

        kwargs = run_svi_mock.call_args.kwargs
        assert kwargs["max_num_epochs"] == 0

    def test_svi_checkpoint_always_get_posterior_true(self, tmp_path, mocker):
        """_run_svi is called with always_get_posterior=True."""
        svi_params = {"p_loc": jnp.array(0.0)}
        chk_path, ri, run_svi_mock = self._setup(mocker, tmp_path, svi_params)

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            always_get_posterior=False,   # should be overridden
        )

        kwargs = run_svi_mock.call_args.kwargs
        assert kwargs["always_get_posterior"] is True

    def test_no_seed_required_for_svi_posterior(self, tmp_path, mocker):
        """seed=None is allowed for SVI posterior mode when a checkpoint is provided."""
        svi_params = {"p_loc": jnp.array(0.0)}
        chk_path, ri, run_svi_mock = self._setup(mocker, tmp_path, svi_params)

        # Should not raise even though seed is None.
        run_growth_analysis(
            config_file="dummy.yaml",
            seed=None,
            checkpoint_file=chk_path,
            analysis_method="posterior",
        )

        run_svi_mock.assert_called_once()
