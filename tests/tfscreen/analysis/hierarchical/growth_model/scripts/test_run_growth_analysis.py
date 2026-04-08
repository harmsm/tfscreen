"""
Tests for run_growth_analysis.py:
  - _init_from_map_params()
  - run_growth_analysis() analysis_method="posterior" branch
"""
import os
import pytest
import dill
import jax.numpy as jnp
from unittest.mock import MagicMock, patch, call

from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import (
    _init_from_map_params,
    run_growth_analysis,
)


class _FakeSVIState:
    """Minimal picklable stand-in for a Numpyro SVIState."""
    def __init__(self):
        self.optim_state = object()


# ---------------------------------------------------------------------------
# _init_from_map_params
# ---------------------------------------------------------------------------

class TestInitFromMapParams:

    def test_scalar_auto_loc_remapped(self):
        params = {"foo_auto_loc": jnp.array(1.5)}
        result = _init_from_map_params(params)
        assert "foo_loc" in result
        assert float(result["foo_loc"]) == pytest.approx(1.5)

    def test_auto_loc_suffix_stripped_correctly(self):
        params = {"theta_log_hill_K_hyper_loc_auto_loc": jnp.array(-4.1)}
        result = _init_from_map_params(params)
        assert "theta_log_hill_K_hyper_loc_loc" in result
        assert "theta_log_hill_K_hyper_loc_auto_loc" not in result

    def test_array_auto_loc_excluded(self):
        # 1-D array params should NOT be remapped (per-genotype offsets).
        params = {"offset_auto_loc": jnp.array([0.0, 1.0, 2.0])}
        result = _init_from_map_params(params)
        assert result == {}

    def test_non_auto_loc_key_excluded(self):
        params = {"foo_loc": jnp.array(1.5), "bar_scale": jnp.array(0.1)}
        result = _init_from_map_params(params)
        assert result == {}

    def test_mixed_params(self):
        params = {
            "a_auto_loc": jnp.array(1.0),           # scalar → remapped
            "b_auto_loc": jnp.array([1.0, 2.0]),    # array  → excluded
            "c_loc": jnp.array(0.5),                 # wrong suffix → excluded
        }
        result = _init_from_map_params(params)
        assert list(result.keys()) == ["a_loc"]
        assert float(result["a_loc"]) == pytest.approx(1.0)

    def test_empty_params(self):
        assert _init_from_map_params({}) == {}

    def test_zero_dim_scalar_included(self):
        # jnp.array(x) with a Python float gives a 0-dim array
        params = {"hyper_scale_auto_loc": jnp.array(0.0)}
        result = _init_from_map_params(params)
        assert "hyper_scale_loc" in result


# ---------------------------------------------------------------------------
# run_growth_analysis — posterior branch
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ri(tmp_path):
    """Minimal RunInference mock."""
    ri = MagicMock()

    # setup_svi returns an svi object whose optim.get_params returns a param dict
    svi_obj = MagicMock()
    ri.setup_svi.return_value = svi_obj

    # run_optimization returns (svi_state, params, converged)
    ri.run_optimization.return_value = (MagicMock(), {}, True)
    ri._iterations_per_epoch = 1

    return ri


def _write_checkpoint(path, params):
    """Write a minimal dill checkpoint file with the given param dict."""
    svi_state = MagicMock()
    # We use a real dict so the optim.get_params call in the code can return it
    chk = {"svi_state": svi_state}
    with open(path, "wb") as f:
        dill.dump(chk, f)
    return svi_state, chk


class TestRunGrowthAnalysisPosteriorBranch:

    # ------------------------------------------------------------------
    # Raises ValueError when no checkpoint supplied
    # ------------------------------------------------------------------

    def _common_patches(self, mocker):
        """Patch read_configuration and RunInference for all posterior tests."""
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=MagicMock(_iterations_per_epoch=1),
        )
        # Prevent the pre-flight FileExistsError that fires when checkpoint_file
        # is None and a checkpoint already exists in the working directory.
        mocker.patch("os.path.exists", return_value=False)

    def test_raises_if_checkpoint_file_is_none(self, tmp_path, mocker):
        self._common_patches(mocker)
        with pytest.raises(ValueError, match="requires an existing MAP or SVI checkpoint"):
            run_growth_analysis(
                config_file="dummy.yaml",
                seed=1,
                checkpoint_file=None,
                analysis_method="posterior",
            )

    def test_raises_if_checkpoint_file_does_not_exist(self, tmp_path, mocker):
        self._common_patches(mocker)
        missing = str(tmp_path / "nonexistent.pkl")
        with pytest.raises(ValueError, match="requires an existing MAP or SVI checkpoint"):
            run_growth_analysis(
                config_file="dummy.yaml",
                seed=1,
                checkpoint_file=missing,
                analysis_method="posterior",
            )

    # ------------------------------------------------------------------
    # MAP checkpoint detected → SVI seeded from remapped params
    # ------------------------------------------------------------------

    def test_map_checkpoint_detected_and_remapped(self, tmp_path, mocker):
        chk_path = str(tmp_path / "map.pkl")

        # MAP params contain _auto_loc keys (scalars)
        map_params = {
            "hyper_loc_auto_loc": jnp.array(1.0),
            "hyper_scale_auto_loc": jnp.array(0.5),
        }

        chk = {"svi_state": _FakeSVIState()}
        with open(chk_path, "wb") as f:
            dill.dump(chk, f)

        svi_obj_mock = MagicMock()
        svi_obj_mock.optim.get_params.return_value = map_params

        ri_mock = MagicMock(_iterations_per_epoch=1)
        ri_mock.setup_svi.return_value = svi_obj_mock
        ri_mock.run_optimization.return_value = (MagicMock(), {}, True)

        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=ri_mock,
        )
        run_svi_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis._run_svi",
            return_value=(MagicMock(), {}, True),
        )

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            max_num_epochs=500,
        )

        call_kwargs = run_svi_mock.call_args
        # checkpoint_file passed to _run_svi should be None (MAP → not loadable as svi_state)
        assert call_kwargs.kwargs["checkpoint_file"] is None
        # init_params should contain remapped _loc keys
        init_p = call_kwargs.kwargs["init_params"]
        assert "hyper_loc_loc" in init_p
        assert "hyper_scale_loc" in init_p
        # svi_epochs should be max_num_epochs (not 0)
        assert call_kwargs.kwargs["max_num_epochs"] == 500

    # ------------------------------------------------------------------
    # SVI checkpoint detected → load directly, 0 epochs
    # ------------------------------------------------------------------

    def test_svi_checkpoint_loaded_directly(self, tmp_path, mocker):
        chk_path = str(tmp_path / "svi.pkl")

        # SVI params have _loc keys, no _auto_loc
        svi_params = {
            "hyper_loc_loc": jnp.array(1.0),
            "hyper_scale_loc": jnp.array(0.5),
        }

        chk = {"svi_state": _FakeSVIState()}
        with open(chk_path, "wb") as f:
            dill.dump(chk, f)

        svi_obj_mock = MagicMock()
        svi_obj_mock.optim.get_params.return_value = svi_params

        ri_mock = MagicMock(_iterations_per_epoch=1)
        ri_mock.setup_svi.return_value = svi_obj_mock
        ri_mock.run_optimization.return_value = (MagicMock(), {}, True)

        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis.RunInference",
            return_value=ri_mock,
        )
        run_svi_mock = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_growth_analysis._run_svi",
            return_value=(MagicMock(), {}, True),
        )

        run_growth_analysis(
            config_file="dummy.yaml",
            seed=1,
            checkpoint_file=chk_path,
            analysis_method="posterior",
            max_num_epochs=500,
        )

        call_kwargs = run_svi_mock.call_args
        # checkpoint_file passed to _run_svi should be the original path
        assert call_kwargs.kwargs["checkpoint_file"] == chk_path
        # 0 epochs: checkpoint is already converged SVI state
        assert call_kwargs.kwargs["max_num_epochs"] == 0
        # always_get_posterior should be forced True
        assert call_kwargs.kwargs["always_get_posterior"] is True
