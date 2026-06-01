"""Tests for diagnose_nan_cli.py — NaN debugging tool for SVI optimization."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scalar(value):
    return np.array(value)


def _make_vector(values):
    return np.array(values)


# ---------------------------------------------------------------------------
# Near-zero filtering logic (pure numpy — no JAX needed)
# ---------------------------------------------------------------------------

class TestNearZeroFiltering:
    """The near-zero dict comprehension can be tested with plain numpy arrays."""

    def _filter_near_zero(self, params, threshold):
        return {k: float(v)
                for k, v in params.items()
                if v.ndim == 0 and float(v) < threshold}

    def test_scalar_below_threshold_included(self):
        params = {"scale_a": _make_scalar(0.005)}
        result = self._filter_near_zero(params, threshold=0.01)
        assert "scale_a" in result
        assert result["scale_a"] == pytest.approx(0.005)

    def test_scalar_above_threshold_excluded(self):
        params = {"scale_a": _make_scalar(0.05)}
        result = self._filter_near_zero(params, threshold=0.01)
        assert "scale_a" not in result

    def test_vector_excluded_regardless_of_values(self):
        params = {"vec": _make_vector([0.0, 0.0])}
        result = self._filter_near_zero(params, threshold=1.0)
        assert "vec" not in result

    def test_mixed_params(self):
        params = {
            "bad_scale": _make_scalar(0.001),
            "good_scale": _make_scalar(0.1),
            "loc_vec": _make_vector([0.0, 0.001]),
        }
        result = self._filter_near_zero(params, threshold=0.01)
        assert set(result.keys()) == {"bad_scale"}

    def test_exactly_at_threshold_excluded(self):
        params = {"scale": _make_scalar(0.01)}
        result = self._filter_near_zero(params, threshold=0.01)
        assert "scale" not in result

    def test_empty_params(self):
        assert self._filter_near_zero({}, threshold=0.01) == {}


# ---------------------------------------------------------------------------
# diagnose_nan() — mock all heavy dependencies via module-level patches
# ---------------------------------------------------------------------------

def _make_diagnose_nan_patches(ri_mock, fake_checkpoint):
    """Return a list of context-manager patches for diagnose_nan dependencies."""
    return [
        patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.jax"),
        patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.read_configuration",
              return_value=(MagicMock(), {})),
        patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.RunInference",
              return_value=ri_mock),
        patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.dill") ,
    ]


class TestDiagnoseNanFunction:

    def _run_diagnose(self, tmp_path, seed=0, num_steps=0):
        """Run diagnose_nan with all heavy deps mocked; return the mocked jax."""
        ckpt_path = tmp_path / "ckpt.pkl"
        ckpt_path.write_bytes(b"")

        ri = MagicMock()
        ri.model.data = MagicMock()
        ri.model.get_random_idx.return_value = [0]
        ri.model.get_batch.return_value = MagicMock()
        ri.get_key.return_value = MagicMock()

        svi = MagicMock()
        svi.init.return_value = MagicMock()
        svi.update.return_value = (MagicMock(), 1.0)
        svi.get_params.return_value = {}
        ri.setup_svi.return_value = svi

        with patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.jax") as mock_jax, \
             patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.RunInference",
                   return_value=ri) as MockRI, \
             patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.dill") as mock_dill:

            mock_jax.device_put.return_value = MagicMock()
            mock_dill.load.return_value = {"svi_state": {}}

            from tfscreen.tfmodel.scripts.diagnose_nan_cli import diagnose_nan
            diagnose_nan("cfg.yaml", str(ckpt_path), seed=seed, num_steps=num_steps)

        return mock_jax, MockRI, ri

    def test_sets_jax_debug_nans(self, tmp_path):
        mock_jax, _, _ = self._run_diagnose(tmp_path)
        mock_jax.config.update.assert_any_call("jax_debug_nans", True)

    def test_disables_jit(self, tmp_path):
        mock_jax, _, _ = self._run_diagnose(tmp_path)
        mock_jax.config.update.assert_any_call("jax_disable_jit", True)

    def test_passes_seed_to_run_inference(self, tmp_path):
        _, MockRI, _ = self._run_diagnose(tmp_path, seed=42)
        call_args = MockRI.call_args
        assert call_args[1]["seed"] == 42

    def test_calls_setup_svi(self, tmp_path):
        _, _, ri = self._run_diagnose(tmp_path)
        ri.setup_svi.assert_called_once()


# ---------------------------------------------------------------------------
# main() argument parsing
# ---------------------------------------------------------------------------

class TestDiagnoseNanMain:

    def test_default_args_routed(self):
        captured = {}

        def fake_diagnose_nan(config_file, checkpoint_file, seed, num_steps,
                              near_zero_threshold):
            captured.update(locals())

        with patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.diagnose_nan",
                   side_effect=fake_diagnose_nan), \
             patch("sys.argv", ["tfs-diagnose-nan", "cfg.yaml", "ckpt.pkl"]):
            from tfscreen.tfmodel.scripts.diagnose_nan_cli import main
            main()

        assert captured["config_file"] == "cfg.yaml"
        assert captured["checkpoint_file"] == "ckpt.pkl"
        assert captured["seed"] == 0
        assert captured["num_steps"] == 200
        assert captured["near_zero_threshold"] == pytest.approx(0.01)

    def test_custom_args_routed(self):
        captured = {}

        def fake_diagnose_nan(config_file, checkpoint_file, seed, num_steps,
                              near_zero_threshold):
            captured.update(locals())

        with patch("tfscreen.tfmodel.scripts.diagnose_nan_cli.diagnose_nan",
                   side_effect=fake_diagnose_nan), \
             patch("sys.argv", [
                 "tfs-diagnose-nan", "run.yaml", "run.pkl",
                 "--seed", "7",
                 "--num_steps", "50",
                 "--near_zero_threshold", "0.001",
             ]):
            from tfscreen.tfmodel.scripts.diagnose_nan_cli import main
            main()

        assert captured["seed"] == 7
        assert captured["num_steps"] == 50
        assert captured["near_zero_threshold"] == pytest.approx(0.001)
