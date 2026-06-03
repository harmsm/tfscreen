"""Tests for sample_posterior_cli.py — draws posterior samples from a checkpoint."""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# FileNotFoundError
# ---------------------------------------------------------------------------

class TestSamplePosteriorMissingFile:

    def test_raises_when_checkpoint_missing(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})):
            with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
                sample_posterior("cfg.yaml",
                                 str(tmp_path / "nonexistent.pkl"),
                                 out_prefix=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# NUTS checkpoint routing (payload has "mcmc_samples")
# ---------------------------------------------------------------------------

class TestSamplePosteriorNuts:

    def _run_nuts(self, tmp_path, ri, forward_batch_size=512):
        ckpt_path = str(tmp_path / "nuts.pkl")
        open(ckpt_path, "w").close()
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                   return_value=ri), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill") as mock_dill:

            mock_dill.load.return_value = {
                "mcmc_samples": {"activity": np.zeros((10, 4))}
            }

            def fake_nuts_posteriors(samples, out_prefix, forward_batch_size):
                open(h5_src, "w").close()

            ri.get_nuts_posteriors.side_effect = fake_nuts_posteriors

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"),
                             forward_batch_size=forward_batch_size)

    def test_nuts_checkpoint_calls_get_nuts_posteriors(self, tmp_path):
        ri = MagicMock()
        self._run_nuts(tmp_path, ri)
        ri.get_nuts_posteriors.assert_called_once()

    def test_nuts_output_file_renamed(self, tmp_path):
        ri = MagicMock()
        self._run_nuts(tmp_path, ri)
        assert os.path.isfile(str(tmp_path / "out.h5"))

    def test_nuts_forward_batch_size_passed(self, tmp_path):
        ri = MagicMock()
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")
        ri.get_nuts_posteriors.side_effect = lambda **kw: open(h5_src, "w").close()
        self._run_nuts(tmp_path, ri, forward_batch_size=256)
        _, call_kwargs = ri.get_nuts_posteriors.call_args
        assert call_kwargs["forward_batch_size"] == 256


# ---------------------------------------------------------------------------
# MAP checkpoint routing (AutoDelta: params contain "_auto_loc")
# ---------------------------------------------------------------------------

class TestSamplePosteriorMap:

    def _make_map_ri(self, tmp_path, auto_loc=True):
        ri = MagicMock()
        fake_optim = MagicMock()
        key = "global_p_auto_loc" if auto_loc else "some_param_mean"
        fake_optim.get_params.return_value = {key: np.array(0.5)}
        fake_svi = MagicMock()
        fake_svi.optim = fake_optim
        ri.setup_svi.return_value = fake_svi
        return ri

    def _run_non_nuts(self, tmp_path, ri, extra_patches=None):
        ckpt_path = str(tmp_path / "map.pkl")
        open(ckpt_path, "w").close()
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")

        patches = [
            patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                  return_value=(MagicMock(), {})),
            patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                  return_value=ri),
            patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill"),
        ]
        if extra_patches:
            patches.extend(extra_patches)

        with patches[0], patches[1], patches[2] as mock_dill:
            # svi_state needs .optim_state attribute (MagicMock handles that)
            fake_svi_state = MagicMock()
            mock_dill.load.return_value = {"svi_state": fake_svi_state}

            # Ensure .optim.get_params(svi_state.optim_state) returns the right keys
            ri.setup_svi.return_value.optim.get_params.return_value = (
                ri.setup_svi.return_value.optim.get_params.return_value
            )

            def fake_laplace(**kwargs):
                open(h5_src, "w").close()

            ri.get_laplace_posteriors.side_effect = lambda **kw: (
                open(h5_src, "w").close()
            )

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"))

        return ri

    def test_map_checkpoint_calls_get_laplace_posteriors(self, tmp_path):
        ri = self._make_map_ri(tmp_path, auto_loc=True)
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")

        ckpt_path = str(tmp_path / "map.pkl")
        open(ckpt_path, "w").close()

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                   return_value=ri), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill") as mock_dill:

            mock_dill.load.return_value = {"svi_state": MagicMock()}
            ri.get_laplace_posteriors.side_effect = lambda **kw: (
                open(h5_src, "w").close()
            )

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"))

        ri.get_laplace_posteriors.assert_called_once()

    def test_map_output_file_renamed(self, tmp_path):
        ri = self._make_map_ri(tmp_path, auto_loc=True)
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")
        ckpt_path = str(tmp_path / "map.pkl")
        open(ckpt_path, "w").close()

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                   return_value=ri), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill") as mock_dill:

            mock_dill.load.return_value = {"svi_state": MagicMock()}
            ri.get_laplace_posteriors.side_effect = lambda **kw: (
                open(h5_src, "w").close()
            )

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"))

        assert os.path.isfile(str(tmp_path / "out.h5"))

    def test_map_detection_requires_auto_loc_key(self, tmp_path):
        """Without '_auto_loc' keys the SVI branch must be taken, not MAP."""
        ri = self._make_map_ri(tmp_path, auto_loc=False)  # no _auto_loc key
        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")
        ckpt_path = str(tmp_path / "svi.pkl")
        open(ckpt_path, "w").close()

        fake_svi_obj = MagicMock()
        fake_svi_state = MagicMock()

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                   return_value=ri), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill") as mock_dill, \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli._run_svi",
                   return_value=(fake_svi_obj, fake_svi_state, {}, False)) as mock_svi:

            mock_dill.load.return_value = {"svi_state": MagicMock()}
            ri.get_posteriors.side_effect = lambda **kw: open(h5_src, "w").close()

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"))

        ri.get_laplace_posteriors.assert_not_called()
        mock_svi.assert_called_once()


# ---------------------------------------------------------------------------
# SVI checkpoint routing (no "_auto_loc" keys)
# ---------------------------------------------------------------------------

class TestSamplePosteriorSvi:

    def _run_svi_test(self, tmp_path):
        ri = MagicMock()
        fake_svi_state = MagicMock()
        fake_svi_obj = MagicMock()
        ri.setup_svi.return_value = fake_svi_obj

        h5_src = str(tmp_path / "out_tmp_posterior_posterior.h5")
        ckpt_path = str(tmp_path / "svi.pkl")
        open(ckpt_path, "w").close()

        captured = {}

        def fake_run_svi(ri_obj, init_params, checkpoint_file, out_prefix,
                         max_num_epochs, **kwargs):
            captured["max_num_epochs"] = max_num_epochs
            return fake_svi_obj, fake_svi_state, {}, False

        def fake_get_posteriors(**kwargs):
            captured["get_posteriors_called"] = True
            captured["get_posteriors_kwargs"] = kwargs
            open(h5_src, "w").close()

        ri.get_posteriors.side_effect = fake_get_posteriors

        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.RunInference",
                   return_value=ri), \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli.dill") as mock_dill, \
             patch("tfscreen.tfmodel.scripts.sample_posterior_cli._run_svi",
                   side_effect=fake_run_svi):

            mock_dill.load.return_value = {"svi_state": MagicMock()}

            from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
            sample_posterior("cfg.yaml", ckpt_path,
                             out_prefix=str(tmp_path / "out"))

        return captured

    def test_svi_checkpoint_calls_run_svi_with_zero_epochs(self, tmp_path):
        captured = self._run_svi_test(tmp_path)
        assert captured["max_num_epochs"] == 0

    def test_svi_calls_get_posteriors_directly(self, tmp_path):
        captured = self._run_svi_test(tmp_path)
        assert captured.get("get_posteriors_called") is True

    def test_svi_output_renamed(self, tmp_path):
        self._run_svi_test(tmp_path)
        assert os.path.isfile(str(tmp_path / "out.h5"))


# ---------------------------------------------------------------------------
# main() routes to generalized_main
# ---------------------------------------------------------------------------

class TestSamplePosteriorMain:

    def test_main_calls_generalized_main(self):
        with patch("tfscreen.tfmodel.scripts.sample_posterior_cli.generalized_main") as mock_gm:
            from tfscreen.tfmodel.scripts import sample_posterior_cli
            sample_posterior_cli.main()
        mock_gm.assert_called_once()
        args, kwargs = mock_gm.call_args
        assert args[0] is sample_posterior_cli.sample_posterior
