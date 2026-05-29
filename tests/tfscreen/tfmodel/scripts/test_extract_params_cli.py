"""
Tests for extract_params_cli.py — unified parameter extraction from checkpoints and posteriors.
"""
import os
import dill
import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from unittest.mock import MagicMock, patch
from flax import struct
from numpyro.infer.svi import SVIState

from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.scripts.extract_params_cli import extract_params


# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------

@struct.dataclass
class ExtractData:
    num_genotype: int
    batch_idx: jnp.ndarray


def _extract_jax_model(data, priors):
    global_p = numpyro.sample("global_p", dist.Normal(0., 1.))
    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        numpyro.sample("geno_p", dist.Normal(global_p, 1.))


class ExtractModel:
    def __init__(self, num_genotype=4):
        self.data = ExtractData(num_genotype=num_genotype,
                                batch_idx=jnp.arange(num_genotype))
        self.priors = {}
        self.jax_model = _extract_jax_model
        self.jax_model_guide = lambda data, priors: None

    def get_batch(self, data, indices):
        return ExtractData(num_genotype=len(indices), batch_idx=indices)

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _make_map_checkpoint(tmp_path, model, step=1000, name="ckpt"):
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    ri._current_step = step
    ckpt_path = str(tmp_path / f"{name}_checkpoint.pkl")
    chk = {"svi_state": svi_state, "main_key": ri._main_key, "current_step": step}
    with open(ckpt_path, "wb") as f:
        dill.dump(chk, f)
    return ckpt_path


class FakeTFModel:
    def __init__(self, model):
        self._em = model
        self.growth_tm = MagicMock()
        self.growth_tm.df = pd.DataFrame({
            "genotype": ["wt"] * 4,
            "titrant_name": ["IPTG"] * 4,
            "titrant_conc": [0.0, 0.0, 1.0, 1.0],
        })
        self.mut_labels = []
        self.pair_labels = []
        self._growth_shares_replicates = False
        self.data = model.data
        self.priors = model.priors
        self.jax_model = model.jax_model
        self.jax_model_guide = model.jax_model_guide

    def get_batch(self, data, indices):
        return self._em.get_batch(data, indices)

    def get_random_idx(self, key=None, num_batches=1):
        return self._em.get_random_idx(key, num_batches)


# ---------------------------------------------------------------------------
# .pkl path — point estimate output
# ---------------------------------------------------------------------------

class TestExtractParamsCheckpoint:

    def test_creates_csv_files(self, tmp_path):
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=500)
        out_prefix = str(tmp_path / "out")
        fake_gm = FakeTFModel(model)

        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.extract_parameters",
            return_value={"activity": pd.DataFrame({"genotype": ["wt"], "point_est": [0.5]})}
        ):
            extract_params("cfg.yaml", ckpt_path, out_prefix=out_prefix)

        assert os.path.exists(f"{out_prefix}_activity.csv")

    def test_passes_point_est_q_to_get(self, tmp_path):
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=100)
        out_prefix = str(tmp_path / "out")
        captured = {}

        def fake_extract(gm, posteriors, q_to_get=None):
            captured["q_to_get"] = q_to_get
            return {"activity": pd.DataFrame({"genotype": ["wt"], "point_est": [0.5]})}

        fake_gm = FakeTFModel(model)
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpt_path, out_prefix=out_prefix)

        assert captured["q_to_get"] == {"point_est": 0.5}

    def test_posteriors_have_leading_sample_dim(self, tmp_path):
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=100)
        captured = {}

        def fake_extract(gm, posteriors, q_to_get=None):
            captured["posteriors"] = posteriors
            return {"activity": pd.DataFrame({"genotype": ["wt"], "point_est": [0.5]})}

        fake_gm = FakeTFModel(model)
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpt_path, out_prefix=str(tmp_path / "out"))

        for k, v in captured["posteriors"].items():
            assert v.shape[0] == 1, f"Expected leading dim=1 for '{k}', got {v.shape}"


# ---------------------------------------------------------------------------
# .h5 path — quantile output
# ---------------------------------------------------------------------------

class TestExtractParamsPosterior:

    def test_calls_extract_with_default_quantiles(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        posterior_file = str(tmp_path / "post.h5")
        captured = {}

        import h5py
        with h5py.File(posterior_file, "w") as f:
            f.create_dataset("dummy_param", data=np.ones((10, 4)))

        def fake_extract(gm, posteriors, q_to_get=None):
            captured["q_to_get"] = q_to_get
            return {"activity": pd.DataFrame({"genotype": ["wt"], "median": [0.5]})}

        fake_gm = FakeTFModel(ExtractModel())
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", posterior_file, out_prefix=out_prefix)

        assert captured["q_to_get"] is None

    def test_creates_csv_from_posterior(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        posterior_file = str(tmp_path / "post.h5")

        import h5py
        with h5py.File(posterior_file, "w") as f:
            f.create_dataset("dummy_param", data=np.ones((10, 4)))

        fake_gm = FakeTFModel(ExtractModel())
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.extract_parameters",
            return_value={"activity": pd.DataFrame({"genotype": ["wt"], "median": [0.5]})}
        ):
            extract_params("cfg.yaml", posterior_file, out_prefix=out_prefix)

        assert os.path.exists(f"{out_prefix}_activity.csv")
        df = pd.read_csv(f"{out_prefix}_activity.csv")
        assert "median" in df.columns


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestExtractParamsErrors:

    def test_missing_pkl_raises(self, tmp_path):
        fake_gm = FakeTFModel(ExtractModel())
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
                extract_params("cfg.yaml", str(tmp_path / "missing.pkl"),
                               out_prefix=str(tmp_path / "out"))

    def test_missing_h5_raises(self, tmp_path):
        fake_gm = FakeTFModel(ExtractModel())
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(FileNotFoundError, match="Posterior file not found"):
                extract_params("cfg.yaml", str(tmp_path / "missing.h5"),
                               out_prefix=str(tmp_path / "out"))

    def test_nuts_checkpoint_raises(self, tmp_path):
        ckpt_path = str(tmp_path / "nuts.pkl")
        with open(ckpt_path, "wb") as f:
            dill.dump({"mcmc_samples": {"x": np.zeros((10, 4))}, "current_step": 0}, f)

        fake_gm = FakeTFModel(ExtractModel())
        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(ValueError, match="NUTS checkpoint"):
                extract_params("cfg.yaml", ckpt_path, out_prefix=str(tmp_path / "out"))

    def test_svi_checkpoint_raises(self, tmp_path):
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=500, name="svi_fake")
        fake_gm = FakeTFModel(model)

        with patch(
            "tfscreen.tfmodel.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.tfmodel.inference.run_inference.RunInference.setup_svi",
        ) as mock_setup:
            mock_svi = MagicMock()
            mock_svi.optim.get_params.return_value = {"some_param_mean": np.array(1.0)}
            mock_setup.return_value = mock_svi
            with pytest.raises(ValueError, match="SVI.*checkpoint"):
                extract_params("cfg.yaml", ckpt_path, out_prefix=str(tmp_path / "out"))
