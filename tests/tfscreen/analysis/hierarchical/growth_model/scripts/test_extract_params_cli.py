"""
Tests for extract_params_cli.py — MAP point estimate extraction from checkpoints.
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

from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model.scripts.extract_params_cli import extract_params


# ---------------------------------------------------------------------------
# Minimal model fixtures (reuse LaplaceModel pattern from test_run_inference)
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
    """Write a MAP checkpoint and return its path."""
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    ri._current_step = step

    ckpt_path = str(tmp_path / f"{name}_checkpoint.pkl")
    chk = {"svi_state": svi_state, "main_key": ri._main_key, "current_step": step}
    with open(ckpt_path, "wb") as f:
        dill.dump(chk, f)
    return ckpt_path


def _write_checkpoints_file(tmp_path, paths, name="checkpoints.txt"):
    """Write a plain-text file listing checkpoint paths."""
    ckpts_file = str(tmp_path / name)
    with open(ckpts_file, "w") as f:
        f.write("\n".join(paths) + "\n")
    return ckpts_file


# ---------------------------------------------------------------------------
# Shared mock for read_configuration
# ---------------------------------------------------------------------------

class FakeGrowthModel:
    """Minimal GrowthModel stand-in that satisfies extract_parameters."""

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
        # Model attributes needed by RunInference
        self.data = model.data
        self.priors = model.priors
        self.jax_model = model.jax_model
        self.jax_model_guide = model.jax_model_guide

    def get_batch(self, data, indices):
        return self._em.get_batch(data, indices)

    def get_random_idx(self, key=None, num_batches=1):
        return self._em.get_random_idx(key, num_batches)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractParamsSingleCheckpoint:

    def test_creates_csv_files(self, tmp_path):
        """A single checkpoint produces at least one CSV file."""
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=500)
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt_path])
        out_prefix = str(tmp_path / "out")

        fake_gm = FakeGrowthModel(model)
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.extract_parameters",
            return_value={"activity": pd.DataFrame({"genotype": ["wt"], "step_500": [0.5]})}
        ):
            extract_params("cfg.yaml", ckpts_file, out_prefix=out_prefix)

        assert os.path.exists(f"{out_prefix}_activity.csv")

    def test_step_column_in_output(self, tmp_path):
        """The CSV column is named step_{step_number}."""
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=12345)
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt_path])
        out_prefix = str(tmp_path / "out")

        captured = {}

        def fake_extract(gm, posteriors, q_to_get=None):
            captured["q_to_get"] = q_to_get
            return {"activity": pd.DataFrame({"genotype": ["wt"], "step_12345": [0.5]})}

        fake_gm = FakeGrowthModel(model)
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpts_file, out_prefix=out_prefix)

        assert "step_12345" in captured["q_to_get"]
        assert captured["q_to_get"]["step_12345"] == 0.5

    def test_posteriors_passed_as_dict_with_leading_dim(self, tmp_path):
        """extract_parameters receives a dict with shape (1, *) per parameter."""
        model = ExtractModel(num_genotype=4)
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=100)
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt_path])

        captured = {}

        def fake_extract(gm, posteriors, q_to_get=None):
            captured["posteriors"] = posteriors
            return {"activity": pd.DataFrame({"genotype": ["wt"], "step_100": [0.5]})}

        fake_gm = FakeGrowthModel(model)
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpts_file, out_prefix=str(tmp_path / "out"))

        for k, v in captured["posteriors"].items():
            assert v.ndim >= 1
            assert v.shape[0] == 1, f"Expected leading dim=1 for '{k}', got {v.shape}"


class TestExtractParamsMultipleCheckpoints:

    def test_multiple_checkpoints_produce_multiple_columns(self, tmp_path):
        """Two checkpoints produce two step_N columns in the merged CSV."""
        model = ExtractModel(num_genotype=4)
        ckpt1 = _make_map_checkpoint(tmp_path, model, step=1000, name="ckpt1")
        ckpt2 = _make_map_checkpoint(tmp_path, model, step=2000, name="ckpt2")
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt1, ckpt2])
        out_prefix = str(tmp_path / "out")

        call_count = [0]

        def fake_extract(gm, posteriors, q_to_get=None):
            call_count[0] += 1
            col = list(q_to_get.keys())[0]
            return {"activity": pd.DataFrame({"genotype": ["wt"], col: [float(call_count[0])]})}

        fake_gm = FakeGrowthModel(model)
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpts_file, out_prefix=out_prefix)

        df = pd.read_csv(f"{out_prefix}_activity.csv")
        assert "step_1000" in df.columns
        assert "step_2000" in df.columns

    def test_merged_csv_preserves_identifier_columns(self, tmp_path):
        """Identifier columns (genotype etc.) appear once in the merged CSV."""
        model = ExtractModel(num_genotype=4)
        ckpt1 = _make_map_checkpoint(tmp_path, model, step=100, name="c1")
        ckpt2 = _make_map_checkpoint(tmp_path, model, step=200, name="c2")
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt1, ckpt2])
        out_prefix = str(tmp_path / "out")

        def fake_extract(gm, posteriors, q_to_get=None):
            col = list(q_to_get.keys())[0]
            return {"activity": pd.DataFrame({
                "genotype": ["wt", "A1B"],
                col: [0.5, 0.6],
            })}

        fake_gm = FakeGrowthModel(model)
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.extract_parameters",
            side_effect=fake_extract,
        ):
            extract_params("cfg.yaml", ckpts_file, out_prefix=out_prefix)

        df = pd.read_csv(f"{out_prefix}_activity.csv")
        # genotype column should appear exactly once
        assert df.columns.tolist().count("genotype") == 1
        assert len(df) == 2


class TestExtractParamsErrorHandling:

    def test_missing_checkpoint_file_raises(self, tmp_path):
        """FileNotFoundError when a listed checkpoint does not exist."""
        ckpts_file = _write_checkpoints_file(tmp_path, ["/nonexistent/path.pkl"])
        fake_gm = FakeGrowthModel(ExtractModel())
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
                extract_params("cfg.yaml", ckpts_file, out_prefix=str(tmp_path / "out"))

    def test_empty_checkpoints_file_raises(self, tmp_path):
        """ValueError when the checkpoints file is empty."""
        ckpts_file = str(tmp_path / "empty.txt")
        with open(ckpts_file, "w") as f:
            f.write("# just a comment\n\n")

        fake_gm = FakeGrowthModel(ExtractModel())
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(ValueError, match="No checkpoint paths found"):
                extract_params("cfg.yaml", ckpts_file, out_prefix=str(tmp_path / "out"))

    def test_nuts_checkpoint_raises(self, tmp_path):
        """ValueError for NUTS (mcmc_samples) checkpoints."""
        ckpt_path = str(tmp_path / "nuts.pkl")
        with open(ckpt_path, "wb") as f:
            dill.dump({"mcmc_samples": {"x": np.zeros((10, 4))},
                       "current_step": 0}, f)
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt_path])

        fake_gm = FakeGrowthModel(ExtractModel())
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ):
            with pytest.raises(ValueError, match="NUTS checkpoint"):
                extract_params("cfg.yaml", ckpts_file, out_prefix=str(tmp_path / "out"))

    def test_svi_checkpoint_raises(self, tmp_path):
        """ValueError for SVI (non-_auto_loc) checkpoints."""
        model = ExtractModel(num_genotype=4)
        # Write a real (picklable) MAP checkpoint, then patch get_params to
        # return non-_auto_loc keys, simulating an SVI checkpoint.
        ckpt_path = _make_map_checkpoint(tmp_path, model, step=500, name="svi_fake")
        ckpts_file = _write_checkpoints_file(tmp_path, [ckpt_path])
        fake_gm = FakeGrowthModel(model)

        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".extract_params_cli.read_configuration",
            return_value=(fake_gm, {}),
        ), patch(
            "tfscreen.analysis.hierarchical.run_inference.RunInference.setup_svi",
        ) as mock_setup:
            mock_svi = MagicMock()
            mock_svi.optim.get_params.return_value = {"some_param_mean": np.array(1.0)}
            mock_setup.return_value = mock_svi
            with pytest.raises(ValueError, match="SVI.*checkpoint"):
                extract_params("cfg.yaml", ckpts_file, out_prefix=str(tmp_path / "out"))
