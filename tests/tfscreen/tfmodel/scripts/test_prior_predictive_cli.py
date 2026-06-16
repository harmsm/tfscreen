"""Tests for sample_prior_cli.py — draws and writes synthetic prior datasets."""
import os
import pytest
import numpy as np
import h5py
from unittest.mock import patch, MagicMock
import pandas as pd


# ---------------------------------------------------------------------------
# _write_h5
# ---------------------------------------------------------------------------

class TestWriteH5:

    def test_creates_file(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        _write_h5(path, {"x": np.ones((5, 4))}, num_draws=5)
        assert os.path.isfile(path)

    def test_datasets_present(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        data = {"alpha": np.zeros((3, 2)), "beta": np.ones(10)}
        _write_h5(path, data, num_draws=3)
        with h5py.File(path, "r") as hf:
            assert "alpha" in hf
            assert "beta" in hf

    def test_data_values_correct(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        arr = np.arange(12, dtype=float).reshape(3, 4)
        _write_h5(path, {"arr": arr}, num_draws=3)
        with h5py.File(path, "r") as hf:
            np.testing.assert_array_equal(hf["arr"][:], arr)

    def test_num_samples_attr(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        _write_h5(path, {"x": np.ones(5)}, num_draws=7)
        with h5py.File(path, "r") as hf:
            assert hf.attrs["num_samples"] == 7

    def test_gzip_compression(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        _write_h5(path, {"x": np.ones((100, 50))}, num_draws=100)
        with h5py.File(path, "r") as hf:
            assert hf["x"].compression == "gzip"

    def test_1d_array_chunked(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import _write_h5
        path = str(tmp_path / "out.h5")
        arr = np.zeros(20)
        _write_h5(path, {"flat": arr}, num_draws=20)
        with h5py.File(path, "r") as hf:
            assert hf["flat"].chunks is not None


# ---------------------------------------------------------------------------
# sample_prior
# ---------------------------------------------------------------------------

class TestSamplePrior:

    def _mock_draw_prior(self):
        predictions = {}
        latent_params = {"activity": np.array([[1.0, 2.0]])}
        return predictions, latent_params

    def test_writes_csv_and_h5_for_one_dataset(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        fake_gm = MagicMock()
        fake_growth_df = pd.DataFrame({"ln_cfu": [1.0, 2.0]})
        latent_params = {"activity": np.array([[1.0]])}

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(fake_gm, {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   return_value=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=1, seed=0)

        assert os.path.isfile(f"{out_prefix}_000_growth.csv")
        assert os.path.isfile(f"{out_prefix}_000_ground_truth.h5")

    def test_writes_multiple_datasets(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        fake_gm = MagicMock()
        latent_params = {"activity": np.array([[1.0]])}
        fake_growth_df = pd.DataFrame({"ln_cfu": [1.0]})

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(fake_gm, {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   return_value=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=3, seed=0)

        for i in range(3):
            assert os.path.isfile(f"{out_prefix}_00{i}_growth.csv")
            assert os.path.isfile(f"{out_prefix}_00{i}_ground_truth.h5")

    def test_zero_padded_index_width(self, tmp_path):
        """Indices must be zero-padded to at least 3 digits."""
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        latent_params = {"activity": np.array([[1.0]])}
        fake_growth_df = pd.DataFrame({"ln_cfu": [1.0]})

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   return_value=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=1, seed=0)

        assert os.path.isfile(f"{out_prefix}_000_growth.csv")

    def test_each_dataset_uses_different_seed(self, tmp_path):
        """draw_prior must be called with seed+i for dataset i."""
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        seeds_used = []
        latent_params = {"activity": np.array([[1.0]])}
        fake_growth_df = pd.DataFrame({"ln_cfu": [1.0]})

        def fake_draw_prior(orchestrator, rng_key, num_draws):
            seeds_used.append(rng_key)
            return {}, latent_params

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   side_effect=fake_draw_prior), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   return_value=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=3, seed=5)

        assert seeds_used == [5, 6, 7]

    def test_no_noise_passes_none_rng(self, tmp_path):
        """noise=False must pass noise_rng=None to growth_df_from_prior."""
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        captured = {}
        latent_params = {"activity": np.array([[1.0]])}

        def fake_growth_df(orchestrator, params, draw_idx, noise_rng):
            captured["noise_rng"] = noise_rng
            return pd.DataFrame({"ln_cfu": [1.0]})

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   side_effect=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=1,
                         noise=False, seed=0)

        assert captured["noise_rng"] is None

    def test_with_noise_passes_rng(self, tmp_path):
        """noise=True must pass a non-None rng to growth_df_from_prior."""
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        captured = {}
        latent_params = {"activity": np.array([[1.0]])}

        def fake_growth_df(orchestrator, params, draw_idx, noise_rng):
            captured["noise_rng"] = noise_rng
            return pd.DataFrame({"ln_cfu": [1.0]})

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   side_effect=fake_growth_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=1,
                         noise=True, seed=0)

        assert captured["noise_rng"] is not None

    def test_csv_content_written(self, tmp_path):
        from tfscreen.tfmodel.scripts.sample_prior_cli import sample_prior

        latent_params = {"activity": np.array([[1.0]])}
        fake_df = pd.DataFrame({"genotype": ["wt", "A1G"], "ln_cfu": [5.0, 4.5]})

        with patch("tfscreen.tfmodel.scripts.sample_prior_cli.read_configuration",
                   return_value=(MagicMock(), {})), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.draw_prior",
                   return_value=({}, latent_params)), \
             patch("tfscreen.tfmodel.scripts.sample_prior_cli.growth_df_from_prior",
                   return_value=fake_df):

            out_prefix = str(tmp_path / "prior")
            sample_prior("cfg.yaml", out_prefix=out_prefix, num_datasets=1, seed=0)

        df = pd.read_csv(f"{out_prefix}_000_growth.csv")
        assert list(df.columns) == ["genotype", "ln_cfu"]
        assert len(df) == 2
