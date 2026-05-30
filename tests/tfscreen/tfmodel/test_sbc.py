"""
Tests for tfscreen.tfmodel.analysis.sbc.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from tfscreen.tfmodel.analysis.sbc import (
    _find_pairs,
    _load_h5_params,
    compute_sbc_ranks,
    summarize_sbc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_h5(path, arrays):
    """Write a dict of numpy arrays to an HDF5 file."""
    with h5py.File(path, "w") as hf:
        for key, val in arrays.items():
            hf.create_dataset(key, data=val)


# ---------------------------------------------------------------------------
# _load_h5_params
# ---------------------------------------------------------------------------

class TestLoadH5Params:

    def test_round_trips_arrays(self, tmp_path):
        data = {"alpha": np.array([1.0, 2.0, 3.0]), "beta": np.array([[4.0, 5.0]])}
        p = str(tmp_path / "test.h5")
        _write_h5(p, data)
        result = _load_h5_params(p)
        np.testing.assert_allclose(result["alpha"], data["alpha"])
        np.testing.assert_allclose(result["beta"], data["beta"])

    def test_returns_all_keys(self, tmp_path):
        data = {"x": np.ones(3), "y": np.zeros(5)}
        p = str(tmp_path / "test.h5")
        _write_h5(p, data)
        result = _load_h5_params(p)
        assert set(result.keys()) == {"x", "y"}

    def test_empty_file(self, tmp_path):
        p = str(tmp_path / "empty.h5")
        _write_h5(p, {})
        result = _load_h5_params(p)
        assert result == {}


# ---------------------------------------------------------------------------
# compute_sbc_ranks
# ---------------------------------------------------------------------------

class TestComputeSBCRanks:

    def _make_pair(self, tmp_path, gt_val, posterior_samples, name="param"):
        """Write ground-truth (shape (1, *)) and posterior (shape (S, *)) files."""
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {name: np.array(gt_val)[np.newaxis, :]})
        _write_h5(post_path, {name: np.array(posterior_samples)})
        return gt_path, post_path

    def test_rank_zero_when_gt_below_all_posterior(self, tmp_path):
        # gt = -100; all posterior > -100 → rank = 0
        gt_path, post_path = self._make_pair(
            tmp_path,
            gt_val=[-100.0],
            posterior_samples=np.ones((20, 1)),
        )
        ranks = compute_sbc_ranks(gt_path, post_path)
        assert "param" in ranks
        np.testing.assert_allclose(ranks["param"], [0.0])

    def test_rank_one_when_gt_above_all_posterior(self, tmp_path):
        # gt = 100; all posterior < 100 → rank = 1
        gt_path, post_path = self._make_pair(
            tmp_path,
            gt_val=[100.0],
            posterior_samples=-np.ones((20, 1)),
        )
        ranks = compute_sbc_ranks(gt_path, post_path)
        np.testing.assert_allclose(ranks["param"], [1.0])

    def test_rank_half_when_gt_at_median(self, tmp_path):
        # posterior is [-10, -9, ..., -1, 0, 1, ..., 10] (21 values), gt = 0
        # 10 values < 0, 11 >= 0 → rank = 10/21
        post = np.arange(-10, 11, dtype=float).reshape(21, 1)
        gt_path, post_path = self._make_pair(
            tmp_path, gt_val=[0.0], posterior_samples=post
        )
        ranks = compute_sbc_ranks(gt_path, post_path)
        np.testing.assert_allclose(ranks["param"], [10 / 21])

    def test_multidimensional_param(self, tmp_path):
        # param shape (3,): three independent elements
        rng = np.random.default_rng(0)
        post = rng.normal(size=(100, 3))
        gt = np.zeros((1, 3))
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {"param": gt})
        _write_h5(post_path, {"param": post})
        ranks = compute_sbc_ranks(gt_path, post_path)
        assert ranks["param"].shape == (3,)
        assert np.all((ranks["param"] >= 0) & (ranks["param"] <= 1))

    def test_missing_posterior_param_skipped(self, tmp_path):
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {"param_a": np.ones((1, 2)), "param_b": np.ones((1, 2))})
        _write_h5(post_path, {"param_a": np.ones((10, 2))})
        ranks = compute_sbc_ranks(gt_path, post_path)
        assert "param_a" in ranks
        assert "param_b" not in ranks

    def test_returns_empty_when_no_common_params(self, tmp_path):
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {"a": np.ones((1, 2))})
        _write_h5(post_path, {"b": np.ones((10, 2))})
        ranks = compute_sbc_ranks(gt_path, post_path)
        assert ranks == {}

    def test_rank_values_in_unit_interval(self, tmp_path):
        rng = np.random.default_rng(42)
        post = rng.normal(size=(50, 4))
        gt = rng.normal(size=(1, 4))
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {"theta": gt})
        _write_h5(post_path, {"theta": post})
        ranks = compute_sbc_ranks(gt_path, post_path)
        assert np.all((ranks["theta"] >= 0) & (ranks["theta"] <= 1))


# ---------------------------------------------------------------------------
# _find_pairs
# ---------------------------------------------------------------------------

class TestFindPairs:

    def test_finds_paired_files(self, tmp_path):
        for prefix in ("run001", "run002"):
            (tmp_path / f"{prefix}_ground_truth.h5").touch()
            (tmp_path / f"{prefix}_posterior.h5").touch()
        pairs = _find_pairs(str(tmp_path))
        assert len(pairs) == 2
        run_ids = {p[0] for p in pairs}
        assert run_ids == {"run001", "run002"}

    def test_missing_posterior_gives_none(self, tmp_path):
        (tmp_path / "run001_ground_truth.h5").touch()
        pairs = _find_pairs(str(tmp_path))
        assert len(pairs) == 1
        run_id, gt_path, post_path = pairs[0]
        assert run_id == "run001"
        assert post_path is None

    def test_returns_empty_when_no_gt_files(self, tmp_path):
        (tmp_path / "run001_posterior.h5").touch()
        pairs = _find_pairs(str(tmp_path))
        assert pairs == []

    def test_gt_path_is_absolute(self, tmp_path):
        (tmp_path / "run001_ground_truth.h5").touch()
        (tmp_path / "run001_posterior.h5").touch()
        pairs = _find_pairs(str(tmp_path))
        _, gt_path, post_path = pairs[0]
        assert os.path.isabs(gt_path)
        assert os.path.isabs(post_path)


# ---------------------------------------------------------------------------
# summarize_sbc (integration)
# ---------------------------------------------------------------------------

def _make_calibrated_sbc_dir(tmp_path, n_runs=20, n_samples=200, seed=0):
    """
    Create a synthetic SBC directory where ranks are genuinely uniform.

    For each run: draw gt ~ N(0,1), draw posterior ~ N(0,1) (S samples).
    The ranks will be approximately uniform under this prior-predictive setup.
    """
    rng = np.random.default_rng(seed)
    sbc_dir = tmp_path / "sbc"
    sbc_dir.mkdir()
    for i in range(n_runs):
        gt = rng.normal(size=(1, 3))
        post = rng.normal(size=(n_samples, 3))
        run_id = f"run{i:04d}"
        _write_h5(str(sbc_dir / f"{run_id}_ground_truth.h5"), {"alpha": gt})
        _write_h5(str(sbc_dir / f"{run_id}_posterior.h5"), {"alpha": post})
    return str(sbc_dir)


class TestSummarizeSBC:

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        df = summarize_sbc(sbc_dir)
        assert not df.empty
        for col in ("param", "n_runs", "n_ranks", "mean_rank", "ks_stat", "ks_pval"):
            assert col in df.columns

    def test_one_row_per_parameter(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        df = summarize_sbc(sbc_dir)
        assert len(df) == 1  # only "alpha"
        assert df.iloc[0]["param"] == "alpha"

    def test_n_runs_correct(self, tmp_path):
        n_runs = 15
        sbc_dir = _make_calibrated_sbc_dir(tmp_path, n_runs=n_runs)
        df = summarize_sbc(sbc_dir)
        assert df.iloc[0]["n_runs"] == n_runs

    def test_n_ranks_correct(self, tmp_path):
        # 10 runs × 3 elements each = 30 rank values
        sbc_dir = _make_calibrated_sbc_dir(tmp_path, n_runs=10)
        df = summarize_sbc(sbc_dir)
        assert df.iloc[0]["n_ranks"] == 30

    def test_writes_summary_csv(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        summarize_sbc(sbc_dir)
        assert os.path.exists(os.path.join(sbc_dir, "sbc_sbc_summary.csv"))

    def test_writes_ranks_csv(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        summarize_sbc(sbc_dir)
        assert os.path.exists(os.path.join(sbc_dir, "sbc_sbc_ranks.csv"))

    def test_writes_histogram_pdf(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        summarize_sbc(sbc_dir)
        assert os.path.exists(os.path.join(sbc_dir, "sbc_rank_hist.pdf"))

    def test_custom_out_prefix(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path)
        prefix = str(tmp_path / "myprefix")
        summarize_sbc(sbc_dir, out_prefix=prefix)
        assert os.path.exists(f"{prefix}_sbc_summary.csv")
        assert os.path.exists(f"{prefix}_sbc_ranks.csv")

    def test_empty_dir_returns_empty_dataframe(self, tmp_path):
        sbc_dir = str(tmp_path / "empty")
        os.makedirs(sbc_dir)
        df = summarize_sbc(sbc_dir)
        assert df.empty

    def test_missing_posterior_skipped_without_crash(self, tmp_path):
        sbc_dir = tmp_path / "sbc"
        sbc_dir.mkdir()
        rng = np.random.default_rng(1)
        # Run 0: has both files
        gt = rng.normal(size=(1, 2))
        post = rng.normal(size=(50, 2))
        _write_h5(str(sbc_dir / "run0000_ground_truth.h5"), {"alpha": gt})
        _write_h5(str(sbc_dir / "run0000_posterior.h5"), {"alpha": post})
        # Run 1: only ground truth
        _write_h5(str(sbc_dir / "run0001_ground_truth.h5"), {"alpha": rng.normal(size=(1, 2))})
        df = summarize_sbc(str(sbc_dir))
        assert not df.empty
        assert df.iloc[0]["n_runs"] == 1

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarize_sbc(str(tmp_path / "does_not_exist"))

    def test_mean_rank_near_half_for_calibrated_model(self, tmp_path):
        # With many runs, mean rank should be close to 0.5
        sbc_dir = _make_calibrated_sbc_dir(tmp_path, n_runs=100, n_samples=500)
        df = summarize_sbc(sbc_dir)
        assert abs(df.iloc[0]["mean_rank"] - 0.5) < 0.05
