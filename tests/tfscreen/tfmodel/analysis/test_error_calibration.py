"""
Tests for tfscreen.tfmodel.analysis.error_calibration.

Covers all public functions and the private SBC helpers that are tested by
the now-deleted test_sbc.py.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.analysis.error_calibration import (
    # Core
    pit_from_samples,
    pit_from_quantiles,
    # Stats
    calibration_curve,
    pit_uniformity_test,
    # Plots
    plot_pit_histogram,
    plot_calibration_curve,
    # Single-run
    calibration_summary,
    # SBC
    _find_pairs,
    _load_h5_params,
    compute_sbc_ranks,
    summarize_sbc,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_h5(path, arrays):
    """Write a dict of numpy arrays to an HDF5 file."""
    with h5py.File(path, "w") as hf:
        for key, val in arrays.items():
            hf.create_dataset(key, data=val)


# ============================================================
# Part 1 — pit_from_samples
# ============================================================

class TestPitFromSamples:

    def test_rank_zero_when_gt_below_all_posterior(self):
        # true = -100; all posterior > -100 → PIT = 0
        pit = pit_from_samples(np.array([-100.0]), np.ones((50, 1)))
        np.testing.assert_allclose(pit, [0.0])

    def test_rank_one_when_gt_above_all_posterior(self):
        # true = 100; all posterior < 100 → PIT = 1
        pit = pit_from_samples(np.array([100.0]), -np.ones((50, 1)))
        np.testing.assert_allclose(pit, [1.0])

    def test_rank_half_at_median(self):
        # posterior is -10..10 (21 values), true = 0 → 10/21
        post = np.arange(-10, 11, dtype=float).reshape(21, 1)
        pit = pit_from_samples(np.array([0.0]), post)
        np.testing.assert_allclose(pit, [10 / 21])

    def test_nan_true_value_gives_nan(self):
        post = np.ones((20, 1))
        pit = pit_from_samples(np.array([np.nan]), post)
        assert np.isnan(pit[0])

    def test_mixed_nan_and_valid(self):
        true_vals = np.array([0.0, np.nan, 1000.0])
        post = np.zeros((20, 3))  # all samples = 0
        pit = pit_from_samples(true_vals, post)
        # true=0: mean(0 < 0) = 0
        np.testing.assert_allclose(pit[0], 0.0)
        assert np.isnan(pit[1])
        # true=1000: mean(0 < 1000) = 1
        np.testing.assert_allclose(pit[2], 1.0)

    def test_multidimensional_true_vals(self):
        rng = np.random.default_rng(0)
        post = rng.normal(size=(100, 4))
        true_vals = np.zeros(4)
        pit = pit_from_samples(true_vals, post)
        assert pit.shape == (4,)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(42)
        post = rng.normal(size=(200, 10))
        true_vals = rng.normal(size=10)
        pit = pit_from_samples(true_vals, post)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_scalar_true_val(self):
        # true_vals can be a scalar if posterior is shape (S,)
        post = np.linspace(-1, 1, 101)
        pit = pit_from_samples(0.0, post)
        # 50 values < 0 → 50/101
        np.testing.assert_allclose(pit, 50 / 101)


# ============================================================
# Part 1 — pit_from_quantiles
# ============================================================

class TestPitFromQuantiles:

    def _make_gaussian_quantiles(self, n=5):
        """Return quantile_matrix and levels for N(0,1) at 9 levels."""
        from scipy.stats import norm
        levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        qmat = np.tile(norm.ppf(levels), (n, 1))  # same distribution for all n
        return qmat, levels

    def test_true_at_stored_quantile_returns_that_level(self):
        from scipy.stats import norm
        levels = np.array([0.1, 0.5, 0.9])
        qmat = norm.ppf(levels).reshape(1, 3)
        # True value exactly at the 50th percentile
        pit = pit_from_quantiles(np.array([0.0]), qmat, levels)
        np.testing.assert_allclose(pit, [0.5])

    def test_true_below_all_quantiles_returns_zero(self):
        levels = np.array([0.1, 0.5, 0.9])
        qmat = np.array([[1.0, 2.0, 3.0]])
        pit = pit_from_quantiles(np.array([-999.0]), qmat, levels)
        np.testing.assert_allclose(pit, [0.0])

    def test_true_above_all_quantiles_returns_one(self):
        levels = np.array([0.1, 0.5, 0.9])
        qmat = np.array([[1.0, 2.0, 3.0]])
        pit = pit_from_quantiles(np.array([999.0]), qmat, levels)
        np.testing.assert_allclose(pit, [1.0])

    def test_linear_interpolation_between_levels(self):
        # quantile values 0,1,2,3,4 at levels 0,0.25,0.50,0.75,1.0
        levels = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        qmat = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        # True = 1.5: halfway between q=1.0 (level 0.25) and q=2.0 (level 0.5)
        pit = pit_from_quantiles(np.array([1.5]), qmat, levels)
        np.testing.assert_allclose(pit, [0.375], atol=1e-10)

    def test_nan_true_value_gives_nan(self):
        levels = np.array([0.1, 0.5, 0.9])
        qmat = np.array([[1.0, 2.0, 3.0]])
        pit = pit_from_quantiles(np.array([np.nan]), qmat, levels)
        assert np.isnan(pit[0])

    def test_nan_in_quantile_row_gives_nan(self):
        levels = np.array([0.1, 0.5, 0.9])
        qmat = np.array([[1.0, np.nan, 3.0]])
        pit = pit_from_quantiles(np.array([2.0]), qmat, levels)
        assert np.isnan(pit[0])

    def test_shape_preserved(self):
        qmat, levels = self._make_gaussian_quantiles(n=7)
        true_vals = np.zeros(7)
        pit = pit_from_quantiles(true_vals, qmat, levels)
        assert pit.shape == (7,)

    def test_output_in_unit_interval_for_gaussian(self):
        rng = np.random.default_rng(7)
        qmat, levels = self._make_gaussian_quantiles(n=20)
        true_vals = rng.normal(size=20)
        pit = pit_from_quantiles(true_vals, qmat, levels)
        finite = pit[np.isfinite(pit)]
        assert np.all((finite >= 0) & (finite <= 1))

    def test_multiple_rows_independent(self):
        # Each row has different quantile values; all true values at median
        levels = np.array([0.1, 0.5, 0.9])
        qmat = np.array([
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
        ])
        true_vals = np.array([1.0, 20.0])  # each at the median
        pit = pit_from_quantiles(true_vals, qmat, levels)
        np.testing.assert_allclose(pit, [0.5, 0.5])


# ============================================================
# Part 2 — calibration_curve
# ============================================================

class TestCalibrationCurve:

    def test_uniform_pit_gives_empirical_near_nominal(self):
        rng = np.random.default_rng(0)
        pit = rng.uniform(0, 1, size=20_000)
        curve = calibration_curve(pit)
        for nominal, empirical in curve.items():
            assert abs(empirical - nominal) < 0.03, (
                f"nominal={nominal:.2f}, empirical={empirical:.3f}"
            )

    def test_overconfident_model_below_diagonal(self):
        # All PIT near 0 or 1 (U-shaped) → CIs too narrow → empirical < nominal
        rng = np.random.default_rng(1)
        pit = np.concatenate([
            rng.uniform(0.0, 0.05, 500),
            rng.uniform(0.95, 1.0, 500),
        ])
        curve = calibration_curve(pit, levels=[0.9])
        # For 90% CI we expect ~90% coverage but get ~0% (all PIT outside [0.05, 0.95])
        assert curve[0.9] < 0.5

    def test_underconfident_model_above_diagonal(self):
        # All PIT near 0.5 → posterior spread too wide → empirical > nominal for tight CIs
        pit = np.full(1000, 0.5)
        curve = calibration_curve(pit, levels=[0.5])
        # 50% CI spans [0.25, 0.75]; all PIT = 0.5 are inside → empirical = 1.0 > 0.5
        assert curve[0.5] > 0.9

    def test_returns_dict_with_default_keys(self):
        pit = np.linspace(0.01, 0.99, 100)
        curve = calibration_curve(pit)
        assert set(curve.keys()) == {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}

    def test_custom_levels(self):
        pit = np.linspace(0, 1, 200)
        curve = calibration_curve(pit, levels=[0.8, 0.9])
        assert set(curve.keys()) == {0.8, 0.9}

    def test_nan_in_pit_ignored(self):
        rng = np.random.default_rng(2)
        pit = rng.uniform(0, 1, size=1000)
        pit_with_nan = np.concatenate([pit, np.full(50, np.nan)])
        curve_clean = calibration_curve(pit)
        curve_nan = calibration_curve(pit_with_nan)
        for k in curve_clean:
            np.testing.assert_allclose(curve_clean[k], curve_nan[k])

    def test_values_in_unit_interval(self):
        pit = np.linspace(0.01, 0.99, 200)
        curve = calibration_curve(pit)
        for v in curve.values():
            assert 0.0 <= v <= 1.0

    def test_empty_pit_after_nan_removal(self):
        pit = np.full(10, np.nan)
        with pytest.warns(RuntimeWarning):
            curve = calibration_curve(pit)
        # np.mean of an empty slice returns NaN
        for v in curve.values():
            assert np.isnan(v)


# ============================================================
# Part 2 — pit_uniformity_test
# ============================================================

class TestPitUniformityTest:

    def test_uniform_input_high_pvalue(self):
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=2000)
        result = pit_uniformity_test(pit)
        assert result["ks_pval"] > 0.05

    def test_nonuniform_input_low_pvalue(self):
        # Bimodal near 0 and 1 — clearly not uniform
        pit = np.concatenate([np.full(200, 0.01), np.full(200, 0.99)])
        result = pit_uniformity_test(pit)
        assert result["ks_pval"] < 0.001

    def test_returns_expected_keys(self):
        pit = np.linspace(0, 1, 50)
        result = pit_uniformity_test(pit)
        assert set(result.keys()) == {"ks_stat", "ks_pval", "mean_pit", "n_vals"}

    def test_empty_input_returns_nan(self):
        result = pit_uniformity_test(np.array([]))
        assert np.isnan(result["ks_stat"])
        assert np.isnan(result["ks_pval"])
        assert np.isnan(result["mean_pit"])
        assert result["n_vals"] == 0

    def test_all_nan_returns_nan(self):
        result = pit_uniformity_test(np.full(10, np.nan))
        assert np.isnan(result["ks_stat"])
        assert result["n_vals"] == 0

    def test_mean_pit_near_half_for_uniform(self):
        rng = np.random.default_rng(5)
        pit = rng.uniform(0, 1, size=5000)
        result = pit_uniformity_test(pit)
        assert abs(result["mean_pit"] - 0.5) < 0.05

    def test_n_vals_excludes_nan(self):
        pit = np.array([0.1, 0.5, 0.9, np.nan, np.nan])
        result = pit_uniformity_test(pit)
        assert result["n_vals"] == 3


# ============================================================
# Part 3 — plot_pit_histogram
# ============================================================

class TestPlotPitHistogram:

    def test_returns_axes(self):
        from matplotlib.axes import Axes
        pit = np.linspace(0.05, 0.95, 50)
        ax = plot_pit_histogram(pit)
        assert isinstance(ax, Axes)

    def test_accepts_existing_axes(self):
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes
        fig, ax = plt.subplots()
        pit = np.linspace(0.1, 0.9, 30)
        returned = plot_pit_histogram(pit, ax=ax)
        assert returned is ax
        plt.close(fig)

    def test_title_set(self):
        from matplotlib import pyplot as plt
        pit = np.linspace(0.1, 0.9, 30)
        ax = plot_pit_histogram(pit, title="my title")
        assert ax.get_title() == "my title"
        plt.close(ax.get_figure())

    def test_nan_filtered_no_crash(self):
        pit = np.array([0.1, np.nan, 0.5, np.nan, 0.9])
        ax = plot_pit_histogram(pit)
        from matplotlib import pyplot as plt
        plt.close(ax.get_figure())

    def test_custom_bins(self):
        from matplotlib import pyplot as plt
        pit = np.linspace(0.01, 0.99, 100)
        ax = plot_pit_histogram(pit, n_bins=5)
        # Just checks it runs without error
        plt.close(ax.get_figure())


# ============================================================
# Part 3 — plot_calibration_curve
# ============================================================

class TestPlotCalibrationCurve:

    def test_returns_axes(self):
        from matplotlib.axes import Axes
        curve = {0.5: 0.48, 0.9: 0.88, 0.95: 0.93}
        ax = plot_calibration_curve(curve)
        assert isinstance(ax, Axes)

    def test_accepts_existing_axes(self):
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes
        fig, ax = plt.subplots()
        curve = {0.5: 0.5, 0.9: 0.9}
        returned = plot_calibration_curve(curve, ax=ax)
        assert returned is ax
        plt.close(fig)

    def test_empty_dict_no_crash(self):
        from matplotlib import pyplot as plt
        ax = plot_calibration_curve({})
        plt.close(ax.get_figure())

    def test_label_appears_in_legend(self):
        from matplotlib import pyplot as plt
        curve = {0.5: 0.5, 0.9: 0.9}
        ax = plot_calibration_curve(curve, label="TestModel")
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "TestModel" in legend_texts
        plt.close(ax.get_figure())


# ============================================================
# Part 4 — calibration_summary
# ============================================================

def _make_calibration_inputs(n=40, seed=0):
    """Return (true_vals, quantile_matrix, quantile_levels) for N(0,1).

    Stores 9 quantile levels; true values are drawn from N(0,1) so the PIT
    values should be approximately uniform.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    true_vals = rng.normal(size=n)
    # All observations share the same N(0,1) posterior
    qmat = np.tile(norm.ppf(levels), (n, 1))
    return true_vals, qmat, levels


class TestCalibrationSummary:

    def test_writes_pit_csv(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        assert os.path.exists(f"{prefix}_pit.csv")

    def test_pit_csv_has_correct_columns(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        df = pd.read_csv(f"{prefix}_pit.csv")
        assert "true_val" in df.columns
        assert "pit" in df.columns

    def test_pit_csv_has_correct_length(self, tmp_path):
        n = 25
        true_vals, qmat, levels = _make_calibration_inputs(n=n)
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        df = pd.read_csv(f"{prefix}_pit.csv")
        assert len(df) == n

    def test_writes_calibration_curve_csv(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        assert os.path.exists(f"{prefix}_calibration_curve.csv")

    def test_calibration_curve_csv_columns(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        df = pd.read_csv(f"{prefix}_calibration_curve.csv")
        assert "nominal" in df.columns
        assert "empirical" in df.columns

    def test_writes_pdf(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        assert os.path.exists(f"{prefix}_calibration.pdf")

    def test_returns_dict_with_expected_keys(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        result = calibration_summary(true_vals, qmat, levels, prefix)
        for key in ("n_vals", "ks_stat", "ks_pval", "mean_pit", "calibration_curve"):
            assert key in result

    def test_calibration_curve_in_result(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs()
        prefix = str(tmp_path / "cal")
        result = calibration_summary(true_vals, qmat, levels, prefix)
        assert isinstance(result["calibration_curve"], dict)
        assert len(result["calibration_curve"]) > 0

    def test_n_vals_matches_finite_pit(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs(n=30)
        prefix = str(tmp_path / "cal")
        result = calibration_summary(true_vals, qmat, levels, prefix)
        assert result["n_vals"] == 30

    def test_nan_true_vals_handled_gracefully(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs(n=20)
        true_vals[::4] = np.nan  # introduce NaNs
        prefix = str(tmp_path / "cal")
        result = calibration_summary(true_vals, qmat, levels, prefix)
        assert result["n_vals"] < 20

    def test_label_used_in_no_crash(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs(n=10)
        prefix = str(tmp_path / "cal")
        # Should not raise
        calibration_summary(true_vals, qmat, levels, prefix, label="theta_high")

    def test_creates_output_directory(self, tmp_path):
        true_vals, qmat, levels = _make_calibration_inputs(n=10)
        prefix = str(tmp_path / "nested" / "deep" / "cal")
        calibration_summary(true_vals, qmat, levels, prefix)
        assert os.path.exists(str(tmp_path / "nested" / "deep"))

    def test_well_calibrated_model_passes_ks(self, tmp_path):
        # N(0,1) true values with N(0,1) posterior → PIT ~ Uniform
        true_vals, qmat, levels = _make_calibration_inputs(n=200, seed=99)
        prefix = str(tmp_path / "cal")
        result = calibration_summary(true_vals, qmat, levels, prefix)
        # With 200 samples the KS test should not flag a calibrated model
        assert result["ks_pval"] > 0.01


# ============================================================
# Part 5 — _load_h5_params (migrated from test_sbc.py)
# ============================================================

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


# ============================================================
# Part 5 — compute_sbc_ranks (migrated from test_sbc.py)
# ============================================================

class TestComputeSBCRanks:

    def _make_pair(self, tmp_path, gt_val, posterior_samples, name="param"):
        gt_path = str(tmp_path / "run_ground_truth.h5")
        post_path = str(tmp_path / "run_posterior.h5")
        _write_h5(gt_path, {name: np.array(gt_val)[np.newaxis, :]})
        _write_h5(post_path, {name: np.array(posterior_samples)})
        return gt_path, post_path

    def test_rank_zero_when_gt_below_all_posterior(self, tmp_path):
        gt_path, post_path = self._make_pair(
            tmp_path, gt_val=[-100.0], posterior_samples=np.ones((20, 1))
        )
        ranks = compute_sbc_ranks(gt_path, post_path)
        np.testing.assert_allclose(ranks["param"], [0.0])

    def test_rank_one_when_gt_above_all_posterior(self, tmp_path):
        gt_path, post_path = self._make_pair(
            tmp_path, gt_val=[100.0], posterior_samples=-np.ones((20, 1))
        )
        ranks = compute_sbc_ranks(gt_path, post_path)
        np.testing.assert_allclose(ranks["param"], [1.0])

    def test_rank_half_when_gt_at_median(self, tmp_path):
        post = np.arange(-10, 11, dtype=float).reshape(21, 1)
        gt_path, post_path = self._make_pair(tmp_path, gt_val=[0.0], posterior_samples=post)
        ranks = compute_sbc_ranks(gt_path, post_path)
        np.testing.assert_allclose(ranks["param"], [10 / 21])

    def test_multidimensional_param(self, tmp_path):
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


# ============================================================
# Part 5 — _find_pairs (migrated from test_sbc.py)
# ============================================================

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


# ============================================================
# Part 5 — summarize_sbc (migrated from test_sbc.py)
# ============================================================

def _make_calibrated_sbc_dir(tmp_path, n_runs=20, n_samples=200, seed=0):
    """Create an SBC directory with approximately uniform ranks.

    For each run: gt ~ N(0,1), posterior ~ N(0,1) — ranks are Uniform(0,1).
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
        assert len(df) == 1
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
        gt = rng.normal(size=(1, 2))
        post = rng.normal(size=(50, 2))
        _write_h5(str(sbc_dir / "run0000_ground_truth.h5"), {"alpha": gt})
        _write_h5(str(sbc_dir / "run0000_posterior.h5"), {"alpha": post})
        _write_h5(str(sbc_dir / "run0001_ground_truth.h5"),
                  {"alpha": rng.normal(size=(1, 2))})
        df = summarize_sbc(str(sbc_dir))
        assert not df.empty
        assert df.iloc[0]["n_runs"] == 1

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarize_sbc(str(tmp_path / "does_not_exist"))

    def test_mean_rank_near_half_for_calibrated_model(self, tmp_path):
        sbc_dir = _make_calibrated_sbc_dir(tmp_path, n_runs=100, n_samples=500)
        df = summarize_sbc(sbc_dir)
        assert abs(df.iloc[0]["mean_rank"] - 0.5) < 0.05
