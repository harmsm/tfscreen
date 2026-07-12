"""
Tests for summarize_grid_cli.py — grid summary CSV generation and
fit_summary.json ingestion.
"""
import json
import os

import pandas as pd
import pytest

from tfscreen.tfmodel.scripts.summarize_grid_cli import (
    _flatten_fit_summary,
    summarize_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_combo(subdir, configure_model=None, template=None):
    """Write a minimal combo.json into subdir."""
    combo = {
        "configure_model": configure_model or {"theta": "hill_geno"},
        "template": template or {},
    }
    with open(os.path.join(subdir, "combo.json"), "w") as fh:
        json.dump(combo, fh)


def _make_config_yaml(subdir):
    """Touch the sentinel file that marks configure as complete."""
    open(os.path.join(subdir, "tfs_configure_config.yaml"), "w").close()


def _make_fit_summary(subdir, training_rmse=0.05, test_rmse=None,
                      growth_rmse=None, n_parameters=100,
                      final_loss=-5000.0, filename="tfs_summarize_fit_summary.json"):
    """Write a realistic fit_summary JSON into subdir."""
    theta_training = {"rmse": training_rmse, "pearson_r": 0.95, "spearman_r": 0.94,
                      "r_squared": 0.90, "mean_error": 0.001,
                      "pct_success": 1.0, "normalized_rmse": 0.1,
                      "residual_corr": 0.02, "residual_corr_p_value": 0.4,
                      "bp_p_value": 0.3, "coverage_prob": None}
    theta_test = None
    if test_rmse is not None:
        theta_test = dict(theta_training)
        theta_test["rmse"] = test_rmse

    growth_training = None
    if growth_rmse is not None:
        growth_training = dict(theta_training)
        growth_training["rmse"] = growth_rmse

    data = {
        "metadata": {
            "n_parameters": n_parameters,
            "n_theta_training_points": 200,
            "n_theta_test_points": 50 if test_rmse is not None else None,
            "n_growth_training_points": 500 if growth_rmse is not None else None,
            "final_loss": final_loss,
        },
        "theta": {"training": theta_training, "test": theta_test},
        "growth": {"training": growth_training},
    }
    with open(os.path.join(subdir, filename), "w") as fh:
        json.dump(data, fh)
    return data


# ---------------------------------------------------------------------------
# _flatten_fit_summary
# ---------------------------------------------------------------------------

class TestFlattenFitSummary:

    def test_metadata_scalars_present(self):
        data = _make_fit_summary.__wrapped__ if hasattr(_make_fit_summary, "__wrapped__") \
            else None
        raw = {
            "metadata": {"n_parameters": 42, "final_loss": -1234.5,
                         "n_theta_training_points": 100, "n_theta_test_points": None,
                         "n_growth_training_points": 300},
            "theta": {"training": {"rmse": 0.05}, "test": None},
            "growth": {"training": None},
        }
        flat = _flatten_fit_summary(raw)
        assert flat["n_parameters"] == 42
        assert flat["final_loss"] == pytest.approx(-1234.5)
        assert flat["n_theta_training_points"] == 100

    def test_theta_training_stats_prefixed(self):
        raw = {
            "metadata": {},
            "theta": {"training": {"rmse": 0.07, "pearson_r": 0.9}, "test": None},
            "growth": {"training": None},
        }
        flat = _flatten_fit_summary(raw)
        assert flat["theta_training_rmse"] == pytest.approx(0.07)
        assert flat["theta_training_pearson_r"] == pytest.approx(0.9)

    def test_theta_test_stats_prefixed(self):
        raw = {
            "metadata": {},
            "theta": {"training": None,
                      "test": {"rmse": 0.12, "spearman_r": 0.88}},
            "growth": {"training": None},
        }
        flat = _flatten_fit_summary(raw)
        assert flat["theta_test_rmse"] == pytest.approx(0.12)
        assert flat["theta_test_spearman_r"] == pytest.approx(0.88)

    def test_growth_training_stats_prefixed(self):
        raw = {
            "metadata": {},
            "theta": {"training": None, "test": None},
            "growth": {"training": {"rmse": 0.3, "pearson_r": 0.85}},
        }
        flat = _flatten_fit_summary(raw)
        assert flat["growth_training_rmse"] == pytest.approx(0.3)

    def test_null_sections_produce_no_keys(self):
        raw = {
            "metadata": {"n_parameters": 10},
            "theta": {"training": None, "test": None},
            "growth": {"training": None},
        }
        flat = _flatten_fit_summary(raw)
        assert not any(k.startswith("theta_training_") for k in flat)
        assert not any(k.startswith("theta_test_") for k in flat)
        assert not any(k.startswith("growth_training_") for k in flat)

    def test_missing_sections_handled_gracefully(self):
        flat = _flatten_fit_summary({})
        assert flat == {}

    def test_run_dir_and_timestamp_excluded(self):
        raw = {
            "metadata": {"run_dir": "/some/path", "timestamp": "2025-01-01",
                         "final_loss": -999.0},
            "theta": {"training": None, "test": None},
            "growth": {"training": None},
        }
        flat = _flatten_fit_summary(raw)
        assert "run_dir" not in flat
        assert "timestamp" not in flat
        assert "final_loss" in flat


# ---------------------------------------------------------------------------
# summarize_grid — basic behaviour (no fit summary)
# ---------------------------------------------------------------------------

class TestSummarizeGridBasic:

    def test_returns_dataframe(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_csv_written(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert os.path.exists(str(tmp_path / "out.csv"))

    def test_no_combo_json_skipped(self, tmp_path):
        (tmp_path / "run_0001").mkdir()
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert len(df) == 0

    def test_configure_complete_flag(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        _make_config_yaml(str(sub))
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert df.loc[0, "configure_complete"] is True or df.loc[0, "configure_complete"] == True


# ---------------------------------------------------------------------------
# summarize_grid — fit summary ingestion
# ---------------------------------------------------------------------------

class TestSummarizeGridFitSummary:

    def test_fit_summary_columns_present(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        _make_fit_summary(str(sub))
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert "final_loss" in df.columns
        assert "n_parameters" in df.columns
        assert "theta_training_rmse" in df.columns

    def test_fit_summary_values_correct(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        _make_fit_summary(str(sub), training_rmse=0.07, final_loss=-9999.0,
                          n_parameters=55)
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert df.loc[0, "theta_training_rmse"] == pytest.approx(0.07)
        assert df.loc[0, "final_loss"] == pytest.approx(-9999.0)
        assert df.loc[0, "n_parameters"] == 55

    def test_test_stats_included_when_present(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        _make_fit_summary(str(sub), test_rmse=0.15)
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert "theta_test_rmse" in df.columns
        assert df.loc[0, "theta_test_rmse"] == pytest.approx(0.15)

    def test_growth_stats_included_when_present(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        _make_fit_summary(str(sub), growth_rmse=0.4)
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert "growth_training_rmse" in df.columns
        assert df.loc[0, "growth_training_rmse"] == pytest.approx(0.4)

    def test_run_without_fit_summary_gets_nan(self, tmp_path):
        sub1 = tmp_path / "run_0001"
        sub2 = tmp_path / "run_0002"
        sub1.mkdir()
        sub2.mkdir()
        _make_combo(str(sub1))
        _make_fit_summary(str(sub1), training_rmse=0.05)
        _make_combo(str(sub2))
        # run_0002 has no fit summary
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert len(df) == 2
        row1 = df[df["run"] == "run_0001"].iloc[0]
        row2 = df[df["run"] == "run_0002"].iloc[0]
        assert row1["theta_training_rmse"] == pytest.approx(0.05)
        assert pd.isna(row2["theta_training_rmse"])

    def test_malformed_json_skipped_gracefully(self, tmp_path):
        sub = tmp_path / "run_0001"
        sub.mkdir()
        _make_combo(str(sub))
        with open(os.path.join(str(sub), "tfs_summarize_fit_summary.json"), "w") as fh:
            fh.write("{ not valid json }")
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert len(df) == 1
        # No fit summary columns — just the base combo fields
        assert "theta_training_rmse" not in df.columns or pd.isna(
            df.loc[0, "theta_training_rmse"]
        )

    def test_multiple_runs_all_have_fit_summary(self, tmp_path):
        for i, rmse in enumerate([0.03, 0.07, 0.12], start=1):
            sub = tmp_path / f"run_{i:04d}"
            sub.mkdir()
            _make_combo(str(sub))
            _make_fit_summary(str(sub), training_rmse=rmse)
        df = summarize_grid(str(tmp_path), out_prefix=str(tmp_path / "out"))
        assert len(df) == 3
        rmse_vals = sorted(df["theta_training_rmse"].tolist())
        assert rmse_vals == pytest.approx([0.03, 0.07, 0.12])
