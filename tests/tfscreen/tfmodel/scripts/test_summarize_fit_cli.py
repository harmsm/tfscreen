"""
Tests for summarize_fit_cli.py.

All tests use real file I/O in tmp_path so that we exercise the full
read/parse/stats/write pipeline without mocking the heavy ML stack.
Matplotlib is forced to the Agg backend so no display is required.
"""
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import yaml

from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.scripts.summarize_fit_cli import (
    _find_params_or_posterior,
    _find_unique,
    _json_safe,
    _read_all_losses,
    _read_final_loss,
    _resolve_path,
    _try_plot_theta_fits,
    _try_plot_trajectories,
    summarize_fit,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GENOTYPES = ["wt", "A1B", "C2D", "E4F"]
TITRANT_NAME = "IPTG"
TITRANT_CONCS = [0.0, 10.0, 100.0, 1000.0]

# Synthetic "true" theta values; predictions are slightly off
THETA_OBS = [0.05, 0.25, 0.75, 0.95]
THETA_PRED = [0.07, 0.28, 0.71, 0.93]


def _make_binding_csv(path):
    rows = []
    for g, theta in zip(GENOTYPES, THETA_OBS):
        for tc in TITRANT_CONCS:
            rows.append({
                "genotype": g,
                "titrant_name": TITRANT_NAME,
                "titrant_conc": tc,
                "theta_obs": theta + tc * 0.0001,  # tiny variation across conc
                "theta_std": 0.02,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_pred_csv(path, n_training=3, extra_genotypes=None):
    """Write a theta_pred CSV.  First n_training genotypes are in_training_data=1."""
    rows = []
    genotypes = list(GENOTYPES[:n_training])
    if extra_genotypes:
        genotypes += extra_genotypes
    theta_map = dict(zip(GENOTYPES, THETA_PRED))
    for i, g in enumerate(genotypes):
        in_train = 1 if g in GENOTYPES[:n_training] else 0
        theta = theta_map.get(g, 0.5)
        for tc in TITRANT_CONCS:
            rows.append({
                "genotype": g,
                "titrant_name": TITRANT_NAME,
                "titrant_conc": tc,
                "median": theta + tc * 0.0001,
                "lower_95": theta - 0.05,
                "upper_95": theta + 0.05,
                "in_training_data": in_train,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_guesses_csv(path, n_params=20):
    pd.DataFrame({
        "parameter": [f"p{i}" for i in range(n_params)],
        "value": np.ones(n_params),
        "flat_index": range(n_params),
    }).to_csv(path, index=False)


def _make_losses_txt(path, final_loss=-5432.1):
    lines = ["epoch,loss,relative_change"]
    for i in range(1, 10):
        lines.append(f"{i*100},{-10000 + i*50:.1f},{1.0 / i:.6f}")
    lines.append(f"900,{final_loss},0.000010")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _make_config_yaml(path, binding_csv_path, guesses_name, priors_name="test_priors.csv"):
    config = {
        "tfscreen_version": "0.0.0",
        "data": {"binding": binding_csv_path},
        "components": {},
        "priors_file": priors_name,
        "guesses_file": guesses_name,
    }
    with open(path, "w") as fh:
        yaml.dump(config, fh)


@pytest.fixture
def run_dir(tmp_path):
    """Populated run directory with all standard outputs."""
    binding_path = str(tmp_path / "binding.csv")
    _make_binding_csv(binding_path)

    _make_pred_csv(str(tmp_path / "run_theta_pred.csv"))
    _make_guesses_csv(str(tmp_path / "run_guesses.csv"), n_params=25)
    _make_losses_txt(str(tmp_path / "run_losses.txt"), final_loss=-4321.0)
    _make_config_yaml(
        str(tmp_path / "run_config.yaml"),
        binding_csv_path=binding_path,
        guesses_name="run_guesses.csv",
    )
    # Minimal priors file (needed by read_configuration but not by summarize_fit)
    pd.DataFrame({"parameter": [], "value": []}).to_csv(
        str(tmp_path / "test_priors.csv"), index=False
    )
    return str(tmp_path)


def _make_ground_truth_csv(path, theta_col="theta_obs"):
    """Write a ground-truth CSV for the 4th genotype using the given column name."""
    rows = []
    for tc in TITRANT_CONCS:
        rows.append({
            "genotype": "E4F",
            "titrant_name": TITRANT_NAME,
            "titrant_conc": tc,
            theta_col: THETA_OBS[3] + tc * 0.0001,
            "theta_std": 0.02,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def ground_truth_file(tmp_path):
    """Ground-truth file containing the 4th genotype (not in training data)."""
    path = str(tmp_path / "ground_truth.csv")
    _make_ground_truth_csv(path, theta_col="theta_obs")
    return path


# ---------------------------------------------------------------------------
# _find_unique
# ---------------------------------------------------------------------------

class TestFindUnique:

    def test_finds_single_match(self, tmp_path):
        (tmp_path / "run_config.yaml").touch()
        result = _find_unique(str(tmp_path), "_config.yaml", "config")
        assert result == str(tmp_path / "run_config.yaml")

    def test_returns_none_when_missing(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _find_unique(str(tmp_path), "_config.yaml", "config")
        assert result is None
        assert any("No config" in str(x.message) for x in w)

    def test_no_warning_when_warn_missing_false(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _find_unique(str(tmp_path), "_losses.txt", "losses", warn_missing=False)
        assert result is None
        assert not any("losses" in str(x.message) for x in w)

    def test_warns_and_uses_first_on_multiple_matches(self, tmp_path):
        (tmp_path / "a_config.yaml").touch()
        (tmp_path / "b_config.yaml").touch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _find_unique(str(tmp_path), "_config.yaml", "config")
        assert result is not None
        assert any("Multiple" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:

    def test_absolute_path_exists(self, tmp_path):
        p = str(tmp_path / "data.csv")
        open(p, "w").close()
        assert _resolve_path(p, "/some/other/dir") == p

    def test_absolute_path_missing_returns_none(self, tmp_path):
        assert _resolve_path("/nonexistent/file.csv", str(tmp_path)) is None

    def test_relative_resolved_against_run_dir(self, tmp_path):
        (tmp_path / "binding.csv").touch()
        assert _resolve_path("binding.csv", str(tmp_path)) == str(tmp_path / "binding.csv")

    def test_none_input_returns_none(self, tmp_path):
        assert _resolve_path(None, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _read_final_loss
# ---------------------------------------------------------------------------

class TestReadAllLosses:

    def test_returns_all_values_new_format(self, tmp_path):
        """New epoch,loss,relative_change format — loss is column 1."""
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("epoch,loss,relative_change\n")
            fh.write("100,-9000.0,0.5\n200,-8000.0,0.3\n300,-7000.0,0.1\n")
        epochs, losses = _read_all_losses(p)
        assert losses == pytest.approx([-9000.0, -8000.0, -7000.0])
        assert epochs == [100, 200, 300]

    def test_returns_all_values_whitespace(self, tmp_path):
        """Backward-compat: whitespace-delimited step loss."""
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("100 -9000.0\n200 -8000.0\n300 -7000.0\n")
        epochs, losses = _read_all_losses(p)
        assert losses == pytest.approx([-9000.0, -8000.0, -7000.0])

    def test_returns_all_values_old_comma(self, tmp_path):
        """Backward-compat: old comma format loss,other (loss is column 0)."""
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("9000.0,0.002\n8000.0,0.0018\n7000.0,0.0015\n")
        epochs, losses = _read_all_losses(p)
        assert losses == pytest.approx([9000.0, 8000.0, 7000.0])

    def test_header_line_skipped(self, tmp_path):
        """Header row (non-numeric first token) is silently skipped."""
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("epoch,loss,relative_change\n1,-5000.0,0.01\n")
        _, losses = _read_all_losses(p)
        assert len(losses) == 1
        assert losses[0] == pytest.approx(-5000.0)

    def test_raises_on_empty(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("# only comments\n")
        with pytest.raises(ValueError):
            _read_all_losses(p)


class TestReadFinalLoss:

    def test_reads_last_value_from_new_format(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("epoch,loss,relative_change\n")
            fh.write("100,-9000.0,0.5\n200,-8000.0,0.3\n300,-7654.3,0.1\n")
        assert _read_final_loss(p) == pytest.approx(-7654.3)

    def test_reads_last_value_from_whitespace_file(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("100 -9000.0\n200 -8000.0\n300 -7654.3\n")
        assert _read_final_loss(p) == pytest.approx(-7654.3)

    def test_reads_last_value_from_single_column_file(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("-9000.0\n-8000.0\n-7654.3\n")
        assert _read_final_loss(p) == pytest.approx(-7654.3)

    def test_skips_comment_lines(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("# header\n100 -1234.5\n")
        assert _read_final_loss(p) == pytest.approx(-1234.5)

    def test_reads_loss_from_old_comma_format(self, tmp_path):
        """Old comma format: loss is first column (float, not int epoch)."""
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("9000.0,0.002\n8000.0,0.0018\n7654.3,0.0015\n")
        assert _read_final_loss(p) == pytest.approx(7654.3)

    def test_raises_on_no_numeric_content(self, tmp_path):
        p = str(tmp_path / "losses.txt")
        with open(p, "w") as fh:
            fh.write("# only comments\n")
        with pytest.raises(ValueError):
            _read_final_loss(p)


# ---------------------------------------------------------------------------
# _json_safe
# ---------------------------------------------------------------------------

class TestJsonSafe:

    def test_converts_numpy_float(self):
        assert _json_safe(np.float64(3.14)) == pytest.approx(3.14)

    def test_converts_numpy_int(self):
        assert _json_safe(np.int64(7)) == 7

    def test_nan_becomes_none(self):
        assert _json_safe(np.nan) is None
        assert _json_safe(np.float64(np.nan)) is None

    def test_dict_recursion(self):
        result = _json_safe({"a": np.float64(1.0), "b": np.nan})
        assert result == {"a": 1.0, "b": None}


# ---------------------------------------------------------------------------
# summarize_fit — complete run
# ---------------------------------------------------------------------------

class TestSummarizeFitComplete:

    def test_json_written(self, run_dir):
        summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json"))

    def test_summary_dir_created(self, run_dir):
        summarize_fit(run_dir)
        assert os.path.isdir(os.path.join(run_dir, "summary"))

    def test_pdf_written(self, run_dir):
        summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_theta_corr.pdf"))

    def test_json_is_valid(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert "metadata" in data
        assert "theta" in data
        assert "growth" in data

    def test_json_theta_and_growth_have_training_subkey(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert "training" in data["theta"]
        assert "test" in data["theta"]
        assert "training" in data["growth"]

    def test_metadata_fields_populated(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        meta = data["metadata"]
        assert meta["n_parameters"] == 25
        assert meta["n_theta_training_points"] > 0
        assert meta["final_loss"] == pytest.approx(-4321.0)

    def test_theta_training_stats_populated(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        stats = data["theta"]["training"]
        assert stats is not None
        for key in ("rmse", "pearson_r", "spearman_r", "r_squared", "mean_error"):
            assert key in stats, f"Missing key: {key}"

    def test_theta_test_null_when_no_ground_truth(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["theta"]["test"] is None
        assert data["metadata"]["n_theta_test_points"] is None

    def test_growth_training_null_when_no_growth_pred(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["growth"]["training"] is None
        assert data["metadata"]["n_growth_training_points"] is None

    def test_growth_training_stats_populated_when_file_present(self, run_dir):
        pd.DataFrame({
            "genotype": ["wt"] * 6,
            "ln_cfu": np.linspace(8.0, 13.0, 6),
            "median": np.linspace(8.1, 13.1, 6),
        }).to_csv(os.path.join(run_dir, "tfs_growth_pred.csv"), index=False)
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        stats = data["growth"]["training"]
        assert stats is not None
        for key in ("rmse", "pearson_r", "spearman_r", "r_squared", "mean_error"):
            assert key in stats, f"Missing key: {key}"
        assert data["metadata"]["n_growth_training_points"] == 6

    def test_growth_training_drops_nan_ln_cfu(self, run_dir):
        df = pd.DataFrame({
            "genotype": ["wt"] * 6,
            "ln_cfu": [8.0, np.nan, 9.0, np.nan, 10.0, 11.0],
            "median": np.linspace(8.1, 13.1, 6),
        })
        df.to_csv(os.path.join(run_dir, "tfs_growth_pred.csv"), index=False)
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["metadata"]["n_growth_training_points"] == 4

    def test_loss_pdf_written(self, run_dir):
        summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_losses.pdf"))

    def test_growth_corr_pdf_written_when_file_present(self, run_dir):
        # Write a minimal growth_pred CSV
        pd.DataFrame({
            "genotype": ["wt"] * 6,
            "ln_cfu": np.linspace(8.0, 13.0, 6),
            "median": np.linspace(8.1, 13.1, 6),
        }).to_csv(os.path.join(run_dir, "tfs_growth_pred.csv"), index=False)
        summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_growth_corr.pdf"))

    def test_growth_corr_pdf_not_written_when_file_absent(self, run_dir):
        summarize_fit(run_dir)
        assert not os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_growth_corr.pdf"))

    # ------------------------------------------------------------------
    # CSV outputs
    # ------------------------------------------------------------------

    def test_theta_corr_training_csv_written(self, run_dir):
        summarize_fit(run_dir)
        csv_path = os.path.join(run_dir, "summary", "tfs_summarize_theta_corr_training.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert "theta_obs" in df.columns
        assert "median" in df.columns
        assert len(df) > 0

    def test_theta_corr_training_csv_not_written_when_no_binding(self, run_dir):
        config_path = os.path.join(run_dir, "run_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["data"]["binding"] = "/nonexistent/binding.csv"
        with open(config_path, "w") as fh:
            yaml.dump(cfg, fh)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        assert not os.path.exists(
            os.path.join(run_dir, "summary", "tfs_summarize_theta_corr_training.csv")
        )

    def test_theta_corr_test_csv_not_written_without_ground_truth(self, run_dir):
        summarize_fit(run_dir)
        assert not os.path.exists(
            os.path.join(run_dir, "summary", "tfs_summarize_theta_corr_test.csv")
        )

    def test_theta_corr_test_csv_written_with_ground_truth(self, run_dir, ground_truth_file):
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])
        summarize_fit(run_dir, ground_truth_file=ground_truth_file)
        csv_path = os.path.join(run_dir, "summary", "tfs_summarize_theta_corr_test.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert "theta_obs" in df.columns
        assert "median" in df.columns
        assert len(df) == len(TITRANT_CONCS)

    def test_growth_corr_csv_symlink_created(self, run_dir):
        growth_pred_path = os.path.join(run_dir, "tfs_growth_pred.csv")
        pd.DataFrame({
            "genotype": ["wt"] * 6,
            "ln_cfu": np.linspace(8.0, 13.0, 6),
            "median": np.linspace(8.1, 13.1, 6),
        }).to_csv(growth_pred_path, index=False)
        summarize_fit(run_dir)
        csv_path = os.path.join(run_dir, "summary", "tfs_summarize_growth_corr.csv")
        assert os.path.islink(csv_path)
        # symlink must resolve to the actual growth pred file
        assert os.path.realpath(csv_path) == os.path.realpath(growth_pred_path)

    def test_growth_corr_csv_symlink_not_created_when_file_absent(self, run_dir):
        summarize_fit(run_dir)
        assert not os.path.exists(
            os.path.join(run_dir, "summary", "tfs_summarize_growth_corr.csv")
        )

    def test_growth_corr_csv_symlink_overwritten_on_rerun(self, run_dir):
        growth_pred_path = os.path.join(run_dir, "tfs_growth_pred.csv")
        pd.DataFrame({
            "genotype": ["wt"] * 3,
            "ln_cfu": [8.0, 9.0, 10.0],
            "median": [8.1, 9.1, 10.1],
        }).to_csv(growth_pred_path, index=False)
        summarize_fit(run_dir)
        summarize_fit(run_dir)  # second call must not raise
        csv_path = os.path.join(run_dir, "summary", "tfs_summarize_growth_corr.csv")
        assert os.path.islink(csv_path)

    def test_custom_out_prefix(self, run_dir, tmp_path):
        custom_prefix = str(tmp_path / "custom" / "myrun")
        summarize_fit(run_dir, out_prefix=custom_prefix)
        assert os.path.exists(f"{custom_prefix}_fit_summary.json")
        assert os.path.exists(f"{custom_prefix}_theta_corr.pdf")


# ---------------------------------------------------------------------------
# summarize_fit — with ground truth (test statistics)
# ---------------------------------------------------------------------------

class TestSummarizeFitWithGroundTruth:

    def test_theta_test_stats_populated(self, run_dir, ground_truth_file):
        # Add out-of-training genotype to pred CSV
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        summarize_fit(run_dir, ground_truth_file=ground_truth_file)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["theta"]["test"] is not None
        assert data["metadata"]["n_theta_test_points"] == len(TITRANT_CONCS)

    def test_theta_test_pearson_r_reasonable(self, run_dir, ground_truth_file):
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        summarize_fit(run_dir, ground_truth_file=ground_truth_file)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        r = data["theta"]["test"]["pearson_r"]
        # Predictions are close to truth so r should be high
        assert r is not None and r > 0.9

    def test_theta_col_fallback_to_theta(self, run_dir, tmp_path):
        """Ground-truth file with 'theta' column (no 'theta_obs') still works."""
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        gt_path = str(tmp_path / "gt_theta_col.csv")
        _make_ground_truth_csv(gt_path, theta_col="theta")

        summarize_fit(run_dir, ground_truth_file=gt_path)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["theta"]["test"] is not None
        assert data["metadata"]["n_theta_test_points"] == len(TITRANT_CONCS)

    def test_missing_theta_col_warns_and_skips(self, run_dir, tmp_path):
        """Ground-truth file with no theta column issues a warning and skips test stats."""
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        gt_path = str(tmp_path / "gt_no_theta.csv")
        pd.DataFrame({
            "genotype": ["E4F"],
            "titrant_name": [TITRANT_NAME],
            "titrant_conc": [0.0],
            "some_other_col": [0.5],
        }).to_csv(gt_path, index=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            summarize_fit(run_dir, ground_truth_file=gt_path)
        assert any("neither" in str(x.message).lower() for x in w)

        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["theta"]["test"] is None
        assert data["metadata"]["n_theta_test_points"] is None


# ---------------------------------------------------------------------------
# summarize_fit — graceful failure cases
# ---------------------------------------------------------------------------

class TestSummarizeFitGraceful:

    def test_missing_theta_pred_still_writes_json(self, run_dir):
        os.remove(os.path.join(run_dir, "run_theta_pred.csv"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        json_path = os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as fh:
            data = json.load(fh)
        assert data["theta"]["training"] is None

    def test_missing_binding_csv_gives_null_theta_training(self, run_dir, tmp_path):
        config_path = os.path.join(run_dir, "run_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["data"]["binding"] = "/nonexistent/binding.csv"
        with open(config_path, "w") as fh:
            yaml.dump(cfg, fh)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        with open(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["theta"]["training"] is None

    def test_missing_config_still_writes_json(self, run_dir):
        os.remove(os.path.join(run_dir, "run_config.yaml"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "summary", "tfs_summarize_fit_summary.json"))

    def test_empty_run_dir_writes_json_with_nulls(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(empty_dir)
        json_path = os.path.join(empty_dir, "summary", "tfs_summarize_fit_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as fh:
            data = json.load(fh)
        assert data["theta"]["training"] is None
        assert data["theta"]["test"] is None
        assert data["growth"]["training"] is None
        assert data["metadata"]["n_parameters"] is None
        assert data["metadata"]["final_loss"] is None


# ---------------------------------------------------------------------------
# _find_params_or_posterior
# ---------------------------------------------------------------------------

class TestFindParamsOrPosterior:

    def test_returns_none_when_neither_present(self, tmp_path):
        kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind is None
        assert path is None

    def test_finds_posterior_h5(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind == "posterior"
        assert path == str(tmp_path / "run_posterior.h5")

    def test_finds_params_npz(self, tmp_path):
        (tmp_path / "run_params.npz").touch()
        kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind == "params"
        assert path == str(tmp_path / "run_params.npz")

    def test_prefers_posterior_over_params(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        (tmp_path / "run_params.npz").touch()
        kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind == "posterior"

    def test_warns_on_multiple_posterior_files(self, tmp_path):
        (tmp_path / "a_posterior.h5").touch()
        (tmp_path / "b_posterior.h5").touch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind == "posterior"
        assert any("posterior" in str(x.message).lower() for x in w)

    def test_warns_on_multiple_params_files_falls_back_to_first(self, tmp_path):
        (tmp_path / "a_params.npz").touch()
        (tmp_path / "b_params.npz").touch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kind, path = _find_params_or_posterior(str(tmp_path))
        assert kind == "params"
        assert path == str(tmp_path / "a_params.npz")
        assert any("params" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# _try_plot_theta_fits
# ---------------------------------------------------------------------------

_PATCH_PLOT_THETA_FITS = "tfscreen.plot.plot_theta_fits.plot_theta_fits"

_MOCK_BINDING_WITH_STD = pd.DataFrame({
    "genotype": ["wt", "A1B", "C2D"],
    "titrant_name": ["IPTG"] * 3,
    "titrant_conc": [10.0] * 3,
    "theta_obs": [0.1, 0.5, 0.9],
    "theta_std": [0.02, 0.02, 0.02],
})

_MOCK_PRED_FOR_THETA = pd.DataFrame({
    "genotype": ["wt", "A1B", "C2D"],
    "titrant_name": ["IPTG"] * 3,
    "titrant_conc": [10.0] * 3,
    "median": [0.12, 0.48, 0.88],
    "in_training_data": [1, 1, 1],
})


class TestTryPlotThetaFits:

    def _make_mock_ax(self):
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = MagicMock()
        return mock_ax

    # ------------------------------------------------------------------
    # Guard: None inputs
    # ------------------------------------------------------------------

    def test_skips_silently_when_binding_df_is_none(self, tmp_path):
        with patch(_PATCH_PLOT_THETA_FITS) as mock_plot:
            _try_plot_theta_fits(None, _MOCK_PRED_FOR_THETA, str(tmp_path / "out"))
        mock_plot.assert_not_called()

    def test_skips_silently_when_pred_df_is_none(self, tmp_path):
        with patch(_PATCH_PLOT_THETA_FITS) as mock_plot:
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, None, str(tmp_path / "out"))
        mock_plot.assert_not_called()

    # ------------------------------------------------------------------
    # Guard: missing theta_std column
    # ------------------------------------------------------------------

    def test_warns_and_skips_when_theta_std_missing(self, tmp_path):
        binding_no_std = _MOCK_BINDING_WITH_STD.drop(columns=["theta_std"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(_PATCH_PLOT_THETA_FITS) as mock_plot:
                _try_plot_theta_fits(binding_no_std, _MOCK_PRED_FOR_THETA,
                                     str(tmp_path / "out"))
        mock_plot.assert_not_called()
        assert any("theta_std" in str(x.message).lower() for x in w)

    # ------------------------------------------------------------------
    # Happy path: one call per genotype, savefig called with correct paths
    # ------------------------------------------------------------------

    def test_calls_plot_once_per_genotype(self, tmp_path):
        mock_ax = self._make_mock_ax()
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax) as mock_plot:
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA,
                                 str(tmp_path / "out"))
        assert mock_plot.call_count == 3

    def test_savefig_called_with_correct_paths(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        mock_ax = self._make_mock_ax()
        mock_fig = mock_ax.get_figure.return_value
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax):
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA, out_prefix)
        saved_paths = {call.args[0] for call in mock_fig.savefig.call_args_list}
        for geno in ["wt", "A1B", "C2D"]:
            assert f"{out_prefix}_{geno}_theta_fits.pdf" in saved_paths

    def test_title_set_to_genotype_name(self, tmp_path):
        mock_ax = self._make_mock_ax()
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax) as mock_plot:
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA,
                                 str(tmp_path / "out"))
        titles = {call.kwargs.get("title") or call.args[1]
                  for call in mock_plot.call_args_list
                  if mock_plot.call_args_list}
        # title is passed as kwarg
        titles = {call.kwargs["title"] for call in mock_plot.call_args_list}
        assert titles == {"wt", "A1B", "C2D"}

    # ------------------------------------------------------------------
    # Filename sanitization
    # ------------------------------------------------------------------

    def test_slash_in_genotype_sanitized_in_filename(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        binding = _MOCK_BINDING_WITH_STD.copy()
        binding["genotype"] = binding["genotype"].replace("wt", "wt/mut")
        pred = _MOCK_PRED_FOR_THETA.copy()
        pred["genotype"] = pred["genotype"].replace("wt", "wt/mut")
        mock_ax = self._make_mock_ax()
        mock_fig = mock_ax.get_figure.return_value
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax):
            _try_plot_theta_fits(binding, pred, out_prefix)
        saved_paths = {call.args[0] for call in mock_fig.savefig.call_args_list}
        assert f"{out_prefix}_wt_mut_theta_fits.pdf" in saved_paths

    def test_space_in_genotype_sanitized_in_filename(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        binding = _MOCK_BINDING_WITH_STD.copy()
        binding["genotype"] = binding["genotype"].replace("wt", "wt mut")
        pred = _MOCK_PRED_FOR_THETA.copy()
        pred["genotype"] = pred["genotype"].replace("wt", "wt mut")
        mock_ax = self._make_mock_ax()
        mock_fig = mock_ax.get_figure.return_value
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax):
            _try_plot_theta_fits(binding, pred, out_prefix)
        saved_paths = {call.args[0] for call in mock_fig.savefig.call_args_list}
        assert f"{out_prefix}_wt_mut_theta_fits.pdf" in saved_paths

    # ------------------------------------------------------------------
    # Only genotypes present in both dfs are plotted
    # ------------------------------------------------------------------

    def test_only_genotypes_in_both_dfs_are_plotted(self, tmp_path):
        pred_subset = _MOCK_PRED_FOR_THETA[_MOCK_PRED_FOR_THETA["genotype"] != "C2D"].copy()
        mock_ax = self._make_mock_ax()
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax) as mock_plot:
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, pred_subset,
                                 str(tmp_path / "out"))
        assert mock_plot.call_count == 2

    # ------------------------------------------------------------------
    # CSV output: one CSV per genotype alongside each PDF
    # ------------------------------------------------------------------

    def test_writes_csv_per_genotype(self, tmp_path):
        out_prefix = str(tmp_path / "out")
        mock_ax = self._make_mock_ax()
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax):
            _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA, out_prefix)
        for geno in ["wt", "A1B", "C2D"]:
            csv_path = f"{out_prefix}_{geno}_theta_fits.csv"
            assert os.path.exists(csv_path), f"Missing CSV for {geno}"
            df = pd.read_csv(csv_path)
            assert "theta_obs" in df.columns
            assert "median" in df.columns
            assert all(df["genotype"] == geno)

    def test_csv_written_before_pdf(self, tmp_path):
        """CSV must exist even if savefig raises."""
        out_prefix = str(tmp_path / "out")
        mock_ax = self._make_mock_ax()
        mock_ax.get_figure.return_value.savefig.side_effect = RuntimeError("render failed")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax):
                _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA, out_prefix)
        # savefig raised inside the try block so warn should fire, but the
        # CSV for the first genotype (A1B, alphabetically first) must exist
        # because it is written before savefig is called
        assert os.path.exists(f"{out_prefix}_A1B_theta_fits.csv")

    # ------------------------------------------------------------------
    # Guard: plot raises
    # ------------------------------------------------------------------

    def test_warns_gracefully_when_plot_raises(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(_PATCH_PLOT_THETA_FITS, side_effect=RuntimeError("render failed")):
                _try_plot_theta_fits(_MOCK_BINDING_WITH_STD, _MOCK_PRED_FOR_THETA,
                                     str(tmp_path / "out"))
        assert any("theta fit" in str(x.message).lower() for x in w)

    # ------------------------------------------------------------------
    # Integration: summarize_fit calls _try_plot_theta_fits
    # ------------------------------------------------------------------

    def test_summarize_fit_calls_plot_theta_fits_per_training_genotype(self, run_dir):
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = MagicMock()
        with patch(_PATCH_PLOT_THETA_FITS, return_value=mock_ax) as mock_plot:
            summarize_fit(run_dir)
        # run_dir fixture has 3 training genotypes
        assert mock_plot.call_count == 3

    def test_summarize_fit_skips_theta_fits_when_no_pred_csv(self, run_dir):
        os.remove(os.path.join(run_dir, "run_theta_pred.csv"))
        with patch(_PATCH_PLOT_THETA_FITS) as mock_plot:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                summarize_fit(run_dir)
        mock_plot.assert_not_called()


# ---------------------------------------------------------------------------
# _try_plot_trajectories
# ---------------------------------------------------------------------------

_MOCK_BINDING_DF = pd.DataFrame({
    "genotype": ["wt", "A1B", "C2D"],
    "titrant_name": ["IPTG"] * 3,
    "titrant_conc": [0.0] * 3,
    "theta_obs": [0.1, 0.5, 0.9],
})

_CONFIG_WITH_GROWTH = {"data": {"growth": "growth.csv", "binding": "binding.csv"}}
_CONFIG_NO_GROWTH   = {"data": {"binding": "binding.csv"}}

_PATCH_PREDICT_DF = "tfscreen.plot.geno_trajectory.predict_geno_trajectory_df"
_PATCH_PLOT_GENO  = "tfscreen.plot.geno_trajectory.plot_geno_trajectory"
_PATCH_READ_CONFIG = "tfscreen.tfmodel.configuration_io.read_configuration"

# DataFrame returned by the mock predict_geno_trajectory_df — one row per
# genotype so the per-genotype loop fires exactly len(GENOTYPES) times.
_MOCK_PRED_DF = pd.DataFrame({
    "genotype": list(_MOCK_BINDING_DF["genotype"]),
})


class TestTryPlotTrajectories:

    # ------------------------------------------------------------------
    # Guard: no growth data in config
    # ------------------------------------------------------------------

    def test_skips_silently_when_no_growth_key(self, tmp_path):
        with patch(_PATCH_PREDICT_DF) as mock_pdf:
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_NO_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=_MOCK_BINDING_DF,
            )
        mock_pdf.assert_not_called()

    def test_skips_silently_when_config_yaml_is_none(self, tmp_path):
        with patch(_PATCH_PREDICT_DF) as mock_pdf:
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=None,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=_MOCK_BINDING_DF,
            )
        mock_pdf.assert_not_called()

    # ------------------------------------------------------------------
    # Guard: no params / posterior file
    # ------------------------------------------------------------------

    def test_warns_and_skips_when_no_pred_file(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(_PATCH_PREDICT_DF) as mock_pdf:
                _try_plot_trajectories(
                    config_file=str(tmp_path / "cfg.yaml"),
                    config_yaml=_CONFIG_WITH_GROWTH,
                    run_dir=str(tmp_path),
                    out_prefix=str(tmp_path / "out"),
                    binding_df=_MOCK_BINDING_DF,
                )
        mock_pdf.assert_not_called()
        assert any("posterior" in str(x.message).lower() or
                   "params" in str(x.message).lower() for x in w)

    # ------------------------------------------------------------------
    # Guard: read_configuration raises
    # ------------------------------------------------------------------

    def test_warns_and_skips_when_orchestrator_load_fails(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(_PATCH_READ_CONFIG, side_effect=RuntimeError("load failed")), \
                 patch(_PATCH_PREDICT_DF) as mock_pdf:
                _try_plot_trajectories(
                    config_file=str(tmp_path / "cfg.yaml"),
                    config_yaml=_CONFIG_WITH_GROWTH,
                    run_dir=str(tmp_path),
                    out_prefix=str(tmp_path / "out"),
                    binding_df=_MOCK_BINDING_DF,
                )
        mock_pdf.assert_not_called()
        assert any("trajectory" in str(x.message).lower() for x in w)

    # ------------------------------------------------------------------
    # Happy path: posterior.h5 present — passes h5 path as second arg
    # ------------------------------------------------------------------

    def test_calls_predict_df_with_h5_path(self, tmp_path):
        h5_path = str(tmp_path / "run_posterior.h5")
        (tmp_path / "run_posterior.h5").touch()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF) as mock_pdf, \
             patch(_PATCH_PLOT_GENO, return_value=MagicMock()):
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=_MOCK_BINDING_DF,
            )
        mock_pdf.assert_called_once()
        args, _ = mock_pdf.call_args
        assert args[1] == h5_path

    # ------------------------------------------------------------------
    # Happy path: only params.npz present (fallback) — passes npz path
    # ------------------------------------------------------------------

    def test_calls_predict_df_with_npz_path(self, tmp_path):
        npz_path = str(tmp_path / "run_params.npz")
        (tmp_path / "run_params.npz").touch()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF) as mock_pdf, \
             patch(_PATCH_PLOT_GENO, return_value=MagicMock()):
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=_MOCK_BINDING_DF,
            )
        mock_pdf.assert_called_once()
        args, _ = mock_pdf.call_args
        assert args[1] == npz_path

    # ------------------------------------------------------------------
    # Happy path: one PDF saved per genotype
    # ------------------------------------------------------------------

    def test_saves_per_genotype_pdfs(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        out_prefix = str(tmp_path / "out")
        mock_fig = MagicMock()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF), \
             patch(_PATCH_PLOT_GENO, return_value=mock_fig) as mock_plot:
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=out_prefix,
                binding_df=_MOCK_BINDING_DF,
            )
        genotypes = sorted(_MOCK_PRED_DF["genotype"].unique().tolist(), key=str)
        assert mock_plot.call_count == len(genotypes)
        saved_paths = {call.args[0] for call in mock_fig.savefig.call_args_list}
        for geno in genotypes:
            safe = geno.replace("/", "_").replace(" ", "_")
            assert f"{out_prefix}_{safe}_trajectory.pdf" in saved_paths

    # ------------------------------------------------------------------
    # Genotype list derived from binding_df
    # ------------------------------------------------------------------

    def test_passes_genotypes_from_binding_df(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF) as mock_pdf, \
             patch(_PATCH_PLOT_GENO, return_value=MagicMock()):
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=_MOCK_BINDING_DF,
            )
        _, kwargs = mock_pdf.call_args
        assert set(kwargs["genotypes"]) == {"wt", "A1B", "C2D"}

    def test_passes_none_genotypes_when_binding_df_is_none(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF) as mock_pdf, \
             patch(_PATCH_PLOT_GENO, return_value=MagicMock()):
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=str(tmp_path / "out"),
                binding_df=None,
            )
        _, kwargs = mock_pdf.call_args
        assert kwargs["genotypes"] is None

    # ------------------------------------------------------------------
    # CSV output: one CSV per genotype alongside each trajectory PDF
    # ------------------------------------------------------------------

    def test_writes_csv_per_genotype(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        out_prefix = str(tmp_path / "out")
        mock_fig = MagicMock()
        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF), \
             patch(_PATCH_PLOT_GENO, return_value=mock_fig):
            _try_plot_trajectories(
                config_file=str(tmp_path / "cfg.yaml"),
                config_yaml=_CONFIG_WITH_GROWTH,
                run_dir=str(tmp_path),
                out_prefix=out_prefix,
                binding_df=_MOCK_BINDING_DF,
            )
        for geno in sorted(_MOCK_PRED_DF["genotype"].unique().tolist(), key=str):
            safe = geno.replace("/", "_").replace(" ", "_")
            csv_path = f"{out_prefix}_{safe}_trajectory.csv"
            assert os.path.exists(csv_path), f"Missing CSV for {geno}"
            df = pd.read_csv(csv_path)
            assert all(df["genotype"] == geno)

    # ------------------------------------------------------------------
    # Guard: predict_geno_trajectory_df raises
    # ------------------------------------------------------------------

    def test_warns_gracefully_when_plot_raises(self, tmp_path):
        (tmp_path / "run_posterior.h5").touch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
                 patch(_PATCH_PREDICT_DF, side_effect=RuntimeError("plot failed")):
                _try_plot_trajectories(
                    config_file=str(tmp_path / "cfg.yaml"),
                    config_yaml=_CONFIG_WITH_GROWTH,
                    run_dir=str(tmp_path),
                    out_prefix=str(tmp_path / "out"),
                    binding_df=_MOCK_BINDING_DF,
                )
        assert any("trajectory" in str(x.message).lower() for x in w)

    # ------------------------------------------------------------------
    # Integration: summarize_fit calls _try_plot_trajectories
    # ------------------------------------------------------------------

    def test_summarize_fit_calls_predict_df_when_h5_and_growth_config_present(
        self, run_dir
    ):
        config_path = os.path.join(run_dir, "run_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["data"]["growth"] = "growth.csv"
        with open(config_path, "w") as fh:
            yaml.dump(cfg, fh)
        (Path(run_dir) / "run_posterior.h5").touch()

        with patch(_PATCH_READ_CONFIG, return_value=(MagicMock(), {})), \
             patch(_PATCH_PREDICT_DF, return_value=_MOCK_PRED_DF) as mock_pdf, \
             patch(_PATCH_PLOT_GENO, return_value=MagicMock()):
            summarize_fit(run_dir)

        mock_pdf.assert_called_once()

    def test_summarize_fit_skips_plot_when_no_pred_file(self, run_dir):
        config_path = os.path.join(run_dir, "run_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["data"]["growth"] = "growth.csv"
        with open(config_path, "w") as fh:
            yaml.dump(cfg, fh)
        # No posterior.h5 or params.npz written.

        with patch(_PATCH_PREDICT_DF) as mock_pdf:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                summarize_fit(run_dir)

        mock_pdf.assert_not_called()
