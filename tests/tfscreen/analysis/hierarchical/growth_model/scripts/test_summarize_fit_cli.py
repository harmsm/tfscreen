"""
Tests for summarize_fit_cli.py.

All tests use real file I/O in tmp_path so that we exercise the full
read/parse/stats/write pipeline without mocking the heavy ML stack.
Matplotlib is forced to the Agg backend so no display is required.
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.analysis.hierarchical.growth_model.scripts.summarize_fit_cli import (
    _find_unique,
    _json_safe,
    _read_final_loss,
    _resolve_path,
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
    lines = [f"{i*100} {-10000 + i*50:.1f}" for i in range(1, 10)]
    lines.append(f"900 {final_loss}")
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


@pytest.fixture
def ground_truth_file(tmp_path):
    """Ground-truth file containing the 4th genotype (not in training data)."""
    rows = []
    for tc in TITRANT_CONCS:
        rows.append({
            "genotype": "E4F",
            "titrant_name": TITRANT_NAME,
            "titrant_conc": tc,
            "theta_obs": THETA_OBS[3] + tc * 0.0001,
            "theta_std": 0.02,
        })
    path = str(tmp_path / "ground_truth.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
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

class TestReadFinalLoss:

    def test_reads_last_value_from_two_column_file(self, tmp_path):
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

    def test_reads_first_column_from_comma_delimited_file(self, tmp_path):
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
        assert os.path.exists(os.path.join(run_dir, "tfs_summarize_fit_summary.json"))

    def test_pdf_written(self, run_dir):
        summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "tfs_summarize_theta_corr.pdf"))

    def test_json_is_valid(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert "metadata" in data
        assert "training" in data
        assert "test" in data

    def test_metadata_fields_populated(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        meta = data["metadata"]
        assert meta["n_parameters"] == 25
        assert meta["n_training_points"] > 0
        assert meta["final_loss"] == pytest.approx(-4321.0)

    def test_training_stats_populated(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        stats = data["training"]
        assert stats is not None
        for key in ("rmse", "pearson_r", "spearman_r", "r_squared", "mean_error"):
            assert key in stats, f"Missing key: {key}"

    def test_test_stats_null_when_no_ground_truth(self, run_dir):
        summarize_fit(run_dir)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["test"] is None
        assert data["metadata"]["n_test_points"] is None

    def test_custom_out_prefix(self, run_dir, tmp_path):
        custom_prefix = str(tmp_path / "custom" / "myrun")
        os.makedirs(os.path.dirname(custom_prefix), exist_ok=True)
        summarize_fit(run_dir, out_prefix=custom_prefix)
        assert os.path.exists(f"{custom_prefix}_fit_summary.json")
        assert os.path.exists(f"{custom_prefix}_theta_corr.pdf")


# ---------------------------------------------------------------------------
# summarize_fit — with ground truth (test statistics)
# ---------------------------------------------------------------------------

class TestSummarizeFitWithGroundTruth:

    def test_test_stats_populated(self, run_dir, ground_truth_file):
        # Add out-of-training genotype to pred CSV
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        summarize_fit(run_dir, ground_truth_file=ground_truth_file)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["test"] is not None
        assert data["metadata"]["n_test_points"] == len(TITRANT_CONCS)

    def test_test_pearson_r_reasonable(self, run_dir, ground_truth_file):
        pred_path = os.path.join(run_dir, "run_theta_pred.csv")
        _make_pred_csv(pred_path, n_training=3, extra_genotypes=["E4F"])

        summarize_fit(run_dir, ground_truth_file=ground_truth_file)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        r = data["test"]["pearson_r"]
        # Predictions are close to truth so r should be high
        assert r is not None and r > 0.9


# ---------------------------------------------------------------------------
# summarize_fit — graceful failure cases
# ---------------------------------------------------------------------------

class TestSummarizeFitGraceful:

    def test_missing_theta_pred_still_writes_json(self, run_dir):
        os.remove(os.path.join(run_dir, "run_theta_pred.csv"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        json_path = os.path.join(run_dir, "tfs_summarize_fit_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as fh:
            data = json.load(fh)
        assert data["training"] is None

    def test_missing_binding_csv_gives_null_training(self, run_dir, tmp_path):
        # Point config at a non-existent binding file
        config_path = os.path.join(run_dir, "run_config.yaml")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["data"]["binding"] = "/nonexistent/binding.csv"
        with open(config_path, "w") as fh:
            yaml.dump(cfg, fh)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        with open(os.path.join(run_dir, "tfs_summarize_fit_summary.json")) as fh:
            data = json.load(fh)
        assert data["training"] is None

    def test_missing_config_still_writes_json(self, run_dir):
        os.remove(os.path.join(run_dir, "run_config.yaml"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(run_dir)
        assert os.path.exists(os.path.join(run_dir, "tfs_summarize_fit_summary.json"))

    def test_empty_run_dir_writes_json_with_nulls(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summarize_fit(empty_dir)
        json_path = os.path.join(empty_dir, "tfs_summarize_fit_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as fh:
            data = json.load(fh)
        assert data["training"] is None
        assert data["test"] is None
        assert data["metadata"]["n_parameters"] is None
        assert data["metadata"]["final_loss"] is None
