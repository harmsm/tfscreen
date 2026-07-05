"""Tests for tfs-report-cfu0 CLI (report_cfu0)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from tfscreen.simulate.scripts.report_cfu0_cli import (
    report_cfu0,
    _classify_genotypes,
)
from tfscreen.util.cli.generalized_main import generalized_main


# ---------------------------------------------------------------------------
# _classify_genotypes
# ---------------------------------------------------------------------------

def test_classify_genotypes_basic_classes():
    library_df = pd.DataFrame({
        "genotype": ["wt", "wt", "A1V", "A1V/B2C", "C3D", "wt"],
        "library_origin": ["single-1", "double-1-2", "single-1", "double-1-2",
                           "spiked", "spiked"],
    })
    result = _classify_genotypes(library_df)
    assert result["wt"] == "wt"
    assert result["A1V"] == "single"
    assert result["A1V/B2C"] == "double"
    assert result["C3D"] == "spiked"


def test_classify_genotypes_wt_beats_spiked():
    """A literal 'wt' genotype is always classified 'wt', even if it also
    appears under a 'spiked' library_origin."""
    library_df = pd.DataFrame({
        "genotype": ["wt", "wt"],
        "library_origin": ["single-1", "spiked"],
    })
    result = _classify_genotypes(library_df)
    assert result["wt"] == "wt"


def test_classify_genotypes_triple_mutant_is_other():
    library_df = pd.DataFrame({
        "genotype": ["A1V/B2C/D3E"],
        "library_origin": ["double-1-2"],
    })
    result = _classify_genotypes(library_df)
    assert result["A1V/B2C/D3E"] == "other"


# ---------------------------------------------------------------------------
# report_cfu0
# ---------------------------------------------------------------------------

def _make_library_and_phenotype():
    library_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A1V/B2C", "C3D"],
        "library_origin": ["single-1", "single-1", "double-1-2", "spiked"],
    })
    phenotype_df = pd.DataFrame({"genotype": ["wt", "A1V", "A1V/B2C", "C3D"]})
    theta_df = pd.DataFrame({"genotype": ["wt", "A1V", "A1V/B2C", "C3D"]})
    params_df = pd.DataFrame({
        "genotype": ["wt", "A1V", "A1V/B2C", "C3D"],
        "dk_geno": [0.0, 0.0, 0.0, 0.0],
        "activity": [1.0, 1.0, 1.0, 1.0],
    })
    return library_df, phenotype_df, theta_df, params_df


def _make_selection_experiment_result(rep):
    """Fake one replicate's (sample_df, counts_df), with a distinct ln_cfu_0
    per genotype so we can check pooling/averaging."""
    sample_df = pd.DataFrame(
        [{"sample": 0, "replicate": rep, "library": "kanR"}],
        index=[0],
    )
    counts_df = pd.DataFrame([
        {"sample": 0, "genotype": "wt", "ln_cfu_0": 14.0, "counts": 100},
        {"sample": 0, "genotype": "A1V", "ln_cfu_0": 8.0, "counts": 100},
        {"sample": 0, "genotype": "A1V/B2C", "ln_cfu_0": 1.4, "counts": 100},
        {"sample": 0, "genotype": "C3D", "ln_cfu_0": -np.inf, "counts": 0},
    ])
    return sample_df, counts_df


def test_report_cfu0_prints_expected_classes(capsys):
    library_df, phenotype_df, theta_df, params_df = _make_library_and_phenotype()

    call_count = {"n": 0}

    def fake_selection_experiment(cf, lib_df, pheno_df):
        call_count["n"] += 1
        rep = pheno_df["replicate"].iloc[0]
        return _make_selection_experiment_result(rep)

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.library_prediction",
               return_value=(library_df, phenotype_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.selection_experiment",
               side_effect=fake_selection_experiment):
        report_cfu0("config.yaml", num_replicates=2)

    assert call_count["n"] == 2

    out = capsys.readouterr().out
    assert "wt" in out
    assert "single" in out
    assert "double" in out
    assert "spiked" in out


def test_report_cfu0_excludes_infinite_ln_cfu0_from_mean(capsys):
    """C3D never survives transformation (-inf ln_cfu_0 in every replicate);
    its class ('spiked') should report mean_n_observed == 0 and no crash from
    averaging -inf."""
    library_df, phenotype_df, theta_df, params_df = _make_library_and_phenotype()

    def fake_selection_experiment(cf, lib_df, pheno_df):
        rep = pheno_df["replicate"].iloc[0]
        return _make_selection_experiment_result(rep)

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.library_prediction",
               return_value=(library_df, phenotype_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.selection_experiment",
               side_effect=fake_selection_experiment):
        report_cfu0("config.yaml", num_replicates=3)

    out = capsys.readouterr().out
    spiked_line = [l for l in out.splitlines() if l.startswith("spiked")][0]
    fields = spiked_line.split()
    # class, n_library, mean_n_observed, mean_ln_cfu0, std_ln_cfu0
    assert fields[0] == "spiked"
    assert fields[1] == "1"
    assert fields[2] == "0.0"
    assert fields[3] == "nan"


def test_report_cfu0_seed_overrides_config():
    library_df, phenotype_df, theta_df, params_df = _make_library_and_phenotype()

    captured_seeds = []

    def fake_selection_experiment(cf, lib_df, pheno_df):
        captured_seeds.append(cf["seed"])
        rep = pheno_df["replicate"].iloc[0]
        return _make_selection_experiment_result(rep)

    with patch("tfscreen.util.read_yaml", return_value={"seed": 99, "growth": {}}), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.library_prediction",
               return_value=(library_df, phenotype_df, theta_df, params_df, None)), \
         patch("tfscreen.simulate.scripts.report_cfu0_cli.selection_experiment",
               side_effect=fake_selection_experiment):
        report_cfu0("config.yaml", num_replicates=2, seed=7)

    # base_seed becomes 7 (override), then rep seed = base_seed*num_replicates+rep
    assert captured_seeds == [7 * 2 + 1, 7 * 2 + 2]


# ---------------------------------------------------------------------------
# --seed CLI argument
# ---------------------------------------------------------------------------

def test_seed_cli_parsed_as_int():
    captured = {}

    def fake_run(config_file, num_replicates=5, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml", "--seed", "42"],
        manual_arg_types={"seed": int},
    )
    assert captured["seed"] == 42
    assert isinstance(captured["seed"], int)


def test_seed_cli_defaults_to_none():
    captured = {}

    def fake_run(config_file, num_replicates=5, seed=None):
        captured["seed"] = seed

    generalized_main(
        fake_run,
        argv=["config.yaml"],
        manual_arg_types={"seed": int},
    )
    assert captured["seed"] is None
