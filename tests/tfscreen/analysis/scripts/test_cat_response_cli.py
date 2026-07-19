"""
Tests for cat_response_cli.py -- CSV IO, sigma fallback, column validation,
per-model / prediction output files, and argument wiring.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from tfscreen.analysis.scripts import cat_response_cli
from tfscreen.analysis.scripts.cat_response_cli import (
    cat_response,
    _write_per_model,
    main,
)


def _theta_df(with_titrant=True, with_std=False, with_quantiles=True):
    """A small predict-theta-style table: 2 genotypes x 4 concentrations."""
    concs = np.array([0.0, 1.0, 10.0, 100.0])
    rows = []
    for i, geno in enumerate(["wt", "m1"]):
        center = 0.1 + 0.7 * (concs / concs.max()) + 0.01 * i
        for c, mid in zip(concs, center):
            row = {"genotype": geno, "titrant_conc": float(c), "q0.5": float(mid)}
            if with_titrant:
                row["titrant_name"] = "IPTG"
            if with_quantiles:
                row["q0.841"] = float(mid + 0.05)
                row["q0.159"] = float(mid - 0.05)
            if with_std:
                row["my_std"] = 0.05
            rows.append(row)
    return pd.DataFrame(rows)


def _write(tmp_path, df, name="theta.csv"):
    path = str(tmp_path / name)
    df.to_csv(path, index=False)
    return path


class TestOutputFiles:

    def test_writes_main_permodel_and_predictions(self, tmp_path):
        data = _write(tmp_path, _theta_df())
        out_prefix = str(tmp_path / "out")

        cat_response(data, x_obs="titrant_conc", y_obs="q0.5",
                     out_prefix=out_prefix, models=["flat", "linear"],
                     num_workers=1)

        main = pd.read_csv(f"{out_prefix}.csv")
        assert len(main) == 2  # one row per genotype
        # Column names are cleaned of the '|' delimiter.
        assert not any("|" in c for c in main.columns)

        # One per-model file each, with genotype + est/std columns.
        for model in ["flat", "linear"]:
            mdf = pd.read_csv(f"{out_prefix}_{model}.csv")
            assert len(mdf) == 2
            assert "genotype" in mdf.columns
            assert "is_best_model" in mdf.columns

        preds = pd.read_csv(f"{out_prefix}_predictions.csv")
        assert "genotype" in preds.columns
        assert {"model", "x", "y", "is_best_model"}.issubset(preds.columns)
        # best_only default: predictions restricted to each group's best model.
        assert preds["is_best_model"].all()

        assess = pd.read_csv(f"{out_prefix}_assessment.csv")
        assert "genotype" in assess.columns
        assert {"x", "y_est", "y_std", "z", "sig_nonzero", "direction",
                "equiv_zero"}.issubset(assess.columns)
        # Rollups landed on the main table.
        assert {"omnibus_p", "omnibus_q", "n_nonzero",
                "response_class"}.issubset(main.columns)

    def test_write_all_predictions_flag(self, tmp_path):
        data = _write(tmp_path, _theta_df())
        out_prefix = str(tmp_path / "out")
        cat_response(data, x_obs="titrant_conc", y_obs="q0.5",
                     out_prefix=out_prefix, models=["flat", "linear"],
                     write_all_predictions=True, num_workers=1)
        preds = pd.read_csv(f"{out_prefix}_predictions.csv")
        # Both models present when all predictions are written.
        assert set(preds["model"].unique()) == {"flat", "linear"}


class TestSigmaFallback:

    def test_sigma_from_quantiles_when_no_y_std(self, tmp_path):
        captured = {}

        def fake_core(df, **kwargs):
            captured["y_std"] = kwargs["y_std"]
            captured["sigma_vals"] = list(df[kwargs["y_std"]])
            empty = pd.DataFrame({"genotype": [], "best_model": []})
            return empty, pd.DataFrame({"genotype": []}), \
                pd.DataFrame({"genotype": []}), 0.1

        data = _write(tmp_path, _theta_df(with_quantiles=True))
        with patch.object(cat_response_cli, "_cat_response",
                          side_effect=fake_core), \
             patch.object(cat_response_cli, "_write_per_model"):
            cat_response(data, x_obs="titrant_conc", y_obs="q0.5",
                         out_prefix=str(tmp_path / "out"))

        assert captured["y_std"] == "_sigma"
        # (q0.841 - q0.159)/2 = 0.05 for every row.
        assert captured["sigma_vals"] == pytest.approx([0.05] * 8)

    def test_explicit_y_std_takes_precedence(self, tmp_path):
        captured = {}

        def fake_core(df, **kwargs):
            captured["y_std"] = kwargs["y_std"]
            empty = pd.DataFrame({"genotype": [], "best_model": []})
            return empty, pd.DataFrame({"genotype": []}), \
                pd.DataFrame({"genotype": []}), 0.1

        data = _write(tmp_path, _theta_df(with_std=True, with_quantiles=True))
        with patch.object(cat_response_cli, "_cat_response",
                          side_effect=fake_core), \
             patch.object(cat_response_cli, "_write_per_model"):
            cat_response(data, x_obs="titrant_conc", y_obs="q0.5",
                         y_std="my_std", out_prefix=str(tmp_path / "out"))

        assert captured["y_std"] == "my_std"


class TestValidation:

    def test_missing_y_obs_raises(self, tmp_path):
        data = _write(tmp_path, _theta_df())
        with pytest.raises(ValueError, match="missing required column"):
            cat_response(data, x_obs="titrant_conc", y_obs="nope",
                         out_prefix=str(tmp_path / "out"))

    def test_unknown_model_raises(self, tmp_path):
        data = _write(tmp_path, _theta_df())
        with pytest.raises(ValueError, match="Unknown model"):
            cat_response(data, x_obs="titrant_conc", y_obs="q0.5",
                         models=["nope"], out_prefix=str(tmp_path / "out"))


class TestWritePerModel:

    def test_splits_flat_columns_into_param_table(self, tmp_path):
        results_df = pd.DataFrame({
            "genotype": ["wt", "m1"],
            "best_model": ["linear", "flat"],
            "R2|linear": [0.99, 0.5],
            "AIC_weight|linear": [0.8, 0.2],
            "linear|m|est": [10.0, 5.0],
            "linear|m|std": [0.1, 0.2],
            "linear|b|est": [0.0, 1.0],
            "linear|b|std": [0.1, 0.2],
        })
        out_prefix = str(tmp_path / "out")
        with patch.object(cat_response_cli, "MODEL_LIBRARY",
                          {"linear": {"param_names": ["m", "b"]}}):
            _write_per_model(results_df, ["linear"], ["genotype"], out_prefix)

        mdf = pd.read_csv(f"{out_prefix}_linear.csv")
        assert list(mdf.columns) == ["genotype", "m_est", "b_est",
                                     "m_std", "b_std",
                                     "is_best_model", "R2", "AIC_weight"]
        assert mdf.loc[mdf["genotype"] == "wt", "m_est"].iloc[0] == 10.0
        assert bool(mdf.loc[mdf["genotype"] == "wt", "is_best_model"].iloc[0])
        assert not bool(mdf.loc[mdf["genotype"] == "m1", "is_best_model"].iloc[0])


class TestArgWiring:

    def test_main_parses_positionals_and_flags(self, tmp_path, monkeypatch):
        data = _write(tmp_path, _theta_df())
        out_prefix = str(tmp_path / "out")
        argv = ["cat_response", data, "titrant_conc", "q0.5",
                "--out_prefix", out_prefix,
                "--group_by", "titrant_name",
                "--models", "flat", "linear",
                "--num_workers", "1"]
        monkeypatch.setattr(sys, "argv", argv)

        main()

        assert os.path.exists(f"{out_prefix}.csv")
        assert os.path.exists(f"{out_prefix}_flat.csv")
        assert os.path.exists(f"{out_prefix}_predictions.csv")
