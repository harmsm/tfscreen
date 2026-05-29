"""
Tests for predict_growth_cli.py — genotype/conc union semantics and --only_files.
"""
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from tfscreen.growth_model.scripts.predict_growth_cli import predict_growth
from tfscreen.growth_model.checkpoint_io import resolve_param_file


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_growth_df(genotypes, titrant_concs):
    rows = []
    for g in genotypes:
        for c in titrant_concs:
            rows.append({"genotype": g, "titrant_name": "IPTG",
                         "titrant_conc": c, "ln_cfu": 10.0})
    return pd.DataFrame(rows)


def _write_lines(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(str(x) for x in lines) + "\n")


@pytest.fixture
def mock_gm():
    gm = MagicMock()
    gm.growth_df = _make_growth_df(["wt", "A1B"], [0.0, 1.0])
    return gm


@pytest.fixture
def mock_predict(mock_gm):
    """Patch read_configuration, resolve_param_file, and predict; return captured call kwargs."""
    calls = {}

    def fake_predict(**kwargs):
        calls.update(kwargs)
        genotypes = kwargs.get("genotypes") or mock_gm.growth_df["genotype"].unique().tolist()
        concs = kwargs.get("titrant_conc") or mock_gm.growth_df["titrant_conc"].unique().tolist()
        rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                 "median": 10.0}
                for g in genotypes for c in concs]
        return pd.DataFrame(rows)

    with patch(
        "tfscreen.growth_model.scripts"
        ".predict_growth_cli.read_configuration",
        return_value=(mock_gm, {}),
    ), patch(
        "tfscreen.growth_model.scripts"
        ".predict_growth_cli.resolve_param_file",
        side_effect=lambda pf, gm, op: pf,  # pass-through
    ), patch(
        "tfscreen.growth_model.scripts"
        ".predict_growth_cli.predict",
        side_effect=fake_predict,
    ):
        yield calls


# ---------------------------------------------------------------------------
# Default behaviour (no files)
# ---------------------------------------------------------------------------

class TestPredictGrowthDefaults:

    def test_no_files_passes_none_genotypes(self, mock_predict, tmp_path):
        predict_growth("cfg.yaml", "post.h5", out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"] is None

    def test_no_files_passes_none_concs(self, mock_predict, tmp_path):
        predict_growth("cfg.yaml", "post.h5", out_prefix=str(tmp_path / "out"))
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# Union semantics (only_files=False, the default)
# ---------------------------------------------------------------------------

class TestPredictGrowthUnion:

    def test_genotypes_file_unions_with_training(self, mock_predict, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])  # novel genotype
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        assert "wt" in mock_predict["genotypes"]
        assert "A1B" in mock_predict["genotypes"]
        assert "C2D" in mock_predict["genotypes"]

    def test_genotypes_file_union_preserves_order(self, mock_predict, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        # training genotypes come before file genotypes
        idx_c2d = mock_predict["genotypes"].index("C2D")
        assert idx_c2d > 0

    def test_concs_file_unions_with_training(self, mock_predict, mock_gm, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [5.0])  # novel concentration
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       out_prefix=str(tmp_path / "out"))
        assert 0.0 in mock_predict["titrant_conc"]
        assert 1.0 in mock_predict["titrant_conc"]
        assert 5.0 in mock_predict["titrant_conc"]

    def test_duplicate_concs_not_repeated(self, mock_predict, mock_gm, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [1.0])  # already in training
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       out_prefix=str(tmp_path / "out"))
        concs = mock_predict["titrant_conc"]
        assert concs.count(1.0) == 1

    def test_duplicate_genotypes_not_repeated(self, mock_predict, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["wt"])  # already in training
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"].count("wt") == 1


# ---------------------------------------------------------------------------
# --only_files semantics
# ---------------------------------------------------------------------------

class TestPredictGrowthOnlyFiles:

    def test_only_files_genotypes_restricts_to_file(self, mock_predict, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["A1B"])
        predict_growth("cfg.yaml", "post.h5",
                       genotypes_file=gf,
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"] == ["A1B"]
        assert "wt" not in mock_predict["genotypes"]

    def test_only_files_concs_restricts_to_file(self, mock_predict, tmp_path):
        cf = str(tmp_path / "concs.txt")
        _write_lines(cf, [5.0])
        predict_growth("cfg.yaml", "post.h5",
                       titrant_concs_file=cf,
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["titrant_conc"] == [5.0]
        assert 0.0 not in mock_predict["titrant_conc"]
        assert 1.0 not in mock_predict["titrant_conc"]

    def test_only_files_no_file_falls_through_to_none(self, mock_predict, tmp_path):
        predict_growth("cfg.yaml", "post.h5",
                       only_files=True,
                       out_prefix=str(tmp_path / "out"))
        assert mock_predict["genotypes"] is None
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# titrant_names_file is restrict-only (not unioned)
# ---------------------------------------------------------------------------

class TestPredictGrowthTitrantNamesFilter:

    def test_titrant_names_file_filters_output_rows(self, mock_predict, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        _write_lines(nf, ["IPTG"])
        out = str(tmp_path / "out")
        predict_growth("cfg.yaml", "post.h5",
                       titrant_names_file=nf,
                       out_prefix=out)
        df = pd.read_csv(f"{out}.csv")
        assert set(df["titrant_name"].unique()) <= {"IPTG"}

    def test_titrant_names_file_does_not_affect_predict_call(self, mock_predict, tmp_path):
        nf = str(tmp_path / "names.txt")
        _write_lines(nf, ["IPTG"])
        predict_growth("cfg.yaml", "post.h5",
                       titrant_names_file=nf,
                       out_prefix=str(tmp_path / "out"))
        # titrant_names_file must not influence genotypes or concs passed to predict
        assert mock_predict["genotypes"] is None
        assert mock_predict["titrant_conc"] is None


# ---------------------------------------------------------------------------
# checkpoint (.pkl) param_file support
# ---------------------------------------------------------------------------

class TestPredictGrowthCheckpointInput:

    def _make_fixtures(self, mock_gm, resolved_path="resolved.h5"):
        """Return patch stack that intercepts resolve_param_file."""
        def fake_predict(**kwargs):
            genotypes = kwargs.get("genotypes") or mock_gm.growth_df["genotype"].unique().tolist()
            concs = kwargs.get("titrant_conc") or mock_gm.growth_df["titrant_conc"].unique().tolist()
            rows = [{"genotype": g, "titrant_name": "IPTG", "titrant_conc": c,
                     "median": 10.0}
                    for g in genotypes for c in concs]
            return pd.DataFrame(rows)

        return [
            patch(
                "tfscreen.growth_model.scripts"
                ".predict_growth_cli.read_configuration",
                return_value=(mock_gm, {}),
            ),
            patch(
                "tfscreen.growth_model.scripts"
                ".predict_growth_cli.predict",
                side_effect=fake_predict,
            ),
        ]

    def test_pkl_param_file_calls_resolve(self, mock_gm, tmp_path):
        """resolve_param_file is called when param_file ends with .pkl."""
        resolve_calls = []

        def fake_resolve(pf, gm, op):
            resolve_calls.append(pf)
            return "resolved.h5"

        patches = self._make_fixtures(mock_gm)
        with patches[0], patches[1], patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=fake_resolve,
        ):
            predict_growth("cfg.yaml", "myrun_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert resolve_calls == ["myrun_checkpoint.pkl"]

    def test_h5_param_file_calls_resolve(self, mock_gm, tmp_path):
        """resolve_param_file is called for .h5 files too (pass-through)."""
        resolve_calls = []

        def fake_resolve(pf, gm, op):
            resolve_calls.append(pf)
            return pf

        patches = self._make_fixtures(mock_gm)
        with patches[0], patches[1], patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=fake_resolve,
        ):
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"))

        assert resolve_calls == ["post.h5"]

    def test_resolved_path_passed_to_predict(self, mock_gm, tmp_path):
        """The path returned by resolve_param_file is what predict receives."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["param_posteriors"] = kwargs.get("param_posteriors")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                  "titrant_conc": [0.0], "median": [10.0]})

        patches = self._make_fixtures(mock_gm)
        with patches[0], patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.resolve_param_file",
            return_value="resolved_map.h5",
        ):
            predict_growth("cfg.yaml", "myrun_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["param_posteriors"] == "resolved_map.h5"

    def test_pkl_passes_point_est_q_to_get(self, mock_gm, tmp_path):
        """q_to_get={"point_est": 0.5} is passed to predict when param_file is .pkl."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "point_est": [10.0]})

        patches = self._make_fixtures(mock_gm)
        with patches[0], patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.resolve_param_file",
            return_value="resolved.h5",
        ):
            predict_growth("cfg.yaml", "run_checkpoint.pkl",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["q_to_get"] == {"point_est": 0.5}

    def test_h5_passes_none_q_to_get(self, mock_gm, tmp_path):
        """q_to_get=None is passed to predict when param_file is .h5."""
        predict_calls = {}

        def fake_predict(**kwargs):
            predict_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "median": [10.0]})

        patches = self._make_fixtures(mock_gm)
        with patches[0], patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.predict",
            side_effect=fake_predict,
        ), patch(
            "tfscreen.growth_model.scripts"
            ".predict_growth_cli.resolve_param_file",
            side_effect=lambda pf, gm, op: pf,
        ):
            predict_growth("cfg.yaml", "post.h5",
                           out_prefix=str(tmp_path / "out"))

        assert predict_calls["q_to_get"] is None
