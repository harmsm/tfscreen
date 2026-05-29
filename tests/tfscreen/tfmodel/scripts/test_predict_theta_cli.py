"""
Tests for predict_theta_cli.py — genotype/titrant union semantics and --only_files.
"""
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

from tfscreen.tfmodel.scripts.predict_theta_cli import predict_theta


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_lines(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(str(x) for x in lines) + "\n")


def _make_tm_df(genotypes, titrant_names, titrant_concs):
    rows = []
    for g in genotypes:
        for tn, tc in zip(titrant_names, titrant_concs):
            rows.append({"genotype": g, "titrant_name": tn, "titrant_conc": tc})
    return pd.DataFrame(rows)


@pytest.fixture
def mock_gm():
    gm = MagicMock()
    gm._theta = "hill"
    gm.growth_tm.df = _make_tm_df(
        ["wt", "A1B"],
        ["IPTG", "IPTG"],
        [0.0, 1.0],
    )
    gm.training_tm.df = gm.growth_tm.df
    return gm


def _fake_extract(model, posteriors, **kwargs):
    """Return a minimal result DataFrame covering all requested genotypes."""
    manual = kwargs.get("manual_titrant_df")
    if manual is not None:
        pairs = list(zip(manual["titrant_name"], manual["titrant_conc"]))
    else:
        pairs = [("IPTG", 0.0), ("IPTG", 1.0)]
    target = kwargs.get("target_genotypes") or ["wt", "A1B"]
    rows = [{"genotype": g, "titrant_name": tn, "titrant_conc": tc, "median": 0.5}
            for g in target for tn, tc in pairs]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_extract(mock_gm):
    with patch(
        "tfscreen.tfmodel.scripts"
        ".predict_theta_cli.read_configuration",
        return_value=(mock_gm, {}),
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_theta_cli.resolve_param_file",
        side_effect=lambda pf, gm, op: pf,  # pass-through
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_theta_cli.extract_theta_curves",
        side_effect=lambda **kw: _fake_extract(None, None, **kw),
    ), patch(
        "tfscreen.tfmodel.scripts"
        ".predict_theta_cli.extract_theta_unmeasured",
        side_effect=lambda **kw: _fake_extract(None, None, **kw),
    ):
        yield


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_mismatched_titrant_files_raises(mock_extract, tmp_path):
    nf = str(tmp_path / "names.txt")
    _write_lines(nf, ["IPTG"])
    with pytest.raises(ValueError, match="together"):
        predict_theta("cfg.yaml", "post.h5",
                      titrant_names_file=nf,
                      out_prefix=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Default behaviour (no files)
# ---------------------------------------------------------------------------

class TestPredictThetaDefaults:

    def test_no_files_uses_all_training_genotypes(self, mock_extract, mock_gm, tmp_path):
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["wt"], "titrant_name": ["IPTG"],
                 "titrant_conc": [0.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5", out_prefix=out)
        assert mock_curves.called


# ---------------------------------------------------------------------------
# Union semantics (only_files=False, the default)
# ---------------------------------------------------------------------------

class TestPredictThetaUnion:

    def test_genotypes_file_unions_with_training(self, mock_extract, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])  # out-of-training
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_unmeasured",
        ) as mock_unmeas:
            mock_unmeas.return_value = pd.DataFrame(
                {"genotype": ["wt", "A1B", "C2D"],
                 "titrant_name": ["IPTG"] * 3,
                 "titrant_conc": [0.0] * 3,
                 "median": [0.5] * 3}
            )
            predict_theta("cfg.yaml", "post.h5",
                          genotypes_file=gf, out_prefix=out)
        target = mock_unmeas.call_args.kwargs["target_genotypes"]
        assert "wt" in target
        assert "A1B" in target
        assert "C2D" in target

    def test_titrant_files_union_with_training_grid(self, mock_extract, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        cf = str(tmp_path / "concs.txt")
        _write_lines(nf, ["IPTG"])
        _write_lines(cf, [5.0])  # novel concentration
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["wt"], "titrant_name": ["IPTG"],
                 "titrant_conc": [0.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5",
                          titrant_names_file=nf,
                          titrant_concs_file=cf,
                          out_prefix=out)
        titrant_df = mock_curves.call_args.kwargs["manual_titrant_df"]
        assert 0.0 in titrant_df["titrant_conc"].values
        assert 1.0 in titrant_df["titrant_conc"].values
        assert 5.0 in titrant_df["titrant_conc"].values

    def test_single_titrant_name_broadcast_across_concs(self, mock_extract, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        cf = str(tmp_path / "concs.txt")
        _write_lines(nf, ["IPTG"])
        _write_lines(cf, [0.0, 10.0, 100.0])
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["wt"], "titrant_name": ["IPTG"],
                 "titrant_conc": [0.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5",
                          titrant_names_file=nf,
                          titrant_concs_file=cf,
                          out_prefix=out)
        titrant_df = mock_curves.call_args.kwargs["manual_titrant_df"]
        assert list(titrant_df["titrant_name"]) == ["IPTG", "IPTG", "IPTG", "IPTG"]
        assert set(titrant_df["titrant_conc"]) >= {0.0, 10.0, 100.0}

    def test_mismatched_titrant_file_lengths_raises(self, mock_extract, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        cf = str(tmp_path / "concs.txt")
        _write_lines(nf, ["IPTG", "ATC"])
        _write_lines(cf, [0.0, 10.0, 100.0])
        with pytest.raises(ValueError, match="entries but"):
            predict_theta("cfg.yaml", "post.h5",
                          titrant_names_file=nf,
                          titrant_concs_file=cf,
                          out_prefix=str(tmp_path / "out"))

    def test_duplicate_titrant_pairs_not_repeated(self, mock_extract, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        cf = str(tmp_path / "concs.txt")
        _write_lines(nf, ["IPTG"])
        _write_lines(cf, [1.0])  # already in training
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["wt"], "titrant_name": ["IPTG"],
                 "titrant_conc": [0.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5",
                          titrant_names_file=nf,
                          titrant_concs_file=cf,
                          out_prefix=out)
        titrant_df = mock_curves.call_args.kwargs["manual_titrant_df"]
        dupes = titrant_df[
            (titrant_df["titrant_name"] == "IPTG") &
            (titrant_df["titrant_conc"] == 1.0)
        ]
        assert len(dupes) == 1


# ---------------------------------------------------------------------------
# --only_files semantics
# ---------------------------------------------------------------------------

class TestPredictThetaOnlyFiles:

    def test_only_files_restricts_genotypes_to_file(self, mock_extract, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["A1B"])
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["A1B"], "titrant_name": ["IPTG"],
                 "titrant_conc": [0.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5",
                          genotypes_file=gf,
                          only_files=True,
                          out_prefix=out)
        # extract_theta_curves doesn't receive target_genotypes; the result is
        # filtered post-call — check via output CSV
        df = pd.read_csv(f"{out}.csv")
        assert set(df["genotype"].unique()) <= {"A1B"}

    def test_only_files_restricts_titrant_grid_to_file(self, mock_extract, mock_gm, tmp_path):
        nf = str(tmp_path / "names.txt")
        cf = str(tmp_path / "concs.txt")
        _write_lines(nf, ["IPTG"])
        _write_lines(cf, [5.0])
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
        ) as mock_curves:
            mock_curves.return_value = pd.DataFrame(
                {"genotype": ["wt"], "titrant_name": ["IPTG"],
                 "titrant_conc": [5.0], "median": [0.5]}
            )
            predict_theta("cfg.yaml", "post.h5",
                          titrant_names_file=nf,
                          titrant_concs_file=cf,
                          only_files=True,
                          out_prefix=out)
        titrant_df = mock_curves.call_args.kwargs["manual_titrant_df"]
        assert list(titrant_df["titrant_conc"]) == [5.0]
        assert 0.0 not in titrant_df["titrant_conc"].values
        assert 1.0 not in titrant_df["titrant_conc"].values


# ---------------------------------------------------------------------------
# checkpoint (.pkl) param_file support
# ---------------------------------------------------------------------------

class TestPredictThetaCheckpointInput:

    def _base_patches(self, mock_gm):
        return [
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_theta_cli.read_configuration",
                return_value=(mock_gm, {}),
            ),
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_theta_cli.extract_theta_curves",
                return_value=pd.DataFrame({
                    "genotype": ["wt"], "titrant_name": ["IPTG"],
                    "titrant_conc": [0.0], "median": [0.5],
                }),
            ),
            patch(
                "tfscreen.tfmodel.scripts"
                ".predict_theta_cli.extract_theta_unmeasured",
                return_value=pd.DataFrame({
                    "genotype": ["wt"], "titrant_name": ["IPTG"],
                    "titrant_conc": [0.0], "median": [0.5],
                }),
            ),
        ]

    def test_pkl_calls_resolve_param_file(self, mock_gm, tmp_path):
        """resolve_param_file is invoked when param_file ends with .pkl."""
        resolve_calls = []

        def fake_resolve(pf, gm, op):
            resolve_calls.append(pf)
            return "resolved.h5"

        p = self._base_patches(mock_gm)
        with p[0], p[1], p[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.resolve_param_file",
            side_effect=fake_resolve,
        ):
            predict_theta("cfg.yaml", "run_checkpoint.pkl",
                          out_prefix=str(tmp_path / "out"))

        assert resolve_calls == ["run_checkpoint.pkl"]

    def test_resolved_path_passed_to_extract(self, mock_gm, tmp_path):
        """The path returned by resolve_param_file reaches extract_theta_curves."""
        extract_calls = {}

        def fake_curves(**kwargs):
            extract_calls["posteriors"] = kwargs.get("posteriors")
            return pd.DataFrame({
                "genotype": ["wt"], "titrant_name": ["IPTG"],
                "titrant_conc": [0.0], "median": [0.5],
            })

        p = self._base_patches(mock_gm)
        with p[0], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
            side_effect=fake_curves,
        ), p[2], patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.resolve_param_file",
            return_value="map_posterior.h5",
        ):
            predict_theta("cfg.yaml", "run_checkpoint.pkl",
                          out_prefix=str(tmp_path / "out"))

        assert extract_calls["posteriors"] == "map_posterior.h5"


# ---------------------------------------------------------------------------
# genotype_batch_size threading
# ---------------------------------------------------------------------------

class TestGenotypeBatchSize:
    """genotype_batch_size is forwarded to extract_theta_unmeasured."""

    def test_custom_batch_size_forwarded(self, mock_extract, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])  # out-of-training → unmeasured path
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_unmeasured",
        ) as mock_unmeas:
            mock_unmeas.return_value = pd.DataFrame({
                "genotype": ["wt", "A1B", "C2D"],
                "titrant_name": ["IPTG"] * 3,
                "titrant_conc": [0.0] * 3,
                "median": [0.5] * 3,
            })
            predict_theta("cfg.yaml", "post.h5",
                          genotypes_file=gf,
                          genotype_batch_size=42,
                          out_prefix=out)
        assert mock_unmeas.call_args.kwargs["genotype_batch_size"] == 42

    def test_default_batch_size_is_2000(self, mock_extract, mock_gm, tmp_path):
        gf = str(tmp_path / "genos.txt")
        _write_lines(gf, ["C2D"])
        out = str(tmp_path / "out")
        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_unmeasured",
        ) as mock_unmeas:
            mock_unmeas.return_value = pd.DataFrame({
                "genotype": ["wt", "A1B", "C2D"],
                "titrant_name": ["IPTG"] * 3,
                "titrant_conc": [0.0] * 3,
                "median": [0.5] * 3,
            })
            predict_theta("cfg.yaml", "post.h5",
                          genotypes_file=gf,
                          out_prefix=out)
        assert mock_unmeas.call_args.kwargs["genotype_batch_size"] == 2000


# ---------------------------------------------------------------------------
# q_to_get / point_est dispatch
# ---------------------------------------------------------------------------

class TestPredictThetaQToGet:

    def test_pkl_passes_point_est_q_to_get(self, mock_gm, tmp_path):
        """q_to_get={"point_est": 0.5} is passed to extract_theta_curves for .pkl input."""
        extract_calls = {}

        def fake_curves(**kwargs):
            extract_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "point_est": [0.5]})

        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.read_configuration",
            return_value=(mock_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.resolve_param_file",
            return_value="resolved.h5",
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
            side_effect=fake_curves,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_unmeasured",
            return_value=pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                       "titrant_conc": [0.0], "point_est": [0.5]}),
        ):
            predict_theta("cfg.yaml", "run_checkpoint.pkl",
                          out_prefix=str(tmp_path / "out"))

        assert extract_calls["q_to_get"] == {"point_est": 0.5}

    def test_h5_passes_none_q_to_get(self, mock_gm, tmp_path):
        """q_to_get=None is passed to extract_theta_curves for .h5 input."""
        extract_calls = {}

        def fake_curves(**kwargs):
            extract_calls["q_to_get"] = kwargs.get("q_to_get")
            return pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                 "titrant_conc": [0.0], "median": [0.5]})

        with patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.read_configuration",
            return_value=(mock_gm, {}),
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.resolve_param_file",
            side_effect=lambda pf, gm, op: pf,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_curves",
            side_effect=fake_curves,
        ), patch(
            "tfscreen.tfmodel.scripts"
            ".predict_theta_cli.extract_theta_unmeasured",
            return_value=pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"],
                                       "titrant_conc": [0.0], "median": [0.5]}),
        ):
            predict_theta("cfg.yaml", "post.h5",
                          out_prefix=str(tmp_path / "out"))

        assert extract_calls["q_to_get"] is None
