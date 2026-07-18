"""
Tests for extract_epistasis_cli.py -- CSV IO, column validation, argument wiring.
"""
import sys

import pytest
import pandas as pd

from tfscreen.analysis.scripts.extract_epistasis_cli import (
    extract_epistasis,
    main,
)


def _write_csv(tmp_path, rows, name="data.csv"):
    path = str(tmp_path / name)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _single_cycle_rows(condition=None):
    """wt/singles/double with a clean additive cycle (ep = 1.0)."""
    rows = [
        {"genotype": "wt", "y": 1.0, "y_err": 0.1},
        {"genotype": "A15G", "y": 2.0, "y_err": 0.1},
        {"genotype": "P75K", "y": 3.0, "y_err": 0.1},
        {"genotype": "A15G/P75K", "y": 5.0, "y_err": 0.1},
    ]
    if condition is not None:
        for r in rows:
            r["condition"] = condition
    return rows


class TestHappyPath:

    def test_additive_ep_obs(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", out_prefix=out_prefix)

        out = pd.read_csv(f"{out_prefix}.csv")
        assert len(out) == 1
        assert out["ep_obs"].iloc[0] == pytest.approx(1.0)
        # No y_std passed -> no ep_std column.
        assert "ep_std" not in out.columns

    def test_y_std_propagates(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", y_std="y_err", out_prefix=out_prefix)

        out = pd.read_csv(f"{out_prefix}.csv")
        # sqrt(4 * 0.1**2) = 0.2
        assert out["ep_std"].iloc[0] == pytest.approx(0.2)

    def test_multiplicative_scale(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", scale="mult", out_prefix=out_prefix)

        out = pd.read_csv(f"{out_prefix}.csv")
        # (5/3) / (2/1) = 0.83333...
        assert out["ep_obs"].iloc[0] == pytest.approx((5.0 / 3.0) / (2.0 / 1.0))

    def test_group_by_groups_independently(self, tmp_path):
        rows = _single_cycle_rows(condition="c1") + _single_cycle_rows(condition="c2")
        # Make c2's double mutant produce a different epistasis (ep = 0.0).
        for r in rows:
            if r["condition"] == "c2" and r["genotype"] == "A15G/P75K":
                r["y"] = 4.0  # (4-3) - (2-1) = 0.0
        data = _write_csv(tmp_path, rows)
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y",
                          group_by=["condition"],
                          out_prefix=out_prefix)

        out = pd.read_csv(f"{out_prefix}.csv").sort_values("condition")
        assert len(out) == 2
        by_cond = dict(zip(out["condition"], out["ep_obs"]))
        assert by_cond["c1"] == pytest.approx(1.0)
        assert by_cond["c2"] == pytest.approx(0.0)

    def test_keep_extra_retains_input_columns(self, tmp_path):
        rows = _single_cycle_rows()
        for r in rows:
            r["note"] = "keepme"
        data = _write_csv(tmp_path, rows)
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", keep_extra=True, out_prefix=out_prefix)

        out = pd.read_csv(f"{out_prefix}.csv")
        assert "note" in out.columns


class TestValidation:

    def test_missing_y_obs_raises(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        with pytest.raises(ValueError, match="missing required column"):
            extract_epistasis(data, y_obs="does_not_exist",
                              out_prefix=str(tmp_path / "out"))

    def test_missing_y_std_raises(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        with pytest.raises(ValueError, match="missing required column"):
            extract_epistasis(data, y_obs="y", y_std="nope",
                              out_prefix=str(tmp_path / "out"))

    def test_missing_group_by_raises(self, tmp_path):
        data = _write_csv(tmp_path, _single_cycle_rows())
        with pytest.raises(ValueError, match="missing required column"):
            extract_epistasis(data, y_obs="y", group_by=["nope"],
                              out_prefix=str(tmp_path / "out"))


class TestEmptyResult:

    def test_no_cycles_writes_empty(self, tmp_path, capsys):
        # No double mutant -> no cycles.
        rows = [
            {"genotype": "wt", "y": 1.0},
            {"genotype": "A15G", "y": 2.0},
        ]
        data = _write_csv(tmp_path, rows)
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", out_prefix=out_prefix)

        import os
        assert os.path.exists(f"{out_prefix}.csv")
        captured = capsys.readouterr().out
        assert "no valid mutant cycles" in captured
        assert "Wrote 0 rows" in captured

    def test_forgot_group_by_hints_column(self, tmp_path, capsys):
        # One row per genotype *per condition* (like titrant_conc), run without
        # --group_by: every genotype is non-unique -> all dropped.
        rows = []
        for conc in [0.0, 0.1, 1.0]:
            for r in _single_cycle_rows():
                rows.append({**r, "titrant_conc": conc})
        data = _write_csv(tmp_path, rows)
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", out_prefix=out_prefix)

        captured = capsys.readouterr().out
        assert "dropped as duplicates" in captured
        assert "--group_by titrant_conc" in captured
        # With the suggested selector it succeeds.
        extract_epistasis(data, y_obs="y",
                          group_by=["titrant_conc"],
                          out_prefix=out_prefix)
        out = pd.read_csv(f"{out_prefix}.csv")
        assert len(out) == 3

    def test_no_cycles_does_not_hint_condition(self, tmp_path, capsys):
        # Genotypes already unique -> the empty result is not a duplicate issue,
        # so we must not emit a spurious condition-column hint.
        rows = [
            {"genotype": "wt", "y": 1.0},
            {"genotype": "A15G", "y": 2.0},
        ]
        data = _write_csv(tmp_path, rows)
        out_prefix = str(tmp_path / "out")

        extract_epistasis(data, y_obs="y", out_prefix=out_prefix)

        captured = capsys.readouterr().out
        assert "--group_by" not in captured


class TestArgWiring:

    def test_main_parses_positionals_and_flags(self, tmp_path, monkeypatch):
        data = _write_csv(tmp_path,
                          _single_cycle_rows(condition="c1")
                          + _single_cycle_rows(condition="c2"))
        out_prefix = str(tmp_path / "out")

        argv = ["extract_epistasis", data, "y",
                "--out_prefix", out_prefix,
                "--y_std", "y_err",
                "--group_by", "condition",
                "--scale", "add",
                "--keep_extra"]
        monkeypatch.setattr(sys, "argv", argv)

        main()

        out = pd.read_csv(f"{out_prefix}.csv")
        assert len(out) == 2
        assert "ep_std" in out.columns
        # keep_extra retained the original observable column.
        assert "y" in out.columns
