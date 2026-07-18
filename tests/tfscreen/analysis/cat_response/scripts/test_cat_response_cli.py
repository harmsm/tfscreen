"""
Tests for cat_response_cli.py — theta_col auto-detection, error handling, and
the serial/parallel/chunked dispatch machinery.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from tfscreen.analysis.cat_response.scripts import cat_response_cli
from tfscreen.analysis.cat_response.scripts.cat_response_cli import (
    cat_response,
    _fit_one,
    _fit_chunk,
    _iter_chunks,
)


_FLAT_RESULT = {"best_model": "flat", "status": "success"}


def _make_theta_df(center_col):
    return pd.DataFrame({
        "genotype": ["wt", "wt"],
        "titrant_name": ["IPTG", "IPTG"],
        "titrant_conc": [0.0, 1.0],
        center_col: [0.3, 0.7],
        "q0.841": [0.4, 0.8],
        "q0.159": [0.2, 0.6],
    })


class TestThetaColAutoDetect:
    """theta_col / sigma_col resolution. Driven serially (num_workers=1) so the
    cat_fit mock runs in-process."""

    def _run_capture(self, tmp_path, df, **kwargs):
        f = str(tmp_path / "theta.csv")
        captured = {}

        def fake_fit(x, y, y_std, models_to_run):
            captured["y"] = list(y)
            captured["y_std"] = list(y_std)
            return (dict(_FLAT_RESULT), None)

        with patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.pd.read_csv",
                   return_value=df), \
             patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.cat_fit",
                   side_effect=fake_fit):
            cat_response(f, out_prefix=str(tmp_path / "out"),
                         num_workers=1, **kwargs)
        return captured

    def test_autodetects_q0_5(self, tmp_path):
        """Values from q0.5 column are passed to the fitter."""
        captured = self._run_capture(tmp_path, _make_theta_df("q0.5"))
        assert captured["y"] == pytest.approx([0.3, 0.7])

    def test_autodetects_point_est_values(self, tmp_path):
        """point_est is used when q0.5 is absent."""
        captured = self._run_capture(tmp_path, _make_theta_df("point_est"))
        assert captured["y"] == pytest.approx([0.3, 0.7])

    def test_explicit_theta_col_overrides(self, tmp_path):
        """Explicit theta_col is used even when q0.5 is also present."""
        df = _make_theta_df("q0.5").copy()
        df["my_col"] = [0.11, 0.22]
        captured = self._run_capture(tmp_path, df, theta_col="my_col")
        assert captured["y"] == pytest.approx([0.11, 0.22])

    def test_q0_5_preferred_over_point_est(self, tmp_path):
        """When both q0.5 and point_est are present, q0.5 wins."""
        df = _make_theta_df("q0.5").copy()
        df["point_est"] = [0.9, 0.8]
        captured = self._run_capture(tmp_path, df)
        assert captured["y"] == pytest.approx([0.3, 0.7])

    def test_sigma_from_quantiles(self, tmp_path):
        """When sigma_col is absent, sigma = (q0.841 - q0.159) / 2."""
        captured = self._run_capture(tmp_path, _make_theta_df("q0.5"))
        # (0.4-0.2)/2 = 0.1 ; (0.8-0.6)/2 = 0.1
        assert captured["y_std"] == pytest.approx([0.1, 0.1])

    def test_raises_when_no_theta_col(self, tmp_path):
        """ValueError when neither q0.5 nor point_est is present."""
        f = str(tmp_path / "theta.csv")
        df = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
            "q0.841": [0.5],
            "q0.159": [0.3],
        })
        with patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.pd.read_csv",
                   return_value=df):
            with pytest.raises(ValueError, match="No theta column found"):
                cat_response(f, out_prefix=str(tmp_path / "out"))


class TestChunkHelpers:

    def test_iter_chunks_partitions_exactly(self):
        items = list(range(10))
        chunks = list(_iter_chunks(items, 3))
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_iter_chunks_empty(self):
        assert list(_iter_chunks([], 3)) == []

    def test_iter_chunks_larger_than_input(self):
        assert list(_iter_chunks([1, 2], 100)) == [[1, 2]]

    def test_fit_one_tags_genotype_and_titrant(self):
        with patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.cat_fit",
                   return_value=(dict(_FLAT_RESULT), None)):
            out = _fit_one("g1", "IPTG",
                           np.array([0.0, 1.0]), np.array([0.2, 0.8]),
                           np.array([0.1, 0.1]), ["flat"])
        assert out["genotype"] == "g1"
        assert out["titrant_name"] == "IPTG"
        assert out["best_model"] == "flat"

    def test_fit_chunk_preserves_order(self):
        # cat_fit echoes the number of points so we can verify ordering.
        def fake_fit(x, y, y_std, models_to_run):
            return ({"n": len(x)}, None)

        items = [
            ("g0", "IPTG", np.zeros(2), np.zeros(2), np.ones(2), ["flat"]),
            ("g1", "IPTG", np.zeros(3), np.zeros(3), np.ones(3), ["flat"]),
        ]
        with patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.cat_fit",
                   side_effect=fake_fit):
            out = _fit_chunk(items)
        assert [r["genotype"] for r in out] == ["g0", "g1"]
        assert [r["n"] for r in out] == [2, 3]


class TestDispatchEquivalence:
    """Serial and parallel paths must produce identical, correctly-ordered
    output across chunk boundaries. Uses the real (deterministic) cat_fit so the
    ProcessPoolExecutor path is exercised end-to-end (a parent-process monkeypatch
    would not reach spawned workers)."""

    def _many_genotype_df(self, n_geno):
        # Four concentrations per genotype: enough points for the fitters, and a
        # different (rising) curve per genotype so rows are distinguishable.
        concs = np.array([0.0, 1.0, 10.0, 100.0])
        rows = []
        for i in range(n_geno):
            center = 0.1 + 0.7 * (concs / concs.max()) + 0.001 * i
            for c, mid in zip(concs, center):
                rows.append({"genotype": f"g{i}", "titrant_name": "IPTG",
                             "titrant_conc": float(c), "q0.5": float(mid),
                             "q0.841": float(mid + 0.05),
                             "q0.159": float(mid - 0.05)})
        return pd.DataFrame(rows)

    def _run(self, tmp_path, df, num_workers, out_name):
        f = str(tmp_path / "theta.csv")
        out_prefix = str(tmp_path / out_name)
        with patch("tfscreen.analysis.cat_response.scripts.cat_response_cli.pd.read_csv",
                   return_value=df):
            cat_response(f, out_prefix=out_prefix, num_workers=num_workers)
        return pd.read_csv(out_prefix + ".csv")

    def test_serial_row_order(self, tmp_path):
        # Force multiple chunks by shrinking the chunk size.
        df = self._many_genotype_df(7)
        with patch.object(cat_response_cli, "_CHUNK_SIZE", 2):
            out = self._run(tmp_path, df, num_workers=1, out_name="serial")
        assert list(out["genotype"]) == [f"g{i}" for i in range(7)]
        assert len(out) == 7

    def test_parallel_matches_serial(self, tmp_path):
        # Multiple chunks spread across 2 real worker processes.
        df = self._many_genotype_df(7)
        with patch.object(cat_response_cli, "_CHUNK_SIZE", 2):
            serial = self._run(tmp_path, df, num_workers=1, out_name="serial")
            parallel = self._run(tmp_path, df, num_workers=2, out_name="parallel")
        pd.testing.assert_frame_equal(serial, parallel)
        assert list(parallel["genotype"]) == [f"g{i}" for i in range(7)]
