"""
Tests for cat_response_cli.py — theta_col auto-detection and error handling.
"""
import pytest
import pandas as pd
from unittest.mock import patch

from tfscreen.analysis.cat_response.cat_response_cli import cat_response


# Run ProcessPoolExecutor synchronously so cat_fit mocks work in-process.
from concurrent.futures import Future as _Future

class _SyncFuture(_Future):
    def __init__(self, result_val):
        super().__init__()
        self.set_result(result_val)

class _SyncExecutor:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def submit(self, fn, *args): return _SyncFuture(fn(*args))

_SYNC_EXECUTOR = patch(
    "tfscreen.analysis.cat_response.cat_response_cli.ProcessPoolExecutor",
    return_value=_SyncExecutor(),
)

_FLAT_RESULT = {"genotype": "wt", "titrant_name": "IPTG",
                "best_model": "flat", "status": "success"}


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

    def test_autodetects_q0_5(self, tmp_path):
        """Values from q0.5 column are passed to the fitter."""
        f = str(tmp_path / "theta.csv")
        captured = {}

        def fake_fit(x, y, y_std, models_to_run):
            captured["y"] = list(y)
            return (_FLAT_RESULT, None)

        df = _make_theta_df("q0.5")
        with patch("tfscreen.analysis.cat_response.cat_response_cli.pd.read_csv",
                   return_value=df), \
             patch("tfscreen.analysis.cat_response.cat_response_cli.cat_fit",
                   side_effect=fake_fit), \
             _SYNC_EXECUTOR:
            cat_response(f, out_prefix=str(tmp_path / "out"))

        assert captured["y"] == pytest.approx([0.3, 0.7])

    def test_autodetects_point_est_values(self, tmp_path):
        """Values from point_est column are passed to the fitter when median is absent."""
        f = str(tmp_path / "theta.csv")
        captured = {}

        def fake_fit(x, y, y_std, models_to_run):
            captured["y"] = list(y)
            return (_FLAT_RESULT, None)

        df = _make_theta_df("point_est")
        with patch("tfscreen.analysis.cat_response.cat_response_cli.pd.read_csv",
                   return_value=df), \
             patch("tfscreen.analysis.cat_response.cat_response_cli.cat_fit",
                   side_effect=fake_fit), \
             _SYNC_EXECUTOR:
            cat_response(f, out_prefix=str(tmp_path / "out"))

        assert captured["y"] == pytest.approx([0.3, 0.7])

    def test_explicit_theta_col_overrides(self, tmp_path):
        """Explicit theta_col is used even when median is also present."""
        f = str(tmp_path / "theta.csv")
        captured = {}

        def fake_fit(x, y, y_std, models_to_run):
            captured["y"] = list(y)
            return (_FLAT_RESULT, None)

        df = _make_theta_df("q0.5").copy()
        df["my_col"] = [0.11, 0.22]
        with patch("tfscreen.analysis.cat_response.cat_response_cli.pd.read_csv",
                   return_value=df), \
             patch("tfscreen.analysis.cat_response.cat_response_cli.cat_fit",
                   side_effect=fake_fit), \
             _SYNC_EXECUTOR:
            cat_response(f, theta_col="my_col", out_prefix=str(tmp_path / "out"))

        assert captured["y"] == pytest.approx([0.11, 0.22])

    def test_q0_5_preferred_over_point_est(self, tmp_path):
        """When both q0.5 and point_est are present, q0.5 wins."""
        f = str(tmp_path / "theta.csv")
        captured = {}

        def fake_fit(x, y, y_std, models_to_run):
            captured["y"] = list(y)
            return (_FLAT_RESULT, None)

        df = _make_theta_df("q0.5").copy()
        df["point_est"] = [0.9, 0.8]
        with patch("tfscreen.analysis.cat_response.cat_response_cli.pd.read_csv",
                   return_value=df), \
             patch("tfscreen.analysis.cat_response.cat_response_cli.cat_fit",
                   side_effect=fake_fit), \
             _SYNC_EXECUTOR:
            cat_response(f, out_prefix=str(tmp_path / "out"))

        assert captured["y"] == pytest.approx([0.3, 0.7])

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
        with patch("tfscreen.analysis.cat_response.cat_response_cli.pd.read_csv",
                   return_value=df):
            with pytest.raises(ValueError, match="No theta column found"):
                cat_response(f, out_prefix=str(tmp_path / "out"))
