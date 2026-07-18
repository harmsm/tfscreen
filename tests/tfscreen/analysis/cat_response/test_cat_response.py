"""
Tests for the generic cat_response core: grouping, column selection, validation,
prediction tagging, and serial/parallel equivalence.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import importlib

from tfscreen.analysis.cat_response.cat_response import (
    cat_response,
    _iter_chunks,
)

# The package __init__ rebinds the name ``cat_response`` to the function, which
# shadows the submodule for ``import ... as`` -- grab the module object directly
# so patch.object targets the module's cat_fit / _CHUNK_SIZE.
cat_response_mod = importlib.import_module(
    "tfscreen.analysis.cat_response.cat_response"
)


# --- helpers -----------------------------------------------------------------

def _flat(best="flat"):
    """A minimal flat cat_fit result dict."""
    return {"status": "success", "best_model": best}


def _pred(n=2):
    """A minimal cat_fit prediction frame."""
    return pd.DataFrame({
        "model": ["flat"] * n,
        "x": np.arange(n, dtype=float),
        "y": np.zeros(n),
        "y_std": np.zeros(n),
        "is_best_model": [True] * n,
    })


def _capturing_fit(store):
    """A fake cat_fit that records the (x, y, y_std) it was handed per call."""
    def fake_fit(x, y, y_std, x_pred=None, models_to_run=None, verbose=False):
        store.append({"x": list(x), "y": list(y), "y_std": list(y_std)})
        return _flat(), _pred(len(x))
    return fake_fit


def _basic_df():
    """Two genotypes, two titrants, two points each."""
    rows = []
    for geno in ["wt", "m1"]:
        for titr in ["IPTG", "aTc"]:
            for conc, val in [(0.0, 0.3), (1.0, 0.7)]:
                rows.append({"genotype": geno, "titrant_name": titr,
                             "titrant_conc": conc, "theta": val,
                             "theta_std": 0.1})
    return pd.DataFrame(rows)


# --- grouping ----------------------------------------------------------------

class TestGrouping:

    def test_groups_by_genotype_only_by_default(self):
        store = []
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            results, _ = cat_response(_basic_df(), x_obs="titrant_conc",
                                      y_obs="theta", y_std="theta_std")
        # 2 genotypes -> 2 groups (titrant_name is ignored without group_by).
        assert len(results) == 2
        assert set(results["genotype"]) == {"wt", "m1"}
        assert "titrant_name" not in results.columns

    def test_group_by_adds_a_column(self):
        store = []
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            results, _ = cat_response(_basic_df(), x_obs="titrant_conc",
                                      y_obs="theta", y_std="theta_std",
                                      group_by=["titrant_name"])
        # 2 genotypes x 2 titrants -> 4 groups.
        assert len(results) == 4
        assert set(results.columns[:2]) == {"genotype", "titrant_name"}
        combos = set(zip(results["genotype"], results["titrant_name"]))
        assert combos == {("wt", "IPTG"), ("wt", "aTc"),
                          ("m1", "IPTG"), ("m1", "aTc")}

    def test_works_without_titrant_name_column(self):
        """The original friction: a table with no titrant_name still groups."""
        df = _basic_df().drop(columns=["titrant_name"])
        store = []
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            results, _ = cat_response(df, x_obs="titrant_conc",
                                      y_obs="theta", y_std="theta_std")
        assert len(results) == 2
        assert set(results["genotype"]) == {"wt", "m1"}


# --- column selection --------------------------------------------------------

class TestColumnSelection:

    def test_passes_selected_columns_to_fitter(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", group_by=["titrant_name"])
        # Every call gets the two theta values for its group; 4 groups x 2 pts.
        all_y = sorted(v for call in store for v in call["y"])
        assert all_y == pytest.approx([0.3] * 4 + [0.7] * 4)
        # y_std comes from the named column, not uniform weights.
        assert all(all(s == 0.1 for s in call["y_std"]) for call in store)

    def test_none_y_std_uses_uniform_weights(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta", y_std=None)
        # Uniform weights: every y_std handed to cat_fit is 1.0.
        assert all(all(s == 1.0 for s in call["y_std"]) for call in store)


# --- validation --------------------------------------------------------------

class TestValidation:

    def test_missing_y_obs_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            cat_response(_basic_df(), x_obs="titrant_conc", y_obs="nope")

    def test_missing_group_by_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            cat_response(_basic_df(), x_obs="titrant_conc", y_obs="theta",
                         group_by=["nope"])

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            cat_response(_basic_df(), x_obs="titrant_conc", y_obs="theta",
                         models_to_run=["not_a_model"])


# --- predictions -------------------------------------------------------------

class TestPredictions:

    def test_predictions_tagged_with_group_keys(self):
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit([])):
            _, preds = cat_response(_basic_df(), x_obs="titrant_conc",
                                    y_obs="theta", y_std="theta_std",
                                    group_by=["titrant_name"])
        # Group keys come first, then the cat_fit prediction columns.
        assert list(preds.columns[:2]) == ["genotype", "titrant_name"]
        for col in ["model", "x", "y", "y_std", "is_best_model"]:
            assert col in preds.columns
        # Every prediction row carries a real group key.
        assert not preds["genotype"].isna().any()


# --- chunk helper ------------------------------------------------------------

class TestIterChunks:

    def test_partitions_exactly(self):
        assert list(_iter_chunks(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]

    def test_empty(self):
        assert list(_iter_chunks([], 3)) == []


# --- serial / parallel equivalence (real cat_fit) ----------------------------

class TestDispatchEquivalence:
    """Serial and parallel paths must produce identical, correctly-ordered
    output. Uses the real (deterministic) cat_fit so the ProcessPoolExecutor
    path is exercised end-to-end."""

    def _many_genotype_df(self, n_geno):
        concs = np.array([0.0, 1.0, 10.0, 100.0])
        rows = []
        for i in range(n_geno):
            center = 0.1 + 0.7 * (concs / concs.max()) + 0.001 * i
            for c, mid in zip(concs, center):
                rows.append({"genotype": f"g{i}", "titrant_conc": float(c),
                             "theta": float(mid), "theta_std": 0.05})
        return pd.DataFrame(rows)

    def test_parallel_matches_serial(self):
        df = self._many_genotype_df(7)
        with patch.object(cat_response_mod, "_CHUNK_SIZE", 2):
            serial, serial_pred = cat_response(
                df, x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", num_workers=1)
            parallel, parallel_pred = cat_response(
                df, x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", num_workers=2)

        assert list(serial["genotype"]) == [f"g{i}" for i in range(7)]
        pd.testing.assert_frame_equal(serial, parallel)
        pd.testing.assert_frame_equal(serial_pred, parallel_pred)
