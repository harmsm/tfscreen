"""
Tests for the generic cat_response core: grouping, column selection, validation,
prediction tagging, the post-hoc assessment pass, and serial/parallel
equivalence.
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

def _flat(best="flat", **extra):
    """A minimal flat cat_fit result dict with the assessment rollups."""
    d = {"status": "success", "best_model": best,
         "nonzero_p": 0.5, "omnibus_p": 0.5,
         "n_nonzero": 0, "any_nonzero": False}
    d.update(extra)
    return d


def _pred(n=2):
    """A minimal cat_fit prediction frame."""
    return pd.DataFrame({
        "model": ["flat"] * n,
        "x": np.arange(n, dtype=float),
        "y_model": np.zeros(n),
        "y_model_std": np.zeros(n),
        "is_best_model": [True] * n,
    })


def _assess(n=2, y_model=0.0, y_model_std=0.1):
    """A minimal cat_fit per-point assessment frame."""
    return pd.DataFrame({
        "model": ["flat"] * n,
        "x": np.arange(n, dtype=float),
        "y_obs": np.full(n, y_model, dtype=float),
        "y_std": np.full(n, y_model_std, dtype=float),
        "y_model": np.full(n, y_model, dtype=float),
        "y_model_std": np.full(n, y_model_std, dtype=float),
        "z": np.full(n, y_model / y_model_std if y_model_std else 0.0),
        "sig_nonzero": np.zeros(n, dtype=bool),
    })


def _capturing_fit(store, **fit_kwargs):
    """A fake cat_fit that records the (x, y, y_std) it was handed per call."""
    def fake_fit(x, y, y_std, x_pred=None, models_to_run=None,
                 best_only=True, alpha=0.05, select_by="shape",
                 adequacy_alpha=0.05, curvy_cutoff=0.1, verbose=False):
        store.append({"x": list(x), "y": list(y), "y_std": list(y_std),
                      "best_only": best_only, "alpha": alpha,
                      "select_by": select_by, "adequacy_alpha": adequacy_alpha,
                      "curvy_cutoff": curvy_cutoff,
                      "models_to_run": models_to_run})
        return _flat(**fit_kwargs), _pred(len(x)), _assess(len(np.unique(x)))
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
            results, _, _, _ = cat_response(_basic_df(), x_obs="titrant_conc",
                                            y_obs="theta", y_std="theta_std")
        # 2 genotypes -> 2 groups (titrant_name is ignored without group_by).
        assert len(results) == 2
        assert set(results["genotype"]) == {"wt", "m1"}
        assert "titrant_name" not in results.columns

    def test_group_by_adds_a_column(self):
        store = []
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            results, _, _, _ = cat_response(_basic_df(), x_obs="titrant_conc",
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
            results, _, _, _ = cat_response(df, x_obs="titrant_conc",
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

    def test_best_only_and_alpha_threaded_to_fitter(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", best_only=False, alpha=0.01)
        assert all(call["best_only"] is False for call in store)
        assert all(call["alpha"] == 0.01 for call in store)

    def test_adequacy_alpha_threaded_to_fitter(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", adequacy_alpha=0.2)
        assert all(call["adequacy_alpha"] == 0.2 for call in store)

    def test_select_by_threaded_to_fitter(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", select_by="adequacy")
        assert all(call["select_by"] == "adequacy" for call in store)
        # Default is the shape classifier.
        store.clear()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std")
        assert all(call["select_by"] == "shape" for call in store)

    def test_curvy_cutoff_threaded_to_fitter(self):
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", curvy_cutoff=0.25)
        assert all(call["curvy_cutoff"] == 0.25 for call in store)

    def test_shape_mode_defaults_to_shape_models(self):
        from tfscreen.mle.curve_models import SHAPE_MODELS
        store = []
        df = _basic_df()
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit(store)):
            cat_response(df, x_obs="titrant_conc", y_obs="theta",
                         y_std="theta_std", select_by="shape")
        assert all(call["models_to_run"] == list(SHAPE_MODELS)
                   for call in store)


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
            _, preds, _, _ = cat_response(_basic_df(), x_obs="titrant_conc",
                                          y_obs="theta", y_std="theta_std",
                                          group_by=["titrant_name"])
        # Group keys come first, then the cat_fit prediction columns.
        assert list(preds.columns[:2]) == ["genotype", "titrant_name"]
        for col in ["model", "x", "y_model", "y_model_std", "is_best_model"]:
            assert col in preds.columns
        # Every prediction row carries a real group key.
        assert not preds["genotype"].isna().any()


# --- post-hoc assessment pass ------------------------------------------------

class TestAssessmentPass:

    def test_assessment_tagged_and_has_fittable(self):
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit([])):
            _, _, assess, rope = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", group_by=["titrant_name"])
        assert list(assess.columns[:2]) == ["genotype", "titrant_name"]
        assert "fittable" in assess.columns
        assert assess["fittable"].dtype == bool
        assert "equiv_zero" not in assess.columns   # dropped from the output
        assert np.isfinite(rope)

    def test_fittable_column_next_to_model(self):
        # fittable is its own bool column immediately after model; the model
        # name and its fitted values are left intact.
        fit = _capturing_fit([])

        def fake(*a, **k):
            flat, pred, _ = fit(*a, **k)
            x = a[0]
            return flat, pred, _assess(len(np.unique(x)), y_model=0.0,
                                       y_model_std=5.0)   # huge observed std
        with patch.object(cat_response_mod, "cat_fit", side_effect=fake):
            _, _, assess, _ = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=0.5)
        # not distinguishable from zero -> fittable False; model name preserved.
        assert set(assess["fittable"]) == {False}
        assert set(assess["model"]) == {"flat"}
        cols = list(assess.columns)
        assert cols[cols.index("model") + 1] == "fittable"

    def test_rope_defaults_to_median_times_multiplier(self):
        # Fake assessment always reports y_std=0.1 -> median=0.1 -> rope=0.2.
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit([])):
            _, _, _, rope = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_multiplier=2.0)
        assert rope == pytest.approx(0.2)

    def test_explicit_rope_cutoff_is_used(self):
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit([])):
            _, _, _, rope = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=0.75)
        assert rope == 0.75

    def test_not_fittable_but_confident_zero(self):
        # Observed y=0 with tiny error -> all_equiv_zero True; nonzero_p high
        # (0.5) -> not distinguishable -> fittable False (recoverable as the old
        # "confident_zero" via all_equiv_zero=True).
        fit = _capturing_fit([])

        def fake(*a, **k):
            flat, pred, _ = fit(*a, **k)
            x = a[0]
            return flat, pred, _assess(len(np.unique(x)), y_model=0.0,
                                       y_model_std=1e-4)
        with patch.object(cat_response_mod, "cat_fit", side_effect=fake):
            results, _, _, _ = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=0.5)
        assert set(results["fittable"]) == {False}
        assert set(results["all_equiv_zero"]) == {True}

    def test_fittable_true_even_inside_rope(self):
        # Distinguishable from zero (nonzero_p tiny) -> fittable True, even when
        # every observed point also sits inside the ROPE.
        fit = _capturing_fit([])

        def fake(*a, **k):
            flat, pred, _ = fit(*a, **k)
            flat["nonzero_p"] = 1e-9
            x = a[0]
            return flat, pred, _assess(len(np.unique(x)), y_model=0.0,
                                       y_model_std=1e-4)
        with patch.object(cat_response_mod, "cat_fit", side_effect=fake):
            results, _, _, _ = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=0.5)
        assert set(results["fittable"]) == {True}

    def test_not_fittable_and_not_equiv_is_indeterminate(self):
        # Not distinguishable from zero and error bars too wide for the ROPE ->
        # fittable False, all_equiv_zero False (the old "indeterminate").
        fit = _capturing_fit([])

        def fake(*a, **k):
            flat, pred, _ = fit(*a, **k)
            x = a[0]
            return flat, pred, _assess(len(np.unique(x)), y_model=0.0,
                                       y_model_std=5.0)   # huge observed std
        with patch.object(cat_response_mod, "cat_fit", side_effect=fake):
            results, _, _, _ = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=0.5)
        assert set(results["fittable"]) == {False}
        assert set(results["all_equiv_zero"]) == {False}

    def test_fittable_from_low_q(self):
        with patch.object(cat_response_mod, "cat_fit",
                          side_effect=_capturing_fit([], nonzero_p=1e-8)):
            results, _, _, _ = cat_response(
                _basic_df(), x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", rope_cutoff=1e-6)
        assert set(results["fittable"]) == {True}
        assert (results["nonzero_q"] < 0.05).all()


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
            serial, serial_pred, serial_assess, serial_delta = cat_response(
                df, x_obs="titrant_conc", y_obs="theta",
                y_std="theta_std", num_workers=1)
            parallel, parallel_pred, parallel_assess, parallel_delta = \
                cat_response(df, x_obs="titrant_conc", y_obs="theta",
                             y_std="theta_std", num_workers=2)

        assert list(serial["genotype"]) == [f"g{i}" for i in range(7)]
        assert serial_delta == pytest.approx(parallel_delta)
        pd.testing.assert_frame_equal(serial, parallel)
        pd.testing.assert_frame_equal(serial_pred, parallel_pred)
        pd.testing.assert_frame_equal(serial_assess, parallel_assess)
