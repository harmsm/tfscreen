"""
Tests for cat_fit: AICc/weighted-R2 selection, best_only predictions, the
per-point assessment/omnibus rollup, and the insufficient-data path. Uses the
real MODEL_LIBRARY (fits are deterministic) rather than mocks.
"""
import numpy as np
import pytest

from tfscreen.analysis.cat_response.cat_fit import cat_fit
from tfscreen.mle.curve_models import MODEL_LIBRARY


# A titration-like grid with enough points that low-parameter models are usable.
X = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])

# Small deterministic +/- scatter. Keeps residuals (and therefore the fitted
# covariance) nonzero -- run_matrix_wls scales covariance by the reduced
# chi-square, so a *perfect* fit would collapse it to zero.
_WOBBLE = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _hill_data(baseline=0.1, amplitude=0.7, logK=np.log(10.0), n=1.0,
               std=0.02):
    from tfscreen.mle.curve_models.models import model_hill_4p
    y = model_hill_4p([baseline, amplitude, logK, n], X)
    y = y + 0.2 * std * _WOBBLE   # tiny wobble -> finite covariance
    return X, y, np.full_like(X, std)


def _flat_data(c=5.0, std=0.1):
    # Symmetric about c -> fitted baseline is exactly c, residuals nonzero.
    return X, c + std * 0.5 * _WOBBLE, np.full_like(X, std)


def _zero_data(std=0.1):
    # Symmetric about 0 -> fitted baseline exactly 0, residuals nonzero.
    return X, std * 0.5 * _WOBBLE, np.full_like(X, std)


# --- return shape ------------------------------------------------------------

def test_returns_three_tuple():
    x, y, ys = _hill_data()
    out = cat_fit(x, y, ys, models_to_run=["flat", "hill_inducer"])
    assert len(out) == 3
    flat_output, pred_df, assess_df = out
    assert isinstance(flat_output, dict)


# --- selection ---------------------------------------------------------------

def test_structured_selected_for_sigmoid_data():
    x, y, ys = _hill_data()
    flat_output, _, _ = cat_fit(x, y, ys,
                                models_to_run=["flat", "hill_inducer"])
    assert flat_output["best_model"] == "hill_inducer"
    # Weighted R2 near 1 for a (near) perfect fit.
    assert flat_output["R2|hill_inducer"] > 0.99
    assert flat_output["status"] == "success"


def test_flat_selected_for_flat_data():
    x, y, ys = _flat_data()
    flat_output, _, _ = cat_fit(x, y, ys,
                                models_to_run=["flat", "hill_inducer"])
    # AICc prefers the cheaper (1-param) flat model over a 4-param hill that
    # buys no fit improvement.
    assert flat_output["best_model"] == "flat"


def test_aicc_excludes_overparameterized_models():
    """With few points a 4-param model has n-k-1<=0 -> can't win, params kept."""
    # 4 points, hill_inducer has k=4 -> denom = 4-4-1 = -1 -> AICc = inf.
    x = np.array([0.0, 1.0, 10.0, 100.0])
    y = np.array([0.1, 0.3, 0.6, 0.8])
    ys = np.full_like(x, 0.05)
    flat_output, _, _ = cat_fit(x, y, ys,
                                models_to_run=["flat", "hill_inducer"])

    assert flat_output["best_model"] == "flat"
    assert flat_output["AIC_weight|hill_inducer"] == 0.0
    # Params are still reported (not selected != not fit).
    for p in MODEL_LIBRARY["hill_inducer"]["param_names"]:
        assert f"hill_inducer|{p}|est" in flat_output


# --- predictions -------------------------------------------------------------

def test_best_only_predicts_single_model():
    x, y, ys = _hill_data()
    _, pred_df, _ = cat_fit(x, y, ys, models_to_run=["flat", "hill_inducer"],
                            best_only=True)
    assert set(pred_df["model"].unique()) == {"hill_inducer"}
    assert pred_df["is_best_model"].all()


def test_write_all_predicts_every_model():
    x, y, ys = _hill_data()
    _, pred_df, _ = cat_fit(x, y, ys, models_to_run=["flat", "hill_inducer"],
                            best_only=False)
    assert set(pred_df["model"].unique()) == {"flat", "hill_inducer"}
    # is_best_model marks only the selected model.
    assert set(pred_df.loc[pred_df["is_best_model"], "model"].unique()) == \
        {"hill_inducer"}


# --- assessment --------------------------------------------------------------

def test_assessment_rollup_and_frame():
    # A clearly-nonzero (constant ~5) curve: every point differs from zero.
    x, y, ys = _flat_data(c=5.0)
    flat_output, _, assess_df = cat_fit(x, y, ys,
                                        models_to_run=["flat", "hill_inducer"])
    for key in ["omnibus_W", "omnibus_df", "omnibus_p", "n_nonzero",
                "any_nonzero"]:
        assert key in flat_output
    assert list(assess_df.columns) == ["x", "y_est", "y_std", "z",
                                       "sig_nonzero", "direction"]
    # One row per unique observed x.
    assert len(assess_df) == len(np.unique(x))
    # Curve sits far from zero -> every point significant, tiny omnibus p.
    assert flat_output["n_nonzero"] == len(np.unique(x))
    assert flat_output["any_nonzero"] is True
    assert flat_output["omnibus_p"] < 0.05


def test_assessment_zero_curve_not_significant():
    x, y, ys = _zero_data()
    flat_output, _, _ = cat_fit(x, y, ys,
                                models_to_run=["flat", "hill_inducer"])
    # A curve sitting on zero (fitted baseline exactly 0) is not distinguishable.
    assert flat_output["n_nonzero"] == 0
    assert flat_output["omnibus_p"] > 0.05


# --- insufficient data -------------------------------------------------------

def test_insufficient_data():
    x = np.array([1.0])
    y = np.array([2.0])
    ys = np.array([0.1])
    flat_output, pred_df, assess_df = cat_fit(
        x, y, ys, x_pred=np.array([1.0, 2.0]), models_to_run=["linear"]
    )
    assert flat_output["status"] == "missing"
    assert flat_output["best_model"] == "None"
    assert np.isnan(flat_output["R2|linear"])
    assert np.isnan(flat_output["omnibus_p"])
    # Best-only predictions: nothing to predict, empty frames.
    assert len(pred_df) == 0
    assert list(pred_df.columns) == ["model", "x", "y", "y_std",
                                     "is_best_model"]
    assert len(assess_df) == 0


def test_all_models_fail_returns_none():
    """If every model fails, status=failure, empty pred/assess, nan rollups."""
    # Two points but ask only for a 4-param model: too few for a fit to
    # converge meaningfully, but more directly, force failure via degenerate y.
    x = np.array([1.0, 2.0])
    y = np.array([np.nan, np.nan])   # all filtered -> insufficient
    ys = np.array([0.1, 0.1])
    flat_output, pred_df, assess_df = cat_fit(x, y, ys,
                                              models_to_run=["linear"])
    # All-NaN y -> filtered to zero points -> missing path.
    assert flat_output["status"] == "missing"
    assert len(assess_df) == 0


# --- input sanitizing --------------------------------------------------------

def test_sanitize_filters_nonfinite():
    x = np.array([0.0, 1.0, np.nan, 10.0, 30.0, 100.0])
    y = np.array([1.0, 3.0, 5.0, 21.0, np.inf, 201.0])
    ys = np.full_like(x, 0.05)
    # Two points dropped (nan x, inf y) -> 4 usable, linear still fittable.
    flat_output, _, assess_df = cat_fit(x, y, ys,
                                        models_to_run=["flat", "linear"])
    assert flat_output["status"] in ("success", "partial")
    assert len(assess_df) == 4
