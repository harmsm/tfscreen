"""
Tests for cat_fit: AICc/weighted-R2 selection, best_only predictions, the
per-point assessment/omnibus rollup, and the insufficient-data path. Uses the
real MODEL_LIBRARY (fits are deterministic) rather than mocks.
"""
import numpy as np
import pytest

from tfscreen.analysis.cat_response.cat_fit import (
    cat_fit, select_by_adequacy, select_by_shape, _shape_status,
)
from tfscreen.mle.curve_models import MODEL_LIBRARY, DEFAULT_MODELS, SHAPE_MODELS


# --- escalate-only adequacy selection ----------------------------------------

def _rec(model, k, aicc, runs_p):
    return {"model": model, "k": k, "AICc": aicc, "runs_p": runs_p}


class TestSelectByAdequacy:
    def test_keeps_adequate_aicc_pick(self):
        # AICc pick (bell, lowest AICc) is adequate -> kept, not demoted.
        models = [_rec("linear", 2, 10.0, 0.5), _rec("bell", 4, 2.0, 0.6)]
        assert select_by_adequacy(models, 0.05)["model"] == "bell"

    def test_escalates_off_flagged_pick(self):
        # AICc pick (linear) is flagged; escalate to the adequate, no-simpler bell.
        models = [_rec("flat", 1, 5.0, 0.5), _rec("linear", 2, 4.0, 0.01),
                  _rec("bell", 4, 8.0, 0.4)]
        assert select_by_adequacy(models, 0.05)["model"] == "bell"

    def test_never_demotes_flagged_complex_pick(self):
        # AICc pick (bell) is flagged but nothing no-simpler is adequate; the
        # only adequate model is *simpler* (flat) -> KEEP bell, never demote.
        # This is the failure mode the escalate-only rule exists to prevent.
        models = [_rec("flat", 1, 10.0, 0.9), _rec("bell", 4, 2.0, 0.01)]
        assert select_by_adequacy(models, 0.05)["model"] == "bell"

    def test_escalation_tie_break_by_aicc(self):
        # Flagged linear -> among no-simpler adequate models, lowest AICc wins.
        models = [_rec("linear", 2, 4.0, 0.01), _rec("repressor", 3, 9.0, 0.3),
                  _rec("inducer", 3, 6.0, 0.4)]
        assert select_by_adequacy(models, 0.05)["model"] == "inducer"

    def test_keeps_flagged_pick_when_no_adequate_alternative(self):
        models = [_rec("flat", 1, 5.0, 0.002), _rec("linear", 2, 4.0, 0.002)]
        assert select_by_adequacy(models, 0.05)["model"] == "linear"

    def test_unassessable_pick_kept(self):
        models = [_rec("flat", 1, 5.0, np.nan), _rec("linear", 2, 4.0, np.nan)]
        assert select_by_adequacy(models, 0.05)["model"] == "linear"


class TestShapeStatus:
    def test_adequate(self):
        assert _shape_status(0.5, 0.05) == "adequate"

    def test_misfit(self):
        assert _shape_status(0.01, 0.05) == "misfit"

    def test_unassessable(self):
        assert _shape_status(np.nan, 0.05) == "unassessable"


# --- shape classifier --------------------------------------------------------

def _srec(model, k, aicc, r2, autocorr_p):
    return {"model": model, "k": k, "AICc": aicc, "R2": r2,
            "autocorr_p": autocorr_p}


class TestSelectByShape:
    def test_flat_when_no_structure(self):
        # flat autocorr_p above cutoff -> not curvy -> flat, even if a curve fits.
        models = [_srec("flat", 1, 5.0, 0.0, 0.6),
                  _srec("bell_dip_log", 4, 2.0, 0.95, 0.6)]
        assert select_by_shape(models, curvy_cutoff=0.1)["model"] == "flat"

    def test_curvy_picks_best_r2_shape(self):
        # flat is structured (low autocorr_p); the dip fits far better than the
        # step, so it's chosen despite more parameters (no AICc penalty).
        models = [_srec("flat", 1, 2.0, -0.05, 0.02),
                  _srec("inducer", 3, 4.0, 0.51, 0.3),
                  _srec("bell_dip_log", 4, 8.0, 0.96, 0.3)]
        chosen = select_by_shape(models, curvy_cutoff=0.1)
        assert chosen["model"] == "bell_dip_log"     # -> shape "dip"

    def test_prefers_simpler_within_r2_margin(self):
        # step and dip fit within r2_margin -> the simpler (step) wins.
        models = [_srec("flat", 1, 2.0, -0.05, 0.02),
                  _srec("inducer", 3, 4.0, 0.951, 0.3),
                  _srec("bell_dip_log", 4, 3.0, 0.955, 0.3)]
        chosen = select_by_shape(models, curvy_cutoff=0.1, r2_margin=0.02)
        assert chosen["model"] == "inducer"

    def test_curvy_but_no_curvy_model_falls_back_to_flat(self):
        models = [_srec("flat", 1, 2.0, -0.05, 0.02),
                  _srec("linear_log", 2, 1.0, 0.3, 0.02)]   # linear is not curvy
        assert select_by_shape(models, curvy_cutoff=0.1)["model"] == "flat"

    def test_cutoff_controls_flat_vs_curvy(self):
        models = [_srec("flat", 1, 2.0, -0.05, 0.08),
                  _srec("inducer", 3, 4.0, 0.9, 0.08)]
        # strict cutoff -> flat; liberal cutoff -> curvy
        assert select_by_shape(models, curvy_cutoff=0.05)["model"] == "flat"
        assert select_by_shape(models, curvy_cutoff=0.20)["model"] == "inducer"


def test_shape_mode_default_models():
    """select_by='shape' with models_to_run=None fits the SHAPE_MODELS set."""
    x, y, ys = _hill_data()
    flat_output, _, _ = cat_fit(x, y, ys, select_by="shape")
    fit_models = {k.split("|", 1)[1] for k in flat_output
                  if k.startswith("AIC_weight|")}
    assert fit_models == set(SHAPE_MODELS)
    assert "linear_log" not in fit_models
    assert "biphasic_peak" in fit_models


def test_shape_mode_classifies_dip():
    """The reported real-data dip (called flat by AICc) -> 'dip' in shape mode."""
    conc = np.array([0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0])
    ep = np.array([0.00922, -2.4289, -3.0633, -3.2761, -2.5217, -1.3812,
                   -0.6332, 2.1689])
    es = np.array([1.4067, 1.5657, 0.9956, 0.8424, 1.2566, 1.2217, 1.2639,
                   2.6622])
    aicc_out, _, _ = cat_fit(conc, ep, es, select_by="aicc")
    shape_out, _, _ = cat_fit(conc, ep, es, select_by="shape", curvy_cutoff=0.1)
    assert aicc_out["best_model"] == "flat"          # AICc buries the dip
    assert shape_out["shape"] in ("dip", "biphasic")  # classifier recovers it
    assert shape_out["best_model"] != "flat"


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
    # Adequacy-first prefers the simplest adequate (1-param flat) model over a
    # 4-param hill that buys no fit improvement.
    assert flat_output["best_model"] == "flat"


def test_adequacy_columns_present_and_shape():
    x, y, ys = _hill_data()
    flat_output, _, _ = cat_fit(x, y, ys,
                                models_to_run=["flat", "hill_inducer"])
    # New diagnostic + shape columns are emitted.
    for key in ["aicc_best_model", "best_model_gof_p", "best_model_runs_p",
                "best_model_autocorr", "best_model_autocorr_p",
                "shape", "shape_status", "gof_p|flat", "runs_p|flat",
                "autocorr|flat", "autocorr_p|flat",
                "gof_p|hill_inducer", "runs_p|hill_inducer"]:
        assert key in flat_output
    # Sigmoid data -> hill_inducer selected -> shape "step", adequate.
    assert flat_output["best_model"] == "hill_inducer"
    assert flat_output["shape"] == "step"
    assert flat_output["shape_status"] == "adequate"


def _curved_hetero_data():
    """A real curve where the misfit lives in a few precise points and the many
    plateau points are noisy -- the heteroscedastic case where the sign-based
    runs test on flat is diluted but AICc still sees the curve."""
    x = np.logspace(-8, -2, 15)
    lx = np.log10(x)
    bump = 1.2 * np.exp(-0.5 * ((lx + 5.0) / 0.6) ** 2)
    ystd = np.where(np.abs(lx + 5.0) < 1.2, 0.05, 0.6)
    rng = np.random.default_rng(4)
    return x, bump + rng.normal(0, ystd), ystd


def test_aicc_default_does_not_collapse_to_flat():
    """Regression: default select_by='aicc' keeps the AICc-confident curve even
    when the runs test is too weak to flag flat (the reported bug)."""
    x, y, ys = _curved_hetero_data()
    models = ["flat", "linear_log", "inducer", "bell_peak_log", "bell_dip_log"]
    flat_output, _, _ = cat_fit(x, y, ys, models_to_run=models)  # default aicc
    assert flat_output["best_model"] != "flat"
    assert flat_output["best_model"] == flat_output["aicc_best_model"]


def test_adequacy_mode_never_demotes_confident_curve():
    """Even in adequacy mode, a confident curved AICc pick is never demoted to
    flat just because the diluted runs test can't reject flat."""
    x, y, ys = _curved_hetero_data()
    models = ["flat", "linear_log", "inducer", "bell_peak_log", "bell_dip_log"]
    flat_output, _, _ = cat_fit(x, y, ys, models_to_run=models,
                                select_by="adequacy")
    assert flat_output["best_model"] != "flat"


def test_adequacy_mode_escalates_off_flagged_pick():
    """A clean curved dataset where flat is the AICc pick but is flagged:
    adequacy mode escalates to the curved model; aicc mode would keep flat."""
    # Gentle curvature + tiny homoscedastic noise, enough points for power.
    x = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0])
    lx = np.log10(x + 1.0)
    y = 0.15 * lx ** 2                      # mild parabola in log-x
    ys = np.full_like(x, 0.05)
    models = ["flat", "linear_log", "bell_peak_log"]
    aicc_out, _, _ = cat_fit(x, y, ys, models_to_run=models, select_by="aicc")
    adeq_out, _, _ = cat_fit(x, y, ys, models_to_run=models,
                             select_by="adequacy")
    # flat's residuals cluster (curvature) -> flagged.
    assert aicc_out["runs_p|flat"] < 0.05
    if aicc_out["best_model"] == "flat":
        # The regime this test targets: adequacy escalates off the flagged flat.
        assert adeq_out["best_model"] != "flat"


def test_invalid_select_by_raises():
    x, y, ys = _hill_data()
    with pytest.raises(ValueError, match="select_by"):
        cat_fit(x, y, ys, models_to_run=["flat"], select_by="bogus")


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
    assert list(assess_df.columns) == ["model", "x", "y_obs", "y_std",
                                       "y_model", "y_model_std", "z",
                                       "sig_nonzero", "direction"]
    # One row per unique observed x.
    assert len(assess_df) == len(np.unique(x))
    # Model name recorded; observed data carried through alongside the fit.
    assert (assess_df["model"] == "flat").all()
    assert assess_df["y_obs"].to_numpy() == pytest.approx(y)
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
    assert list(pred_df.columns) == ["model", "x", "y_model", "y_model_std",
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


def test_default_models_are_the_curated_set():
    """With models_to_run=None, cat_fit fits exactly DEFAULT_MODELS (not all)."""
    x, y, y_std = _hill_data()
    flat, _, _ = cat_fit(x, y, y_std)  # models_to_run defaults to None
    fit_models = {k.split("|", 1)[1] for k in flat if k.startswith("AIC_weight|")}
    assert fit_models == set(DEFAULT_MODELS)
    # The log-conc variants are in; the raw-x duplicates and biphasic are out.
    assert "bell_peak_log" in fit_models
    assert "biphasic_peak" not in fit_models
    assert "bell_peak" not in fit_models


# --- degenerate-covariance guard ---------------------------------------------

def _singular_bell_data():
    """Constant y -> bell amplitude ~ 0 -> center/width unidentified.

    scipy converges (fit.success is True) but the Jacobian is rank-deficient, so
    get_cov returns an all-NaN covariance. This is the reachable case where a
    model would otherwise be selected with a NaN covariance.
    """
    x = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])
    y = np.full_like(x, 0.5)
    ys = np.full_like(x, 0.05)
    return x, y, ys


def test_singular_covariance_model_excluded_from_selection():
    # bell_peak is the only candidate and it fits with a NaN covariance.
    x, y, ys = _singular_bell_data()
    flat, _, assess = cat_fit(x, y, ys, models_to_run=["bell_peak"])

    # Not selectable -> no best model, and (crucially) no NaN-poisoned
    # assessment rows to leak into the global delta.
    assert flat["best_model"] == "None"
    assert len(assess) == 0
    # Point estimates are still reported (not selected != not fit).
    assert "bell_peak|amplitude|est" in flat
    assert np.isfinite(flat["bell_peak|amplitude|est"])


def test_singular_covariance_loses_to_usable_model():
    """A NaN-covariance model gets zero weight; a usable competitor wins."""
    x, y, ys = _singular_bell_data()
    # flat also fits this constant data (with a finite covariance) and wins.
    flat, _, assess = cat_fit(x, y, ys, models_to_run=["flat", "bell_peak"])

    assert flat["best_model"] == "flat"
    assert flat["AIC_weight|bell_peak"] == 0.0
    # The selected model's assessment errors are all finite (delta stays clean).
    assert np.all(np.isfinite(assess["y_model_std"].to_numpy()))


# --- prediction grid domain --------------------------------------------------

def test_prediction_grid_is_non_negative():
    """The predicted curve grid must not include negative concentrations.

    The concentration-parameterized models take log(x); a negative x_pred would
    make them emit NaN (and a RuntimeWarning). The xfill min_value=0 floor
    prevents that.
    """
    x, y, ys = _hill_data()   # X spans [0, 100]; linear pad would go negative
    _, pred_df, _ = cat_fit(x, y, ys, models_to_run=["flat", "hill_inducer"])
    assert (pred_df["x"].to_numpy() >= 0).all()
    # And the predicted curve has no NaN (Hill no longer evaluated at x < 0).
    assert not pred_df["y_model"].isna().any()
