from tfscreen.mle.curve_models import MODEL_LIBRARY, DEFAULT_MODELS

from tfscreen.mle import (
    run_least_squares,
    predict_with_error,
    run_matrix_wls
)

from tfscreen.util.numerical import xfill
from .cat_assess import assess_best_model, residual_runs_p, goodness_of_fit_p

import numpy as np
import pandas as pd

# Qualitative response shape of each model, surfaced as the ``shape`` column so
# curves can be categorized by form (independent of the zero/magnitude axis).
# A model not listed here maps to "other".
_SHAPE_BY_MODEL = {
    "flat": "flat",
    "linear": "linear", "linear_log": "linear",
    "repressor": "sigmoid", "inducer": "sigmoid",
    "hill_repressor": "sigmoid", "hill_inducer": "sigmoid",
    "bell_peak": "peaked", "bell_dip": "peaked",
    "bell_peak_log": "peaked", "bell_dip_log": "peaked",
    "biphasic_peak": "biphasic", "biphasic_dip": "biphasic",
}


def select_by_adequacy(models, adequacy_alpha):
    """
    Pick the simplest *adequate* model; AICc breaks ties within a complexity tier.

    AICc alone selects the least-bad candidate but never asks whether the winner
    actually fits, so at small n its parsimony penalty can crown a simple model
    whose residuals are clearly systematic. Here a model is *adequate* when the
    residual runs test does not reject randomness (``runs_p >= adequacy_alpha``).
    Walking complexity (number of parameters ``k``) from low to high, the winner
    is the lowest-AICc adequate model in the first tier that has one.

    Parameters
    ----------
    models : list of dict
        One dict per selectable fit, with keys ``model``, ``k``, ``AICc``,
        ``runs_p`` (already filtered to successful fits with a usable covariance
        and finite AICc).
    adequacy_alpha : float
        Runs-test significance threshold. ``runs_p`` below this flags
        systematic residuals (inadequate shape).

    Returns
    -------
    best_model : str or None
        Selected model name (None if ``models`` is empty).
    shape_status : str
        ``"adequate"`` if an adequate model was found; otherwise the fallback is
        the overall lowest-AICc model, tagged ``"unmodeled"`` (some fit could be
        assessed but none passed) or ``"unassessable"`` (no fit had a computable
        runs test), or ``"none"`` when there are no models.
    """
    if not models:
        return None, "none"

    aicc_best = min(models, key=lambda m: m["AICc"])["model"]

    adequate = [m for m in models
                if np.isfinite(m["runs_p"]) and m["runs_p"] >= adequacy_alpha]
    if adequate:
        min_k = min(m["k"] for m in adequate)
        tier = [m for m in adequate if m["k"] == min_k]
        return min(tier, key=lambda m: m["AICc"])["model"], "adequate"

    assessable = any(np.isfinite(m["runs_p"]) for m in models)
    return aicc_best, ("unmodeled" if assessable else "unassessable")

# Keys returned by assess_best_model's per-point dict (the model curve + tests).
_PER_POINT_COLS = ["x", "y_model", "y_model_std", "z", "sig_nonzero",
                   "direction"]

# Columns of the per-point assessment frame, in order. Kept as a module
# constant so empty-result paths emit an identically-shaped (empty) frame. This
# is the self-contained record: the best model's name, the observed data
# (``y_obs`` and its input error ``y_std``), the fitted curve at the observed x
# (``y_model`` and its propagated error ``y_model_std``), and the zero tests.
_ASSESS_COLS = ["model", "x", "y_obs", "y_std", "y_model", "y_model_std",
                "z", "sig_nonzero", "direction"]

# Non-float dtypes for the empty-frame path.
_ASSESS_DTYPES = {"model": object, "sig_nonzero": bool, "direction": int}

# Rollup keys added to flat_output by the post-hoc assessment. Listed here so
# the insufficient-data / all-fail paths can emit them as NaN.
_ROLLUP_KEYS = ["omnibus_W", "omnibus_df", "omnibus_p", "n_nonzero",
                "any_nonzero"]


def _set_no_best_model(flat_output):
    """Emit the best-model / shape keys for a curve with no selectable model."""
    flat_output["best_model"] = "None"
    flat_output["aicc_best_model"] = "None"
    flat_output["best_model_R2"] = np.nan
    flat_output["best_model_AIC_weight"] = np.nan
    flat_output["best_model_gof_p"] = np.nan
    flat_output["best_model_runs_p"] = np.nan
    flat_output["shape"] = "none"
    flat_output["shape_status"] = "none"


def _empty_assess_df():
    return pd.DataFrame({
        c: pd.Series(dtype=_ASSESS_DTYPES.get(c, float)) for c in _ASSESS_COLS
    })


def cat_fit(x, y, y_std, x_pred=None, models_to_run=None, best_only=True,
            alpha=0.05, adequacy_alpha=0.05, verbose=False):
    """
    Fits multiple models to a single dataset and returns a flat dictionary
    of all results, suitable for aggregation.

    Model selection is *adequacy-first*: each model's AICc (the small-sample-
    corrected AIC on the weighted residuals) is still reported, but the selected
    ``best_model`` is the **simplest model whose residuals are not systematically
    structured** -- judged by a Wald-Wolfowitz runs test (``runs_p >=
    adequacy_alpha``) -- with AICc breaking ties within a complexity tier (see
    :func:`select_by_adequacy`). This avoids AICc's small-sample tendency to
    crown a simple model (e.g. linear) that clearly misfits. A weighted-chi2
    goodness-of-fit p-value is also reported per model (``gof_p|*``) but does not
    gate selection. After selection, the best model is evaluated at the observed
    x and tested against zero (see :mod:`cat_assess`).

    Parameters
    ----------
    x, y, y_std : np.ndarray
        The independent variable, dependent variable, and standard error of the
        dependent variable.
    x_pred : np.ndarray, optional
        array at which to predict x after fitting each model. If not specified,
        fill in values within x.
    models_to_run : list of str, optional
        A list of model names to test. If None (default), the curated
        ``DEFAULT_MODELS`` set is used.
    best_only : bool, optional
        If True (default), only the selected best model's predicted curve is
        emitted in the returned prediction frame. If False, every successfully
        fit model's curve is emitted (the larger, "all models" output).
    alpha : float, optional
        Two-sided significance level for the per-point ``sig_nonzero`` test.
        Default 0.05.
    adequacy_alpha : float, optional
        Runs-test threshold for the adequacy-first selection. A model whose
        ``runs_p`` falls below this is treated as having systematic (structured)
        residuals and is not selected unless no adequate model exists. Default
        0.05.
    verbose : bool, optional
        If True, prints warnings to the console when a model fails to fit.
        Defaults to False.

    Returns
    -------
    dict
        A single, flat dictionary containing the best model summary, AICc
        weights for all tested models, all parameter estimates and standard
        errors for all tested models, and the best-model omnibus rollup
        (``omnibus_W/df/p``, ``n_nonzero``, ``any_nonzero``).
    pd.DataFrame
        Model predictions with columns ``model``, ``x``, ``y_model``,
        ``y_model_std``, ``is_best_model``. Restricted to the best model unless
        ``best_only`` is False.
    pd.DataFrame
        Self-contained per-point assessment of the best model at the observed
        (unique) x, with columns ``model``, ``x``, ``y_obs`` (observed value),
        ``y_std`` (observed input error), ``y_model`` (fitted curve),
        ``y_model_std`` (propagated fit error), ``z``, ``sig_nonzero``,
        ``direction``. Empty if no model could be fit. (``equiv_zero`` is added
        downstream in ``cat_response`` once the global delta is known.)
    """

    if models_to_run is None:
        models_to_run = list(DEFAULT_MODELS)

    flat_output = {}

    # Sanitize inputs
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_std) & (y_std > 0)
    x = x[finite_mask]
    y = y[finite_mask]
    y_std = y_std[finite_mask]

    if x_pred is None:
        # min_value=0 avoids negative concentrations in the pad (the
        # concentration-parameterized models take log(x) and NaN on negative x).
        x_pred = xfill(x, min_value=0.0)

    # Assessment / omnibus grid: the unique observed concentrations (the shared
    # titration series), one row per concentration.
    x_assess = np.unique(x)

    # Handle insufficient data case by returning nan dict and empty pred/assess.
    if len(x) < 2:

        flat_output['status'] = "missing"
        _set_no_best_model(flat_output)
        for key in _ROLLUP_KEYS:
            flat_output[key] = np.nan
        for name in models_to_run:
            param_names = MODEL_LIBRARY[name]["param_names"]
            flat_output[f"R2|{name}"] = np.nan
            flat_output[f"AIC_weight|{name}"] = np.nan
            flat_output[f"gof_p|{name}"] = np.nan
            flat_output[f"runs_p|{name}"] = np.nan
            for p_name in param_names:
                flat_output[f"{name}|{p_name}|est"] = np.nan
                flat_output[f"{name}|{p_name}|std"] = np.nan

        pred_df = pd.DataFrame({"model": pd.Series(dtype=object),
                                "x": pd.Series(dtype=float),
                                "y_model": pd.Series(dtype=float),
                                "y_model_std": pd.Series(dtype=float),
                                "is_best_model": pd.Series(dtype=bool)})
        return flat_output, pred_df, _empty_assess_df()

    # Iterate through models and fit
    n = len(y)
    summary_results = []
    param_results = {}
    for name in models_to_run:

        model_func = MODEL_LIBRARY[name]["model_func"]
        guess_func = MODEL_LIBRARY[name]["guess_func"]
        param_names = MODEL_LIBRARY[name]["param_names"]
        bounds = MODEL_LIBRARY[name]["bounds"]
        k = len(param_names)

        cov_matrix = None
        try:

            # Get guesses
            guesses = guess_func(x, y)

            # If this is a 1D array of values, we have normal guesses. Solve by
            # nonlinear weighted least squares.
            if len(guesses.shape) == 1:

                params, std_err, cov_matrix, fit_obj = run_least_squares(
                    model_func, y, y_std, guesses, bounds[0], bounds[1], args=(x,)
                )
                # On "SVD did not converge" run_least_squares returns the
                # exception (not an OptimizeResult) as fit_obj, so guard the
                # attribute access -- treat a missing .success as a failed fit.
                if not getattr(fit_obj, "success", False):
                    msg = getattr(fit_obj, "message", fit_obj)
                    raise RuntimeError(f"Fit failed: {msg}")

            # If this is a 2D array of values, this is a full design matrix.
            # Solve by weighted least squares.
            else:
                params, std_err, cov_matrix, _ = run_matrix_wls(guesses,y,1/y_std)

            y_fit = model_func(params, x)

            # Weighted residuals: fits are weighted, so selection and R2 should
            # be too. chi2 is the weighted residual sum of squares.
            resid = (y - y_fit) / y_std
            chi2 = float(np.sum(resid ** 2))

            # Adequacy diagnostics. runs_p (primary, scale-robust) tests the
            # sign order of the residuals along x for systematic structure;
            # gof_p is the absolute weighted-chi2 lack-of-fit p (reported only).
            order = np.argsort(x, kind="stable")
            runs_p = residual_runs_p(resid[order])
            gof_p = goodness_of_fit_p(chi2, n, k)

            w = 1.0 / (y_std ** 2)
            y_wmean = np.average(y, weights=w)
            ss_tot = float(np.sum(w * (y - y_wmean) ** 2))
            r2 = 1 - (chi2 / ss_tot) if ss_tot > 0 else 0.0

            # AIC from the known-variance Gaussian likelihood (the model-
            # independent -0.5*sum(log 2*pi*sigma^2) constant cancels in
            # weights/deltas and is dropped). AICc adds the small-sample
            # correction; when n - k - 1 <= 0 the model is unusable for
            # selection (aicc = inf) but its params are still reported.
            aic = 2 * k + chi2
            denom = n - k - 1
            aicc = aic + (2 * k * (k + 1) / denom) if denom > 0 else np.inf

            # A converged fit can still have a singular Jacobian, in which case
            # get_cov returns an all-NaN covariance (and NaN std errors) even
            # though fit.success is True. Such a model has finite params/chi2 and
            # would otherwise be selectable, but its prediction/assessment errors
            # would be NaN -- and if it won it would poison the global delta.
            # Treat it as unusable for selection: force aicc = inf (zero weight,
            # can't win) and drop the covariance, while still reporting the point
            # estimates. This reuses the same path as the n - k - 1 <= 0 case.
            cov_usable = (cov_matrix is not None
                          and np.all(np.isfinite(cov_matrix))
                          and np.all(np.isfinite(std_err)))
            if not cov_usable:
                aicc = np.inf
                cov_matrix = None

            summary_results.append({"model": name, "k": k, "R2": r2,
                                    "chi2": chi2, "AIC": aic, "AICc": aicc,
                                    "gof_p": gof_p, "runs_p": runs_p,
                                    "success": True})
            param_results[name] = {"params": params, "std_err": std_err,
                                   "cov": cov_matrix, "names": param_names,
                                   "model_func": model_func}

        except (RuntimeError, ValueError) as e:

            if verbose:
                print(f"Warning: Model '{name}' failed to fit. Reason: {e}")
            summary_results.append({"model": name, "k": k, "R2": np.nan,
                                    "chi2": np.nan, "AIC": np.nan,
                                    "AICc": np.nan, "gof_p": np.nan,
                                    "runs_p": np.nan, "success": False})
            param_results[name] = {
                "params": np.full(k, np.nan), "std_err": np.full(k, np.nan),
                "cov": None, "names": param_names, "model_func": model_func
            }

    # Post-process and flatten results. AICc drives selection and weights.
    summary_df = pd.DataFrame(summary_results)
    valid = summary_df.loc[summary_df['success'] & np.isfinite(summary_df['AICc'])]
    if not valid.empty:
        min_aicc = valid['AICc'].min()
        summary_df['delta_AICc'] = summary_df['AICc'] - min_aicc
        relative_likelihood = np.exp(-0.5 * summary_df['delta_AICc'])
        # Models with infinite AICc (n - k - 1 <= 0) get zero weight.
        relative_likelihood = relative_likelihood.where(
            np.isfinite(summary_df['AICc']), 0.0
        )
        sum_likelihoods = relative_likelihood.sum()
        summary_df['AIC_weight'] = relative_likelihood / sum_likelihoods
    else:
        summary_df['AIC_weight'] = np.nan

    summary_df = summary_df.sort_values(by="AICc").reset_index(drop=True)

    # Populate the flat output dictionary
    for _, row in summary_df.iterrows():
        model_name = row['model']
        flat_output[f"AIC_weight|{model_name}"] = row['AIC_weight']
        flat_output[f"R2|{model_name}"] = row['R2']
        flat_output[f"gof_p|{model_name}"] = row['gof_p']
        flat_output[f"runs_p|{model_name}"] = row['runs_p']

        p_res = param_results[model_name]
        for i, p_name in enumerate(p_res['names']):
            flat_output[f"{model_name}|{p_name}|est"] = p_res['params'][i]
            flat_output[f"{model_name}|{p_name}|std"] = p_res['std_err'][i]

    # Overall status and best model info.
    success_states = summary_df['success'].unique()
    if len(success_states) == 1 and success_states[0] == True:
        flat_output['status'] = "success"
    elif len(success_states) == 1 and success_states[0] == False:
        flat_output['status'] = "failure"
    else:
        flat_output['status'] = "partial"

    if not valid.empty:
        # AICc still ranks the models (summary_df is sorted by it), but the
        # selected best_model is the simplest *adequate* one; AICc only breaks
        # ties within a complexity tier. aicc_best_model records what AICc alone
        # would have chosen, for transparency when the two diverge.
        aicc_best_model = summary_df.iloc[0]['model']
        model_records = valid[['model', 'k', 'AICc', 'runs_p']].to_dict('records')
        best_model, shape_status = select_by_adequacy(model_records,
                                                      adequacy_alpha)

        best_row = summary_df.set_index('model').loc[best_model]
        flat_output['best_model'] = best_model
        flat_output['aicc_best_model'] = aicc_best_model
        flat_output['best_model_R2'] = best_row['R2']
        flat_output['best_model_AIC_weight'] = best_row['AIC_weight']
        flat_output['best_model_gof_p'] = best_row['gof_p']
        flat_output['best_model_runs_p'] = best_row['runs_p']
        flat_output['shape'] = _SHAPE_BY_MODEL.get(best_model, "other")
        flat_output['shape_status'] = shape_status
    else:
        best_model = None
        _set_no_best_model(flat_output)

    # Predicted curves. Only successfully-fit models can be predicted, and only
    # the best model unless best_only is False.
    if best_only:
        models_to_predict = [best_model] if best_model is not None else []
    else:
        models_to_predict = [m for m in summary_df['model']
                             if param_results[m]['cov'] is not None]

    pred_rows = []
    for name in models_to_predict:
        p_res = param_results[name]
        y_pred, y_pred_std = predict_with_error(p_res['model_func'],
                                                p_res['params'],
                                                p_res['cov'],
                                                args=[x_pred])
        pred_rows.append(pd.DataFrame({
            "model": name,
            "x": x_pred,
            "y_model": y_pred,
            "y_model_std": y_pred_std,
        }))

    if pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
    else:
        pred_df = pd.DataFrame({"model": pd.Series(dtype=object),
                                "x": pd.Series(dtype=float),
                                "y_model": pd.Series(dtype=float),
                                "y_model_std": pd.Series(dtype=float)})
    pred_df["is_best_model"] = pred_df["model"] == flat_output["best_model"]

    # Per-point assessment + omnibus rollup for the best model.
    if best_model is not None:
        p_res = param_results[best_model]
        per_point, rollup = assess_best_model(
            p_res['model_func'], p_res['params'], p_res['cov'], x_assess,
            alpha=alpha
        )
        assess_df = pd.DataFrame({c: per_point[c] for c in _PER_POINT_COLS})

        # Attach the observed data aggregated to the assessment grid. With one
        # observation per concentration (the shared-grid assumption) this is a
        # straight alignment; replicate concentrations are mean-collapsed.
        obs = (pd.DataFrame({"x": x, "y_obs": y, "y_std": y_std})
               .groupby("x", sort=True).mean().reindex(x_assess))
        assess_df["y_obs"] = obs["y_obs"].to_numpy()
        assess_df["y_std"] = obs["y_std"].to_numpy()
        assess_df["model"] = best_model
        assess_df = assess_df[_ASSESS_COLS]

        for key in _ROLLUP_KEYS:
            flat_output[key] = rollup[key]
    else:
        assess_df = _empty_assess_df()
        for key in _ROLLUP_KEYS:
            flat_output[key] = np.nan

    return flat_output, pred_df, assess_df
