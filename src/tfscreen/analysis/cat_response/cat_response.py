"""
Generic per-group categorical-response engine.

Fits a family of candidate models to a ``y_obs`` vs ``x_obs`` curve independently
within each group and selects the best by AICc weight. Grouping mirrors
``extract_epistasis``: the ``genotype`` column is always the primary axis, with
any additional ``group_by`` columns partitioning the analysis further.

After fitting, a post-hoc pass grades each group's best curve against zero: a
per-point ``sig_nonzero`` test, an ``equiv_zero`` (region-of-practical-
equivalence) test using a global ``delta``, a per-curve omnibus chi-square test,
and a Benjamini-Hochberg FDR correction across curves (see :mod:`cat_assess`).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tqdm

from .cat_fit import cat_fit
from .cat_assess import compute_delta, classify_equiv, benjamini_hochberg
from tfscreen.mle.curve_models import MODEL_LIBRARY, DEFAULT_MODELS
from tfscreen.util import resolve_workers
from tfscreen.util.numerical import xfill

# Number of groups bundled into each worker task. Larger chunks amortize the
# per-task pickle/IPC overhead of ProcessPoolExecutor, which matters a lot when
# there are hundreds of thousands of groups.
_CHUNK_SIZE = 200


def _fit_one(group_key, x, y, y_std, x_pred, models_to_run, best_only, alpha,
             verbose):
    """Run cat_fit for one group and tag the results with the group key."""
    flat_out, pred_df, assess_df = cat_fit(x, y, y_std,
                                           x_pred=x_pred,
                                           models_to_run=models_to_run,
                                           best_only=best_only,
                                           alpha=alpha,
                                           verbose=verbose)
    for col, val in group_key.items():
        flat_out[col] = val
        pred_df[col] = val
        assess_df[col] = val
    return flat_out, pred_df, assess_df


def _fit_chunk(chunk):
    """Worker: run cat_fit for a list of work items, preserving order."""
    return [_fit_one(*item) for item in chunk]


def _iter_chunks(work_items, chunk_size):
    """Yield successive length-``chunk_size`` slices of ``work_items``."""
    for start in range(0, len(work_items), chunk_size):
        yield work_items[start:start + chunk_size]


def cat_response(df,
                 x_obs,
                 y_obs,
                 y_std=None,
                 group_by=None,
                 models_to_run=None,
                 best_only=True,
                 alpha=0.05,
                 delta=None,
                 delta_c=2.0,
                 num_workers=1,
                 progress=True,
                 verbose=False):
    """
    Classify each group's ``y_obs`` vs ``x_obs`` curve using categorical models.

    The DataFrame is partitioned into groups keyed by ``genotype`` plus any
    ``group_by`` columns. For each group, every model in ``models_to_run`` is fit
    to the (``x_obs``, ``y_obs``, ``y_std``) data and the best is selected by AICc
    weight. Fits are embarrassingly parallel across groups. A post-hoc pass then
    grades each best curve against zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form data. Must contain a 'genotype' column, ``x_obs``, ``y_obs``,
        and (if given) ``y_std`` and every column in ``group_by``.
    x_obs : str
        Name of the column holding the independent variable.
    y_obs : str
        Name of the column holding the observable (dependent variable).
    y_std : str or None, optional
        Name of the column holding the standard error of ``y_obs``. If None
        (default), an unweighted fit is performed (uniform weights).
    group_by : list of str or None, optional
        Additional column(s) that, together with 'genotype', define a group. One
        curve is fit per group. If None (default), groups are defined by
        'genotype' alone.
    models_to_run : list of str or None, optional
        Model names to test. If None (default), the curated ``DEFAULT_MODELS``
        set (see ``tfscreen.mle.curve_models``).
    best_only : bool, optional
        If True (default), the returned prediction frame holds only each group's
        best-model curve. If False, it holds every fit model's curve.
    alpha : float, optional
        Two-sided significance level for the per-point ``sig_nonzero`` test and
        the confidence level of the ``equiv_zero`` interval. Also the threshold
        applied to omnibus q-values when assigning ``response_class``. Default
        0.05.
    delta : float or None, optional
        Region-of-practical-equivalence half-width around zero. If None
        (default), computed globally as ``delta_c * median(predicted y_std)``
        over all points (a detectability threshold). Pass a value for a fixed,
        biologically-meaningful region.
    delta_c : float, optional
        Multiplier used when ``delta`` is derived from the median predicted
        standard error. Default 2.0.
    num_workers : int, optional
        Number of worker processes. ``1`` runs serially in-process; ``-1`` uses
        ``os.cpu_count() - 1``; ``N`` uses ``N`` processes. Default 1.
    progress : bool, optional
        If True (default), show a tqdm progress bar over the per-group model
        fits.
    verbose : bool, optional
        If True, print progress and per-model fit warnings. Default False.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per group. Group-key columns, then the flat cat_fit output
        (``status``, ``best_model``, per-model ``AIC_weight|*`` / ``R2|*``, and
        ``<model>|<param>|est`` / ``std`` columns), then the assessment rollups
        (``omnibus_W``, ``omnibus_df``, ``omnibus_p``, ``omnibus_q``,
        ``n_nonzero``, ``any_nonzero``, ``all_equiv_zero``, ``response_class``).
        Column names keep the ``|`` delimiter; presentation is left to callers.
    predictions_df : pandas.DataFrame
        Predicted curves, concatenated across groups. Columns are the group-key
        columns followed by ``model``, ``x``, ``y_model``, ``y_model_std``,
        ``is_best_model``. Restricted to the best model per group unless
        ``best_only`` is False.
    assessment_df : pandas.DataFrame
        Self-contained per-point best-model assessment at the observed (unique)
        x. Group-key columns followed by ``model``, ``x``, ``y_obs``, ``y_std``,
        ``y_model``, ``y_model_std``, ``z``, ``sig_nonzero``, ``direction``,
        ``equiv_zero``.
    delta : float
        The equivalence half-width actually used (resolved from ``delta`` /
        ``delta_c`` if not supplied).
    """
    if models_to_run is None:
        models_to_run = list(DEFAULT_MODELS)

    bad = [m for m in models_to_run if m not in MODEL_LIBRARY]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Valid: {list(MODEL_LIBRARY)}")

    group_cols = ["genotype"] + (list(group_by) if group_by else [])

    needed = list(dict.fromkeys(group_cols + [x_obs, y_obs]
                                + ([y_std] if y_std is not None else [])))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # A shared prediction grid spanning all observed x-values, so every group's
    # predicted curve is evaluated on the same axis. min_value=0 keeps the pad
    # from producing negative concentrations (the concentration-parameterized
    # models take log(x) and would NaN on negative x).
    x_pred = xfill(pd.unique(df[x_obs]), num_points=100, min_value=0.0)

    # observed=True so unused categorical combinations do not create empty groups.
    work_items = []
    for keys, group in df.groupby(group_cols, sort=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_key = dict(zip(group_cols, keys))

        x = group[x_obs].to_numpy(dtype=float)
        y = group[y_obs].to_numpy(dtype=float)
        if y_std is not None:
            ys = group[y_std].to_numpy(dtype=float)
        else:
            ys = np.ones(len(x), dtype=float)

        work_items.append((group_key, x, y, ys, x_pred, models_to_run,
                           best_only, alpha, verbose))

    n_total = len(work_items)
    if n_total == 0:
        empty = pd.DataFrame(columns=group_cols)
        resolved_delta = delta if delta is not None else np.nan
        return empty.copy(), empty.copy(), empty.copy(), resolved_delta

    workers = resolve_workers(num_workers)
    chunks = list(_iter_chunks(work_items, _CHUNK_SIZE))

    if verbose:
        print(f"Fitting {n_total} group(s) with {workers} worker(s)...",
              flush=True)

    results = []
    with tqdm.tqdm(total=n_total, desc="Fitting groups", unit="group",
                   disable=not progress) as pbar:
        if workers == 1:
            # Serial fast-path: run in-process, no pickling/IPC overhead.
            for chunk in chunks:
                chunk_result = _fit_chunk(chunk)
                results.extend(chunk_result)
                pbar.update(len(chunk_result))
        else:
            # executor.map preserves input order, so results stay aligned with
            # work_items without an explicit index map.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for chunk_result in executor.map(_fit_chunk, chunks):
                    results.extend(chunk_result)
                    pbar.update(len(chunk_result))

    flat_list = [r[0] for r in results]
    pred_list = [r[1] for r in results]
    assess_list = [r[2] for r in results]

    results_df = pd.DataFrame(flat_list)

    predictions_df = pd.concat(pred_list, ignore_index=True)
    pred_other = [c for c in predictions_df.columns if c not in group_cols]
    predictions_df = predictions_df[group_cols + pred_other]

    assessment_df = pd.concat(assess_list, ignore_index=True)

    # --- Post-hoc pass: global delta, equivalence, FDR, response class -------
    resolved_delta = delta
    if resolved_delta is None:
        resolved_delta = compute_delta(
            assessment_df.get("y_model_std", []), delta_c
        )

    if len(assessment_df):
        assessment_df["equiv_zero"] = classify_equiv(
            assessment_df["y_model"].to_numpy(dtype=float),
            assessment_df["y_model_std"].to_numpy(dtype=float),
            resolved_delta,
            alpha=alpha,
        )
    else:
        assessment_df["equiv_zero"] = pd.Series(dtype=bool)

    # Per-group all_equiv_zero rollup (a curve is "confidently flat at zero"
    # only if *every* point's CI sits inside the equivalence region).
    if len(assessment_df):
        all_equiv = (assessment_df.groupby(group_cols, sort=False,
                                            observed=True)["equiv_zero"]
                     .all())
        results_df = results_df.merge(
            all_equiv.rename("all_equiv_zero").reset_index(),
            on=group_cols, how="left"
        )
    else:
        results_df["all_equiv_zero"] = pd.Series(dtype=bool)

    # FDR across curves on the omnibus p-values.
    results_df["omnibus_q"] = benjamini_hochberg(
        results_df.get("omnibus_p", pd.Series(np.nan, index=results_df.index))
    )

    # Three-valued response class: real / confident_zero / indeterminate.
    results_df["response_class"] = _response_class(results_df, alpha)

    # Order columns: group keys first.
    other = [c for c in results_df.columns if c not in group_cols]
    results_df = results_df[group_cols + other]

    assess_other = [c for c in assessment_df.columns if c not in group_cols]
    assessment_df = assessment_df[group_cols + assess_other]

    return results_df, predictions_df, assessment_df, resolved_delta


def _response_class(results_df, alpha):
    """
    Assign each curve to confident_zero / real / indeterminate.

    Equivalence takes precedence over significance: a curve whose every point
    sits inside the region of practical equivalence is ``confident_zero`` even
    if the omnibus test finds a statistically significant but practically
    negligible offset (tiny error bars make a hair-off-zero baseline
    "significant"). The ROPE is what makes "real" mean *meaningfully* nonzero.

    - ``confident_zero``: every point's CI lies within [-delta, delta].
    - ``real``: not confident_zero, and omnibus q-value < alpha (distinguishable
      from zero by more than the equivalence margin somewhere on the curve).
    - ``indeterminate``: neither (too noisy to call, or unassessable).
    """
    q = results_df.get("omnibus_q",
                       pd.Series(np.nan, index=results_df.index)).to_numpy()
    all_equiv = results_df.get(
        "all_equiv_zero", pd.Series(False, index=results_df.index)
    ).fillna(False).to_numpy(dtype=bool)

    out = np.full(len(results_df), "indeterminate", dtype=object)
    real = np.isfinite(q) & (q < alpha)
    out[real] = "real"
    # Equivalence wins over a barely-significant offset.
    out[all_equiv] = "confident_zero"
    return out
