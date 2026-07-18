"""
Generic per-group categorical-response engine.

Fits a family of candidate models to a ``y_obs`` vs ``x_obs`` curve independently
within each group and selects the best by AIC weight. Grouping mirrors
``extract_epistasis``: the ``genotype`` column is always the primary axis, with
any additional ``group_by`` columns partitioning the analysis further.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .cat_fit import cat_fit
from tfscreen.mle.curve_models import MODEL_LIBRARY
from tfscreen.util import resolve_workers
from tfscreen.util.numerical import xfill

# Number of groups bundled into each worker task. Larger chunks amortize the
# per-task pickle/IPC overhead of ProcessPoolExecutor, which matters a lot when
# there are hundreds of thousands of groups.
_CHUNK_SIZE = 200


def _fit_one(group_key, x, y, y_std, x_pred, models_to_run, verbose):
    """Run cat_fit for one group and tag the results with the group key."""
    flat_out, pred_df = cat_fit(x, y, y_std,
                                x_pred=x_pred,
                                models_to_run=models_to_run,
                                verbose=verbose)
    for col, val in group_key.items():
        flat_out[col] = val
        pred_df[col] = val
    return flat_out, pred_df


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
                 num_workers=1,
                 verbose=False):
    """
    Classify each group's ``y_obs`` vs ``x_obs`` curve using categorical models.

    The DataFrame is partitioned into groups keyed by ``genotype`` plus any
    ``group_by`` columns. For each group, every model in ``models_to_run`` is fit
    to the (``x_obs``, ``y_obs``, ``y_std``) data and the best is selected by AIC
    weight. Fits are embarrassingly parallel across groups.

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
        Model names to test. If None (default), all models in MODEL_LIBRARY.
    num_workers : int, optional
        Number of worker processes. ``1`` runs serially in-process; ``-1`` uses
        ``os.cpu_count() - 1``; ``N`` uses ``N`` processes. Default 1.
    verbose : bool, optional
        If True, print progress and per-model fit warnings. Default False.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per group. Columns are the group-key columns followed by the flat
        cat_fit output (``status``, ``best_model``, per-model ``AIC_weight|*`` /
        ``R2|*`` weights, and ``<model>|<param>|est`` / ``<model>|<param>|std``
        parameter columns). Column names keep the ``|`` delimiter; presentation
        (cleanup, per-model splitting) is left to callers.
    predictions_df : pandas.DataFrame
        Predicted curves, concatenated across groups. Columns are the group-key
        columns followed by ``model``, ``x``, ``y``, ``y_std``, ``is_best_model``.
    """
    if models_to_run is None:
        models_to_run = list(MODEL_LIBRARY.keys())

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
    # predicted curve is evaluated on the same axis.
    x_pred = xfill(pd.unique(df[x_obs]), num_points=100)

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

        work_items.append((group_key, x, y, ys, x_pred, models_to_run, verbose))

    n_total = len(work_items)
    if n_total == 0:
        empty = pd.DataFrame(columns=group_cols)
        return empty.copy(), empty.copy()

    workers = resolve_workers(num_workers)
    chunks = list(_iter_chunks(work_items, _CHUNK_SIZE))

    if verbose:
        print(f"Fitting {n_total} group(s) with {workers} worker(s)...",
              flush=True)

    results = []
    if workers == 1:
        # Serial fast-path: run in-process, no pickling/IPC overhead.
        for chunk in chunks:
            results.extend(_fit_chunk(chunk))
    else:
        # executor.map preserves input order, so results stay aligned with
        # work_items without an explicit index map.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk_result in executor.map(_fit_chunk, chunks):
                results.extend(chunk_result)

    flat_list = [r[0] for r in results]
    pred_list = [r[1] for r in results]

    results_df = pd.DataFrame(flat_list)
    other = [c for c in results_df.columns if c not in group_cols]
    results_df = results_df[group_cols + other]

    predictions_df = pd.concat(pred_list, ignore_index=True)
    pred_other = [c for c in predictions_df.columns if c not in group_cols]
    predictions_df = predictions_df[group_cols + pred_other]

    return results_df, predictions_df
