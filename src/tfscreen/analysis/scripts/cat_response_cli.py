"""
CLI for fitting categorical response models to a y-vs-x curve per group.
"""

import pandas as pd

from tfscreen.analysis.cat_response.cat_response import (
    cat_response as _cat_response,
)
from tfscreen.mle.curve_models import MODEL_LIBRARY
from tfscreen.util import resolve_obs_columns
from tfscreen.util.cli import generalized_main


def _write_per_model(results_df, models, group_cols, out_prefix):
    """
    Explode the flat results table into one CSV per model.

    ``results_df`` carries every model's parameters in ``<model>|<param>|est`` /
    ``<model>|<param>|std`` columns (plus ``AIC_weight|<model>`` / ``R2|<model>``).
    For each model this pulls out its columns, renames them to ``<param>_est`` /
    ``<param>_std``, appends per-model fit stats, and writes
    ``{out_prefix}_{model}.csv``.
    """
    for model in models:

        model_cols = [c for c in results_df.columns
                      if c.startswith(f"{model}|")]
        model_df = results_df[group_cols + model_cols].copy()

        # "<model>|<param>|est" -> "<param>_est"
        renamer = {}
        for c in model_cols:
            parts = c.split("|")
            renamer[c] = f"{parts[1]}_{parts[2]}"
        model_df = model_df.rename(columns=renamer)

        # Per-model fit statistics.
        model_df["is_best_model"] = (results_df["best_model"] == model).values
        model_df["R2"] = results_df.get(
            f"R2|{model}", pd.Series(index=results_df.index, dtype=float)
        ).values
        model_df["AIC_weight"] = results_df.get(
            f"AIC_weight|{model}", pd.Series(index=results_df.index, dtype=float)
        ).values

        # Order columns: group keys, est..., std..., stats.
        param_names = MODEL_LIBRARY[model]["param_names"]
        ordered = list(group_cols)
        ordered += [f"{p}_est" for p in param_names]
        ordered += [f"{p}_std" for p in param_names]
        ordered += ["is_best_model", "R2", "AIC_weight"]
        model_df = model_df[ordered]

        model_df.to_csv(f"{out_prefix}_{model}.csv", index=False)


def cat_response(data_file,
                 x_obs,
                 y_obs=None,
                 out_prefix="tfs_cat_response",
                 y_std=None,
                 group_by=None,
                 models=None,
                 alpha=0.05,
                 delta=None,
                 delta_c=2.0,
                 write_all_predictions=False,
                 num_workers=-1):
    """
    Classify each group's response curve using categorical response models.

    Reads a long-form CSV and fits one or more response models to every group.
    Groups are defined by the 'genotype' column plus any --group_by columns. For
    each group the best-fitting model is selected by AICc weight, then graded
    against zero (per-point sig_nonzero / equiv_zero tests + a per-curve omnibus
    chi-square with a Benjamini-Hochberg FDR correction).

    Writes:
      - {out_prefix}.csv             one row per group; all models' weights and
                                     parameter estimates, plus the assessment
                                     rollups (omnibus_p/q, n_nonzero,
                                     all_equiv_zero, response_class).
      - {out_prefix}_{model}.csv     one file per model; that model's parameter
                                     table and per-group fit statistics.
      - {out_prefix}_predictions.csv best-model predicted curves (all models
                                     when --write_all_predictions).
      - {out_prefix}_assessment.csv  per (group, x) best-model assessment.

    Parameters
    ----------
    data_file : str
        Path to the input CSV. Must contain a 'genotype' column, ``x_obs``,
        ``y_obs``, and (if given) ``y_std`` and every column in ``group_by``.
    x_obs : str
        Name of the column holding the independent variable (e.g. 'titrant_conc').
    y_obs : str or None, optional
        Name of the column holding the observable (e.g. 'q0.5', 'point_est'). If
        None (default) and the input has a 'q0.5' column (as written by
        tfs-predict-theta), 'q0.5' is used.
    out_prefix : str, optional
        Prefix for the output CSV files. Default 'tfs_cat_response'.
    y_std : str or None, optional
        Name of the column holding per-row uncertainty (standard deviation). If
        None (default) and both 'q0.841' and 'q0.159' are present, sigma is
        computed as (q0.841 - q0.159) / 2; otherwise the fit is unweighted.
    group_by : list of str or None, optional
        Additional column(s) that, together with 'genotype', define a group. If
        omitted, groups are defined by 'genotype' alone.
    models : list of str or None, optional
        Response models to fit. Defaults to all models in MODEL_LIBRARY.
    alpha : float, optional
        Significance level for the per-point tests and the omnibus q-value
        threshold used to call a curve 'real'. Default 0.05.
    delta : float or None, optional
        Region-of-practical-equivalence half-width around zero. If None
        (default), derived globally as ``delta_c * median(predicted y_std)``.
        Pass a value for a fixed, biologically-meaningful region.
    delta_c : float, optional
        Multiplier used when ``delta`` is auto-derived. Default 2.0.
    write_all_predictions : bool, optional
        If True, write every fit model's predicted curve rather than only the
        best model's. Default False.
    num_workers : int, optional
        Number of parallel worker processes. ``1`` runs serially; ``-1`` (the
        default) uses ``os.cpu_count() - 1``; ``N`` uses ``N`` processes.
    """
    if models is None:
        models = list(MODEL_LIBRARY.keys())

    bad = [m for m in models if m not in MODEL_LIBRARY]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Valid: {list(MODEL_LIBRARY)}")

    print(f"Reading {data_file}...", flush=True)
    df = pd.read_csv(data_file)

    group_cols = ["genotype"] + (list(group_by) if group_by else [])

    # Fill in y_obs/y_std defaults from quantile columns (q0.5 for the point
    # estimate, (q0.841 - q0.159)/2 for the std) when not given explicitly.
    df, y_obs, y_std = resolve_obs_columns(df, y_obs=y_obs, y_std=y_std)

    # Fail fast on missing columns rather than deep inside the group loop.
    required = list(dict.fromkeys(group_cols + [x_obs, y_obs]
                                  + ([y_std] if y_std is not None else [])))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{data_file}' is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if y_std is None:
        print("Warning: no y_std column found or specified; fitting unweighted.",
              flush=True)

    print(f"  Fitting groups defined by {group_cols} "
          f"with {num_workers} worker(s)...", flush=True)

    results_df, predictions_df, assessment_df, resolved_delta = _cat_response(
        df,
        x_obs=x_obs,
        y_obs=y_obs,
        y_std=y_std,
        group_by=group_by,
        models_to_run=models,
        best_only=(not write_all_predictions),
        alpha=alpha,
        delta=delta,
        delta_c=delta_c,
        num_workers=num_workers,
    )

    print(f"  Using equivalence half-width delta = {resolved_delta:.6g}",
          flush=True)

    # Main table: flat, one row per group, with clean column names.
    main_df = results_df.copy()
    main_df.columns = [c.replace("|", "_") for c in main_df.columns]
    out_file = f"{out_prefix}.csv"
    main_df.to_csv(out_file, index=False)
    print(f"Wrote {len(main_df)} rows to {out_file}", flush=True)

    # Per-model parameter tables.
    _write_per_model(results_df, models, group_cols, out_prefix)
    print(f"Wrote {len(models)} per-model file(s) "
          f"({out_prefix}_<model>.csv)", flush=True)

    # Predicted curves.
    pred_file = f"{out_prefix}_predictions.csv"
    predictions_df.to_csv(pred_file, index=False)
    print(f"Wrote {len(predictions_df)} rows to {pred_file}", flush=True)

    # Per-point best-model assessment.
    assess_file = f"{out_prefix}_assessment.csv"
    assessment_df.to_csv(assess_file, index=False)
    print(f"Wrote {len(assessment_df)} rows to {assess_file}", flush=True)


def main():
    generalized_main(cat_response,
                     manual_arg_types={"y_obs": str,
                                       "y_std": str,
                                       "group_by": str,
                                       "models": str,
                                       "delta": float},
                     manual_arg_nargs={"group_by": "+",
                                       "models": "+"})


if __name__ == "__main__":
    main()
