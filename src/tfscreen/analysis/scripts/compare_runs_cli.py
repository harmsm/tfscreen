"""
CLI for measuring cross-run agreement of any quantile-summarized estimate table.
"""

import json
import os

import pandas as pd

from tfscreen.__version__ import __version__ as _tfscreen_version
from tfscreen.analysis.compare_runs import (
    compare_runs as _compare_runs,
    aggregate_runs,
    resolve_schema,
    shared_quantile_levels,
)
from tfscreen.util.cli import generalized_main, read_lines


def _resolve_estimate_paths(estimates):
    """
    Turn the ``estimates`` positional into a list of estimate CSV paths.

    Two or more arguments are always direct paths. A single argument is read as
    a manifest (one path per line, ``#`` comments allowed) *unless* it ends in
    ``.csv``, in which case it is a single direct path -- which is only useful
    in reference mode, where one estimate run is legal.
    """
    if isinstance(estimates, str):
        estimates = [estimates]
    estimates = list(estimates)

    if len(estimates) == 1 and not estimates[0].lower().endswith(".csv"):
        paths = read_lines(estimates[0])
        if not paths:
            raise ValueError(f"No estimate paths found in '{estimates[0]}'.")
        return paths, estimates[0]

    return estimates, None


def compare_runs(estimates,
                 out_prefix="tfs_compare_runs",
                 reference=None,
                 index_by=None,
                 group_by=None,
                 match_by=None,
                 y_obs=None,
                 y_std=None,
                 point_quantile=0.5,
                 sigma_quantiles=(0.159, 0.841),
                 no_aggregate=False):
    """
    Measure cross-run agreement of an estimate across N independent runs.

    Reads N estimate CSVs of the same quantity -- predicted features
    (tfs-predict-theta, tfs-predict-growth, tfs-predict-epistasis) or fitted
    parameters (tfs-extract-params) alike -- and reports, for every entity, how
    much the runs disagree (reproducibility, in native units) and whether that
    disagreement is explained by each run's reported uncertainty
    (self-consistency, unit-free).

    No thresholds are applied and no grades are assigned: every quantity is a
    number, so filter downstream (e.g. `df["overdispersion"] > 2`) where the
    cutline is recorded alongside the analysis that used it.

    Writes {out_prefix}.csv (one row per entity, or per entity x group_by),
    {out_prefix}_aggregate.csv (unless --no_aggregate), and
    {out_prefix}_metadata.json (every resolved setting, so the run is
    reconstructable).

    Parameters
    ----------
    estimates : list of str
        Either two or more estimate CSV paths (e.g. `rep1.csv rep2.csv`), or a
        single path to a manifest file listing one CSV path per line ('#'
        comments allowed). A single argument ending in '.csv' is treated as a
        direct path, not a manifest.
    out_prefix : str, optional
        Prefix for the output files. Default 'tfs_compare_runs'.
    reference : str or None, optional
        Path to a reference-run CSV. If given, switches to reference mode: each
        estimate run is scored by its deviation from this run (e.g. k-fold
        dropouts vs. a full-data fit). If omitted, runs are compared to their
        symmetric cross-run mean.
    index_by : str or None, optional
        The entity column being scored. Default None: auto-detect 'genotype',
        else 'parameter', else the sole match-key column.
    group_by : list of str or None, optional
        Extra columns to break each entity out by (e.g. `titrant_name
        titrant_conc`). Must be part of the match key. This is a statistical
        zoom: the finer the grouping, the fewer degrees of freedom behind every
        statistic. Check the n_rows / n_eff columns.
    match_by : list of str or None, optional
        The columns that make a row 'the same row' across runs. Default None:
        auto-detect every shared non-value, non-bookkeeping column.
    y_obs : str or None, optional
        Point-estimate column. Default None, which uses the --point_quantile
        column (q0.5).
    y_std : str or None, optional
        Uncertainty column. Default None, which uses the symmetric quantile
        half-width (q0.841 - q0.159)/2. If neither is available the
        self-consistency axis is reported as NaN.
    point_quantile : float, optional
        Quantile used as the point estimate. Default 0.5.
    sigma_quantiles : sequence of float, optional
        Two ascending quantiles bracketing one sigma. Default (0.159, 0.841).
    no_aggregate : bool, optional
        Suppress {out_prefix}_aggregate.csv, the long-form table combining the
        N runs as an equal-weight mixture of their per-run posteriors (folding
        in both per-run posterior width and run-to-run spread; it does not
        shrink with N). Only the N estimate runs are mixed -- a reference run is
        never included. Aggregation needs at least two shared q<level> columns
        and is skipped with a warning when they are absent. It is also the
        slowest step on a large library, so this is the escape hatch.
    """
    paths, manifest = _resolve_estimate_paths(estimates)

    print(f"Reading {len(paths)} estimate file(s)...", flush=True)
    estimate_dfs = [pd.read_csv(p) for p in paths]

    reference_df = None
    if reference is not None:
        print(f"Reading reference {reference}...", flush=True)
        reference_df = pd.read_csv(reference)

    sigma_quantiles = tuple(sigma_quantiles)

    # Resolve the schema up front so the metadata records exactly what the
    # comparison used, rather than a re-derivation that could drift.
    all_dfs = estimate_dfs + ([reference_df] if reference_df is not None
                              else [])
    schema = resolve_schema(all_dfs,
                            index_by=index_by, group_by=group_by,
                            match_by=match_by, y_obs=y_obs, y_std=y_std,
                            point_quantile=point_quantile,
                            sigma_quantiles=sigma_quantiles)
    # Only the resolved *names* are needed here; compare_runs re-resolves the
    # tables itself, so drop these copies rather than holding two sets of a
    # possibly very large frame alive at once.
    schema.pop("resolved", None)

    print(f"  match_by:    {schema['match_by']}", flush=True)
    print(f"  index_by:    {schema['index_by']}", flush=True)
    print(f"  group_by:    {schema['group_by']}", flush=True)
    print(f"  residual:    {schema['residual']}", flush=True)
    print(f"  y_obs:       {schema['y_obs']}", flush=True)
    print(f"  y_std:       {schema['y_std']}", flush=True)
    if schema["y_std"] is None:
        print("  (no uncertainty column resolved; the self-consistency axis "
              "will be NaN)", flush=True)

    # The resolved keys are safe to pass back in, but y_obs/y_std are not: a
    # derived sigma column ('_sigma') exists only on the resolved copies, so
    # the user's originals go through and compare_runs re-derives identically.
    result = _compare_runs(
        estimate_dfs,
        reference_df=reference_df,
        index_by=schema["index_by"],
        group_by=schema["group_by"],
        match_by=schema["match_by"],
        y_obs=y_obs,
        y_std=y_std,
        point_quantile=point_quantile,
        sigma_quantiles=sigma_quantiles,
    )

    out_file = f"{out_prefix}.csv"
    result.to_csv(out_file, index=False)
    print(f"Wrote {len(result)} rows to {out_file}", flush=True)

    aggregate_file = None
    if not no_aggregate:
        levels = shared_quantile_levels(estimate_dfs)
        if not levels:
            print(
                "Warning: skipping the aggregate table -- it mixes the per-run "
                "posteriors and needs at least 2 shared q<level> columns, but "
                "the estimate tables share fewer than 2.",
                flush=True,
            )
        elif len(estimate_dfs) < 2:
            print(
                "Warning: skipping the aggregate table -- mixing needs at "
                "least 2 estimate runs.",
                flush=True,
            )
        else:
            print("Building aggregate table...", flush=True)
            aggregate = aggregate_runs(estimate_dfs,
                                       match_by=schema["match_by"])
            aggregate_file = f"{out_prefix}_aggregate.csv"
            aggregate.to_csv(aggregate_file, index=False)
            print(f"Wrote {len(aggregate)} aggregate rows to {aggregate_file}",
                  flush=True)

    metadata = {
        "tfscreen_version": _tfscreen_version,
        "mode": "reference" if reference_df is not None else "mean",
        "n_runs": len(estimate_dfs),
        "estimates_manifest": (os.path.abspath(manifest) if manifest
                               else None),
        "estimate_files": [os.path.abspath(p) for p in paths],
        "reference_file": (os.path.abspath(reference) if reference else None),
        "match_by": schema["match_by"],
        "index_by": schema["index_by"],
        "group_by": schema["group_by"],
        "report_keys": schema["report_keys"],
        "residual": schema["residual"],
        "y_obs": schema["y_obs"],
        "y_std": schema["y_std"],
        # Non-null when y_std was derived as a symmetric quantile half-width
        # rather than read from a column of the input.
        "y_std_derived_from": (list(sigma_quantiles)
                               if schema["y_std"] == "_sigma" else None),
        "point_quantile": point_quantile,
        "sigma_quantiles": list(sigma_quantiles),
        "spread_estimator": (result["spread_estimator"].iloc[0]
                             if len(result) else None),
        "output_files": {
            "compare": os.path.abspath(out_file),
            "aggregate": (os.path.abspath(aggregate_file) if aggregate_file
                          else None),
        },
        # Reserved for future weighted aggregation; see compare_runs.py's
        # module docstring on why run-axis weights must be design weights.
        "run_weights": None,
        "row_weight_col": None,
    }
    metadata_file = f"{out_prefix}_metadata.json"
    with open(metadata_file, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"Wrote run metadata to {metadata_file}", flush=True)


def main():
    generalized_main(compare_runs,
                     manual_arg_types={"estimates": str,
                                       "reference": str,
                                       "index_by": str,
                                       "group_by": str,
                                       "match_by": str,
                                       "y_obs": str,
                                       "y_std": str,
                                       "sigma_quantiles": float},
                     manual_arg_nargs={"estimates": "+",
                                       "group_by": "+",
                                       "match_by": "+",
                                       "sigma_quantiles": 2})


if __name__ == "__main__":
    main()
