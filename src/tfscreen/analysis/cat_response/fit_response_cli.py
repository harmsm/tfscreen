"""
CLI for fitting cat_response models to theta-vs-titrant data per genotype.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from tfscreen.analysis.cat_response.cat_fit import cat_fit
from tfscreen.mle.curve_models import MODEL_LIBRARY

DEFAULT_MODELS = ["flat", "repressor", "inducer", "hill_repressor", "hill_inducer"]


def _fit_one(args):
    """Worker: run cat_fit for one (genotype, titrant_name) pair."""
    genotype, titrant_name, x, y, y_std, models_to_run = args
    flat_out, _ = cat_fit(x, y, y_std, models_to_run=models_to_run)
    flat_out["genotype"] = genotype
    flat_out["titrant_name"] = titrant_name
    return flat_out


def fit_response(df,
                 theta_col=None,
                 sigma_col=None,
                 models_to_run=None,
                 workers=1):
    """
    Run cat_fit for every (genotype, titrant_name) group in df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: genotype, titrant_name, titrant_conc, theta_col, and
        either sigma_col or both upper_std and lower_std.
    theta_col : str or None
        Column holding theta values passed to the fitter as y.  If ``None``,
        auto-detected: ``median`` if present, then ``point_est``.
    sigma_col : str or None
        Column holding per-observation sigma.  If None and df has
        upper_std / lower_std, sigma = (upper_std - lower_std) / 2.
    models_to_run : list of str or None
        Subset of MODEL_LIBRARY keys to fit.  Defaults to DEFAULT_MODELS.
    workers : int
        Parallel worker processes.

    Returns
    -------
    pd.DataFrame
        One row per (genotype, titrant_name).  Columns include best_model,
        status, AIC weights, and all parameter estimates/stds for every model,
        with '|' replaced by '_' in column names.
    """
    if models_to_run is None:
        models_to_run = DEFAULT_MODELS

    bad = [m for m in models_to_run if m not in MODEL_LIBRARY]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Valid: {list(MODEL_LIBRARY)}")

    if theta_col is None:
        if "q0.5" in df.columns:
            theta_col = "q0.5"
        elif "point_est" in df.columns:
            theta_col = "point_est"
        else:
            raise ValueError(
                "No theta column found. Expected 'q0.5' (posterior median) or "
                "'point_est' (MAP). Pass theta_col explicitly to override."
            )

    if sigma_col is None:
        if "q0.841" in df.columns and "q0.159" in df.columns:
            df = df.copy()
            df["_sigma"] = (df["q0.841"] - df["q0.159"]) / 2
            sigma_col = "_sigma"
        else:
            raise ValueError(
                "No sigma_col specified and df lacks q0.841/q0.159 columns. "
                "Provide --sigma_col or ensure q0.841 and q0.159 are present."
            )

    work_items = []
    for (genotype, titrant_name), group in df.groupby(["genotype", "titrant_name"],
                                                       sort=False):
        x = group["titrant_conc"].to_numpy(dtype=float)
        y = group[theta_col].to_numpy(dtype=float)
        y_std = group[sigma_col].to_numpy(dtype=float)
        work_items.append((genotype, titrant_name, x, y, y_std, models_to_run))

    n_total = len(work_items)
    results = [None] * n_total
    idx_map = {id(item): i for i, item in enumerate(work_items)}

    print(f"  Fitting {n_total} (genotype, titrant_name) pairs "
          f"with {workers} worker(s)...", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fit_one, item): idx_map[id(item)]
                   for item in work_items}
        n_done = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            n_done += 1
            if n_done % 5000 == 0 or n_done == n_total:
                print(f"  {n_done}/{n_total} fits complete", flush=True)

    out_df = pd.DataFrame(results)
    id_cols = ["genotype", "titrant_name"]
    other_cols = [c for c in out_df.columns if c not in id_cols]
    out_df = out_df[id_cols + other_cols].copy()
    out_df.columns = [c.replace("|", "_") for c in out_df.columns]
    return out_df


def main():
    """CLI entry point for fitting cat_response models to theta data."""

    parser = argparse.ArgumentParser(
        prog="tfs-fit-response",
        description=fit_response.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV with columns: genotype, titrant_name, titrant_conc, "
             "and the theta / sigma columns.",
    )
    parser.add_argument(
        "--theta_col",
        type=str,
        default=None,
        help="Column name for theta values. Auto-detected if omitted "
             "('median' if present, else 'point_est').",
    )
    parser.add_argument(
        "--sigma_col",
        type=str,
        default=None,
        help="Column name for per-observation sigma.  If omitted, computed "
             "as (upper_std - lower_std) / 2.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Models to fit (default: flat repressor inducer hill_repressor "
             "hill_inducer).  Valid choices: " + " ".join(MODEL_LIBRARY),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fit_response.csv",
        help="Output CSV path (default: fit_response.csv).",
    )

    args = parser.parse_args()

    print(f"Reading {args.input}...", flush=True)
    df = pd.read_csv(args.input)

    result_df = fit_response(
        df,
        theta_col=args.theta_col,
        sigma_col=args.sigma_col,
        models_to_run=args.models,
        workers=args.workers,
    )

    result_df.to_csv(args.out, index=False)
    print(f"Wrote {len(result_df)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
