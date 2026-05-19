"""
CLI for fitting cat_response models to theta-vs-titrant data per genotype.
"""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from tfscreen.analysis.cat_response.cat_fit import cat_fit
from tfscreen.models.generic import MODEL_LIBRARY
from tfscreen.util.cli.generalized_main import generalized_main

DEFAULT_MODELS = ["flat", "repressor", "inducer", "hill_repressor", "hill_inducer"]


def _fit_one(args):
    """Worker: run cat_fit for one (genotype, titrant_name) pair."""
    genotype, titrant_name, x, y, y_std, models_to_run = args
    flat_out, _ = cat_fit(x, y, y_std, models_to_run=models_to_run)
    flat_out["genotype"] = genotype
    flat_out["titrant_name"] = titrant_name
    return flat_out


def cat_response(theta_file,
                 out_prefix="tfs_cat_response",
                 theta_col="median",
                 sigma_col=None,
                 models=None,
                 workers=1):
    """
    Classify each genotype's theta-vs-titrant curve using categorical response models.

    Reads the CSV output of tfs-predict-theta and fits one or more response models
    to every (genotype, titrant_name) group. For each group the best-fitting model
    is selected by AIC weight. Results include best_model, AIC weights, and
    parameter estimates for every fitted model. Writes one row per
    (genotype, titrant_name) pair to {out_prefix}.csv.

    Parameters
    ----------
    theta_file : str
        Path to the CSV file produced by tfs-predict-theta. Must contain columns:
        genotype, titrant_name, titrant_conc, and the theta and sigma columns.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_cat_response'.
    theta_col : str, optional
        Name of the column holding theta point estimates passed to the fitter.
        Default 'median'.
    sigma_col : str or None, optional
        Name of the column holding per-row theta uncertainty (standard deviation).
        If None, sigma is computed as (upper_std - lower_std) / 2, which requires
        upper_std and lower_std columns to be present.
    models : list of str or None, optional
        Response models to fit. Defaults to all five: flat, repressor, inducer,
        hill_repressor, hill_inducer.
    workers : int, optional
        Number of parallel worker processes (default 1).
    """
    if models is None:
        models = DEFAULT_MODELS

    bad = [m for m in models if m not in MODEL_LIBRARY]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Valid: {list(MODEL_LIBRARY)}")

    print(f"Reading {theta_file}...", flush=True)
    df = pd.read_csv(theta_file)

    if sigma_col is None:
        if "upper_std" in df.columns and "lower_std" in df.columns:
            df = df.copy()
            df["_sigma"] = (df["upper_std"] - df["lower_std"]) / 2
            sigma_col = "_sigma"
        else:
            raise ValueError(
                "No sigma_col specified and df lacks upper_std/lower_std columns. "
                "Provide --sigma_col or ensure upper_std and lower_std are present."
            )

    work_items = []
    for (genotype, titrant_name), group in df.groupby(["genotype", "titrant_name"],
                                                       sort=False):
        x = group["titrant_conc"].to_numpy(dtype=float)
        y = group[theta_col].to_numpy(dtype=float)
        y_std = group[sigma_col].to_numpy(dtype=float)
        work_items.append((genotype, titrant_name, x, y, y_std, models))

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

    out_file = f"{out_prefix}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Wrote {len(out_df)} rows to {out_file}", flush=True)


def main():
    generalized_main(cat_response,
                     manual_arg_types={"sigma_col": str,
                                       "models": str},
                     manual_arg_nargs={"models": "+"})


if __name__ == "__main__":
    main()
