"""
CLI for fitting cat_response models to theta-vs-titrant data per genotype.
"""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from tfscreen.analysis.cat_response.cat_fit import cat_fit
from tfscreen.mle.curve_models import MODEL_LIBRARY
from tfscreen.util import resolve_workers
from tfscreen.util.cli.generalized_main import generalized_main

# Number of (genotype, titrant_name) pairs bundled into each worker task. Larger
# chunks amortize the per-task pickle/IPC overhead of ProcessPoolExecutor, which
# matters a lot when there are hundreds of thousands of pairs.
_CHUNK_SIZE = 200


def _fit_one(genotype, titrant_name, x, y, y_std, models_to_run):
    """Run cat_fit for one (genotype, titrant_name) pair."""
    flat_out, _ = cat_fit(x, y, y_std, models_to_run=models_to_run)
    flat_out["genotype"] = genotype
    flat_out["titrant_name"] = titrant_name
    return flat_out


def _fit_chunk(chunk):
    """Worker: run cat_fit for a list of work items, preserving order."""
    return [_fit_one(*item) for item in chunk]


def _iter_chunks(work_items, chunk_size):
    """Yield successive length-``chunk_size`` slices of ``work_items``."""
    for start in range(0, len(work_items), chunk_size):
        yield work_items[start:start + chunk_size]


def cat_response(theta_file,
                 out_prefix="tfs_cat_response",
                 theta_col=None,
                 sigma_col=None,
                 models=None,
                 num_workers=-1):
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
    theta_col : str or None, optional
        Name of the column holding theta values passed to the fitter.  If
        ``None`` (default), the column is auto-detected: ``median`` is used if
        present, then ``point_est``.  Pass explicitly to override.
    sigma_col : str or None, optional
        Name of the column holding per-row theta uncertainty (standard deviation).
        If None, sigma is computed as (upper_std - lower_std) / 2, which requires
        upper_std and lower_std columns to be present.
    models : list of str or None, optional
        Response models to fit. Defaults to all models in MODEL_LIBRARY.
    num_workers : int, optional
        Number of parallel worker processes. ``1`` runs serially in-process;
        ``-1`` (the default) uses ``os.cpu_count() - 1``; ``N`` uses ``N``
        processes. The per-pair fits are embarrassingly parallel, so this scales
        nearly linearly on large libraries.
    """
    if models is None:
        models = list(MODEL_LIBRARY.keys())

    bad = [m for m in models if m not in MODEL_LIBRARY]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Valid: {list(MODEL_LIBRARY)}")

    print(f"Reading {theta_file}...", flush=True)
    df = pd.read_csv(theta_file)

    if theta_col is None:
        if "q0.5" in df.columns:
            theta_col = "q0.5"
        elif "point_est" in df.columns:
            theta_col = "point_est"
        else:
            raise ValueError(
                "No theta column found. Expected 'q0.5' (posterior median) or "
                "'point_est' (MAP). Use --theta_col to specify a column explicitly."
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
        work_items.append((genotype, titrant_name, x, y, y_std, models))

    n_total = len(work_items)
    workers = resolve_workers(num_workers)
    chunks = list(_iter_chunks(work_items, _CHUNK_SIZE))

    print(f"  Fitting {n_total} (genotype, titrant_name) pairs "
          f"with {workers} worker(s)...", flush=True)

    results = []
    n_done = 0

    def _report(n_done):
        if n_done % 5000 < _CHUNK_SIZE or n_done == n_total:
            print(f"  {n_done}/{n_total} fits complete", flush=True)

    if workers == 1:
        # Serial fast-path: run in-process, no pickling/IPC overhead.
        for chunk in chunks:
            results.extend(_fit_chunk(chunk))
            n_done += len(chunk)
            _report(n_done)
    else:
        # executor.map preserves input order, so results stay aligned with
        # work_items without an explicit index map.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk_result in executor.map(_fit_chunk, chunks):
                results.extend(chunk_result)
                n_done += len(chunk_result)
                _report(n_done)

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
                     manual_arg_types={"theta_col": str,
                                       "sigma_col": str,
                                       "models": str},
                     manual_arg_nargs={"models": "+"})


if __name__ == "__main__":
    main()
