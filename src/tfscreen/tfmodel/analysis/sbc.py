"""
Simulation-Based Calibration (SBC) rank statistics.

Theory
------
For a correctly specified model, the posterior rank of the true parameter
value should be uniformly distributed on [0, 1].  Given N ground-truth /
posterior pairs (each drawn from the same generative model), we:

  1. For each parameter array θ of shape (*D), compute the rank of the
     ground-truth value θ_true among the S posterior samples::

         rank[...] = mean(posterior_samples < θ_true, axis=0)   ∈ [0, 1]

  2. Pool all ranks across runs and parameter elements.
  3. Test uniformity with a KS test and summarise with a rank histogram.

References
----------
Talts et al. (2018) "Validating Bayesian Inference Algorithms with
Simulation-Based Calibration". arXiv:1804.06788.
"""

import glob
import os
import warnings
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_1samp, uniform


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_h5_params(path: str) -> Dict[str, np.ndarray]:
    """Load all dataset arrays from an HDF5 file."""
    out = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            out[key] = hf[key][:]
    return out


def compute_sbc_ranks(ground_truth_path: str,
                      posterior_path: str) -> Dict[str, np.ndarray]:
    """
    Compute the posterior rank of each ground-truth parameter value.

    For each parameter present in *both* files the rank is the fraction of
    posterior samples that are strictly less than the ground-truth value::

        rank[i] = mean(posterior_samples[:, i] < gt_value[i])   ∈ [0, 1]

    Parameters
    ----------
    ground_truth_path : str
        HDF5 file produced by ``tfs-sample-prior``.  Each dataset has shape
        ``(1, *param_shape)`` (num_draws=1).
    posterior_path : str
        HDF5 file produced by ``tfs-sample-posterior``.  Each dataset has
        shape ``(S, *param_shape)`` with S posterior samples.

    Returns
    -------
    dict
        Mapping from parameter name to a 1-D rank array (one value per
        scalar element of the parameter).  Only parameters present in both
        files are included.
    """
    gt_params = _load_h5_params(ground_truth_path)
    post_params = _load_h5_params(posterior_path)

    ranks = {}
    for name, gt_val in gt_params.items():
        if name not in post_params:
            continue

        # Ground truth: shape (1, *param_shape) → squeeze to (*param_shape)
        gt = gt_val[0] if gt_val.ndim >= 1 else gt_val
        post = post_params[name]    # (S, *param_shape)

        if post.ndim < 1 or post.shape[0] == 0:
            continue

        try:
            gt_f = np.asarray(gt, dtype=float)
            post_f = np.asarray(post, dtype=float)
        except (TypeError, ValueError):
            continue   # skip non-numeric parameters

        # Broadcast: (S, *param_shape) vs (*param_shape)
        rank = np.mean(post_f < gt_f, axis=0).ravel()   # shape: (n_elements,)
        ranks[name] = rank

    return ranks


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def _find_pairs(sbc_dir: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Find (run_id, ground_truth_path, posterior_path_or_None) triples.

    Scans *sbc_dir* for files matching ``*_ground_truth.h5``.  For each,
    looks for a corresponding ``*_posterior.h5`` with the same prefix
    (replacing ``_ground_truth`` with ``_posterior``).
    """
    gt_files = sorted(glob.glob(os.path.join(sbc_dir, "*_ground_truth.h5")))
    pairs = []
    for gt_path in gt_files:
        basename = os.path.basename(gt_path)
        run_id = basename[: -len("_ground_truth.h5")]
        post_path = os.path.join(sbc_dir, f"{run_id}_posterior.h5")
        pairs.append((run_id, gt_path,
                      post_path if os.path.exists(post_path) else None))
    return pairs


# ---------------------------------------------------------------------------
# Main summary function
# ---------------------------------------------------------------------------

def summarize_sbc(sbc_dir: str, out_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Scan *sbc_dir* for SBC run pairs and compute rank-uniformity statistics.

    For each ``*_ground_truth.h5`` / ``*_posterior.h5`` pair found, compute
    the posterior rank of every ground-truth parameter value.  Pool ranks
    across runs and write three output files:

    ``{out_prefix}_sbc_summary.csv``
        One row per parameter with mean rank, KS statistic, and p-value.
        Mean rank near 0.5 and high p-value indicate good calibration.

    ``{out_prefix}_sbc_ranks.csv``
        Long-form table of every individual rank value, for custom analysis.
        Columns: ``run_id``, ``param``, ``element_idx``, ``rank``.

    ``{out_prefix}_rank_hist.pdf``
        One-panel-per-parameter rank histogram.  A flat histogram indicates
        uniform ranks (good calibration).

    Parameters
    ----------
    sbc_dir : str
        Directory containing ``*_ground_truth.h5`` and ``*_posterior.h5``
        files produced by ``tfs-sample-prior`` and ``tfs-sample-posterior``.
    out_prefix : str, optional
        Prefix for all output files.  Defaults to ``{sbc_dir}/sbc``.

    Returns
    -------
    pd.DataFrame
        Per-parameter summary table (same content as the CSV).
    """
    sbc_dir = os.path.abspath(sbc_dir)
    if not os.path.isdir(sbc_dir):
        raise FileNotFoundError(f"SBC directory not found: {sbc_dir}")

    if out_prefix is None:
        out_prefix = os.path.join(sbc_dir, "sbc")

    pairs = _find_pairs(sbc_dir)
    if not pairs:
        print(f"No *_ground_truth.h5 files found in {sbc_dir}/", flush=True)
        return pd.DataFrame()

    n_total = len(pairs)
    n_posterior = sum(1 for _, _, p in pairs if p is not None)
    print(
        f"Found {n_total} ground-truth file(s), "
        f"{n_posterior} with matching posterior.",
        flush=True,
    )
    if n_posterior == 0:
        warnings.warn(
            "No posterior files found. Run tfs-sample-posterior for each "
            "synthetic dataset before calling tfs-summarize-sbc."
        )
        return pd.DataFrame()

    # Accumulate ranks: param_name → list of rank arrays (one per run)
    all_ranks: Dict[str, List[np.ndarray]] = {}
    rank_rows = []

    for run_id, gt_path, post_path in pairs:
        if post_path is None:
            continue
        try:
            run_ranks = compute_sbc_ranks(gt_path, post_path)
        except Exception as exc:
            warnings.warn(f"Skipping run '{run_id}': {exc}")
            continue

        for param, rank_arr in run_ranks.items():
            all_ranks.setdefault(param, []).append(rank_arr)
            for idx, r in enumerate(rank_arr):
                rank_rows.append({
                    "run_id": run_id,
                    "param": param,
                    "element_idx": idx,
                    "rank": float(r),
                })

    if not all_ranks:
        warnings.warn("No matching parameters found across ground-truth and posterior files.")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # Per-parameter summary: mean rank, KS test against Uniform(0,1)
    # -----------------------------------------------------------------
    summary_rows = []
    for param in sorted(all_ranks):
        pooled = np.concatenate(all_ranks[param])
        pooled = pooled[np.isfinite(pooled)]
        n_runs = len(all_ranks[param])
        n_vals = len(pooled)
        if n_vals == 0:
            continue
        mean_rank = float(np.mean(pooled))
        ks_stat, ks_pval = ks_1samp(pooled, uniform(0, 1).cdf)
        summary_rows.append({
            "param": param,
            "n_runs": n_runs,
            "n_ranks": n_vals,
            "mean_rank": mean_rank,
            "expected_rank": 0.5,
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
        })

    summary_df = pd.DataFrame(summary_rows)

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    summary_csv = f"{out_prefix}_sbc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote per-parameter summary → {summary_csv}", flush=True)

    if rank_rows:
        ranks_csv = f"{out_prefix}_sbc_ranks.csv"
        ranks_df = pd.DataFrame(rank_rows)
        ranks_df.to_csv(ranks_csv, index=False)
        print(f"Wrote all rank values → {ranks_csv}", flush=True)
    else:
        ranks_df = pd.DataFrame()

    _plot_rank_histograms(all_ranks, out_prefix)

    print(
        f"\nSummary: {len(summary_df)} parameter(s) across "
        f"{n_posterior} run(s).",
        flush=True,
    )
    low_pval = summary_df[summary_df["ks_pval"] < 0.05] if len(summary_df) else pd.DataFrame()
    if len(low_pval):
        print(
            f"  {len(low_pval)} parameter(s) with KS p-value < 0.05 "
            f"(possible miscalibration):",
            flush=True,
        )
        for _, row in low_pval.iterrows():
            print(
                f"    {row['param']:40s}  KS={row['ks_stat']:.3f}  p={row['ks_pval']:.3g}",
                flush=True,
            )
    else:
        print("  All parameters pass KS uniformity test (p ≥ 0.05).", flush=True)

    return summary_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_HIST_BINS = 10


def _plot_rank_histograms(all_ranks: Dict[str, List[np.ndarray]],
                          out_prefix: str) -> None:
    """Write a PDF with one rank-histogram panel per parameter."""
    params = sorted(all_ranks)
    n = len(params)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for ax_idx, param in enumerate(params):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        pooled = np.concatenate(all_ranks[param])
        pooled = pooled[np.isfinite(pooled)]
        n_vals = len(pooled)

        ax.hist(pooled, bins=_HIST_BINS, range=(0, 1),
                density=True, color="steelblue", alpha=0.75, edgecolor="white")
        ax.axhline(1.0, color="firebrick", lw=1.5, ls="--", label="uniform")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior rank")
        ax.set_ylabel("Density")

        # Truncate long param names for the title
        title = param if len(param) <= 30 else f"…{param[-27:]}"
        _, ks_pval = ks_1samp(pooled, uniform(0, 1).cdf)
        ax.set_title(f"{title}\n(n={n_vals}, p={ks_pval:.2g})", fontsize=9)

    # Hide unused panels
    for ax_idx in range(n, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    pdf_path = f"{out_prefix}_rank_hist.pdf"
    try:
        fig.savefig(pdf_path)
        print(f"Wrote rank histograms → {pdf_path}", flush=True)
    except Exception as exc:
        warnings.warn(f"Could not write rank histogram PDF: {exc}")
    finally:
        plt.close(fig)
