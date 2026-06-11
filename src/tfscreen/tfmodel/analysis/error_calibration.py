"""
Error calibration diagnostics for posterior distributions.

Theory
------
A well-calibrated posterior assigns the correct probability to every interval:
if the model says the 90 % credible interval covers the true value 90 % of the
time, it should.  The central diagnostic tool is the **Probability Integral
Transform (PIT)**: for a single parameter with posterior CDF F and true value
θ*, the PIT is F(θ*).  For a perfectly calibrated model, PIT values drawn
across many parameters or many independent datasets are Uniform(0, 1).

This module supports two calibration workflows that share the same core:

1. **Single-run calibration** (``calibration_summary``): one inference run on
   simulated data with known ground truth.  True values come from the fixed
   simulation; posterior uncertainty comes from the marginal quantile CSVs
   written by ``tfs-param-quantiles``.  The PIT for each genotype/parameter
   is interpolated from the stored quantiles.

2. **Multi-run SBC** (``summarize_sbc``, Simulation-Based Calibration per
   Talts et al. 2018): many independent inference runs, each with a dataset
   simulated from a fresh prior draw.  The PIT for each parameter element is
   computed from the full posterior sample array.  Pooling PIT values across
   runs and testing uniformity validates that the inference algorithm correctly
   implements the generative model's prior.

Functional layers
-----------------
- **Core**   : ``pit_from_samples``, ``pit_from_quantiles``
- **Stats**  : ``calibration_curve``, ``pit_uniformity_test``
- **Plots**  : ``plot_pit_histogram``, ``plot_calibration_curve``
- **Single-run** : ``calibration_summary``
- **SBC**    : ``compute_sbc_ranks``, ``summarize_sbc``

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


# ============================================================
# Part 1 — Core PIT computation
# ============================================================

def pit_from_samples(true_vals, posterior_samples):
    """Compute PIT values from raw posterior samples.

    For each element the PIT is the fraction of posterior samples strictly
    less than the true value — equivalently, the posterior rank of the true
    value.  For a perfectly calibrated model these values are Uniform(0, 1).

    Parameters
    ----------
    true_vals : array-like, shape (*D)
        True (ground-truth) parameter values.  NaN entries propagate as NaN
        in the output.
    posterior_samples : array-like, shape (S, *D)
        S posterior samples for each element of *D*.

    Returns
    -------
    ndarray, shape (*D)
        PIT values in [0, 1], NaN where ``true_vals`` is NaN.
    """
    true_vals = np.asarray(true_vals, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)

    pit = np.mean(posterior_samples < true_vals, axis=0).astype(float)

    nan_mask = np.isnan(true_vals)
    if np.any(nan_mask):
        pit = np.where(nan_mask, np.nan, pit)

    return pit


def pit_from_quantiles(true_vals, quantile_matrix, quantile_levels):
    """Interpolate PIT values from stored posterior quantiles.

    For each observation i the PIT is estimated by linear interpolation of the
    posterior CDF at ``true_vals[i]``.  Values below the lowest stored
    quantile receive PIT = 0; values above the highest receive PIT = 1.

    Parameters
    ----------
    true_vals : array-like, shape (N,)
        True (ground-truth) values, one per observation.
    quantile_matrix : array-like, shape (N, Q)
        Posterior quantile values; row i contains the Q quantile estimates for
        observation i.  Columns must be in ascending order of ``quantile_levels``.
    quantile_levels : array-like, shape (Q,)
        Probability levels corresponding to the columns of ``quantile_matrix``,
        e.g. ``[0.025, 0.25, 0.50, 0.75, 0.975]``.  Must be sorted ascending.

    Returns
    -------
    ndarray, shape (N,)
        Interpolated PIT values in [0, 1].  NaN where ``true_vals[i]`` is NaN
        or any quantile for row i is NaN.

    Notes
    -----
    Accuracy depends on the density of stored quantile levels; coarse grids
    (e.g. only 2.5 % and 97.5 %) produce step-function PIT estimates.
    """
    true_vals = np.asarray(true_vals, dtype=float)
    quantile_matrix = np.asarray(quantile_matrix, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    N = len(true_vals)
    pit = np.full(N, np.nan)

    for i in range(N):
        tv = true_vals[i]
        qv = quantile_matrix[i]
        if np.isnan(tv) or np.any(np.isnan(qv)):
            continue
        # np.interp: xp must be increasing; quantile values are non-decreasing
        pit[i] = np.interp(tv, qv, quantile_levels, left=0.0, right=1.0)

    return pit


# ============================================================
# Part 2 — Calibration statistics
# ============================================================

_DEFAULT_CALIBRATION_LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def calibration_curve(pit_vals, levels=None):
    """Compute empirical coverage at multiple nominal credible-interval levels.

    For each nominal level α the empirical coverage is the fraction of PIT
    values that fall in the symmetric interval [(1−α)/2, (1+α)/2].  For a
    perfectly calibrated model the empirical coverage equals α at every level,
    so plotting empirical vs. nominal should lie on the diagonal.

    Parameters
    ----------
    pit_vals : array-like
        PIT values in [0, 1].  Non-finite values are ignored.
    levels : array-like, optional
        Nominal coverage levels to evaluate, e.g. ``[0.5, 0.9, 0.95]``.
        Defaults to ``[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]``.

    Returns
    -------
    dict
        Mapping ``{nominal_level (float): empirical_coverage (float)}``.
    """
    if levels is None:
        levels = _DEFAULT_CALIBRATION_LEVELS
    pit = np.asarray(pit_vals, dtype=float)
    pit = pit[np.isfinite(pit)]

    result = {}
    for alpha in levels:
        lower = (1.0 - float(alpha)) / 2.0
        upper = 1.0 - lower
        empirical = float(np.mean((pit >= lower) & (pit <= upper)))
        result[float(alpha)] = empirical
    return result


def pit_uniformity_test(pit_vals):
    """One-sample KS test of PIT values against Uniform(0, 1).

    Parameters
    ----------
    pit_vals : array-like
        PIT values.  Non-finite values are ignored.

    Returns
    -------
    dict
        ``ks_stat`` (float), ``ks_pval`` (float), ``mean_pit`` (float),
        ``n_vals`` (int).  All numeric fields are NaN when no finite values
        are present.
    """
    pit = np.asarray(pit_vals, dtype=float)
    pit = pit[np.isfinite(pit)]
    n = int(len(pit))
    if n == 0:
        return {"ks_stat": np.nan, "ks_pval": np.nan, "mean_pit": np.nan, "n_vals": 0}
    mean_pit = float(np.mean(pit))
    ks_stat, ks_pval = ks_1samp(pit, uniform(0, 1).cdf)
    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "mean_pit": mean_pit,
        "n_vals": n,
    }


# ============================================================
# Part 3 — Plotting
# ============================================================

def plot_pit_histogram(pit_vals, ax=None, title=None, n_bins=10):
    """Plot a PIT histogram with a uniform-density reference line.

    Parameters
    ----------
    pit_vals : array-like
        PIT values.  Non-finite values are silently dropped.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    title : str, optional
        Axes title.
    n_bins : int
        Number of histogram bins over [0, 1] (default 10).

    Returns
    -------
    matplotlib.axes.Axes
    """
    pit = np.asarray(pit_vals, dtype=float)
    pit = pit[np.isfinite(pit)]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
            color="steelblue", alpha=0.75, edgecolor="white")
    ax.axhline(1.0, color="firebrick", lw=1.5, ls="--", label="Uniform")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_calibration_curve(calibration_dict, ax=None, label=None):
    """Plot empirical vs. nominal coverage (calibration curve).

    A perfectly calibrated model lies on the diagonal.  Points above the
    diagonal indicate underconfidence (CIs too wide); points below indicate
    overconfidence (CIs too narrow).

    Parameters
    ----------
    calibration_dict : dict
        ``{nominal_level: empirical_coverage}`` as returned by
        ``calibration_curve``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    label : str, optional
        Legend label for the model curve.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    if not calibration_dict:
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        return ax

    nominals = np.array(sorted(calibration_dict.keys()))
    empiricals = np.array([calibration_dict[k] for k in nominals])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    ax.plot(nominals, empiricals, "o-", color="steelblue",
            label=label or "Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.legend(fontsize=8)
    return ax


# ============================================================
# Part 4 — Single-run calibration summary
# ============================================================

def calibration_summary(true_vals, quantile_matrix, quantile_levels,
                        out_prefix, label=""):
    """Compute and write calibration diagnostics for a single inference run.

    Computes PIT values via interpolation from stored quantiles, then writes
    a PIT CSV, a calibration-curve CSV, and a two-panel PDF.

    **CSV files are the authoritative outputs.** The PDF is a human-readable
    summary; the CSVs are the record of truth for downstream analysis.

    Output files
    ------------
    ``{out_prefix}_pit.csv``
        Columns: ``true_val``, ``pit``.
    ``{out_prefix}_calibration_curve.csv``
        Columns: ``nominal``, ``empirical``.
    ``{out_prefix}_calibration.pdf``
        Two panels: PIT histogram (left) and calibration curve (right).

    Parameters
    ----------
    true_vals : array-like, shape (N,)
        Known ground-truth values from simulation.
    quantile_matrix : array-like, shape (N, Q)
        Posterior quantile values; row i is quantile estimates for observation
        i at levels ``quantile_levels``.
    quantile_levels : array-like, shape (Q,)
        Probability levels (sorted ascending) corresponding to columns of
        ``quantile_matrix``.
    out_prefix : str
        File-system prefix for all output files.
    label : str, optional
        Human-readable label used in plot titles.

    Returns
    -------
    dict
        ``{"n_vals": int, "ks_stat": float, "ks_pval": float,
           "mean_pit": float, "calibration_curve": {nominal: empirical, ...}}``
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    true_vals = np.asarray(true_vals, dtype=float)
    pit_vals = pit_from_quantiles(true_vals, quantile_matrix, quantile_levels)
    cal_curve = calibration_curve(pit_vals)
    uniformity = pit_uniformity_test(pit_vals)

    # --- PIT CSV ---
    pit_csv = f"{out_prefix}_pit.csv"
    try:
        pd.DataFrame({"true_val": true_vals, "pit": pit_vals}).to_csv(pit_csv, index=False)
        print(f"Wrote PIT values to {pit_csv}")
    except Exception as exc:
        warnings.warn(f"Could not write PIT CSV to {pit_csv}: {exc}")

    # --- Calibration-curve CSV ---
    cal_csv = f"{out_prefix}_calibration_curve.csv"
    try:
        pd.DataFrame({
            "nominal": list(cal_curve.keys()),
            "empirical": list(cal_curve.values()),
        }).to_csv(cal_csv, index=False)
        print(f"Wrote calibration curve to {cal_csv}")
    except Exception as exc:
        warnings.warn(f"Could not write calibration curve CSV to {cal_csv}: {exc}")

    # --- Two-panel PDF ---
    pdf_path = f"{out_prefix}_calibration.pdf"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        hist_title = f"PIT histogram — {label}" if label else "PIT histogram"
        curve_title = f"Calibration curve — {label}" if label else "Calibration curve"
        plot_pit_histogram(pit_vals, ax=axes[0], title=hist_title)
        plot_calibration_curve(cal_curve, ax=axes[1])
        axes[1].set_title(curve_title)
        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote calibration plot to {pdf_path}")
    except Exception as exc:
        warnings.warn(f"Could not write calibration PDF to {pdf_path}: {exc}")

    return {
        "n_vals": uniformity["n_vals"],
        "ks_stat": uniformity["ks_stat"],
        "ks_pval": uniformity["ks_pval"],
        "mean_pit": uniformity["mean_pit"],
        "calibration_curve": cal_curve,
    }


# ============================================================
# Part 5 — Multi-run SBC (Simulation-Based Calibration)
# ============================================================

def _load_h5_params(path: str) -> Dict[str, np.ndarray]:
    """Load all dataset arrays from an HDF5 file."""
    out = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            out[key] = hf[key][:]
    return out


def compute_sbc_ranks(ground_truth_path: str,
                      posterior_path: str) -> Dict[str, np.ndarray]:
    """Compute the posterior rank (PIT) of each ground-truth parameter value.

    For each parameter present in both files the rank is the fraction of
    posterior samples strictly less than the ground-truth value::

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
        Mapping from parameter name to a 1-D rank array (one value per scalar
        element of the parameter).  Only parameters present in both files are
        included.
    """
    gt_params = _load_h5_params(ground_truth_path)
    post_params = _load_h5_params(posterior_path)

    ranks = {}
    for name, gt_val in gt_params.items():
        if name not in post_params:
            continue

        gt = gt_val[0] if gt_val.ndim >= 1 else gt_val
        post = post_params[name]

        if post.ndim < 1 or post.shape[0] == 0:
            continue

        try:
            gt_f = np.asarray(gt, dtype=float)
            post_f = np.asarray(post, dtype=float)
        except (TypeError, ValueError):
            continue

        ranks[name] = pit_from_samples(gt_f, post_f).ravel()

    return ranks


def _find_pairs(sbc_dir: str) -> List[Tuple[str, str, Optional[str]]]:
    """Find (run_id, ground_truth_path, posterior_path_or_None) triples.

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


_SBC_HIST_BINS = 10


def _plot_sbc_rank_histograms(all_ranks: Dict[str, List[np.ndarray]],
                               out_prefix: str) -> None:
    """Write a PDF with one PIT-histogram panel per parameter."""
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

        plot_pit_histogram(pooled, ax=ax, n_bins=_SBC_HIST_BINS)

        title = param if len(param) <= 30 else f"…{param[-27:]}"
        _, ks_pval = ks_1samp(pooled, uniform(0, 1).cdf) if len(pooled) > 0 else (np.nan, np.nan)
        ax.set_title(f"{title}\n(n={len(pooled)}, p={ks_pval:.2g})", fontsize=9)

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


def summarize_sbc(sbc_dir: str, out_prefix: Optional[str] = None) -> pd.DataFrame:
    """Scan *sbc_dir* for SBC run pairs and compute rank-uniformity statistics.

    For each ``*_ground_truth.h5`` / ``*_posterior.h5`` pair found, compute
    the posterior rank (PIT) of every ground-truth parameter value.  Pool
    ranks across runs and write three output files:

    ``{out_prefix}_sbc_summary.csv``
        One row per parameter with mean rank, KS statistic, and p-value.
        Mean rank near 0.5 and high p-value indicate good calibration.

    ``{out_prefix}_sbc_ranks.csv``
        Long-form table of every individual rank value.
        Columns: ``run_id``, ``param``, ``element_idx``, ``rank``.

    ``{out_prefix}_rank_hist.pdf``
        One-panel-per-parameter PIT histogram.  A flat histogram indicates
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

    summary_rows = []
    for param in sorted(all_ranks):
        pooled = np.concatenate(all_ranks[param])
        uniformity = pit_uniformity_test(pooled)
        n_runs = len(all_ranks[param])
        summary_rows.append({
            "param": param,
            "n_runs": n_runs,
            "n_ranks": uniformity["n_vals"],
            "mean_rank": uniformity["mean_pit"],
            "expected_rank": 0.5,
            "ks_stat": uniformity["ks_stat"],
            "ks_pval": uniformity["ks_pval"],
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = f"{out_prefix}_sbc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote per-parameter summary → {summary_csv}", flush=True)

    if rank_rows:
        ranks_csv = f"{out_prefix}_sbc_ranks.csv"
        ranks_df = pd.DataFrame(rank_rows)
        ranks_df.to_csv(ranks_csv, index=False)
        print(f"Wrote all rank values → {ranks_csv}", flush=True)

    _plot_sbc_rank_histograms(all_ranks, out_prefix)

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
