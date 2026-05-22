#!/usr/bin/env python3
"""
summarize_grid.py — Summarize model grid results.

Scans every immediate subdirectory of a grid directory for combo.json, then for
each completed run:
  - generates a two-panel diagnostic PDF (growth and theta correlation plots)
  - computes fit statistics via stats_test_suite
  - computes approximate AIC (Gaussian MLE) using k from tfs_configure_priors.csv

Writes a summary CSV to <out_prefix>.csv with one row per run.

Usage:
    python scripts/summarize_grid.py output/   [--out_prefix results/grid_stats]

AIC formula (Gaussian log-likelihood at the MLE residual variance):
    AIC = 2·k + n·(1 + ln(2π·RMSE²))
where k ≈ row count of tfs_configure_priors.csv (proxy for model complexity)
and n = number of observations used in the RMSE.

AIC weight:
    Δᵢ = AICᵢ − min(AIC)
    wᵢ = exp(−Δᵢ/2) / Σ exp(−Δⱼ/2)

AIC weights are computed separately for growth and theta fits.
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tfscreen.fitting.stats_test_suite import stats_test_suite


# ---------------------------------------------------------------------------
# AIC helpers
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi


def _compute_aic(n_params, n_obs, rmse):
    """AIC under Gaussian MLE; returns NaN if any input is missing or invalid."""
    if n_params is None or n_obs is None or rmse is None:
        return float("nan")
    try:
        n_params, n_obs, rmse = float(n_params), float(n_obs), float(rmse)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(n_params) and math.isfinite(n_obs) and math.isfinite(rmse)):
        return float("nan")
    if n_obs <= 0 or rmse <= 0:
        return float("nan")
    return 2.0 * n_params + n_obs * (1.0 + math.log(_TWO_PI * rmse ** 2))


def _compute_aic_weights(aic_values):
    """AIC weights as a numpy array; NaN where AIC is non-finite."""
    aic = np.asarray(aic_values, dtype=float)
    finite = np.isfinite(aic)
    weights = np.full(len(aic), float("nan"))
    if finite.sum() == 0:
        return weights
    delta = aic[finite] - aic[finite].min()
    w = np.exp(-0.5 * delta)
    w /= w.sum()
    weights[finite] = w
    return weights


# ---------------------------------------------------------------------------
# Run directory discovery
# ---------------------------------------------------------------------------

def find_run_dirs(grid_dir):
    """Return sorted list of subdirectories that contain combo.json."""
    return sorted(d for d in Path(grid_dir).iterdir()
                  if d.is_dir() and (d / "combo.json").exists())


def load_combo(run_dir):
    with open(run_dir / "combo.json") as f:
        return json.load(f)


def count_params(run_dir):
    """Row count of tfs_configure_priors.csv as a proxy for k, or None."""
    priors = run_dir / "tfs_configure_priors.csv"
    if not priors.exists():
        return None
    try:
        return len(pd.read_csv(priors))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path, label):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  WARNING: could not read {label}: {e}", file=sys.stderr)
        return None


def load_growth(run_dir):
    return _load_csv(run_dir / "tfs_growth_pred.csv", "tfs_growth_pred.csv")


def load_theta(run_dir):
    return _load_csv(run_dir / "tfs_theta_pred.csv", "tfs_theta_pred.csv")


def load_binding(run_dir, combo):
    raw = combo.get("BINDING_CSV_FILE", "")
    if not raw:
        return None
    return _load_csv(run_dir / raw, f"binding CSV ({raw})")


def merge_theta_obs(theta_df, binding_df):
    """Inner-join theta predictions with theta_obs (and theta_std if present)."""
    if theta_df is None or binding_df is None:
        return None
    merge_keys = ["genotype", "titrant_name", "titrant_conc"]
    for df, label in [(theta_df, "theta_pred"), (binding_df, "binding")]:
        missing = [k for k in merge_keys if k not in df.columns]
        if missing:
            print(f"  WARNING: merge keys missing from {label}: {missing}", file=sys.stderr)
            return None
    if "theta_obs" not in binding_df.columns:
        print("  WARNING: 'theta_obs' column not in binding CSV", file=sys.stderr)
        return None
    obs_cols = [c for c in ["theta_obs", "theta_std"] if c in binding_df.columns]
    return theta_df.merge(binding_df[merge_keys + obs_cols], on=merge_keys, how="inner")


# ---------------------------------------------------------------------------
# Fit statistics
# ---------------------------------------------------------------------------

def _sym_std(df, upper_col, lower_col):
    """Symmetric std: half-width of the 16th–84th percentile interval."""
    return (df[upper_col] - df[lower_col]).values / 2.0


def compute_stats(df, pred_col, obs_col, upper_col="upper_std", lower_col="lower_std"):
    """
    Run stats_test_suite on a prediction dataframe.

    pred_col   — posterior median (our 'estimate')
    obs_col    — observed value   (our 'truth')
    Returns (stats_dict, n_obs) or (None, 0) on failure.
    """
    if df is None:
        return None, 0
    required = [pred_col, obs_col, upper_col, lower_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: columns missing for stats: {missing}", file=sys.stderr)
        return None, 0
    subset = df[[pred_col, obs_col, upper_col, lower_col]].dropna()
    if len(subset) < 3:
        return None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats_test_suite(
            subset[pred_col].values,
            _sym_std(subset, upper_col, lower_col),
            subset[obs_col].values,
        )
    return result, len(subset)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _hexbin_panel(ax, obs, pred, title, xlabel, ylabel, trim_quantile=0.001):
    """Single scatter correlation panel with 1:1 dashed line and equal axes.

    Removes the top and bottom trim_quantile fraction of points (applied to
    the joint distribution) before plotting to suppress extreme outliers.
    """
    finite = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[finite], pred[finite]
    if len(obs) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return

    # Outlier removal: keep points within [trim_quantile, 1-trim_quantile]
    # on both axes independently, then take the intersection.
    q_lo_obs,  q_hi_obs  = np.quantile(obs,  [trim_quantile, 1 - trim_quantile])
    q_lo_pred, q_hi_pred = np.quantile(pred, [trim_quantile, 1 - trim_quantile])
    keep = (obs  >= q_lo_obs)  & (obs  <= q_hi_obs) \
         & (pred >= q_lo_pred) & (pred <= q_hi_pred)
    obs, pred = obs[keep], pred[keep]
    if len(obs) == 0:
        ax.text(0.5, 0.5, "no data after trimming",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return

    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    margin = (hi - lo) * 0.05
    lo -= margin
    hi += margin

    ax.scatter(obs, pred, facecolor="none", edgecolor="black", linewidths=0.5,
               s=10, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _theta_curves_panel(ax, theta_df, binding_df):
    """Plot predicted theta curves per genotype overlaid with binding observations.

    Predictions are filtered to only concentrations present in the binding data.
    The x-axis is log-scaled; any titrant_conc == 0 is replaced with
    min(positive concentrations) / 100 before plotting.

    For each (genotype, titrant_name) group:
      - fill_between lower_95 / upper_95  (alpha=0.2, outer CI)
      - fill_between lower_std / upper_std (alpha=0.5, inner CI)
      - median line
      - observed scatter points (no connecting lines)
    """
    merge_keys = ["genotype", "titrant_name", "titrant_conc"]
    pred_needed = ["median", "lower_95", "upper_95", "lower_std", "upper_std"]

    has_pred = (theta_df is not None
                and all(c in theta_df.columns for c in merge_keys + pred_needed))
    has_obs  = (binding_df is not None
                and "theta_obs" in binding_df.columns
                and all(c in binding_df.columns for c in merge_keys))

    if not has_pred and not has_obs:
        ax.text(0.5, 0.5, "no theta data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("Theta")
        return

    # Filter predictions to concentrations present in binding data only
    if has_pred and has_obs:
        binding_concs = binding_df[merge_keys].drop_duplicates()
        theta_df = theta_df.merge(binding_concs, on=merge_keys, how="inner")
        if len(theta_df) == 0:
            has_pred = False

    # Determine zero-replacement value from all positive concentrations
    all_concs = np.concatenate([
        df["titrant_conc"].values
        for df, flag in [(theta_df, has_pred), (binding_df, has_obs)]
        if flag
    ])
    pos = all_concs[all_concs > 0]
    zero_sub = pos.min() / 100.0 if len(pos) > 0 else 1e-6

    def _fix_zero(s):
        return s.where(s != 0.0, zero_sub)

    # Collect group keys from the binding data only (it drives what gets plotted)
    if has_obs:
        group_keys = sorted(set(zip(binding_df["genotype"], binding_df["titrant_name"])))
    else:
        group_keys = sorted(set(zip(theta_df["genotype"], theta_df["titrant_name"])))

    n = len(group_keys)
    cmap = plt.cm.get_cmap("tab20" if n <= 20 else "turbo", max(n, 1))

    for idx, (genotype, titrant_name) in enumerate(group_keys):
        color = cmap(idx)

        if has_pred:
            mask = ((theta_df["genotype"] == genotype) &
                    (theta_df["titrant_name"] == titrant_name))
            gdf = theta_df[mask].copy()
            gdf["titrant_conc"] = _fix_zero(gdf["titrant_conc"])
            gdf = gdf.sort_values("titrant_conc")
            if len(gdf):
                x = gdf["titrant_conc"].values
                ax.fill_between(x, gdf["lower_95"], gdf["upper_95"],
                                color=color, alpha=0.2)
                ax.fill_between(x, gdf["lower_std"], gdf["upper_std"],
                                color=color, alpha=0.5)
                ax.plot(x, gdf["median"], color=color, linewidth=1,
                        label=f"{genotype} / {titrant_name}")

        if has_obs:
            mask = ((binding_df["genotype"] == genotype) &
                    (binding_df["titrant_name"] == titrant_name))
            bdf = binding_df[mask].copy()
            bdf["titrant_conc"] = _fix_zero(bdf["titrant_conc"])
            if len(bdf):
                yerr = bdf["theta_std"].values if "theta_std" in bdf.columns else None
                ax.errorbar(bdf["titrant_conc"], bdf["theta_obs"],
                            yerr=yerr, fmt="o", color=color,
                            markersize=3, lw=0, elinewidth=1,
                            capsize=2, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("titrant_conc")
    ax.set_ylabel("θ")
    ax.set_title("Theta")
    if n <= 20:
        ax.legend(fontsize=6, loc="best", framealpha=0.5)


def make_plots(run_dir, growth_df, theta_df, binding_df):
    """Write diagnostics.pdf to run_dir: growth scatter + theta curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if (growth_df is not None
            and "median" in growth_df.columns
            and "ln_cfu" in growth_df.columns):
        _hexbin_panel(axes[0],
                      growth_df["ln_cfu"].values,
                      growth_df["median"].values,
                      title="Growth",
                      xlabel="ln_cfu (observed)",
                      ylabel="median (predicted)")
    else:
        axes[0].text(0.5, 0.5, "no growth data",
                     transform=axes[0].transAxes, ha="center", va="center")
        axes[0].set_title("Growth")

    _theta_curves_panel(axes[1], theta_df, binding_df)

    plt.tight_layout()
    fig.savefig(run_dir / "diagnostics.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(run_dir):
    """Process one run directory. Returns a flat dict for the summary row."""
    combo = load_combo(run_dir)

    growth_df = load_growth(run_dir)
    theta_df = load_theta(run_dir)
    binding_df = load_binding(run_dir, combo)
    theta_merged = merge_theta_obs(theta_df, binding_df)

    make_plots(run_dir, growth_df, theta_df, binding_df)

    growth_stats, growth_n = compute_stats(growth_df, "median", "ln_cfu")
    theta_stats, theta_n = compute_stats(theta_merged, "median", "theta_obs")

    k = count_params(run_dir)
    growth_aic = _compute_aic(k, growth_n, growth_stats["rmse"] if growth_stats else None)
    theta_aic = _compute_aic(k, theta_n, theta_stats["rmse"] if theta_stats else None)

    row = {"run_dir": run_dir.name}
    row.update(combo)
    row["n_params"] = k

    if growth_stats:
        for key, val in growth_stats.items():
            row[f"growth_{key}"] = val
    row["growth_aic"] = growth_aic

    if theta_stats:
        for key, val in theta_stats.items():
            row[f"theta_{key}"] = val
    row["theta_aic"] = theta_aic

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Summarize model grid: diagnostic plots, fit statistics, AIC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("grid_dir", help="Grid root directory (parent of run subdirectories)")
    ap.add_argument("--out_prefix", default=None,
                    help="Output CSV prefix (default: <grid_dir>/grid_stats)")
    args = ap.parse_args()

    grid_dir = Path(args.grid_dir).resolve()
    if not grid_dir.is_dir():
        sys.exit(f"ERROR: grid directory not found: {grid_dir}")

    out_prefix = args.out_prefix or str(grid_dir / "grid_stats")

    run_dirs = find_run_dirs(grid_dir)
    if not run_dirs:
        sys.exit(f"ERROR: no subdirectories with combo.json found in {grid_dir}")

    rows = []
    for run_dir in run_dirs:
        print(f"  {run_dir.name}")
        try:
            rows.append(process_run(run_dir))
        except Exception as e:
            print(f"  WARNING: failed to process {run_dir.name}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("ERROR: no runs were successfully processed")

    df = pd.DataFrame(rows)

    # AIC weights computed across all runs; insert each right after its AIC column
    for prefix in ("growth", "theta"):
        aic_col = f"{prefix}_aic"
        weight_col = f"{prefix}_aic_weight"
        if aic_col in df.columns:
            df[weight_col] = _compute_aic_weights(df[aic_col].to_numpy())
            # Move weight_col to immediately follow aic_col
            cols = list(df.columns)
            cols.remove(weight_col)
            cols.insert(cols.index(aic_col) + 1, weight_col)
            df = df[cols]

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary written to {csv_path} ({len(df)} run(s))")


if __name__ == "__main__":
    main()
