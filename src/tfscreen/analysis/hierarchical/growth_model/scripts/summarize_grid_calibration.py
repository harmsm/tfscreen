"""
Summarize pre-fit calibration results across a model grid.

Scans every immediate subdirectory of a grid directory created by
``tfs-setup-grid-calibration`` for a ``*_calib_stats.json`` file written by
``tfs-prefit-calibration``.  For each completed run it reads the component
settings from ``run_config.yaml``, computes an AIC-based model comparison, and
writes a ranked summary CSV.

AIC formula (Gaussian log-likelihood at the MLE residual variance):

    AIC = 2·k + n·(1 + ln(2π·RMSE²))

where k = n_params and n = n_obs.

AIC weight:

    Δᵢ = AICᵢ − min(AIC)
    wᵢ = exp(−Δᵢ/2) / Σ exp(−Δⱼ/2)
"""

import glob
import json
import math
import os

import numpy as np
import pandas as pd
import yaml

from tfscreen.util.cli import generalized_main


_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# AIC helpers
# ---------------------------------------------------------------------------

def _compute_aic(n_params, n_obs, rmse):
    """Return AIC under a Gaussian likelihood MLE, or NaN if inputs are invalid."""
    if n_params is None or n_obs is None or rmse is None:
        return float("nan")
    try:
        n_params = float(n_params)
        n_obs = float(n_obs)
        rmse = float(rmse)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(n_params) and math.isfinite(n_obs)
            and math.isfinite(rmse)):
        return float("nan")
    if n_obs <= 0 or rmse <= 0:
        return float("nan")
    return 2.0 * n_params + n_obs * (1.0 + math.log(_TWO_PI * rmse ** 2))


def _compute_aic_weights(aic_values):
    """Return AIC weights as a numpy array (NaN where AIC is non-finite)."""
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
# Subdirectory helpers
# ---------------------------------------------------------------------------

def _find_stats_json(subdir):
    """Return the path to the most recent ``*_calib_stats.json`` in *subdir*,
    or ``None`` if none exists."""
    hits = glob.glob(os.path.join(subdir, "*_calib_stats.json"))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _read_components(subdir):
    """Read the three grid-axis component names from ``run_config.yaml``.

    Returns ``(condition_growth, growth_transition, theta_rescale)``;
    any missing key is returned as ``None``.
    """
    yaml_path = os.path.join(subdir, "run_config.yaml")
    if not os.path.exists(yaml_path):
        return None, None, None
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    components = cfg.get("components") or {}
    return (
        components.get("condition_growth"),
        components.get("growth_transition"),
        components.get("theta_rescale"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def summarize_grid_calibration(grid_dir, out_prefix=None):
    """
    Summarise pre-fit calibration results across a model grid.

    Scans every immediate subdirectory of *grid_dir* for a
    ``*_calib_stats.json`` file, reads the component choices from each
    subdirectory's ``run_config.yaml``, computes AIC and AIC weights, and
    writes a summary CSV sorted by AIC weight (highest first).

    Parameters
    ----------
    grid_dir : str
        Path to the grid root directory created by ``tfs-setup-grid-calibration``.
    out_prefix : str or None, optional
        Prefix for the output CSV file; ``{out_prefix}.csv`` is written.
        When ``None`` the file is written to ``{grid_dir}/grid_summary.csv``.

    Returns
    -------
    pd.DataFrame
        Summary table, one row per completed run, sorted by ``aic_weight``
        descending.  An empty DataFrame is returned if no completed runs are
        found.
    """
    grid_dir = os.path.abspath(grid_dir)
    if not os.path.isdir(grid_dir):
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")

    if out_prefix is None:
        out_prefix = os.path.join(grid_dir, "grid_summary")

    # Scan immediate subdirectories.
    entries = sorted(
        d for d in os.listdir(grid_dir)
        if os.path.isdir(os.path.join(grid_dir, d))
    )

    rows = []
    for name in entries:
        subdir = os.path.join(grid_dir, name)
        stats_path = _find_stats_json(subdir)
        if stats_path is None:
            continue

        with open(stats_path) as fh:
            stats = json.load(fh)

        cg, gt, tr = _read_components(subdir)

        row = {
            "subdir": name,
            "condition_growth": cg,
            "growth_transition": gt,
            "theta_rescale": tr,
        }
        row.update(stats)
        rows.append(row)

    if not rows:
        print(f"No completed calibration runs found in {grid_dir}/", flush=True)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["aic"] = [
        _compute_aic(row.get("n_params"), row.get("n_obs"), row.get("rmse"))
        for _, row in df.iterrows()
    ]
    df["aic_weight"] = _compute_aic_weights(df["aic"].to_numpy())

    df = df.sort_values("aic_weight", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary written to {csv_path} ({len(df)} run(s))", flush=True)

    return df


def main():
    generalized_main(
        summarize_grid_calibration,
        manual_arg_types={"grid_dir": str, "out_prefix": str},
    )


if __name__ == "__main__":
    main()
