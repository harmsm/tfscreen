import datetime
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from tfscreen.fitting.stats_test_suite import stats_test_suite
from tfscreen.plot.xy_corr import xy_corr
from tfscreen.util.cli.generalized_main import generalized_main


def _find_unique(run_dir, suffix, label, warn_missing=True):
    """Return single file in run_dir matching *suffix, or None."""
    matches = sorted(glob.glob(os.path.join(run_dir, f"*{suffix}")))
    if not matches:
        if warn_missing:
            warnings.warn(f"No {label} file (*{suffix}) found in {run_dir}")
        return None
    if len(matches) > 1:
        warnings.warn(
            f"Multiple {label} files found in {run_dir}; using {os.path.basename(matches[0])}"
        )
    return matches[0]


def _resolve_path(path, run_dir):
    """Return an existing path given a raw path and a fallback base directory."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    candidate = os.path.join(run_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path if os.path.exists(path) else None


def _read_final_loss(losses_file):
    """Return the last loss value from a losses text file.

    Supports two formats:
    - Comma-delimited  ``loss,other``  (loss is the first column)
    - Whitespace-delimited  ``step loss``  (loss is the last column)
    """
    last_val = None
    with open(losses_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(",")[0] if "," in line else line.split()[-1]
            try:
                last_val = float(token)
            except ValueError:
                pass
    if last_val is None:
        raise ValueError(f"No numeric values found in {losses_file}")
    return last_val


def _run_stats(pred_vals, obs_vals):
    """Return stats dict, or None on failure."""
    try:
        return stats_test_suite(
            param_est=np.array(pred_vals, dtype=float),
            param_real=np.array(obs_vals, dtype=float),
        )
    except Exception as exc:
        warnings.warn(f"stats_test_suite failed: {exc}")
        return None


def _json_safe(obj):
    """Recursively convert numpy scalars/NaNs to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _blank_panel(ax, message):
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray")


def summarize_fit(run_dir,
                  ground_truth_file=None,
                  out_prefix=None):
    """
    Evaluate theta prediction quality for a tfs-fit-model run directory.

    Scans run_dir for a *_config.yaml, a *_theta_pred.csv, and (optionally)
    a *_losses.txt, then computes prediction statistics for both training data
    (binding observations used during fitting) and an optional held-out test
    set.  Results are written to {out_prefix}_fit_summary.json and a two-panel
    correlation plot to {out_prefix}_theta_corr.pdf.

    Parameters
    ----------
    run_dir : str
        Directory containing model fit outputs.
    ground_truth_file : str, optional
        CSV with columns genotype, titrant_name, titrant_conc, theta_obs,
        theta_std containing known theta values for evaluating out-of-sample
        prediction.  When omitted, only training statistics are computed.
    out_prefix : str, optional
        Prefix for output files.  Defaults to {run_dir}/tfs_summarize.
    """
    run_dir = os.path.abspath(run_dir)
    if out_prefix is None:
        out_prefix = os.path.join(run_dir, "tfs_summarize")

    metadata = {
        "run_dir": run_dir,
        "ground_truth_file": ground_truth_file,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_parameters": None,
        "n_training_points": None,
        "n_test_points": None,
        "final_loss": None,
    }
    training_stats = None
    test_stats = None
    train_merged = None
    test_merged = None

    # --- Locate files in run_dir ---
    config_file = _find_unique(run_dir, "_config.yaml", "config")
    theta_pred_file = _find_unique(run_dir, "_theta_pred.csv", "theta predictions")
    losses_file = _find_unique(run_dir, "_losses.txt", "losses", warn_missing=False)

    # --- Parse YAML config once ---
    config_yaml = None
    if config_file is not None:
        try:
            with open(config_file) as fh:
                config_yaml = yaml.safe_load(fh)
        except Exception as exc:
            warnings.warn(f"Could not parse config {config_file}: {exc}")

    # --- Load binding data (training theta observations) ---
    binding_df = None
    if config_yaml is not None:
        binding_path_raw = config_yaml.get("data", {}).get("binding")
        if binding_path_raw:
            binding_path = _resolve_path(binding_path_raw, run_dir)
            if binding_path is None:
                warnings.warn(
                    f"Binding data path '{binding_path_raw}' not found "
                    f"(tried absolute and relative to {run_dir})"
                )
            else:
                try:
                    binding_df = pd.read_csv(binding_path)
                except Exception as exc:
                    warnings.warn(f"Could not load binding data from {binding_path}: {exc}")

    # --- Count parameters via guesses CSV ---
    if config_yaml is not None:
        guesses_name = config_yaml.get("guesses_file")
        if guesses_name:
            guesses_path = os.path.join(os.path.dirname(config_file), guesses_name)
            if os.path.exists(guesses_path):
                try:
                    guesses_df = pd.read_csv(guesses_path)
                    metadata["n_parameters"] = len(guesses_df)
                except Exception as exc:
                    warnings.warn(f"Could not count parameters from {guesses_path}: {exc}")

    # --- Read converged loss ---
    if losses_file is not None:
        try:
            metadata["final_loss"] = _read_final_loss(losses_file)
        except Exception as exc:
            warnings.warn(f"Could not read final loss from {losses_file}: {exc}")

    # --- Load theta predictions ---
    pred_df = None
    if theta_pred_file is not None:
        try:
            pred_df = pd.read_csv(theta_pred_file)
        except Exception as exc:
            warnings.warn(f"Could not load theta predictions from {theta_pred_file}: {exc}")

    # --- Training statistics ---
    if pred_df is not None and binding_df is not None:
        try:
            join_cols = ["genotype", "titrant_name", "titrant_conc"]
            train_pred = pred_df[pred_df["in_training_data"] == 1].copy()
            train_merged = train_pred.merge(
                binding_df[join_cols + ["theta_obs"]],
                on=join_cols,
                how="inner",
            )
            metadata["n_training_points"] = len(train_merged)
            if len(train_merged) > 0:
                training_stats = _run_stats(
                    train_merged["median"].values,
                    train_merged["theta_obs"].values,
                )
        except Exception as exc:
            warnings.warn(f"Could not compute training statistics: {exc}")

    # --- Test statistics ---
    if ground_truth_file is not None and pred_df is not None:
        try:
            gt_df = pd.read_csv(ground_truth_file)
            join_cols = ["genotype", "titrant_name", "titrant_conc"]
            test_merged = pred_df.merge(
                gt_df[join_cols + ["theta_obs"]],
                on=join_cols,
                how="inner",
            )
            metadata["n_test_points"] = len(test_merged)
            if len(test_merged) > 0:
                test_stats = _run_stats(
                    test_merged["median"].values,
                    test_merged["theta_obs"].values,
                )
        except Exception as exc:
            warnings.warn(f"Could not compute test statistics: {exc}")

    # --- Write JSON ---
    results = {"metadata": metadata, "training": training_stats, "test": test_stats}
    json_file = f"{out_prefix}_fit_summary.json"
    try:
        with open(json_file, "w") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"Wrote statistics to {json_file}")
    except Exception as exc:
        warnings.warn(f"Could not write JSON to {json_file}: {exc}")

    # --- Correlation plot ---
    pdf_file = f"{out_prefix}_theta_corr.pdf"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if train_merged is not None and len(train_merged) > 0:
            xy_corr(
                x_values=train_merged["theta_obs"].values,
                y_values=train_merged["median"].values,
                as_hexbin=True,
                ax=axes[0],
            )
            axes[0].set_xlabel("Observed θ")
            axes[0].set_ylabel("Predicted θ")
            axes[0].set_title("Training data")
        else:
            _blank_panel(axes[0], "No training data available")

        if test_merged is not None and len(test_merged) > 0:
            xy_corr(
                x_values=test_merged["theta_obs"].values,
                y_values=test_merged["median"].values,
                as_hexbin=True,
                ax=axes[1],
            )
            axes[1].set_xlabel("Observed θ")
            axes[1].set_ylabel("Predicted θ")
            axes[1].set_title("Test data")
        else:
            _blank_panel(axes[1], "No test data provided")

        fig.tight_layout()
        fig.savefig(pdf_file)
        plt.close(fig)
        print(f"Wrote correlation plot to {pdf_file}")
    except Exception as exc:
        warnings.warn(f"Could not generate correlation plot: {exc}")

    return results


def main():
    return generalized_main(
        summarize_fit,
        manual_arg_types={"ground_truth_file": str},
    )


if __name__ == "__main__":
    main()
