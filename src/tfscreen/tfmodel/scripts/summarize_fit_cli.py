import datetime
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from tfscreen.analysis.stats_test_suite import stats_test_suite
from tfscreen.plot import default_styles  # noqa: F401 — applies global rcParams on import
from tfscreen.plot.default_styles import DEFAULT_COLORS
from tfscreen.plot.xy_corr import xy_corr
from tfscreen.tfmodel.analysis.error_calibration import calibration_summary
from tfscreen.util.cli.generalized_main import generalized_main


def _find_params_or_posterior(run_dir):
    """Return (kind, path) for the best available prediction source in run_dir.

    Preference order: ``*_posterior.h5`` first (richer uncertainty), then
    ``*_params.npz`` (MAP fallback).  Returns ``(None, None)`` when neither
    is found.  Warns when multiple files of the same type are present and
    uses the first alphabetically.
    """
    for suffix, kind in (("_posterior.h5", "posterior"), ("_params.npz", "params")):
        matches = sorted(glob.glob(os.path.join(run_dir, f"*{suffix}")))
        if not matches:
            continue
        if len(matches) > 1:
            warnings.warn(
                f"Multiple {kind} files found in {run_dir}; "
                f"using {os.path.basename(matches[0])}"
            )
        return kind, matches[0]
    return None, None


def _try_plot_theta_fits(binding_df, pred_df, out_prefix):
    """Generate per-genotype theta fit plots merging binding observations with predictions.

    Silently skips when binding_df or pred_df is None.  Warns and returns on
    any failure so that the rest of summarize_fit output is unaffected.
    """
    if binding_df is None or pred_df is None:
        return

    if "theta_std" not in binding_df.columns:
        warnings.warn(
            "binding_df has no 'theta_std' column; skipping theta fit plots."
        )
        return

    try:
        from tfscreen.plot.plot_theta_fits import plot_theta_fits

        join_cols = ["genotype", "titrant_name", "titrant_conc"]
        binding_sel = [c for c in join_cols + ["theta_obs", "theta_std"]
                       if c in binding_df.columns]
        merged = pred_df.merge(binding_df[binding_sel], on=join_cols, how="inner")
        if len(merged) == 0:
            return

        for geno in sorted(merged["genotype"].unique().tolist(), key=str):
            geno_df = merged[merged["genotype"] == geno]
            safe_name = str(geno).replace("/", "_").replace(" ", "_")
            csv_path = f"{out_prefix}_{safe_name}_theta_fits.csv"
            geno_df.to_csv(csv_path, index=False)
            print(f"Wrote theta fit data to {csv_path}")
            ax = plot_theta_fits(geno_df, title=str(geno))
            fig = ax.get_figure()
            pdf_path = f"{out_prefix}_{safe_name}_theta_fits.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote theta fit plot to {pdf_path}")
    except Exception as exc:
        warnings.warn(f"Could not generate theta fit plots: {exc}")


def _try_plot_trajectories(config_file, config_yaml, run_dir, out_prefix, binding_df):
    """Attempt to generate per-genotype growth trajectory plots.

    Silently skips when the run has no growth data.  Warns and returns on any
    other failure so that the rest of summarize_fit output is unaffected.
    """
    if config_yaml is None or not config_yaml.get("data", {}).get("growth"):
        return

    kind, pred_path = _find_params_or_posterior(run_dir)
    if pred_path is None:
        warnings.warn(
            "No *_posterior.h5 or *_params.npz found in "
            f"{run_dir}; skipping trajectory plots."
        )
        return

    try:
        from tfscreen.tfmodel.configuration_io import read_configuration
        from tfscreen.plot.geno_trajectory import predict_geno_trajectory_df, plot_geno_trajectory

        orchestrator, _ = read_configuration(config_file)

        genotypes = list(binding_df["genotype"].unique()) if binding_df is not None else None

        pred_df = predict_geno_trajectory_df(
            orchestrator,
            pred_path,
            genotypes=genotypes,
        )
        for geno in sorted(pred_df["genotype"].unique().tolist(), key=str):
            geno_df = pred_df[pred_df["genotype"] == geno]
            safe_name = str(geno).replace("/", "_").replace(" ", "_")
            csv_path = f"{out_prefix}_{safe_name}_trajectory.csv"
            geno_df.to_csv(csv_path, index=False)
            print(f"Wrote trajectory data to {csv_path}")
            fig = plot_geno_trajectory(geno_df)
            pdf_path = f"{out_prefix}_{safe_name}_trajectory.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote trajectory plot to {pdf_path}")
    except Exception as exc:
        warnings.warn(f"Could not generate trajectory plots: {exc}")


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


def _read_all_losses(losses_file):
    """Return (epochs, losses) from a losses text file.

    Supports three formats:
    - New: ``epoch,loss,relative_change``  (int epoch first, loss second)
    - Old comma: ``loss,other``  (float loss first, no epoch)
    - Whitespace: ``step loss``  (loss is the last column)

    For old formats without a true epoch column, row index is used as epoch.
    """
    epochs = []
    values = []
    with open(losses_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = line.split(",")
                try:
                    epoch = int(parts[0])       # new format: epoch is an integer
                    loss = float(parts[1])
                    epochs.append(epoch)
                    values.append(loss)
                except (ValueError, IndexError):
                    try:
                        values.append(float(parts[0]))  # old format: loss is first
                        epochs.append(len(values) - 1)
                    except ValueError:
                        pass                            # header line or non-numeric
            else:
                parts = line.split()
                try:
                    values.append(float(parts[-1]))
                    epochs.append(len(values) - 1)
                except (ValueError, IndexError):
                    pass
    if not values:
        raise ValueError(f"No numeric values found in {losses_file}")
    return epochs, values


def _read_final_loss(losses_file):
    """Return the last loss value from a losses text file."""
    _, losses = _read_all_losses(losses_file)
    return losses[-1]


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


def _extract_quantile_data(df):
    """Return (quantile_matrix, levels) from a DataFrame with q{level} columns.

    Columns must match ``q{float}`` exactly (e.g. ``q0.025``, ``q0.5``).
    Returns ``(None, None)`` when fewer than two such columns exist.
    """
    q_cols = sorted(
        [c for c in df.columns if re.match(r"^q\d*\.?\d+$", c)],
        key=lambda c: float(c[1:]),
    )
    if len(q_cols) < 2:
        return None, None
    levels = np.array([float(c[1:]) for c in q_cols])
    return df[q_cols].values, levels


def _try_calibration(df, ref_col, out_prefix, label):
    """Run calibration_summary on df[ref_col] vs quantile columns.

    Silently skips when df is None, ref_col is absent, or fewer than two
    quantile columns exist.  Warns and returns on any other failure so the
    rest of the output is unaffected.
    """
    if df is None:
        return
    if ref_col not in df.columns:
        return
    quantile_matrix, levels = _extract_quantile_data(df)
    if quantile_matrix is None:
        return
    try:
        calibration_summary(
            true_vals=df[ref_col].values,
            quantile_matrix=quantile_matrix,
            quantile_levels=levels,
            out_prefix=out_prefix,
            label=label,
        )
    except Exception as exc:
        warnings.warn(f"Could not run calibration summary for {label}: {exc}")


def _build_transform_registry(sim_df):
    """Return a dict mapping parameter name → ref Series indexed by genotype.

    Supports four resolution strategies (tried in this order at registration):

    1. **Compound**: ``logit_delta`` = logit(theta_high) – logit(theta_low),
       registered before the column loop so it is never overwritten.
    2. **Direct**: column name is the parameter name.
    3. **log_ prefix**: ``log_{col}`` → ``np.log(sim[col])``.
    4. **logit_ prefix**: ``logit_{col}`` → logit(sim[col]).  For columns
       that start with ``theta_`` the short form is also registered, e.g.
       ``theta_low`` registers both ``logit_theta_low`` and ``logit_low``.

    ``setdefault`` is used for entries 2–4 so compound entries win.
    """
    sim_lookup = sim_df.set_index("genotype")
    sim_cols = [c for c in sim_df.columns if c != "genotype"]
    registry = {}

    # Compound: logit_delta = logit(theta_high) - logit(theta_low)
    if "theta_high" in sim_cols and "theta_low" in sim_cols:
        high = sim_lookup["theta_high"].clip(1e-9, 1 - 1e-9)
        low = sim_lookup["theta_low"].clip(1e-9, 1 - 1e-9)
        registry["logit_delta"] = np.log(high / (1 - high)) - np.log(low / (1 - low))

    for col in sim_cols:
        col_s = sim_lookup[col]
        clipped = col_s.clip(1e-9, 1 - 1e-9)
        logit_s = np.log(clipped / (1 - clipped))

        registry.setdefault(col, col_s)
        registry.setdefault(f"log_{col}", np.log(col_s))
        registry.setdefault(f"logit_{col}", logit_s)

        # For theta_* columns also register the short logit/log/direct forms,
        # e.g. theta_low → logit_low, log_low, low.
        if col.startswith("theta_"):
            base = col[6:]
            registry.setdefault(f"logit_{base}", logit_s)
            registry.setdefault(f"log_{base}", np.log(col_s))
            registry.setdefault(base, col_s)

    return registry


def _compute_direct_obs(params_df, ref_series):
    """Return ref array for a direct params file keyed on genotype.

    Returns ``None`` when params_df has no ``genotype`` column (e.g. files
    keyed on condition/replicate that cannot be matched to sim_parameters).
    """
    if "genotype" not in params_df.columns:
        return None
    return ref_series.reindex(params_df["genotype"]).values


def _compute_diff_obs(params_df, ref_series):
    """Return ref array for a diff params file keyed on mutation.

    ref = sim[mutation] − sim[wt].  Returns ``None`` when params_df has no
    ``mutation`` column.  Sets ref to NaN when ``wt`` is absent from
    ref_series.
    """
    if "mutation" not in params_df.columns:
        return None
    if "wt" not in ref_series.index:
        warnings.warn("'wt' not found in sim_parameters; setting diff ref to NaN")
        return np.full(len(params_df), np.nan)
    wt_val = ref_series["wt"]
    return ref_series.reindex(params_df["mutation"]).values - wt_val


def _compute_epi_obs(params_df, ref_series, param_name):
    """Return ref array for an epi params file keyed on pair.

    Uses ``extract_epistasis`` (additive scale) to compute per-double-mutant
    epistasis from the ground-truth ref_series.  Genotypes not present in the
    result (triple+, unrecognised pairs) receive NaN.  Returns ``None`` when
    params_df has no ``pair`` column.
    """
    from tfscreen.analysis.extract_epistasis import extract_epistasis

    if "pair" not in params_df.columns:
        return None

    work_df = pd.DataFrame({
        "genotype": ref_series.index,
        "_obs": ref_series.values,
    })
    try:
        ep_result = extract_epistasis(work_df, y_obs="_obs", scale="add")
    except Exception as exc:
        warnings.warn(f"extract_epistasis failed for {param_name}: {exc}")
        return np.full(len(params_df), np.nan)

    if len(ep_result) == 0:
        return np.full(len(params_df), np.nan)

    from tfscreen.genetics import standardize_genotypes

    ep_map = ep_result.set_index("genotype")["ep_obs"]
    std_pairs = pd.Series(standardize_genotypes(params_df["pair"]),
                          index=params_df.index)
    return ep_map.reindex(std_pairs).values


def _try_plot_params_corr(out_df, label, pdf_path):
    """Save a scatter correlation plot of ref vs median for a params CSV.

    Silently skips when fewer than two finite (ref, median) pairs exist.
    Warns and returns on any other failure so the rest of the output is
    unaffected.
    """
    try:
        valid = out_df[["ref", "q0.5"]].dropna()
        if len(valid) < 2:
            return
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        xy_corr(
            x_values=valid["ref"].values,
            y_values=valid["q0.5"].values,
            as_hexbin=False,
            ax=ax,
        )
        ax.set_xlabel(f"Simulated {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote parameter correlation plot to {pdf_path}")
    except Exception as exc:
        warnings.warn(f"Could not generate parameter correlation plot {pdf_path}: {exc}")


def _summarize_params(run_dir, out_prefix):
    """Annotate *_params_*.csv files with ground-truth ref from tfs_sim_parameters.csv.

    Scans run_dir for a ``*_sim_parameters.csv`` file.  If absent the function
    returns silently (real-data runs have no ground truth).

    For each non-genotype column in sim_parameters the function checks that a
    matching ``*_params_{col}.csv`` exists and warns if it does not.

    All ``*_params_*.csv`` files in run_dir are then classified by their
    suffix as *direct* (genotype key), *diff* (``_d_`` infix, mutation key),
    or *epi* (``_epi_`` infix, pair key).  A ``ref`` column is appended to
    each file whose parameter name resolves via the transform registry and
    written to the summary directory under ``{out_prefix}_params_*.csv``.
    """
    sim_path = _find_unique(run_dir, "_sim_parameters.csv", "sim_parameters",
                            warn_missing=False)
    if sim_path is None:
        return

    try:
        sim_df = pd.read_csv(sim_path)
    except Exception as exc:
        warnings.warn(f"Could not load sim parameters from {sim_path}: {exc}")
        return

    if "genotype" not in sim_df.columns:
        warnings.warn(
            f"sim_parameters {sim_path} has no 'genotype' column; "
            "skipping param summaries"
        )
        return

    sim_cols = [c for c in sim_df.columns if c != "genotype"]

    for col in sim_cols:
        if not glob.glob(os.path.join(run_dir, f"*_params_{col}.csv")):
            warnings.warn(f"Expected *_params_{col}.csv in {run_dir} but none found")

    transform_registry = _build_transform_registry(sim_df)

    for param_file in sorted(glob.glob(os.path.join(run_dir, "*_params_*.csv"))):
        basename = os.path.basename(param_file)

        m_epi = re.search(r"_params_epi_(.+)\.csv$", basename)
        m_diff = re.search(r"_params_d_(.+)\.csv$", basename)
        m_direct = re.search(r"_params_(.+)\.csv$", basename)

        if m_epi:
            kind, param_name = "epi", m_epi.group(1)
        elif m_diff:
            kind, param_name = "diff", m_diff.group(1)
        elif m_direct:
            kind, param_name = "direct", m_direct.group(1)
        else:
            continue

        if param_name not in transform_registry:
            continue

        ref_series = transform_registry[param_name]

        try:
            params_df = pd.read_csv(param_file)
        except Exception as exc:
            warnings.warn(f"Could not read {param_file}: {exc}")
            continue

        ref = None
        try:
            if kind == "direct":
                ref = _compute_direct_obs(params_df, ref_series)
            elif kind == "diff":
                ref = _compute_diff_obs(params_df, ref_series)
            else:
                ref = _compute_epi_obs(params_df, ref_series, param_name)
        except Exception as exc:
            warnings.warn(f"Could not compute ref for {basename}: {exc}")
            continue

        if ref is None:
            continue

        out_df = params_df.copy()
        out_df["ref"] = ref

        if kind == "epi":
            out_name = f"{out_prefix}_params_epi_{param_name}.csv"
            plot_label = f"epi_{param_name}"
        elif kind == "diff":
            out_name = f"{out_prefix}_params_d_{param_name}.csv"
            plot_label = f"d_{param_name}"
        else:
            out_name = f"{out_prefix}_params_{param_name}.csv"
            plot_label = param_name

        try:
            out_df.to_csv(out_name, index=False)
            print(f"Wrote parameter summary to {out_name}")
        except Exception as exc:
            warnings.warn(f"Could not write {out_name}: {exc}")
            continue

        pdf_name = out_name.replace(".csv", ".pdf")
        _try_plot_params_corr(out_df, plot_label, pdf_name)
        _try_calibration(out_df, "ref", out_name[:-4], plot_label)


def summarize_fit(run_dir,
                  ref_theta_file=None,
                  out_prefix=None):
    """
    Evaluate theta prediction quality for a tfs-fit-model run directory.

    Scans run_dir for a *_config.yaml, *_theta_pred.csv, *_growth_pred.csv
    (optional), and *_losses.txt (optional).  Computes prediction statistics
    and writes output files to the summary directory.

    **CSV files are the authoritative outputs.**  Every plot written by this
    function has a matched CSV containing the exact data used to generate it.
    The PDFs are human-readable summaries; the CSVs are the record of truth
    for downstream quantitative analysis.

    Output files written:

    - ``{out_prefix}_fit_summary.json`` — nested statistics with top-level
      keys ``metadata``, ``theta`` (training / test sub-keys), and ``growth``
      (training sub-key).
    - ``{out_prefix}_theta_corr_training.csv`` — joined (ref, predicted)
      theta pairs for training genotypes; ``ref`` column holds the training
      observation used as ground truth.
    - ``{out_prefix}_theta_corr_test.csv`` — same for test genotypes (only
      written when a ref theta file is resolved and has matching rows).
    - ``{out_prefix}_theta_corr.pdf`` — two-panel correlation plot for theta
      (training left, test right).
    - ``{out_prefix}_growth_corr.csv`` — relative symlink to the
      *_growth_pred.csv in run_dir (only created when that file is present).
    - ``{out_prefix}_growth_corr.pdf`` — correlation plot for ln_cfu (only
      written when *_growth_pred.csv is present).
    - ``{out_prefix}_{genotype}_theta_fits.csv`` / ``.pdf`` — per-genotype
      joined observation + prediction data and fit plot (one pair per
      genotype present in both binding and prediction CSVs).
    - ``{out_prefix}_{genotype}_trajectory.csv`` / ``.pdf`` — per-genotype
      predicted growth trajectory data and plot (only when growth data is
      configured and a posterior/params file is found).
    - ``{out_prefix}_losses.pdf`` — training loss curve (only written when
      *_losses.txt is present).

    Parameters
    ----------
    run_dir : str
        Directory containing model fit outputs.
    ref_theta_file : str, optional
        CSV with columns genotype, titrant_name, titrant_conc, and a theta
        column containing known theta values for evaluating out-of-sample
        prediction.  When omitted the function looks for a
        ``*_sim_genotype_theta.csv`` file in *run_dir* automatically (the
        file produced by ``tfs-simulate``).  The theta column is resolved
        in order: ``theta_obs`` (preferred) → ``theta`` (fallback); it is
        renamed to ``ref`` in the output CSV.  If neither column is present
        a warning is issued and test statistics are skipped.  Pass this
        argument explicitly to use a theta reference from outside *run_dir*
        or with a non-standard name.
    out_prefix : str, optional
        Prefix for output files.  Defaults to {run_dir}/tfs_summarize.
    """
    run_dir = os.path.abspath(run_dir)
    if out_prefix is None:
        out_prefix = os.path.join(run_dir, "summary", "tfs_summarize")
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    metadata = {
        "run_dir": run_dir,
        "ref_theta_file": None,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_parameters": None,
        "n_theta_training_points": None,
        "n_theta_test_points": None,
        "n_growth_training_points": None,
        "final_loss": None,
    }
    theta_training_stats = None
    theta_test_stats = None
    growth_training_stats = None
    train_merged = None
    test_merged = None
    growth_pred_df = None

    # --- Locate files in run_dir ---
    config_file = _find_unique(run_dir, "_config.yaml", "config")
    theta_pred_file = _find_unique(run_dir, "_theta_pred.csv", "theta predictions")
    losses_file = _find_unique(run_dir, "_losses.txt", "losses", warn_missing=False)
    growth_pred_file = _find_unique(run_dir, "_growth_pred.csv", "growth predictions",
                                    warn_missing=False)

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

    # --- Read converged loss (and keep full history for the loss-curve plot) ---
    all_epochs = None
    all_losses = None
    if losses_file is not None:
        try:
            all_epochs, all_losses = _read_all_losses(losses_file)
            metadata["final_loss"] = all_losses[-1]
        except Exception as exc:
            warnings.warn(f"Could not read final loss from {losses_file}: {exc}")

    # --- Load theta predictions ---
    pred_df = None
    if theta_pred_file is not None:
        try:
            pred_df = pd.read_csv(theta_pred_file)
        except Exception as exc:
            warnings.warn(f"Could not load theta predictions from {theta_pred_file}: {exc}")

    # --- Theta training statistics ---
    if pred_df is not None and binding_df is not None:
        try:
            join_cols = ["genotype", "titrant_name", "titrant_conc"]
            train_pred = pred_df[pred_df["in_training_data"] == 1].copy()
            train_merged = train_pred.merge(
                binding_df[join_cols + ["theta_obs"]],
                on=join_cols,
                how="inner",
            ).rename(columns={"theta_obs": "ref"})
            metadata["n_theta_training_points"] = len(train_merged)
            if len(train_merged) > 0:
                theta_training_stats = _run_stats(
                    train_merged["q0.5"].values,
                    train_merged["ref"].values,
                )
                _try_calibration(
                    train_merged, "ref",
                    f"{out_prefix}_theta_training", "theta training",
                )
        except Exception as exc:
            warnings.warn(f"Could not compute theta training statistics: {exc}")

    # --- Resolve ref_theta_file: explicit arg wins, else auto-discover ---
    if ref_theta_file is None:
        ref_theta_file = _find_unique(run_dir, "_sim_genotype_theta.csv",
                                      "ref theta", warn_missing=False)
    metadata["ref_theta_file"] = ref_theta_file

    # --- Theta test statistics ---
    if ref_theta_file is not None and pred_df is not None:
        try:
            gt_df = pd.read_csv(ref_theta_file)
            join_cols = ["genotype", "titrant_name", "titrant_conc"]
            if "theta_obs" in gt_df.columns:
                theta_col = "theta_obs"
            elif "theta" in gt_df.columns:
                theta_col = "theta"
            else:
                warnings.warn(
                    f"Ref theta file {ref_theta_file} has neither "
                    f"'theta_obs' nor 'theta' column; skipping test statistics"
                )
                theta_col = None
            if theta_col is not None:
                gt_df = gt_df.rename(columns={theta_col: "ref"})
            test_merged = pred_df.merge(
                gt_df[join_cols + ["ref"]],
                on=join_cols,
                how="inner",
            ) if theta_col is not None else None
            if test_merged is not None:
                metadata["n_theta_test_points"] = len(test_merged)
                if len(test_merged) > 0:
                    theta_test_stats = _run_stats(
                        test_merged["q0.5"].values,
                        test_merged["ref"].values,
                    )
                    _try_calibration(
                        test_merged, "ref",
                        f"{out_prefix}_theta_test", "theta test",
                    )
        except Exception as exc:
            warnings.warn(f"Could not compute theta test statistics from {ref_theta_file}: {exc}")

    # --- Growth training statistics ---
    if growth_pred_file is not None:
        try:
            growth_pred_df = pd.read_csv(growth_pred_file)
            # Drop rows where observed ln_cfu is missing (no observation at that timepoint)
            growth_valid = growth_pred_df.dropna(subset=["ln_cfu", "q0.5"])
            metadata["n_growth_training_points"] = len(growth_valid)
            if len(growth_valid) > 0:
                growth_training_stats = _run_stats(
                    growth_valid["q0.5"].values,
                    growth_valid["ln_cfu"].values,
                )
        except Exception as exc:
            warnings.warn(f"Could not compute growth training statistics: {exc}")

    # --- Trajectory plots ---
    _try_plot_trajectories(config_file, config_yaml, run_dir, out_prefix, binding_df)

    # --- Per-genotype theta fit plots ---
    _try_plot_theta_fits(binding_df, pred_df, out_prefix)

    # --- Write JSON ---
    results = {
        "metadata": metadata,
        "theta": {
            "training": theta_training_stats,
            "test": theta_test_stats,
        },
        "growth": {
            "training": growth_training_stats,
        },
    }
    json_file = f"{out_prefix}_fit_summary.json"
    try:
        with open(json_file, "w") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"Wrote statistics to {json_file}")
    except Exception as exc:
        warnings.warn(f"Could not write JSON to {json_file}: {exc}")

    # --- Theta correlation CSVs ---
    if train_merged is not None and len(train_merged) > 0:
        csv_file = f"{out_prefix}_theta_corr_training.csv"
        try:
            train_merged.to_csv(csv_file, index=False)
            print(f"Wrote theta training data to {csv_file}")
        except Exception as exc:
            warnings.warn(f"Could not write theta training CSV to {csv_file}: {exc}")

    if test_merged is not None and len(test_merged) > 0:
        csv_file = f"{out_prefix}_theta_corr_test.csv"
        try:
            test_merged.to_csv(csv_file, index=False)
            print(f"Wrote theta test data to {csv_file}")
        except Exception as exc:
            warnings.warn(f"Could not write theta test CSV to {csv_file}: {exc}")

    # --- Correlation plot ---
    pdf_file = f"{out_prefix}_theta_corr.pdf"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if train_merged is not None and len(train_merged) > 0:
            xy_corr(
                x_values=train_merged["ref"].values,
                y_values=train_merged["q0.5"].values,
                as_hexbin=False,
                ax=axes[0],
            )
            axes[0].set_xlabel("Observed θ")
            axes[0].set_ylabel("Predicted θ")
            axes[0].set_title("Training data")
        else:
            _blank_panel(axes[0], "No training data available")

        if test_merged is not None and len(test_merged) > 0:
            xy_corr(
                x_values=test_merged["ref"].values,
                y_values=test_merged["q0.5"].values,
                as_hexbin=False,
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

    # --- Growth correlation CSV ---
    growth_copy_df = None
    if growth_pred_file is not None:
        growth_csv_file = f"{out_prefix}_growth_corr.csv"
        try:
            growth_copy_df = pd.read_csv(growth_pred_file).rename(
                columns={"ln_cfu": "ref", "ln_cfu_std": "ref_std"}
            )
            growth_copy_df.to_csv(growth_csv_file, index=False)
            print(f"Wrote growth data to {growth_csv_file}")
        except Exception as exc:
            warnings.warn(f"Could not write growth CSV at {growth_csv_file}: {exc}")
        _try_calibration(
            growth_copy_df, "ref",
            f"{out_prefix}_growth", "growth",
        )

    # --- Growth correlation plot ---
    growth_pdf_file = f"{out_prefix}_growth_corr.pdf"
    try:
        if growth_pred_df is not None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            xy_corr(
                x_values=growth_pred_df["ln_cfu"].values,
                y_values=growth_pred_df["q0.5"].values,
                as_hexbin=False,
                ax=ax,
            )
            ax.set_xlabel("Observed ln CFU")
            ax.set_ylabel("Predicted ln CFU")
            ax.set_title("Growth prediction")
            fig.tight_layout()
            fig.savefig(growth_pdf_file)
            plt.close(fig)
            print(f"Wrote growth correlation plot to {growth_pdf_file}")
    except Exception as exc:
        warnings.warn(f"Could not generate growth correlation plot: {exc}")

    # --- Loss curve plot ---
    loss_pdf_file = f"{out_prefix}_losses.pdf"
    try:
        if all_losses is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(all_epochs, all_losses, lw=1.5, color=DEFAULT_COLORS[2])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training loss")
            fig.tight_layout()
            fig.savefig(loss_pdf_file)
            plt.close(fig)
            print(f"Wrote loss curve to {loss_pdf_file}")
    except Exception as exc:
        warnings.warn(f"Could not generate loss curve plot: {exc}")

    # --- Sim-parameter correlation CSVs ---
    _summarize_params(run_dir, out_prefix)

    return results


def main():
    return generalized_main(
        summarize_fit,
        manual_arg_types={"ref_theta_file": str},
    )


if __name__ == "__main__":
    main()
