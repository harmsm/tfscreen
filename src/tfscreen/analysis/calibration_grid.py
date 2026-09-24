"""
Calibration summary across a simulation grid (``tfs-summarize-calibration``).

Every run directory of a ``tfs-setup-sim-grid`` grid holds a ``combo.json``
(its grid variables) and, once the run finishes, the ``tfs-summarize-fit``
outputs in ``summary/``: posterior quantile ladders (``q<level>`` columns)
joined to the simulated truth (``ref``) for θ (``*_theta_corr_test.csv``) and
for every fitted parameter with ground truth (``*_params_*.csv``).  This
module turns those into calibration metrics per run, averages them into arms
(runs sharing every grid variable except the replicate keys), and pairs every
run with a baseline run fit to the same simulated data set.

Metrics (per run x quantity x stratum)
--------------------------------------
- ``coverage_<a>``: fraction of true values inside the central ``a``
  credible interval, for ``a`` in ``CURVE_LEVELS``.  Computed from PIT values
  interpolated from the quantile ladder -- the same calculation as
  ``tfs-summarize-fit`` (``error_calibration``), so numbers agree.
- ``calibration_error``: mean |coverage - nominal| over ``CURVE_LEVELS``.
- ``calibration_bias``: mean (coverage - nominal); negative = intervals too
  narrow (overconfident), positive = too wide.
- ``ks_stat``/``ks_pval``/``mean_pit``: PIT uniformity (one-sample KS).
- ``width_<a>``: mean width of the central ``a`` interval, for ``a`` in
  ``WIDTH_LEVELS`` -- so a guide cannot look calibrated just by being vague.
- ``rmse``/``pearson_r``: posterior median vs truth.

Strata
------
Genotype-indexed quantities are also broken out by ``has_binding`` (the
genotype has binding data: ``*_sim_binding.csv``) and ``purity`` (from the
sub-libraries encoding the genotype in ``*_sim_library.csv``), separately and
crossed.  ``purity`` follows the genotype's ``bulk_fraction``, which sets its
congressed fraction: ``spike`` (only a spiked sequence encodes it,
``bulk_fraction`` 0, never congressed), ``bulk`` (only bulk sub-libraries,
``bulk_fraction`` 1) or ``mixed`` (both, as for wt and the spiked single
mutants, which the bulk sub-libraries also encode).  θ is additionally split
by ``theta_regime``:
``resolvable`` when the true θ lies in ``[regime_eps, 1 - regime_eps]``,
``saturated`` otherwise.  The pooled row carries ``"all"`` in every stratum
column.

No thresholds, no grades: every value is a raw number; filter downstream.
"""

import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

from tfscreen.tfmodel.analysis.error_calibration import (
    calibration_curve,
    pit_from_quantiles,
    pit_uniformity_test,
)

CURVE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
WIDTH_LEVELS = (0.5, 0.8, 0.95)
STRATUM_COLS = ("has_binding", "purity", "theta_regime")
SPIKED_ORIGIN = "spiked"
ALL = "all"

# Metrics compared against the baseline in paired_differences.
PAIRED_METRICS = ("calibration_error", "calibration_bias", "coverage_0.95",
                  "width_0.95", "rmse", "ks_stat")

_Q_COL_RE = re.compile(r"^q(\d*\.?\d+)$")
_NONE_STRINGS = {"none", "null", "~", ""}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def quantile_columns(df):
    """Return (sorted ``q<level>`` column names, their levels as an array)."""
    cols = sorted((c for c in df.columns if _Q_COL_RE.match(c)),
                  key=lambda c: float(c[1:]))
    return cols, np.array([float(c[1:]) for c in cols])


def _interp_quantile(qmat, levels, p):
    """Per-row quantile at probability ``p``, linear between stored levels."""
    j = int(np.searchsorted(levels, p))
    if j < len(levels) and np.isclose(levels[j], p):
        return qmat[:, j]
    if j == 0 or j == len(levels):
        return np.full(qmat.shape[0], np.nan)
    w = (p - levels[j - 1]) / (levels[j] - levels[j - 1])
    return qmat[:, j - 1] + w * (qmat[:, j] - qmat[:, j - 1])


def metric_columns():
    """Names of the metric columns ``calibration_metrics`` returns."""
    return ([f"coverage_{a:g}" for a in CURVE_LEVELS]
            + ["calibration_error", "calibration_bias",
               "ks_stat", "ks_pval", "mean_pit"]
            + [f"width_{a:g}" for a in WIDTH_LEVELS]
            + ["rmse", "pearson_r"])


def calibration_metrics(ref, qmat, levels):
    """
    Calibration, sharpness and accuracy of posterior quantiles vs truth.

    Parameters
    ----------
    ref : array-like, shape (N,)
        True values.
    qmat : array-like, shape (N, Q)
        Posterior quantiles, columns ordered as ``levels``.
    levels : array-like, shape (Q,)
        Ascending quantile levels.

    Returns
    -------
    dict
        ``n`` plus every name in ``metric_columns()``.  Rows with a
        non-finite truth or quantile are dropped first; with no rows left
        every metric is NaN.
    """
    ref = np.asarray(ref, dtype=float)
    qmat = np.asarray(qmat, dtype=float)
    levels = np.asarray(levels, dtype=float)

    ok = np.isfinite(ref) & np.all(np.isfinite(qmat), axis=1)
    ref, qmat = ref[ok], qmat[ok]

    out = {"n": int(len(ref))}
    out.update({k: np.nan for k in metric_columns()})
    if len(ref) == 0:
        return out

    pit = pit_from_quantiles(ref, qmat, levels)
    curve = calibration_curve(pit, CURVE_LEVELS)
    for a in CURVE_LEVELS:
        out[f"coverage_{a:g}"] = curve[float(a)]
    diffs = np.array([curve[float(a)] - a for a in CURVE_LEVELS])
    out["calibration_error"] = float(np.mean(np.abs(diffs)))
    out["calibration_bias"] = float(np.mean(diffs))

    ks = pit_uniformity_test(pit)
    out["ks_stat"] = ks["ks_stat"]
    out["ks_pval"] = ks["ks_pval"]
    out["mean_pit"] = ks["mean_pit"]

    for a in WIDTH_LEVELS:
        lo = _interp_quantile(qmat, levels, (1.0 - a) / 2.0)
        hi = _interp_quantile(qmat, levels, (1.0 + a) / 2.0)
        out[f"width_{a:g}"] = float(np.nanmean(hi - lo))

    median = _interp_quantile(qmat, levels, 0.5)
    out["rmse"] = float(np.sqrt(np.nanmean((median - ref) ** 2)))
    if len(ref) > 1 and np.nanstd(median) > 0 and np.nanstd(ref) > 0:
        out["pearson_r"] = float(np.corrcoef(median, ref)[0, 1])

    return out


# ---------------------------------------------------------------------------
# Reading one run
# ---------------------------------------------------------------------------

def read_combo(run_dir):
    """Flatten a run's ``combo.json`` (simulate + template variables)."""
    with open(os.path.join(run_dir, "combo.json")) as fh:
        combo = json.load(fh)
    flat = dict(combo.get("simulate") or {})
    for key, value in (combo.get("template") or {}).items():
        if key in flat:
            warnings.warn(f"combo.json in {run_dir}: '{key}' is both a simulate "
                          f"and a template variable; using the template value.")
        flat[key] = value
    return flat


def _sim_file(run_dir, name):
    matches = sorted(glob.glob(os.path.join(run_dir, f"*_sim_{name}.csv")))
    return matches[0] if matches else None


def genotype_strata(run_dir):
    """
    Per-genotype strata from the simulation outputs, or None.

    Returns a DataFrame with ``genotype``, ``has_binding`` ("yes"/"no") and
    ``purity`` ("spike"/"mixed"/"bulk").  None when ``*_sim_library.csv`` is
    absent.
    """
    lib_path = _sim_file(run_dir, "library")
    if lib_path is None:
        return None
    lib_df = pd.read_csv(lib_path)
    genotypes = pd.unique(lib_df["genotype"].astype(str))
    is_spiked = lib_df["library_origin"].astype(str) == SPIKED_ORIGIN
    by_genotype = (lib_df.assign(genotype=lib_df["genotype"].astype(str),
                                 _spiked=is_spiked)
                   .groupby("genotype")["_spiked"])
    any_spiked, all_spiked = by_genotype.any(), by_genotype.all()

    binding_path = _sim_file(run_dir, "binding")
    binding = (set(pd.read_csv(binding_path)["genotype"].astype(str))
               if binding_path else set())

    return pd.DataFrame({
        "genotype": genotypes,
        "has_binding": ["yes" if g in binding else "no" for g in genotypes],
        "purity": ["spike" if all_spiked[g] else
                   "mixed" if any_spiked[g] else "bulk" for g in genotypes],
    })


def read_quantities(summary_dir):
    """
    Return [(quantity, DataFrame)] for every calibration-ready summary CSV.

    ``theta_test`` comes from ``*_theta_corr_test.csv``; each
    ``*_params_<name>.csv`` gives quantity ``<name>``.  A file qualifies when
    it has a ``ref`` column and at least two ``q<level>`` columns.
    """
    candidates = [("theta_test", p) for p in
                  sorted(glob.glob(os.path.join(summary_dir, "*_theta_corr_test.csv")))]
    for path in sorted(glob.glob(os.path.join(summary_dir, "*_params_*.csv"))):
        name = re.search(r"_params_(.+)\.csv$", os.path.basename(path)).group(1)
        if name.endswith(("_calibration_curve", "_pit")):
            continue
        candidates.append((name, path))

    out = []
    for quantity, path in candidates:
        df = pd.read_csv(path)
        cols, _ = quantile_columns(df)
        if "ref" in df.columns and len(cols) >= 2:
            out.append((quantity, df))
    return out


def _groupings(df):
    """Stratum groupings present in ``df`` (the pooled one first)."""
    present = [c for c in STRATUM_COLS if c in df.columns and df[c].notna().any()]
    groupings = [()] + [(c,) for c in present]
    if "has_binding" in present and "purity" in present:
        groupings.append(("has_binding", "purity"))
    return groupings


def summarize_run(run_dir, summary_subdir="summary", regime_eps=0.01):
    """
    Calibration metrics for one run.

    Returns
    -------
    rows : list of dict
        One per quantity x stratum: ``quantity``, the ``STRATUM_COLS``
        (``"all"`` where pooled), ``n`` and the metric columns.
    problem : str or None
        Why the run produced no rows, else None.
    """
    summary_dir = os.path.join(run_dir, summary_subdir)
    if not os.path.isdir(summary_dir):
        return [], f"no {summary_subdir}/ directory"
    quantities = read_quantities(summary_dir)
    if not quantities:
        return [], "no calibration outputs"

    strata = genotype_strata(run_dir)
    rows = []
    for quantity, df in quantities:
        cols, levels = quantile_columns(df)
        df = df.copy()
        if "genotype" in df.columns and strata is not None:
            df["genotype"] = df["genotype"].astype(str)
            df = df.merge(strata, on="genotype", how="left")
            unmatched = df["purity"].isna().mean()
            if unmatched > 0:
                warnings.warn(f"{run_dir}: {unmatched:.1%} of '{quantity}' rows "
                              f"have genotypes missing from the sim library; "
                              f"they are pooled but not stratified.")
        if quantity == "theta_test":
            resolvable = df["ref"].between(regime_eps, 1.0 - regime_eps)
            df["theta_regime"] = np.where(resolvable, "resolvable", "saturated")

        for keys in _groupings(df):
            groups = [((), df)] if not keys else df.groupby(list(keys), dropna=True)
            for values, sub in groups:
                values = values if isinstance(values, tuple) else (values,)
                row = {"quantity": quantity, **{c: ALL for c in STRATUM_COLS}}
                row.update(dict(zip(keys, values)))
                row.update(calibration_metrics(sub["ref"], sub[cols], levels))
                rows.append(row)
    return rows, None


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def summarize_grid_runs(grid_dir, summary_subdir="summary", regime_eps=0.01):
    """
    Calibration metrics for every run under ``grid_dir``.

    Returns
    -------
    runs_df : DataFrame
        One row per run x quantity x stratum: ``run``, the grid variables,
        ``quantity``, the strata, ``n`` and the metrics.
    status_df : DataFrame
        One row per run directory: ``run``, grid variables, ``status``
        ("ok" or the problem) and ``n_rows``.
    grid_vars : list of str
        Grid variable names, in first-seen order.
    """
    run_dirs = sorted(d for d in glob.glob(os.path.join(grid_dir, "*"))
                      if os.path.isfile(os.path.join(d, "combo.json")))
    if not run_dirs:
        raise ValueError(f"No run directories with combo.json found in {grid_dir}")

    rows, status, grid_vars = [], [], []
    for run_dir in run_dirs:
        run = os.path.basename(run_dir)
        combo = read_combo(run_dir)
        grid_vars += [k for k in combo if k not in grid_vars]
        run_rows, problem = summarize_run(run_dir, summary_subdir, regime_eps)
        status.append({"run": run, **combo, "status": problem or "ok",
                       "n_rows": len(run_rows)})
        rows += [{"run": run, **combo, **r} for r in run_rows]

    return pd.DataFrame(rows), pd.DataFrame(status), grid_vars


def _metrics_present(df):
    return [c for c in metric_columns() if c in df.columns]


def summarize_arms(runs_df, grid_vars, replicate_keys):
    """
    Average metrics over replicates within each arm.

    An arm is every run sharing all grid variables except ``replicate_keys``.
    Returns one row per arm x quantity x stratum with ``n_runs`` and
    ``<metric>_mean`` / ``<metric>_std`` columns.
    """
    arm_keys = [k for k in grid_vars if k not in replicate_keys]
    group_cols = arm_keys + ["quantity", *STRATUM_COLS]
    metrics = ["n"] + _metrics_present(runs_df)

    grouped = runs_df.groupby(group_cols, dropna=False, sort=True)
    stats = grouped[metrics].agg(["mean", "std"])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    stats.insert(0, "n_runs", grouped.size())
    return stats.reset_index()


def parse_baseline(baseline):
    """
    Parse ``["key=value", ...]`` into a dict.  "None"/"null"/"~" -> None;
    other values are parsed as int, then float, else kept as strings.
    """
    out = {}
    for item in baseline or []:
        if "=" not in item:
            raise ValueError(f"baseline entry '{item}' must be key=value.")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if value.lower() in _NONE_STRINGS:
            out[key] = None
            continue
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = value
    return out


def _matches(series, value):
    if value is None:
        return series.isna()
    return series == value


def paired_differences(runs_df, grid_vars, baseline):
    """
    Pair every non-baseline run with the baseline run of the same design.

    ``baseline`` maps grid variables to their baseline values (e.g.
    ``{"guide_type": "component", "guide_rank": None}``).  A run's partner
    shares every other grid variable -- including the simulation and fit
    seeds -- plus quantity and stratum.  Returns the run rows with
    ``delta_<metric>`` = run - baseline for each of ``PAIRED_METRICS``.
    """
    unknown = [k for k in baseline if k not in grid_vars]
    if unknown:
        raise ValueError(f"baseline keys {unknown} are not grid variables "
                         f"({grid_vars}).")

    is_base = np.ones(len(runs_df), dtype=bool)
    for key, value in baseline.items():
        is_base &= _matches(runs_df[key], value).to_numpy()
    if not is_base.any():
        raise ValueError(f"No runs match baseline {baseline}.")

    match_keys = [k for k in grid_vars if k not in baseline] + ["quantity", *STRATUM_COLS]
    metrics = [m for m in PAIRED_METRICS if m in runs_df.columns]
    base = runs_df.loc[is_base, match_keys + metrics].rename(
        columns={m: f"baseline_{m}" for m in metrics})
    paired = runs_df.loc[~is_base].merge(base, on=match_keys, how="inner")
    for m in metrics:
        paired[f"delta_{m}"] = paired[m] - paired[f"baseline_{m}"]

    keep = ["run", *grid_vars, "quantity", *STRATUM_COLS]
    keep += [f"delta_{m}" for m in metrics]
    return paired[keep]


def summarize_paired(paired_df, grid_vars, replicate_keys):
    """Mean, std, count and standard error of each delta per arm x quantity x stratum."""
    arm_keys = [k for k in grid_vars if k not in replicate_keys]
    group_cols = arm_keys + ["quantity", *STRATUM_COLS]
    deltas = [c for c in paired_df.columns if c.startswith("delta_")]

    grouped = paired_df.groupby(group_cols, dropna=False, sort=True)
    stats = grouped[deltas].agg(["mean", "std", "count"])
    for d in deltas:
        stats[(d, "se")] = stats[(d, "std")] / np.sqrt(stats[(d, "count")])
    stats = stats.sort_index(axis=1)
    stats.columns = [f"{d}_{s}" for d, s in stats.columns]
    stats.insert(0, "n_pairs", grouped.size())
    return stats.reset_index()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _label(row, keys):
    parts = []
    for k in keys:
        v = row[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        parts.append(f"{k}={v}")
    return ", ".join(parts) or "(all)"


def plot_calibration_curves(arms_df, arm_keys, quantity, out_path, facet_by=None):
    """
    Pooled calibration curves (coverage vs nominal) per arm for one quantity.

    Faceted by the ``facet_by`` grid variables; one line per remaining arm.
    Returns the path written, or None when there is nothing to plot.
    """
    from matplotlib import pyplot as plt

    facet_by = list(facet_by or [])
    sub = arms_df[arms_df["quantity"] == quantity]
    for c in STRATUM_COLS:
        sub = sub[sub[c] == ALL]
    if sub.empty:
        return None

    line_keys = [k for k in arm_keys if k not in facet_by]
    facets = ([((), sub)] if not facet_by
              else list(sub.groupby(facet_by, dropna=False, sort=True)))

    fig, axes = plt.subplots(1, len(facets), figsize=(4.6 * len(facets), 4.4),
                             squeeze=False)
    nominal = np.array(CURVE_LEVELS)
    for ax, (fvals, fdf) in zip(axes[0], facets):
        ax.plot([0.45, 1.0], [0.45, 1.0], color="gray", lw=1, ls="--")
        for _, row in fdf.iterrows():
            coverage = [row[f"coverage_{a:g}_mean"] for a in CURVE_LEVELS]
            ax.plot(nominal, coverage, marker="o", ms=3, lw=1.2,
                    label=_label(row, line_keys))
        fvals = fvals if isinstance(fvals, tuple) else (fvals,)
        ax.set_title(", ".join(f"{k}={v}" for k, v in zip(facet_by, fvals))
                     or quantity, fontsize=9)
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{quantity}: pooled calibration by arm", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def summarize_calibration(grid_dir,
                          out_prefix="tfs_calibration",
                          summary_subdir="summary",
                          replicate_keys=None,
                          baseline=None,
                          facet_by=None,
                          plot_quantity="theta_test",
                          regime_eps=0.01):
    """
    Summarize posterior calibration across a simulation grid.

    Parameters
    ----------
    grid_dir : str
        Directory of run subdirectories written by ``tfs-setup-sim-grid``
        (each with ``combo.json`` and, when finished, ``summary/``).
    out_prefix : str, optional
        Prefix for all outputs (default ``tfs_calibration``).
    summary_subdir : str, optional
        Subdirectory holding the ``tfs-summarize-fit`` outputs (default
        ``summary``).
    replicate_keys : list of str, optional
        Grid variables that index replicates rather than arms (default
        ``seed`` and ``fit_seed``).  Keys absent from the grid are ignored.
    baseline : list of str, optional
        ``key=value`` pairs selecting the baseline arm, e.g.
        ``guide_type=component guide_rank=None``.  When given, every other
        run is paired with the baseline run sharing all remaining grid
        variables (same simulated data, same fit seed).
    facet_by : list of str, optional
        Grid variables that split the calibration-curve plot into panels.
    plot_quantity : str, optional
        Quantity plotted (default ``theta_test``).
    regime_eps : float, optional
        θ within ``[regime_eps, 1 - regime_eps]`` is ``resolvable``
        (default 0.01).

    Outputs
    -------
    ``{out_prefix}_runs.csv``            per run x quantity x stratum
    ``{out_prefix}_run_status.csv``      per run: ok / what is missing
    ``{out_prefix}_arms.csv``            per arm: mean/std over replicates
    ``{out_prefix}_paired.csv``          per run: deltas vs baseline (if given)
    ``{out_prefix}_paired_summary.csv``  per arm: mean/std/se of deltas
    ``{out_prefix}_{plot_quantity}_calibration_curves.pdf``
    ``{out_prefix}_metadata.json``       resolved settings and counts
    """
    if replicate_keys is None:
        replicate_keys = ["seed", "fit_seed"]
    baseline_spec = parse_baseline(baseline)

    runs_df, status_df, grid_vars = summarize_grid_runs(
        grid_dir, summary_subdir=summary_subdir, regime_eps=regime_eps)

    # A baseline must name real grid variables and match at least one run in
    # the grid (finished or not) -- otherwise it is a typo.  Matching only
    # unfinished runs is normal mid-grid and just skips the paired outputs.
    if baseline_spec:
        unknown = [k for k in baseline_spec if k not in grid_vars]
        if unknown:
            raise ValueError(f"baseline keys {unknown} are not grid variables "
                             f"({grid_vars}).")
        in_grid = np.ones(len(status_df), dtype=bool)
        for key, value in baseline_spec.items():
            in_grid &= _matches(status_df[key], value).to_numpy()
        if not in_grid.any():
            raise ValueError(f"No run in {grid_dir} matches baseline {baseline_spec}.")

    ignored = [k for k in replicate_keys if k not in grid_vars]
    if ignored:
        warnings.warn(f"replicate keys {ignored} are not grid variables; ignored.")
    replicate_keys = [k for k in replicate_keys if k in grid_vars]
    arm_keys = [k for k in grid_vars if k not in replicate_keys]

    facet_by = list(facet_by or [])
    bad_facets = [k for k in facet_by if k not in arm_keys]
    if bad_facets:
        raise ValueError(f"facet_by keys {bad_facets} must be arm variables "
                         f"({arm_keys}).")

    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    n_ok = int((status_df["status"] == "ok").sum())
    print(f"{n_ok} of {len(status_df)} runs have calibration outputs.", flush=True)
    status_df.to_csv(f"{out_prefix}_run_status.csv", index=False)
    written = [f"{out_prefix}_run_status.csv"]

    metadata = {
        "grid_dir": os.path.abspath(grid_dir),
        "summary_subdir": summary_subdir,
        "grid_vars": grid_vars,
        "replicate_keys": replicate_keys,
        "arm_keys": arm_keys,
        "baseline": baseline_spec,
        "facet_by": facet_by,
        "regime_eps": regime_eps,
        "curve_levels": list(CURVE_LEVELS),
        "width_levels": list(WIDTH_LEVELS),
        "n_runs": int(len(status_df)),
        "n_runs_ok": n_ok,
    }

    if runs_df.empty:
        warnings.warn("No run has calibration outputs yet; only the run status "
                      "table was written.")
    else:
        sort_cols = [c for c in arm_keys + replicate_keys + ["quantity"]
                     if c in runs_df.columns]
        runs_df = runs_df.sort_values(sort_cols, kind="stable")
        runs_df.to_csv(f"{out_prefix}_runs.csv", index=False)
        written.append(f"{out_prefix}_runs.csv")

        arms_df = summarize_arms(runs_df, grid_vars, replicate_keys)
        arms_df.to_csv(f"{out_prefix}_arms.csv", index=False)
        written.append(f"{out_prefix}_arms.csv")

        base_done = np.ones(len(runs_df), dtype=bool)
        for key, value in baseline_spec.items():
            base_done &= _matches(runs_df[key], value).to_numpy()
        if baseline_spec and not base_done.any():
            warnings.warn(f"No finished run matches baseline {baseline_spec} yet; "
                          f"paired outputs skipped.")
        elif baseline_spec:
            paired = paired_differences(runs_df, grid_vars, baseline_spec)
            paired.to_csv(f"{out_prefix}_paired.csv", index=False)
            summarize_paired(paired, grid_vars, replicate_keys).to_csv(
                f"{out_prefix}_paired_summary.csv", index=False)
            written += [f"{out_prefix}_paired.csv",
                        f"{out_prefix}_paired_summary.csv"]
            metadata["n_pairs"] = int(len(paired))

        pdf = plot_calibration_curves(
            arms_df, arm_keys, plot_quantity,
            f"{out_prefix}_{plot_quantity}_calibration_curves.pdf",
            facet_by=facet_by)
        if pdf is not None:
            written.append(pdf)

    with open(f"{out_prefix}_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    written.append(f"{out_prefix}_metadata.json")

    for path in written:
        print(f"Wrote {path}", flush=True)
    return runs_df, status_df
