"""
Cross-run stability statistics for any quantile-summarized estimate table.

Given N independent estimates of the same quantity (e.g. runs that differ only
by random seed, or k-fold dropouts of the training data), this module measures
how much the runs disagree and whether that disagreement is explained by the
uncertainty each run reports. It works on *any* long-form table that carries a
point estimate and (optionally) a spread -- predicted features (theta, growth
rate, epistasis) and fitted parameters (``log_hill_K``, ``dk_geno``,
``growth_k``, ``k_ref``) alike.

Two axes are reported:

  Axis 1 -- reproducibility: how much the point estimates disagree across runs,
            in the estimate's native units (``rms_sd``, ``max_sd``).

  Axis 2 -- self-consistency: whether that disagreement is explained by each
            run's own reported uncertainty (``chi2``, ``dof``,
            ``overdispersion`` and its p/q values). This distinguishes
            "honestly uncertain" entities from the dangerous "overconfident and
            inconsistent" ones.

**No thresholds are applied and no grades are assigned.** Every quantity is
reported as a number so the caller can filter downstream (and record the cutline
where it belongs -- in the analysis, not in a lost command line). Note that
``rms_sd`` is in native units and is *not* comparable across different
parameters; ``overdispersion`` is the unit-free axis and is the one to reach for
when comparing across parameter types.

A caveat for parameters with skewed posteriors (``theta_low``/``theta_high``
pushed against 0/1, ``hill_n`` against its lower bound): the default sigma is
the symmetric quantile half-width ``(q0.841 - q0.159)/2``, which remains a fine
robust *scale* but biases the Gaussian-flavored Axis 2. ``rms_sd`` is unaffected.

Three key sets
--------------
The tool distinguishes three sets of columns, all recorded in the run metadata:

  * **match key** -- what makes a row "the same row" across runs. Auto-detected
    as every column shared by all runs that is not a value column
    (``q<level>``, ``y_obs``, ``y_std``) and not incidental (see
    ``_DROP_COLUMNS``). Override with ``match_by``.
  * **index_by** -- the entity being scored. Auto-detected as ``genotype``, else
    ``parameter``, else the sole match-key column if there is exactly one.
  * **report key** -- ``index_by + group_by``; one output row per distinct value.
    Must be a subset of the match key.

The columns in ``match key - report key`` are the *residual* axes: what each
output row pools over. Running without ``group_by`` pools an entity over its
whole sweep; adding ``group_by titrant_name titrant_conc`` zooms in to one row
per grid point. That zoom is not free -- see the ``group_by`` notes on
:func:`compare_runs`.

Comparison modes
----------------
Agreement is assessed in the estimate's absolute units (no registration or
rescaling of runs).

  * mean mode (``reference_df is None``): symmetric. The target is the cross-run
    mean; every run is treated equally. Intended for same-data/different-seed
    runs.

  * reference mode (``reference_df`` given): asymmetric. The reference run is the
    target and the N estimate runs are measured by their pooled deviation from
    it. Intended for k-fold dropout runs measured against a full-data fit.

Weights
-------
Every reduction in this module is written in weighted form against a per-
observation weight column, currently pinned to 1.0 everywhere. With uniform
weights the formulas reduce exactly to the unweighted ones (``ddof=1`` sample
variance, arithmetic means), so turning weighting on later changes no existing
number. See :func:`_row_statistics` for the algebra.

A caution for whoever wires weights up: on the **run** axis the natural weights
are *design* weights (fold size, run validity), never precision weights.
Inverse-variance weighting across seed/dropout runs would down-weight runs that
honestly report wide posteriors and tilt the result toward the overconfident
ones -- exactly the pathology Axis 2 exists to detect. Precision weights are
defensible on the *row* axis.

The public API operates on already-loaded DataFrames so it is trivially
testable; the ``tfs-compare-runs`` CLI wraps it with file IO.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from tfscreen.analysis.cat_response.cat_assess import benjamini_hochberg
from tfscreen.util.dataframe import resolve_obs_columns

# Preferred entity columns, in priority order, for auto-detecting ``index_by``.
_INDEX_CANDIDATES = ("genotype", "parameter")

# Columns that are never part of the match key: bookkeeping emitted by other
# tfs-* tools, and the sigma column resolve_obs_columns may synthesize.
_DROP_COLUMNS = frozenset({
    "in_training_data",
    "in_regime",
    "n_present",
    "n_runs",
    "n_rows",
    "n_eff",
    "mode",
    "_sigma",
})

# Below this many estimate runs, the sample standard deviation is too noisy to
# trust, so Axis 1 spread falls back to the half-range (max - min)/2.
_SMALL_N_CUTOFF = 5

# Output column order (the report key is prepended at write time).
_STAT_COLUMNS = [
    "n_runs", "n_present", "n_rows", "n_eff",
    "mode", "spread_estimator",
    "rms_sd", "max_sd", "mean_value", "dynamic_range",
    "mean_reported_sigma",
    "chi2", "dof", "overdispersion", "overdispersion_p", "overdispersion_q",
]


def _parse_quantile_columns(df):
    """
    Return ``{level: column_name}`` for columns named ``q<level>`` with a level
    strictly inside (0, 1) (e.g. ``'q0.159' -> 0.159``).
    """
    out = {}
    for c in df.columns:
        if not (isinstance(c, str) and c.startswith("q")):
            continue
        try:
            lvl = float(c[1:])
        except ValueError:
            continue
        if 0.0 < lvl < 1.0:
            out[lvl] = c
    return out


def _is_quantile_col(col):
    """True if ``col`` is a ``q<level>`` column with level in (0, 1)."""
    if not (isinstance(col, str) and col.startswith("q")):
        return False
    try:
        lvl = float(col[1:])
    except ValueError:
        return False
    return 0.0 < lvl < 1.0


def _is_incidental(col):
    """True if ``col`` is bookkeeping rather than an identity axis."""
    return col in _DROP_COLUMNS or (isinstance(col, str)
                                    and col.startswith("Unnamed:"))


def detect_match_keys(dfs, value_columns=(), match_by=None):
    """
    Determine the columns that make a row "the same row" across runs.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        All tables that must share the key (estimate runs, plus the reference
        when there is one).
    value_columns : iterable of str, optional
        Column names holding values rather than identity (the resolved
        ``y_obs``/``y_std``). Quantile columns are excluded automatically.
    match_by : list of str or str or None, optional
        Explicit override. Validated against the shared columns.

    Returns
    -------
    list of str
        The match-key columns, in the first table's column order.

    Raises
    ------
    ValueError
        If ``match_by`` names a column absent from some table, or if
        auto-detection leaves no candidate columns.
    """
    shared = set(dfs[0].columns)
    for df in dfs[1:]:
        shared &= set(df.columns)

    if match_by is not None:
        match_by = [match_by] if isinstance(match_by, str) else list(match_by)
        missing = [c for c in match_by if c not in shared]
        if missing:
            raise ValueError(
                f"match_by column(s) {missing} are not present in every "
                f"table. Columns shared by all tables: {sorted(shared)}"
            )
        return match_by

    value_columns = set(value_columns)
    keys = [c for c in dfs[0].columns
            if c in shared
            and c not in value_columns
            and not _is_incidental(c)
            and not _is_quantile_col(c)]

    if not keys:
        raise ValueError(
            "Could not auto-detect any match-key columns: every shared column "
            "looks like a value or bookkeeping column. Specify match_by "
            f"explicitly. Columns shared by all tables: {sorted(shared)}"
        )
    return keys


def detect_index_by(match_keys, index_by=None):
    """
    Determine the entity column being scored.

    Auto-detection tries ``genotype``, then ``parameter``, then falls back to
    the sole match-key column when there is exactly one.

    Parameters
    ----------
    match_keys : list of str
        The match-key columns.
    index_by : str or None, optional
        Explicit override; must be one of ``match_keys``.

    Returns
    -------
    str
        The entity column.

    Raises
    ------
    ValueError
        If the override is not a match key, or if auto-detection is ambiguous.
    """
    if index_by is not None:
        if index_by not in match_keys:
            raise ValueError(
                f"index_by='{index_by}' is not one of the match-key columns "
                f"{match_keys}."
            )
        return index_by

    for candidate in _INDEX_CANDIDATES:
        if candidate in match_keys:
            return candidate

    if len(match_keys) == 1:
        return match_keys[0]

    raise ValueError(
        f"Could not auto-detect the entity column: none of "
        f"{list(_INDEX_CANDIDATES)} is present and there is more than one "
        f"match-key column ({match_keys}). Specify index_by explicitly."
    )


def _resolve_report_keys(group_by, index_by, match_keys):
    """Validate ``group_by`` and return the report key ``index_by + group_by``."""
    if group_by is None:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]
    else:
        group_by = list(group_by)

    bad = [c for c in group_by if c not in match_keys]
    if bad:
        raise ValueError(
            f"group_by column(s) {bad} are not part of the match key "
            f"{match_keys}. Only columns that identify a row across runs can "
            f"be grouped on."
        )
    if index_by in group_by:
        raise ValueError(f"group_by must not repeat index_by ('{index_by}').")
    return [index_by] + group_by


def _prepare_runs(dfs, y_obs, y_std, point_quantile, sigma_quantiles):
    """
    Resolve the value/sigma columns on every table and check they agree.

    Returns
    -------
    (list of pandas.DataFrame, str, str or None)
        The resolved tables (possibly copies carrying a synthesized ``_sigma``),
        the resolved observable column, and the resolved sigma column (None when
        the tables carry no usable spread).
    """
    out, obs_names, std_names = [], set(), set()
    for i, df in enumerate(dfs):
        try:
            resolved, yo, ys = resolve_obs_columns(
                df, y_obs=y_obs, y_std=y_std,
                point_quantile=point_quantile,
                sigma_quantiles=sigma_quantiles,
            )
        except ValueError as exc:
            raise ValueError(f"Table {i}: {exc}") from exc
        out.append(resolved)
        obs_names.add(yo)
        std_names.add(ys)

    if len(obs_names) > 1:
        raise ValueError(
            f"Tables resolved to different observable columns: "
            f"{sorted(obs_names)}. Specify y_obs explicitly."
        )
    if len(std_names) > 1:
        raise ValueError(
            "Tables resolved to different uncertainty columns: "
            f"{sorted(str(s) for s in std_names)}. Some tables carry the "
            "sigma quantiles and others do not. Specify y_std explicitly."
        )
    return out, obs_names.pop(), std_names.pop()


def resolve_schema(dfs, *,
                   index_by=None,
                   group_by=None,
                   match_by=None,
                   y_obs=None,
                   y_std=None,
                   point_quantile=0.5,
                   sigma_quantiles=(0.159, 0.841)):
    """
    Resolve the value columns and the three key sets for a set of tables.

    This is the single place the schema is decided, so the CLI can record
    exactly what :func:`compare_runs` will use rather than re-deriving it.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        Every table that must share the schema (estimate runs, plus the
        reference when there is one).
    index_by, group_by, match_by, y_obs, y_std, point_quantile, sigma_quantiles
        See :func:`compare_runs`.

    Returns
    -------
    dict
        ``resolved`` (the tables with any synthesized sigma column), ``y_obs``,
        ``y_std``, ``match_by``, ``index_by``, ``group_by``, ``report_keys``,
        and ``residual``.
    """
    resolved, y_obs, y_std = _prepare_runs(
        dfs, y_obs, y_std, point_quantile, sigma_quantiles
    )
    match_keys = detect_match_keys(
        resolved,
        value_columns=[c for c in (y_obs, y_std) if c is not None],
        match_by=match_by,
    )
    index_by = detect_index_by(match_keys, index_by)
    report_keys = _resolve_report_keys(group_by, index_by, match_keys)

    return {
        "resolved": resolved,
        "y_obs": y_obs,
        "y_std": y_std,
        "match_by": match_keys,
        "index_by": index_by,
        "group_by": report_keys[1:],
        "report_keys": report_keys,
        "residual": [c for c in match_keys if c not in report_keys],
    }


def _check_unique(df, match_keys, label):
    """Fail fast if ``match_keys`` does not uniquely identify a row."""
    n_dup = int(df.duplicated(subset=match_keys).sum())
    if n_dup:
        example = df.loc[df.duplicated(subset=match_keys, keep=False),
                         match_keys].head(4)
        raise ValueError(
            f"{label}: the match key {match_keys} does not uniquely identify a "
            f"row ({n_dup} duplicate row(s)). A non-unique key silently turns "
            f"the cross-run comparison into a many-to-many join. Add the "
            f"missing identity column(s) via match_by, or de-duplicate the "
            f"input.\nFirst offending rows:\n{example.to_string(index=False)}"
        )


def _build_long(dfs, match_keys, y_obs, y_std):
    """
    Stack all runs into one long table tagged by run index.

    Returns a frame with ``match_keys + ['_run', 'value', 'sigma', '_w']``.
    Non-finite ``value`` rows are dropped with a warning -- they would otherwise
    poison every downstream reduction with silent NaNs.
    """
    frames = []
    for i, df in enumerate(dfs):
        run = df.loc[:, match_keys].copy()
        run["_run"] = i
        run["value"] = df[y_obs].to_numpy(dtype=float)
        run["sigma"] = (np.nan if y_std is None
                        else df[y_std].to_numpy(dtype=float))
        frames.append(run)

    long = pd.concat(frames, ignore_index=True)

    bad = ~np.isfinite(long["value"].to_numpy(dtype=float))
    if bad.any():
        warnings.warn(
            f"Dropped {int(bad.sum())} row(s) with a non-finite "
            f"'{y_obs}' value.",
            stacklevel=3,
        )
        long = long.loc[~bad].reset_index(drop=True)

    # Uniform weights for now. Every reduction below is written in weighted
    # form so per-run or per-row weights can be substituted here without
    # touching the statistics (see the module docstring).
    long["_w"] = 1.0
    return long


def _row_statistics(long, n_rows, estimator, is_reference):
    """
    Collapse the long table to one record per matched row.

    All reductions are weighted, with ``V1 = sum(w)`` and ``V2 = sum(w**2)``:

      * target   ``mu = sum(w*v) / V1``               (mean mode)
      * spread   ``sqrt(SS / (V1 - V2/V1))``          (mean mode, 'std')
      * spread   ``(max(v) - min(v)) / 2``            (mean mode, half-range)
      * spread   ``sqrt(SS / V1)``                    (reference mode)

    where ``SS = sum(w * dev**2)``. At ``w = 1`` the 'std' denominator is
    ``n - 1`` (the ddof=1 sample variance) and the reference spread is the plain
    RMS deviation, so uniform weights reproduce the unweighted formulas exactly.
    The half-range is a range statistic and ignores weights by construction.

    Returns
    -------
    dict
        ``mu``, ``spread``, ``V1``, ``V2`` -- each a length-``n_rows`` array.
    """
    rid = long["_row"].to_numpy()
    w = long["_w"].to_numpy(dtype=float)
    v = long["value"].to_numpy(dtype=float)

    V1 = np.bincount(rid, weights=w, minlength=n_rows)
    V2 = np.bincount(rid, weights=w * w, minlength=n_rows)

    if is_reference:
        # The target is the reference value, constant within a matched row.
        mu = np.full(n_rows, np.nan)
        mu[rid] = long["value_ref"].to_numpy(dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(
                V1 > 0,
                np.bincount(rid, weights=w * v, minlength=n_rows) / V1,
                np.nan,
            )

    dev = long["_dev"].to_numpy(dtype=float)
    ss = np.bincount(rid, weights=w * dev * dev, minlength=n_rows)

    # More than one effective observation. This is the ddof=1 denominator, and
    # it also guards the half-range: a row seen in a single run has an
    # *undefined* spread, not a zero one.
    denom = V1 - np.divide(V2, V1, out=np.full(n_rows, np.nan), where=V1 > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        if is_reference:
            spread = np.sqrt(np.where(V1 > 0, ss / V1, np.nan))
        elif estimator == "std":
            spread = np.sqrt(np.where(denom > 0, ss / denom, np.nan))
        else:
            hi = np.full(n_rows, -np.inf)
            lo = np.full(n_rows, np.inf)
            np.maximum.at(hi, rid, v)
            np.minimum.at(lo, rid, v)
            spread = np.where(np.isfinite(hi) & np.isfinite(lo) & (denom > 0),
                              (hi - lo) / 2.0, np.nan)

    return {"mu": mu, "spread": spread, "V1": V1, "V2": V2}


def _group_statistics(long, group_frame, row_stats, grp_of_row, n_groups,
                      has_residual, is_reference):
    """
    Pool matched-row statistics and per-observation chi-square onto report groups.

    Axis 1 (``rms_sd``, ``max_sd``, ``mean_value``, ``dynamic_range``) comes from
    the per-row table; Axis 2 (``chi2``, ``dof``, ``overdispersion``) from the
    per-observation table.

    ``dof`` generalizes "each matched row spends one degree of freedom
    estimating its own mean" to weights: ``sum_rows (V1 - V2/V1)`` in mean mode
    and ``sum_rows V1`` in reference mode, restricted to the observations that
    actually contribute a chi-square term. At ``w = 1`` these are
    ``n_terms - n_rows`` and ``n_terms``.
    """
    spread = row_stats["spread"]
    mu = row_stats["mu"]
    n_rows_tot = spread.shape[0]

    # --- Axis 1: pooled over the matched rows in each group -----------------
    # Uniform row weights; the hook for a future row_weight_col goes here.
    rw = np.ones(n_rows_tot)
    finite_spread = np.isfinite(spread)
    sw = np.bincount(grp_of_row[finite_spread], weights=rw[finite_spread],
                     minlength=n_groups)
    ss = np.bincount(grp_of_row[finite_spread],
                     weights=rw[finite_spread] * spread[finite_spread] ** 2,
                     minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_sd = np.sqrt(np.where(sw > 0, ss / sw, np.nan))

    max_sd = np.full(n_groups, -np.inf)
    np.maximum.at(max_sd, grp_of_row[finite_spread], spread[finite_spread])
    max_sd = np.where(np.isfinite(max_sd), max_sd, np.nan)

    finite_mu = np.isfinite(mu)
    mw = np.bincount(grp_of_row[finite_mu], weights=rw[finite_mu],
                     minlength=n_groups)
    msum = np.bincount(grp_of_row[finite_mu],
                       weights=rw[finite_mu] * mu[finite_mu],
                       minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_value = np.where(mw > 0, msum / mw, np.nan)

    # Dynamic range is the spread of the target over the *residual* axes. When
    # the report key is the whole match key there is no residual axis to sweep,
    # so the quantity is undefined rather than zero. Note this pools across all
    # residual axes at once: to get a per-stratum range (e.g. within each
    # titrant_name), put the stratifier in group_by.
    if has_residual:
        hi = np.full(n_groups, -np.inf)
        lo = np.full(n_groups, np.inf)
        np.maximum.at(hi, grp_of_row[finite_mu], mu[finite_mu])
        np.minimum.at(lo, grp_of_row[finite_mu], mu[finite_mu])
        dynamic_range = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo,
                                 np.nan)
    else:
        dynamic_range = np.full(n_groups, np.nan)

    # Pooling depth for Axis 1: matched rows that actually contributed a
    # spread. A row present in only one run has no spread, so it is not counted
    # here (it also contributes zero degrees of freedom below).
    n_rows_per_group = np.bincount(grp_of_row[finite_spread],
                                   minlength=n_groups)

    # --- Axis 2: pooled over the observations in each group -----------------
    rid = long["_row"].to_numpy()
    gid = grp_of_row[rid]
    w = long["_w"].to_numpy(dtype=float)
    dev = long["_dev"].to_numpy(dtype=float)
    var = long["_var"].to_numpy(dtype=float)
    sigma = long["sigma"].to_numpy(dtype=float)

    valid = (np.isfinite(dev) & np.isfinite(var) & (var > 0)
             & np.isfinite(w))
    safe_var = np.where(valid, var, 1.0)
    term = np.where(valid, w * dev * dev / safe_var, 0.0)

    chi2 = np.bincount(gid[valid], weights=term[valid],
                       minlength=n_groups).astype(float)
    v1_valid = np.bincount(gid[valid], weights=w[valid],
                           minlength=n_groups).astype(float)
    v2_valid = np.bincount(gid[valid], weights=w[valid] ** 2,
                           minlength=n_groups).astype(float)
    n_valid = np.bincount(gid[valid], minlength=n_groups)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(v2_valid > 0, v1_valid ** 2 / v2_valid, np.nan)

    if is_reference:
        dof = v1_valid.astype(float)
    else:
        # One degree of freedom per matched row goes to that row's own mean.
        v1_row = np.bincount(rid[valid], weights=w[valid],
                             minlength=n_rows_tot).astype(float)
        v2_row = np.bincount(rid[valid], weights=w[valid] ** 2,
                             minlength=n_rows_tot).astype(float)
        per_row_dof = v1_row - np.divide(
            v2_row, v1_row, out=np.zeros(n_rows_tot, dtype=float),
            where=v1_row > 0
        )
        dof = np.bincount(grp_of_row, weights=per_row_dof, minlength=n_groups)

    with np.errstate(divide="ignore", invalid="ignore"):
        sig_sum = np.bincount(gid[valid], weights=w[valid] * sigma[valid],
                              minlength=n_groups).astype(float)
        mean_sigma = np.where(v1_valid > 0, sig_sum / v1_valid, np.nan)

    chi2 = np.where(n_valid > 0, chi2, np.nan)
    dof = np.where(n_valid > 0, dof, np.nan)
    good = np.isfinite(chi2) & np.isfinite(dof) & (dof > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        overdispersion = np.where(good, chi2 / np.where(good, dof, 1.0),
                                  np.nan)
        p = np.where(good,
                     stats.chi2.sf(np.where(good, chi2, 0.0),
                                   np.where(good, dof, 1.0)),
                     np.nan)

    out = group_frame.copy()
    out["n_rows"] = n_rows_per_group
    out["n_eff"] = n_eff
    out["rms_sd"] = rms_sd
    out["max_sd"] = max_sd
    out["mean_value"] = mean_value
    out["dynamic_range"] = dynamic_range
    out["mean_reported_sigma"] = mean_sigma
    out["chi2"] = chi2
    out["dof"] = dof
    out["overdispersion"] = overdispersion
    out["overdispersion_p"] = p
    out["overdispersion_q"] = benjamini_hochberg(p)
    return out


def compare_runs(estimate_dfs,
                 reference_df=None,
                 *,
                 index_by=None,
                 group_by=None,
                 match_by=None,
                 y_obs=None,
                 y_std=None,
                 point_quantile=0.5,
                 sigma_quantiles=(0.159, 0.841)):
    """
    Measure cross-run agreement of an estimate, one row per report key.

    See the module docstring for the two axes, the three key sets, and the two
    comparison modes. No thresholds are applied: every quantity is reported as a
    number for downstream filtering.

    Parameters
    ----------
    estimate_dfs : list of pandas.DataFrame
        The N estimate tables. In mean mode at least two are required; in
        reference mode at least one.
    reference_df : pandas.DataFrame or None, optional
        If given, switches to reference mode: each estimate run is measured by
        its deviation from this table (which is excluded from the mutual
        comparison). Rows absent from the reference are dropped with a warning.
    index_by : str or None, optional
        The entity column. Default None (auto-detect: ``genotype``, else
        ``parameter``, else the sole match-key column).
    group_by : list of str or str or None, optional
        Extra report-key columns, breaking each entity out further. Must be part
        of the match key.

        This is a statistical *zoom*, not a free refinement. Without it an
        entity's ``rms_sd`` pools every matched row it appears in; grouping down
        to individual grid points leaves only ``n_runs - 1`` degrees of freedom
        per row, where the relative error on a standard deviation is
        ``1/sqrt(2*dof)`` (~35% at 4 dof). The ``n_rows`` and ``n_eff`` columns
        report the pooling depth so this stays visible.
    match_by : list of str or str or None, optional
        Explicit match key. Default None (auto-detect; see
        :func:`detect_match_keys`).
    y_obs : str or None, optional
        Point-estimate column. Default None, which uses the ``point_quantile``
        column.
    y_std : str or None, optional
        Uncertainty column. Default None, which uses the symmetric quantile
        half-width from ``sigma_quantiles``. When neither is available, Axis 2
        is reported as NaN.
    point_quantile : float, optional
        Quantile used as the point estimate. Default 0.5.
    sigma_quantiles : tuple of float, optional
        ``(lo, hi)`` quantiles bracketing one sigma. Default (0.159, 0.841).

    Returns
    -------
    pandas.DataFrame
        One row per report key, sorted by ``rms_sd`` ascending. Columns: the
        report key, then ``n_runs``, ``n_present``, ``n_rows``, ``n_eff``,
        ``mode``, ``spread_estimator``, ``rms_sd``, ``max_sd``, ``mean_value``,
        ``dynamic_range``, ``mean_reported_sigma``, ``chi2``, ``dof``,
        ``overdispersion``, ``overdispersion_p``, ``overdispersion_q``.

        ``n_present`` is how many runs contributed; compare it to ``n_runs`` for
        coverage. ``n_rows`` is the Axis-1 pooling depth -- how many matched rows
        actually contributed a spread, which excludes any row present in only
        one run. ``n_eff`` is
        the Kish effective count of run-by-row observations behind Axis 2 (equal
        to the raw count under uniform weights). ``dynamic_range`` is NaN when
        the report key is the whole match key (no residual axis to sweep).

    Raises
    ------
    ValueError
        If too few runs are supplied, if the match key cannot be resolved or is
        not unique within a run, or if required columns are missing.
    """
    if not isinstance(estimate_dfs, (list, tuple)):
        raise ValueError(
            "estimate_dfs must be a list of DataFrames, "
            f"not {type(estimate_dfs).__name__}."
        )

    mode = "mean" if reference_df is None else "reference"
    min_estimates = 2 if mode == "mean" else 1
    if len(estimate_dfs) < min_estimates:
        raise ValueError(
            f"{mode} mode requires at least {min_estimates} estimate "
            f"table(s); got {len(estimate_dfs)}."
        )

    n_runs = len(estimate_dfs)
    is_reference = mode == "reference"

    all_dfs = list(estimate_dfs) + ([reference_df] if is_reference else [])
    schema = resolve_schema(all_dfs,
                            index_by=index_by, group_by=group_by,
                            match_by=match_by, y_obs=y_obs, y_std=y_std,
                            point_quantile=point_quantile,
                            sigma_quantiles=sigma_quantiles)
    resolved = schema["resolved"]
    y_obs, y_std = schema["y_obs"], schema["y_std"]
    match_keys = schema["match_by"]
    report_keys = schema["report_keys"]
    residual = schema["residual"]

    for i, df in enumerate(resolved):
        label = ("Reference table"
                 if is_reference and i == len(resolved) - 1
                 else f"Estimate run {i}")
        _check_unique(df, match_keys, label)

    long = _build_long(resolved[:n_runs], match_keys, y_obs, y_std)
    if long.empty:
        return _empty_result(report_keys)

    if is_reference:
        ref = resolved[-1].loc[:, match_keys].copy()
        ref["value_ref"] = resolved[-1][y_obs].to_numpy(dtype=float)
        ref["sigma_ref"] = (np.nan if y_std is None
                            else resolved[-1][y_std].to_numpy(dtype=float))
        before = len(long)
        long = long.merge(ref, on=match_keys, how="inner")
        if len(long) < before:
            print(
                f"Warning: dropped {before - len(long)} row(s) absent from "
                f"the reference table.",
                flush=True,
            )
        if long.empty:
            return _empty_result(report_keys)
        long["_var"] = long["sigma"] ** 2 + long["sigma_ref"] ** 2
    else:
        long["_var"] = long["sigma"] ** 2

    long["_row"] = long.groupby(match_keys, sort=False, dropna=False).ngroup()
    n_rows_tot = int(long["_row"].max()) + 1

    if is_reference:
        long["_dev"] = long["value"] - long["value_ref"]
    else:
        rid = long["_row"].to_numpy()
        w = long["_w"].to_numpy(dtype=float)
        v = long["value"].to_numpy(dtype=float)
        v1 = np.bincount(rid, weights=w, minlength=n_rows_tot)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_mean = np.where(
                v1 > 0,
                np.bincount(rid, weights=w * v, minlength=n_rows_tot) / v1,
                np.nan,
            )
        long["_dev"] = v - row_mean[rid]

    estimator = "std" if n_runs >= _SMALL_N_CUTOFF else "half_range"
    spread_label = "rms_dev" if is_reference else estimator
    row_stats = _row_statistics(long, n_rows_tot, estimator, is_reference)

    # One record per matched row, carrying the identity columns.
    rows = long.groupby("_row", sort=True)[match_keys].first()
    if len(report_keys) > 1:
        report_index = pd.MultiIndex.from_frame(rows.loc[:, report_keys])
    else:
        report_index = pd.Index(rows[report_keys[0]])
    grp_of_row, grp_values = pd.factorize(report_index, sort=False)
    n_groups = len(grp_values)

    if len(report_keys) > 1:
        group_frame = pd.DataFrame(list(grp_values), columns=report_keys)
    else:
        group_frame = pd.DataFrame({report_keys[0]: np.asarray(grp_values)})

    result = _group_statistics(long, group_frame, row_stats, grp_of_row,
                               n_groups, bool(residual), is_reference)

    # Coverage: how many distinct runs contributed to each report group.
    n_present = (long.assign(_grp=grp_of_row[long["_row"].to_numpy()])
                     .groupby("_grp", sort=True)["_run"].nunique())
    result["n_runs"] = n_runs
    result["n_present"] = n_present.reindex(range(n_groups)).to_numpy()
    result["mode"] = mode
    result["spread_estimator"] = spread_label

    result = result.loc[:, report_keys + _STAT_COLUMNS]
    result = result.sort_values("rms_sd", kind="mergesort")
    return result.reset_index(drop=True)


def _empty_result(report_keys):
    """Return an empty result frame with the standard columns."""
    return pd.DataFrame(columns=list(report_keys) + _STAT_COLUMNS)


def _monotone_knots(values, probs):
    """
    Collapse a (non-decreasing) quantile ladder to strictly increasing knots.

    Ties in ``values`` (a flat region of the quantile function, i.e. a point
    mass) are merged, keeping the highest cumulative probability at that value
    -- the CDF value just above the mass.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Strictly increasing ``values`` and their (non-decreasing) ``probs``.
    """
    keep_v = []
    keep_p = []
    for x, p in zip(values, probs):
        if keep_v and x <= keep_v[-1]:
            keep_p[-1] = p  # probs are ascending, so this keeps the max
        else:
            keep_v.append(x)
            keep_p.append(p)
    return np.asarray(keep_v, dtype=float), np.asarray(keep_p, dtype=float)


def _mixture_quantiles(value_matrix, probs, weights=None):
    """
    Quantiles of the weighted mixture of N per-run marginal posteriors.

    Each run contributes a marginal distribution described by its quantile
    ladder (``probs`` -> ``value_matrix[i]``). The mixture "pick a run at random
    (with probability proportional to its weight), then draw from its posterior"
    has CDF ``F_mix = sum_i w_i F_i / sum_i w_i``; its variance is the within-run
    variance plus the between-run spread of the point estimates (law of total
    variance). This reads the mixture's quantiles at the same ``probs`` levels by
    averaging the per-run CDFs on the pooled support grid and inverting.

    Parameters
    ----------
    value_matrix : numpy.ndarray, shape (n_present, K)
        Each row is one run's non-decreasing quantile values at ``probs``.
    probs : numpy.ndarray, shape (K,)
        Ascending probability levels in (0, 1), shared across runs.
    weights : array-like, shape (n_present,), optional
        Per-run mixture weights. Default None (equal weights). See the module
        docstring on why these should be design weights, not precision weights.

    Returns
    -------
    numpy.ndarray, shape (K,)
        Mixture quantile values at ``probs``.
    """
    n = value_matrix.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,):
            raise ValueError(
                f"weights must have shape ({n},); got {weights.shape}."
            )

    grid = np.unique(value_matrix)
    if grid.size == 1:
        # Every run is a point mass at the same value -> mixture is that mass.
        return np.full(probs.shape, grid[0], dtype=float)

    # Average the per-run CDFs on the pooled support grid.
    f_sum = np.zeros(grid.size)
    for i in range(n):
        xs, ps = _monotone_knots(value_matrix[i], probs)
        if xs.size == 1:
            # Point-mass run: step from 0 to 1 at its single value.
            f_i = np.where(grid < xs[0], 0.0, 1.0)
        else:
            f_i = np.interp(grid, xs, ps, left=0.0, right=1.0)
        f_sum += weights[i] * f_i
    f_mix = f_sum / weights.sum()

    # Invert F_mix. Collapse ties (flat CDF = zero-density gaps) keeping the
    # smallest value for each CDF level, matching the standard quantile def.
    uf, idx = np.unique(f_mix, return_index=True)
    return np.interp(probs, uf, grid[idx])


def shared_quantile_levels(estimate_dfs):
    """
    Return the quantile levels present in every table, ascending.

    Returns an empty list when the tables share fewer than two levels -- too few
    to reconstruct a distribution to mix.
    """
    levels = None
    for df in estimate_dfs:
        lv = set(_parse_quantile_columns(df))
        levels = lv if levels is None else (levels & lv)
    levels = sorted(levels or [])
    return levels if len(levels) >= 2 else []


def aggregate_runs(estimate_dfs, *, match_by=None, progress_every=200_000):
    """
    Combine N estimates into one aggregate table, one row per matched row.

    For every matched row the N runs present are combined as an **equal-weight
    mixture** of their per-run marginal posteriors (reconstructed from the stored
    quantile ladder). The aggregate error thus folds in both each run's own
    posterior width *and* the run-to-run spread of the point estimates (law of
    total variance), and does **not** shrink with N -- appropriate because
    different-seed runs are not independent replicates. A bias shared by all N
    runs is invisible to this (only sampled variation -- seed or training-data
    dropout -- is captured).

    This is a row-level combination, not a summary, so it is always keyed on the
    match key and is unaffected by ``group_by``. In reference mode on the CLI
    side, only the N estimate runs are mixed; a reference run is never folded in
    (it is a comparison target, not a sample).

    Parameters
    ----------
    estimate_dfs : list of pandas.DataFrame
        The N estimate tables. Each must carry a quantile ladder (``q<level>``
        columns); the mixture is taken over the levels present in *all* runs.
    match_by : list of str or str or None, optional
        Explicit match key. Default None (auto-detect).
    progress_every : int, optional
        Print a progress line every this many matched-row groups. Default
        200000.

    Returns
    -------
    pandas.DataFrame
        Long-form, one row per matched row, with the shared ``q<level>`` columns
        holding the mixture quantiles and an ``n_present`` column (how many runs
        contributed). Sorted by the match key.

    Raises
    ------
    ValueError
        If fewer than two runs are supplied or the runs share fewer than two
        quantile levels.
    """
    if not isinstance(estimate_dfs, (list, tuple)):
        raise ValueError(
            "estimate_dfs must be a list of DataFrames, "
            f"not {type(estimate_dfs).__name__}."
        )
    if len(estimate_dfs) < 2:
        raise ValueError(
            f"aggregate_runs requires at least 2 estimate tables; "
            f"got {len(estimate_dfs)}."
        )

    levels = shared_quantile_levels(estimate_dfs)
    if not levels:
        raise ValueError(
            "Estimate tables share fewer than 2 quantile (q<level>) columns; "
            "cannot reconstruct per-run distributions to mix."
        )
    probs = np.asarray(levels, dtype=float)
    qmap0 = _parse_quantile_columns(estimate_dfs[0])
    out_cols = [qmap0[lvl] for lvl in levels]

    keys = detect_match_keys(estimate_dfs, match_by=match_by)

    # Stack keys and the aligned value matrix across all runs.
    key_frames = []
    value_blocks = []
    for df in estimate_dfs:
        qmap = _parse_quantile_columns(df)
        key_frames.append(df.loc[:, keys])
        value_blocks.append(
            df.loc[:, [qmap[lvl] for lvl in levels]].to_numpy(dtype=float)
        )
    long_keys = pd.concat(key_frames, ignore_index=True)
    values = np.vstack(value_blocks)

    groups = long_keys.groupby(keys, sort=False).indices
    group_keys = list(groups.keys())
    n_groups = len(group_keys)

    out_values = np.empty((n_groups, len(levels)), dtype=float)
    n_present = np.empty(n_groups, dtype=int)
    for gi, gkey in enumerate(group_keys):
        pos = groups[gkey]
        n_present[gi] = pos.size
        out_values[gi] = _mixture_quantiles(values[pos], probs)
        if progress_every and (gi + 1) % progress_every == 0:
            print(f"  aggregated {gi + 1}/{n_groups} groups...", flush=True)

    if len(keys) == 1:
        out = pd.DataFrame({keys[0]: group_keys})
    else:
        out = pd.DataFrame(group_keys, columns=keys)
    for j, col in enumerate(out_cols):
        out[col] = out_values[:, j]
    out["n_present"] = n_present
    return out.sort_values(keys).reset_index(drop=True)
