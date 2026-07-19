"""
Cross-run stability grading for per-genotype theta estimates.

Given N independent theta estimates of the same library (e.g. runs that differ
only by random seed, or k-fold dropouts of the binding training data), this
module scores every genotype on two independent axes and assigns a graded
stability tier:

  Axis 1 -- reproducibility: how much the point estimates (``q0.5``) of theta
            disagree across runs, measured in native theta units. This is the
            axis the tier (A/B/C/D) is graded on.

  Axis 2 -- self-consistency: whether the run-to-run disagreement is explained
            by each run's own reported uncertainty (derived from the stored
            quantiles). Reported as an ``overdispersion`` statistic and a flag,
            *not* folded into the tier -- it distinguishes "honestly uncertain"
            genotypes from the dangerous "overconfident and inconsistent" ones.

Agreement is assessed in absolute theta units (no registration/rescaling of
runs). Two comparison modes are supported:

  * mean mode (``reference_df is None``): symmetric. The target is the cross-run
    mean; every run is treated equally. Intended for same-data/different-seed
    runs.

  * reference mode (``reference_df`` given): asymmetric. The reference run is the
    target and the N estimate runs are measured by their pooled deviation from
    it. Intended for k-fold dropout runs measured against a full-data fit.

The public API operates on already-loaded DataFrames so it is trivially
testable; the ``tfs-compare-theta`` CLI wraps it with file IO.
"""

import numpy as np
import pandas as pd

# Genotype-plus-condition key columns, in priority order. ``titrant_name`` is
# only included when present in the input (see ``_detect_keys``).
_GENOTYPE_KEY = "genotype"
_NAME_KEY = "titrant_name"
_CONC_KEY = "titrant_conc"

# Below this many estimate runs, the sample standard deviation is too noisy to
# trust, so Axis 1 spread falls back to the half-range (max - min)/2.
_SMALL_N_CUTOFF = 5


def _quantile_col(q):
    """Column name storing quantile ``q`` (e.g. 0.159 -> 'q0.159')."""
    return f"q{q}"


def _detect_keys(df):
    """
    Return the join-key columns for an estimate table.

    ``[genotype, titrant_name, titrant_conc]`` when ``titrant_name`` is present,
    otherwise ``[genotype, titrant_conc]``.

    Parameters
    ----------
    df : pandas.DataFrame
        An estimate table.

    Returns
    -------
    list of str
        The key columns, guaranteed to include ``genotype`` and
        ``titrant_conc``.

    Raises
    ------
    ValueError
        If ``genotype`` or ``titrant_conc`` is missing.
    """
    missing = [c for c in (_GENOTYPE_KEY, _CONC_KEY) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Estimate table is missing required key column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    keys = [_GENOTYPE_KEY]
    if _NAME_KEY in df.columns:
        keys.append(_NAME_KEY)
    keys.append(_CONC_KEY)
    return keys


def _condition_grid(df, keys):
    """Return the set of unique condition-key tuples (all keys except genotype)."""
    cond_keys = [k for k in keys if k != _GENOTYPE_KEY]
    return set(map(tuple, df.loc[:, cond_keys].drop_duplicates().to_numpy()))


def _extract_run(df, keys, point_quantile, sigma_quantiles):
    """
    Reduce one estimate table to key columns plus ``theta`` and ``sigma``.

    ``theta`` is the ``point_quantile`` column (median by default); ``sigma`` is
    the symmetric 1-sigma half-width ``(q_hi - q_lo)/2`` from ``sigma_quantiles``.

    Parameters
    ----------
    df : pandas.DataFrame
        An estimate table with the standard quantile columns.
    keys : list of str
        Key columns (from ``_detect_keys``).
    point_quantile : float
        Quantile used as the point estimate (e.g. 0.5).
    sigma_quantiles : tuple of float
        ``(lo, hi)`` quantiles bracketing one sigma (e.g. (0.159, 0.841)).

    Returns
    -------
    pandas.DataFrame
        Columns ``keys + ['theta', 'sigma']``.

    Raises
    ------
    ValueError
        If any required quantile column is absent.
    """
    lo, hi = sigma_quantiles
    point_col = _quantile_col(point_quantile)
    lo_col = _quantile_col(lo)
    hi_col = _quantile_col(hi)

    needed = [point_col, lo_col, hi_col]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(
            f"Estimate table is missing required quantile column(s): {absent}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.loc[:, keys].copy()
    out["theta"] = df[point_col].to_numpy(dtype=float)
    out["sigma"] = (df[hi_col].to_numpy(dtype=float)
                    - df[lo_col].to_numpy(dtype=float)) / 2.0
    return out


def _grid_label(df, keys):
    """
    Return a per-row string label for the condition (non-genotype) keys.

    Used to name the per-grid-point ``sd_*`` output columns. Includes
    ``titrant_name`` when it is one of the keys.
    """
    conc = df[_CONC_KEY].astype(str)
    if _NAME_KEY in keys:
        return "sd[" + df[_NAME_KEY].astype(str) + "," + conc + "]"
    return "sd[" + conc + "]"


def _assign_tier(rms_sd, n_present, n_runs, min_coverage, sd_tier_edges):
    """
    Map a genotype's Axis-1 score to a stability tier.

    Genotypes present in fewer than ``ceil(min_coverage * n_runs)`` runs are
    labelled ``low_coverage`` (not graded). Otherwise ``rms_sd`` is binned by
    ``sd_tier_edges`` into ``A`` (best) through ``D`` (worst).

    Parameters
    ----------
    rms_sd : float
        Root-mean-square run-to-run spread over the condition grid.
    n_present : int
        Number of runs in which the genotype appears.
    n_runs : int
        Total number of estimate runs.
    min_coverage : float
        Minimum fraction of runs a genotype must appear in to be graded.
    sd_tier_edges : sequence of float
        Three ascending cutlines separating tiers A|B, B|C, C|D.

    Returns
    -------
    str
        One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'low_coverage'``.
    """
    min_runs = int(np.ceil(min_coverage * n_runs))
    if n_present < min_runs:
        return "low_coverage"

    if not np.isfinite(rms_sd):
        return "low_coverage"

    e0, e1, e2 = sd_tier_edges
    if rms_sd < e0:
        return "A"
    if rms_sd < e1:
        return "B"
    if rms_sd < e2:
        return "C"
    return "D"


def compare_theta(estimate_dfs,
                  reference_df=None,
                  *,
                  min_coverage=0.5,
                  sd_tier_edges=(0.02, 0.05, 0.10),
                  overdispersion_threshold=2.0,
                  point_quantile=0.5,
                  sigma_quantiles=(0.159, 0.841)):
    """
    Grade per-genotype theta stability across N independent estimate runs.

    See the module docstring for the two axes and the two comparison modes.

    Parameters
    ----------
    estimate_dfs : list of pandas.DataFrame
        The N estimate tables. Each must share the same key columns
        (``[genotype, titrant_conc]`` or with ``titrant_name``) and the standard
        quantile columns (at least ``point_quantile`` and ``sigma_quantiles``).
        In mean mode at least two are required.
    reference_df : pandas.DataFrame or None, optional
        If given, switches to reference mode: each estimate run is measured by
        its deviation from this table (excluded from the mutual comparison).
        Genotypes absent from the reference are dropped with a warning.
    min_coverage : float, optional
        Minimum fraction of estimate runs a genotype must appear in to be
        graded. Default 0.5.
    sd_tier_edges : tuple of float, optional
        Ascending cutlines on ``rms_sd`` for tiers A|B, B|C, C|D. Default
        ``(0.02, 0.05, 0.10)``.
    overdispersion_threshold : float, optional
        ``overdispersion`` above this sets the ``overdispersed`` flag. Default
        2.0. Does not affect the tier.
    point_quantile : float, optional
        Quantile used as the point estimate. Default 0.5.
    sigma_quantiles : tuple of float, optional
        ``(lo, hi)`` quantiles bracketing one sigma. Default (0.159, 0.841).

    Returns
    -------
    pandas.DataFrame
        One row per genotype, sorted by ``rms_sd``. Columns: ``genotype``,
        ``n_present``, ``mode``, ``tier``, ``rms_sd``, ``max_sd``,
        ``dynamic_range``, ``overdispersion``, ``overdispersed``,
        ``mean_reported_sigma``, ``spread_estimator``, and one ``sd[...]``
        column per condition-grid point.

        The ``tier`` is the graded Axis-1 label. When the result is passed to
        :func:`stability_crosstabs`, the tiers collapse onto the crosstab rows
        as follows:

          * ``A`` and ``B``  -> ``reproducible``
          * ``C`` and ``D``  -> ``unstable``
          * ``low_coverage`` -> dropped (never graded, so not in any crosstab)

        The crosstab *columns* come from the other two per-genotype fields, not
        from the tier: ``overdispersed`` (``overconfident`` vs ``consistent``)
        for ``tier_vs_overdispersion``, and ``dynamic_range`` (``informative``
        vs ``flat``) for ``tier_vs_dynamic_range``.

    Raises
    ------
    ValueError
        If fewer than the minimum number of runs are supplied, if the key sets
        disagree across runs, or if required columns are missing.
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

    # Detect keys from the first run and require every other table (including
    # the reference) to agree -- a silent key mismatch would corrupt the join.
    keys = _detect_keys(estimate_dfs[0])
    for i, df in enumerate(estimate_dfs[1:], start=1):
        if _detect_keys(df) != keys:
            raise ValueError(
                f"Estimate run {i} has key columns {_detect_keys(df)}, "
                f"which differ from run 0's {keys}."
            )
    if reference_df is not None and _detect_keys(reference_df) != keys:
        raise ValueError(
            f"Reference table has key columns {_detect_keys(reference_df)}, "
            f"which differ from the estimate runs' {keys}."
        )

    # The condition grid (concentrations, and titrant names if present) must
    # agree across runs. Missing *genotypes* are fine -- coverage handles those
    # -- but a mismatched concentration grid signals unit/config errors.
    grid0 = _condition_grid(estimate_dfs[0], keys)
    tables = list(estimate_dfs) + (
        [reference_df] if reference_df is not None else []
    )
    for i, df in enumerate(tables[1:], start=1):
        if _condition_grid(df, keys) != grid0:
            raise ValueError(
                f"Table {i} has a different condition grid (titrant "
                f"concentrations/names) than run 0. All runs must share the "
                f"same grid."
            )

    estimator = "std" if n_runs >= _SMALL_N_CUTOFF else "half_range"

    # Stack all runs long, tagged by run index.
    runs = []
    for i, df in enumerate(estimate_dfs):
        run = _extract_run(df, keys, point_quantile, sigma_quantiles)
        run["_run"] = i
        runs.append(run)
    long = pd.concat(runs, ignore_index=True)

    # Coverage is measured over the estimate runs, before any inner filtering.
    n_present = long.groupby(_GENOTYPE_KEY)["_run"].nunique()

    if mode == "reference":
        ref = _extract_run(reference_df, keys, point_quantile, sigma_quantiles)
        ref = ref.rename(columns={"theta": "theta_ref", "sigma": "sigma_ref"})
        before = long[_GENOTYPE_KEY].nunique()
        long = long.merge(ref, on=keys, how="inner")
        after = long[_GENOTYPE_KEY].nunique()
        if after < before:
            print(
                f"Warning: dropped {before - after} genotype(s) absent from "
                f"the reference table.",
                flush=True,
            )
        long["_target"] = long["theta_ref"]
        long["_dev"] = long["theta"] - long["theta_ref"]
        long["_var"] = long["sigma"] ** 2 + long["sigma_ref"] ** 2
    else:
        # Symmetric: the per-grid mean is the target.
        long["_target"] = long.groupby(keys)["theta"].transform("mean")
        long["_dev"] = long["theta"] - long["_target"]
        long["_var"] = long["sigma"] ** 2

    if long.empty:
        return _empty_result(keys)

    spread_label = "rms_dev" if mode == "reference" else estimator
    per_grid = _per_grid_table(long, keys, estimator)
    per_geno = _aggregate_genotypes(per_grid, long, keys, spread_label)

    # Attach coverage, tier, and the overdispersion flag.
    per_geno["n_present"] = per_geno[_GENOTYPE_KEY].map(n_present).astype(int)
    per_geno["mode"] = mode
    per_geno["tier"] = [
        _assign_tier(r, n, n_runs, min_coverage, sd_tier_edges)
        for r, n in zip(per_geno["rms_sd"], per_geno["n_present"])
    ]
    per_geno["overdispersed"] = per_geno["overdispersion"] > overdispersion_threshold

    # Wide per-grid sd columns.
    sd_wide = _sd_wide(per_grid)
    per_geno = per_geno.merge(sd_wide, on=_GENOTYPE_KEY, how="left")

    front = [
        _GENOTYPE_KEY, "n_present", "mode", "tier",
        "rms_sd", "max_sd", "dynamic_range",
        "overdispersion", "overdispersed", "mean_reported_sigma",
        "spread_estimator",
    ]
    sd_cols = [c for c in per_geno.columns if c.startswith("sd[")]
    per_geno = per_geno.loc[:, front + sorted(sd_cols)]
    per_geno = per_geno.sort_values("rms_sd", kind="mergesort").reset_index(drop=True)
    return per_geno


def _per_grid_table(long, keys, estimator):
    """
    Collapse the long run table to one row per condition-grid point.

    Returns a frame keyed by ``keys`` with:
      * ``spread`` -- Axis-1 dispersion at that grid point (std or half-range in
        mean mode; RMS deviation from the reference in reference mode).
      * ``mu`` -- the target theta curve value (cross-run mean or reference).
      * ``grid_label`` -- the ``sd[...]`` output-column name for this point.
    """
    grp = long.groupby(keys, sort=False)

    if "theta_ref" in long.columns:
        # Reference mode: RMS deviation of the runs from the reference.
        spread = np.sqrt(grp["_dev"].apply(lambda d: np.nanmean(np.square(d))))
    elif estimator == "std":
        spread = grp["theta"].std(ddof=1)
    else:
        spread = (grp["theta"].max() - grp["theta"].min()) / 2.0

    mu = grp["_target"].first()

    per_grid = pd.DataFrame({"spread": spread, "mu": mu}).reset_index()
    per_grid["grid_label"] = _grid_label(per_grid, keys)
    return per_grid


def _aggregate_genotypes(per_grid, long, keys, spread_label):
    """
    Collapse per-grid statistics and per-observation chi-square to per genotype.

    Computes ``rms_sd``/``max_sd`` (Axis 1) from ``per_grid``, ``dynamic_range``
    from the target curve (per ``titrant_name`` when present, then the max
    range), and ``overdispersion``/``mean_reported_sigma`` (Axis 2) from the
    long table. ``spread_label`` records which Axis-1 estimator was used
    (``'std'``, ``'half_range'``, or ``'rms_dev'``).
    """
    # Axis 1: pooled spread over the grid.
    def _rms(s):
        return np.sqrt(np.nanmean(np.square(s)))

    g = per_grid.groupby(_GENOTYPE_KEY, sort=False)
    rms_sd = g["spread"].apply(_rms)
    max_sd = g["spread"].apply(lambda s: np.nanmax(s.to_numpy()))
    per_geno = pd.DataFrame({"rms_sd": rms_sd, "max_sd": max_sd}).reset_index()

    # Dynamic range: measured within each titrant, then the widest across
    # titrants, so a genotype responsive in *any* titrant is not called flat.
    if _NAME_KEY in keys:
        by = per_grid.groupby([_GENOTYPE_KEY, _NAME_KEY])["mu"]
        rng = (by.max() - by.min()).groupby(_GENOTYPE_KEY).max()
    else:
        by = per_grid.groupby(_GENOTYPE_KEY)["mu"]
        rng = by.max() - by.min()
    per_geno["dynamic_range"] = per_geno[_GENOTYPE_KEY].map(rng)

    # Axis 2: overdispersion = chi2 / dof.
    #   chi2 = sum of (dev)^2 / var over valid observations.
    #   dof  = (# valid terms) - (# grid points) in mean mode (each grid point
    #          spends one dof estimating its own mean); (# valid terms) in
    #          reference mode (the target is fixed, not estimated).
    var = long["_var"].to_numpy(dtype=float)
    dev = long["_dev"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(var > 0, np.square(dev) / var, np.nan)
    chi_tbl = pd.DataFrame({
        _GENOTYPE_KEY: long[_GENOTYPE_KEY].to_numpy(),
        "term": term,
        "sigma": long["sigma"].to_numpy(dtype=float),
    })
    chi_g = chi_tbl.groupby(_GENOTYPE_KEY, sort=False)
    chi2 = chi_g["term"].sum(min_count=1)
    n_terms = chi_g["term"].apply(lambda s: int(np.isfinite(s).sum()))
    mean_sigma = chi_g["sigma"].mean()

    is_reference = "theta_ref" in long.columns
    n_grids = long.groupby(_GENOTYPE_KEY)[keys[1:]].apply(
        lambda d: len(d.drop_duplicates())
    )
    dof = n_terms if is_reference else (n_terms - n_grids.reindex(n_terms.index))
    with np.errstate(divide="ignore", invalid="ignore"):
        overdispersion = chi2 / dof.where(dof > 0)

    per_geno["overdispersion"] = per_geno[_GENOTYPE_KEY].map(overdispersion)
    per_geno["mean_reported_sigma"] = per_geno[_GENOTYPE_KEY].map(mean_sigma)
    per_geno["spread_estimator"] = spread_label
    return per_geno


def _sd_wide(per_grid):
    """Pivot the per-grid ``spread`` values to one ``sd[...]`` column each."""
    wide = per_grid.pivot_table(
        index=_GENOTYPE_KEY,
        columns="grid_label",
        values="spread",
        aggfunc="first",
    )
    wide.columns.name = None
    return wide.reset_index()


def _empty_result(keys):
    """Return an empty result frame with the standard columns."""
    cols = [
        _GENOTYPE_KEY, "n_present", "mode", "tier",
        "rms_sd", "max_sd", "dynamic_range",
        "overdispersion", "overdispersed", "mean_reported_sigma",
        "spread_estimator",
    ]
    return pd.DataFrame(columns=cols)


def stability_crosstabs(result,
                        overdispersion_threshold=2.0,
                        flat_range_threshold=0.1):
    """
    Build the two 2x2 interpretation tables from a ``compare_theta`` result.

    ``low_coverage`` genotypes are excluded (they were never graded).

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`compare_theta`.
    overdispersion_threshold : float, optional
        Split between "consistent" (<=) and "overconfident" (>). Default 2.0.
    flat_range_threshold : float, optional
        Split between "flat" (dynamic_range <) and "informative" (>=). Default
        0.1.

    Returns
    -------
    dict of str -> pandas.DataFrame
        ``'tier_vs_overdispersion'``: reproducible/unstable x
        consistent/overconfident. ``'tier_vs_dynamic_range'``:
        reproducible/unstable x informative/flat. Cells are genotype counts.
    """
    graded = result.loc[result["tier"].isin(["A", "B", "C", "D"])].copy()

    reproducible = np.where(
        graded["tier"].isin(["A", "B"]), "reproducible", "unstable"
    )
    consistent = np.where(
        graded["overdispersion"] > overdispersion_threshold,
        "overconfident", "consistent",
    )
    informative = np.where(
        graded["dynamic_range"] >= flat_range_threshold, "informative", "flat"
    )

    tier_vs_over = pd.crosstab(
        pd.Series(reproducible, name="reproducibility"),
        pd.Series(consistent, name="self_consistency"),
    )
    tier_vs_range = pd.crosstab(
        pd.Series(reproducible, name="reproducibility"),
        pd.Series(informative, name="dynamic_range"),
    )
    return {
        "tier_vs_overdispersion": tier_vs_over,
        "tier_vs_dynamic_range": tier_vs_range,
    }


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


def _mixture_quantiles(value_matrix, probs):
    """
    Quantiles of the equal-weight mixture of N per-run marginal posteriors.

    Each run contributes a marginal distribution described by its quantile
    ladder (``probs`` -> ``value_matrix[i]``). The mixture "pick a run at random,
    then draw from its posterior" has CDF ``F_mix = mean_i F_i``; its variance is
    the within-run variance plus the between-run spread of the point estimates
    (law of total variance). This reads the mixture's quantiles at the same
    ``probs`` levels by averaging the per-run CDFs on the pooled support grid and
    inverting.

    Parameters
    ----------
    value_matrix : numpy.ndarray, shape (n_present, K)
        Each row is one run's non-decreasing quantile values at ``probs``.
    probs : numpy.ndarray, shape (K,)
        Ascending probability levels in (0, 1), shared across runs.

    Returns
    -------
    numpy.ndarray, shape (K,)
        Mixture quantile values at ``probs``.
    """
    n = value_matrix.shape[0]
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
        f_sum += f_i
    f_mix = f_sum / n

    # Invert F_mix. Collapse ties (flat CDF = zero-density gaps) keeping the
    # smallest theta for each CDF level, matching the standard quantile def.
    uf, idx = np.unique(f_mix, return_index=True)
    return np.interp(probs, uf, grid[idx])


def aggregate_theta(estimate_dfs, *, progress_every=200_000):
    """
    Combine N theta estimates into one aggregate theta-vs-condition table.

    For every ``(genotype, [titrant_name,] titrant_conc)`` the N runs present are
    combined as an **equal-weight mixture** of their per-run marginal posteriors
    (reconstructed from the stored quantile ladder). The aggregate error thus
    folds in both each run's own posterior width *and* the run-to-run spread of
    the point estimates (law of total variance), and does **not** shrink with N
    -- appropriate because different-seed runs are not independent replicates.
    A bias shared by all N runs is invisible to this (only sampled variation --
    seed or training-data dropout -- is captured).

    In reference mode on the CLI side, only the N estimate runs are mixed; a
    reference run is never folded in (it is a comparison target, not a sample).

    Parameters
    ----------
    estimate_dfs : list of pandas.DataFrame
        The N estimate tables. Each must share the same key columns and carry a
        quantile ladder (``q<level>`` columns). The mixture is taken over the
        probability levels present in *all* runs.
    progress_every : int, optional
        Print a progress line every this many genotype-condition groups. Default
        200000.

    Returns
    -------
    pandas.DataFrame
        Long-form, one row per ``(genotype, [titrant_name,] titrant_conc)``, with
        the shared ``q<level>`` columns holding the mixture quantiles and an
        ``n_present`` column (how many runs contributed). Sorted by the keys.

    Raises
    ------
    ValueError
        If fewer than two runs are supplied, the key sets disagree, or the runs
        share fewer than two quantile levels.
    """
    if not isinstance(estimate_dfs, (list, tuple)):
        raise ValueError(
            "estimate_dfs must be a list of DataFrames, "
            f"not {type(estimate_dfs).__name__}."
        )
    if len(estimate_dfs) < 2:
        raise ValueError(
            f"aggregate_theta requires at least 2 estimate tables; "
            f"got {len(estimate_dfs)}."
        )

    keys = _detect_keys(estimate_dfs[0])

    # Intersect the quantile levels across runs; keep run 0's column names.
    levels = None
    qmap0 = _parse_quantile_columns(estimate_dfs[0])
    for i, df in enumerate(estimate_dfs):
        if _detect_keys(df) != keys:
            raise ValueError(
                f"Estimate run {i} has key columns {_detect_keys(df)}, "
                f"which differ from run 0's {keys}."
            )
        lv = set(_parse_quantile_columns(df))
        levels = lv if levels is None else (levels & lv)
    levels = sorted(levels)
    if len(levels) < 2:
        raise ValueError(
            "Estimate tables share fewer than 2 quantile (q<level>) columns; "
            "cannot reconstruct per-run distributions to mix."
        )
    probs = np.asarray(levels, dtype=float)
    out_cols = [qmap0[lvl] for lvl in levels]

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
