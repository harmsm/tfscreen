"""
Diagnostic for the per-tube sample offsets (``sample_offset: level``).

A level offset should be tube noise: an error in a tube's supplied total or
its composition, independent from tube to tube. On the dev data the MAP's
better optimum put the offsets in a pattern instead (the "offset mode",
``planning/offset-mode-growth-transition.md``): within each selection
condition they followed IPTG, from +2 to -4 ln units across 0 to 1 mM, and
held constant over time, carrying growth the model could not express. This
module reports the offsets in units of their prior SD and tests, within each
condition, whether they trend with titrant or with time.

``tfs-summarize-fit`` calls ``tube_offset_diagnostic`` on the
``*_sample_offset_offset.csv`` that ``tfs-extract-params`` writes.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from tfscreen.analysis.cat_response.cat_assess import benjamini_hochberg

GROUP_COLUMNS = ["library", "condition_pre", "condition_sel", "titrant_name"]


def _spearman(x, y):
    """Spearman rho and p, NaN when either side is constant or n < 3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def _r2_by(values, labels):
    """Fraction of the variance of ``values`` explained by the means by label."""
    values = np.asarray(values, dtype=float)
    total = np.sum((values - values.mean()) ** 2)
    if total == 0 or len(np.unique(labels)) < 2:
        return np.nan
    means = pd.Series(values).groupby(np.asarray(labels)).transform("mean")
    return float(np.sum((means.to_numpy() - values.mean()) ** 2) / total)


def tube_offset_diagnostic(offsets_df, sigma=None, alpha=0.05,
                           value_column="q0.5"):
    """
    Size and structure of the per-tube offsets.

    Parameters
    ----------
    offsets_df : pandas.DataFrame
        One row per tube: the design columns (``condition_sel``,
        ``titrant_conc``, ``t_sel`` and whichever of ``library``,
        ``condition_pre``, ``titrant_name``, ``replicate`` are present) and
        ``value_column``, the offset's point estimate.
    sigma : float or None
        The offsets' prior SD (held ``sigma_fixed`` or the fit's sigma). With
        None, sizes are reported in ln units only.
    alpha : float
        A condition is called structured when the Benjamini-Hochberg q of its
        titrant or its time trend is below ``alpha`` (q over all conditions
        and both trends at once).
    value_column : str
        Column holding the offset.

    Returns
    -------
    tubes : pandas.DataFrame
        ``offsets_df`` with ``offset`` and, given ``sigma``, ``z`` (offset in
        prior SDs).
    trends : pandas.DataFrame
        One row per condition (``library``, ``condition_pre``,
        ``condition_sel``, ``titrant_name``): ``n_tubes``,
        ``n_titrant_conc``, ``mean``, ``sd``, ``mean_z``, ``sd_z``,
        Spearman ``rho_titrant``/``p_titrant``/``q_titrant`` (offset against
        titrant concentration), ``rho_time``/``p_time``/``q_time`` (against
        ``t_sel``), ``r2_titrant`` (variance explained by the mean at each
        concentration, any shape) and ``structured``.
    summary : dict
        ``n_tubes``, ``sigma``, ``sd``, ``sd_over_sigma``, ``abs_z_median``,
        ``abs_z_q95``, ``abs_z_max``, ``range``, ``alpha``,
        ``n_conditions``, ``n_structured`` and ``structured`` (any
        structured condition), with the structured conditions listed.
    """
    if value_column not in offsets_df.columns:
        raise ValueError(f"offsets table has no '{value_column}' column")
    for col in ("condition_sel", "titrant_conc", "t_sel"):
        if col not in offsets_df.columns:
            raise ValueError(f"offsets table has no '{col}' column")

    tubes = offsets_df.copy()
    tubes["offset"] = tubes[value_column].astype(float)
    have_sigma = sigma is not None and np.isfinite(sigma) and sigma > 0
    if have_sigma:
        tubes["z"] = tubes["offset"] / float(sigma)

    group_cols = [c for c in GROUP_COLUMNS if c in tubes.columns]
    rows = []
    for key, g in tubes.groupby(group_cols, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        off = g["offset"].to_numpy()
        row["n_tubes"] = len(g)
        row["n_titrant_conc"] = int(g["titrant_conc"].nunique())
        row["mean"] = float(off.mean())
        row["sd"] = float(off.std(ddof=1)) if len(off) > 1 else np.nan
        row["mean_z"] = row["mean"] / sigma if have_sigma else np.nan
        row["sd_z"] = row["sd"] / sigma if have_sigma else np.nan
        row["rho_titrant"], row["p_titrant"] = _spearman(g["titrant_conc"], off)
        row["rho_time"], row["p_time"] = _spearman(g["t_sel"], off)
        row["r2_titrant"] = _r2_by(off, g["titrant_conc"].to_numpy())
        rows.append(row)
    trends = pd.DataFrame(rows)

    n = len(trends)
    q = benjamini_hochberg(np.concatenate([trends["p_titrant"].to_numpy(),
                                           trends["p_time"].to_numpy()]))
    trends["q_titrant"] = q[:n]
    trends["q_time"] = q[n:]
    trends["structured"] = ((trends["q_titrant"] < alpha)
                            | (trends["q_time"] < alpha))
    order = group_cols + ["n_tubes", "n_titrant_conc", "mean", "sd",
                          "mean_z", "sd_z", "rho_titrant", "p_titrant",
                          "q_titrant", "r2_titrant", "rho_time", "p_time",
                          "q_time", "structured"]
    trends = trends[order]

    off = tubes["offset"].to_numpy()
    sd = float(off.std(ddof=1)) if len(off) > 1 else np.nan
    abs_z = np.abs(tubes["z"].to_numpy()) if have_sigma else None
    structured = trends.loc[trends["structured"], group_cols]
    summary = {
        "n_tubes": int(len(tubes)),
        "sigma": float(sigma) if have_sigma else None,
        "sd": sd,
        "sd_over_sigma": sd / sigma if have_sigma else None,
        "abs_z_median": float(np.median(abs_z)) if have_sigma else None,
        "abs_z_q95": float(np.quantile(abs_z, 0.95)) if have_sigma else None,
        "abs_z_max": float(abs_z.max()) if have_sigma else None,
        "range": [float(off.min()), float(off.max())],
        "alpha": float(alpha),
        "n_conditions": int(n),
        "n_structured": int(trends["structured"].sum()),
        "structured": bool(trends["structured"].any()),
        "structured_conditions": structured.astype(str).to_dict("records"),
    }
    return tubes, trends, summary


def format_tube_offset_summary(summary, trends):
    """A few printable lines: size, then each structured condition."""
    lines = [f"Tube offsets: {summary['n_tubes']} tubes, SD {summary['sd']:.3g}, "
             f"range {summary['range'][0]:.3g} to {summary['range'][1]:.3g}"]
    if summary["sigma"] is not None:
        lines[0] += (f"; prior SD {summary['sigma']:.3g}, SD/prior "
                     f"{summary['sd_over_sigma']:.3g}, max |z| "
                     f"{summary['abs_z_max']:.3g}")
    if summary["structured"]:
        lines.append(f"  {summary['n_structured']} of {summary['n_conditions']} "
                     f"conditions trend with titrant or time (BH q < "
                     f"{summary['alpha']:g}); the offsets carry structure "
                     f"(the offset mode):")
        for _, r in trends[trends["structured"]].iterrows():
            lines.append(f"    {r['condition_sel']}: rho titrant "
                         f"{r['rho_titrant']:.2f} (q {r['q_titrant']:.2g}), "
                         f"rho time {r['rho_time']:.2f} (q {r['q_time']:.2g}), "
                         f"mean {r['mean']:.3g}")
    else:
        lines.append("  no condition trends with titrant or time "
                     f"(BH q < {summary['alpha']:g})")
    return "\n".join(lines)
