"""
Tube QC: does each selection tube's genotype composition match its titrant
label?

A tube grown at more titrant shifts the library toward the genotypes that
titrant favors, so a tube's composition says roughly which concentration it
saw. ``score_tubes`` turns that into an apparent-titrant score per tube and
flags tubes whose score is out of order within their row of tubes, the mark
of a tube or sample-sheet swap.

Method, within each panel (one library x condition_pre x condition_sel x
titrant_name; rows are replicate x t_pre x t_sel, one tube per
concentration):

1. Keep the genotypes with at least ``min_counts`` reads in every tube of the
   panel and take each tube's centered log-ratio composition (log counts
   less their mean over genotypes). Tube totals and read depth cancel.
2. Fit every genotype's row x concentration table robustly as
   ``L[g, r, c] = a[g, r] + b[r] * d[g, c] + residual`` (``_fit_profiles``:
   a median polish, then alternating medians). The row effect ``a`` holds
   what a row shares (its time, its replicate's starting pool), the
   concentration profile ``d`` is the titrant's effect, and the row scale
   ``b`` lets that effect grow with time; without it a late or early row
   is shifted. Medians make a swapped pair an outlier rather than part of
   the profile, and a row missing some concentrations gets its row effect
   from the profiles of the ones it has, so it is not shifted either (the
   bias of centering each row on its own mean).
3. A tube's score is its composition less its row's at the lowest
   concentration, ``L - a - b * d[:, lowest]``, projected on the axis from
   the lowest concentration's profile to the highest's and scaled so the
   profiles score 0 and 1; a row of scale ``b`` scores about
   ``b * expected``, where ``expected`` is the score of the label's own
   profile.
4. Two tests per tube, both leave-one-out within its row. *Order*: fit
   the rest of the row by isotonic (increasing) regression on
   concentration; ``order_violation`` is how far the tube's score lies
   outside the fitted values of its neighbors in concentration (0 between
   them). *Level*: ``deviation`` is ``score - b * expected``, with ``b``
   the rest of the row's scale (the median of score/expected over its
   tubes with expected >= 0.5; 1 if there is none). ``sigma``, one tube's
   score noise, is 1.4826 x the MAD of every tube's deviation in the panel,
   each divided by its ``deviation_sd`` (the leave-one-out ``b`` carries
   noise of its own, more for a high-concentration tube); ``order_z`` is
   ``order_violation / sigma`` and ``deviation_z`` is
   ``deviation / (sigma * deviation_sd)``. Of the tubes with either
   ``|z|`` above ``flag_z``, the one with the largest
   ``hypot(order_z, deviation_z)`` is flagged and left out (the two tubes
   of an adjacent swap have equal order violations; the level test says
   which is off), and the row is tested again until no tube exceeds it.
   The order test needs no scale;
   the level test also catches a high-concentration tube that sits low in
   a row whose other tubes are all low. Rows of fewer than three tubes are
   not flagged (a pair cannot say which tube is wrong).
5. Refit steps 2-4 with the flagged tubes left out of the fit (but still
   scored and tested) until the flags stop changing, so a bad tube does
   not set its own row's scale.

The scores are relative to the panel's own profiles, so a panel needs rows
that span its concentrations. A row without any tube at a concentration of
expected >= 0.5 has no scale of its own and is judged at ``b = 1``: a late
row of only low concentrations can then look high.
"""

import numpy as np
import pandas as pd

from tfscreen.util.dataframe import check_columns

# Panel and row keys, used when present in the growth table.
PANEL_COLUMNS = ["library", "condition_pre", "condition_sel", "titrant_name"]
ROW_COLUMNS = ["replicate", "t_pre", "t_sel"]

_UNKNOWN = "__unknown__"
_MAD_TO_SD = 1.4826

# Tubes whose label's expected score is below this do not set a row's scale.
SCALE_MIN_EXPECTED = 0.5


def _isotonic(y):
    """Increasing least-squares fit of ``y`` (pool adjacent violators)."""
    blocks = []  # [mean, weight, length]
    for v in y:
        blocks.append([float(v), 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


def order_violations(scores):
    """
    Leave-one-out order violation of each value in a sequence.

    ``scores`` are ordered by increasing concentration. Each value is
    compared with an isotonic fit of the others: positive if it lies above
    the fitted value of the next higher concentration, negative if below
    that of the next lower, 0 between them.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    out = np.zeros(n)
    if n < 2:
        return out
    for i in range(n):
        fit = _isotonic(np.delete(scores, i))
        lower = fit[i - 1] if i > 0 else -np.inf
        upper = fit[i] if i < n - 1 else np.inf
        if scores[i] > upper:
            out[i] = scores[i] - upper
        elif scores[i] < lower:
            out[i] = scores[i] - lower
    return out


def _row_scale(score, expected, default=1.0):
    """
    How strongly a row shows the titrant's effect, relative to the profile.

    The median of score/expected over the tubes whose expected score is at
    least ``SCALE_MIN_EXPECTED``; the ratio is unstable for the low
    concentrations. Unweighted, so one bad high tube cannot set the scale.
    ``default`` if the row has no such tube. Never below 0.
    """
    score = np.asarray(score, dtype=float)
    expected = np.asarray(expected, dtype=float)
    keep = (expected >= SCALE_MIN_EXPECTED) & np.isfinite(score)
    if not keep.any():
        return default
    return max(float(np.median(score[keep] / expected[keep])), 0.0)


def deviations(score, expected):
    """
    Each tube's score less its expectation, ``score - b * expected``.

    ``b`` is the row's scale (``_row_scale``) computed without the tube, so
    a tube cannot set its own expectation. Also returns each deviation's SD
    in units of one tube's score noise: ``b`` is itself estimated from k
    tubes, so ``sqrt(1 + expected^2 var(b))``, with ``var(b)`` the variance
    of a median of k ratios score/expected (``pi/2`` x the mean's for k >= 3)
    and 0 where ``b`` is the default. A row of one tube has deviation 0.
    """
    score = np.asarray(score, dtype=float)
    expected = np.asarray(expected, dtype=float)
    dev = np.zeros(len(score))
    sd = np.ones(len(score))
    if len(score) < 2:
        return dev, sd
    for i in range(len(score)):
        s_i, e_i = np.delete(score, i), np.delete(expected, i)
        dev[i] = score[i] - _row_scale(s_i, e_i) * expected[i]
        used = e_i[e_i >= SCALE_MIN_EXPECTED]
        if len(used):
            var_b = np.mean(1.0 / used ** 2) / len(used)
            if len(used) >= 3:
                var_b *= np.pi / 2
            sd[i] = np.sqrt(1.0 + expected[i] ** 2 * var_b)
    return dev, sd


def _median_polish(L, max_iter=50, tol=1e-10):
    """
    Median polish of a (genotype, row, conc) array with NaN for missing tubes.

    Returns the row effects ``a`` (G, R) and concentration effects ``d``
    (G, C) such that ``L ~ a[:, :, None] + d[:, None, :]``.
    """
    resid = L.copy()
    a = np.zeros(L.shape[:2])
    d = np.zeros((L.shape[0], L.shape[2]))
    for _ in range(max_iter):
        rm = np.nanmedian(resid, axis=2)
        a += rm
        resid -= rm[:, :, None]
        cm = np.nanmedian(resid, axis=1)
        d += cm
        resid -= cm[:, None, :]
        if max(np.nanmax(np.abs(rm)), np.nanmax(np.abs(cm))) < tol:
            break
    return a, d


def _project(L, a, d, b=None):
    """
    Scores (R, C) of every cell and expected scores (C,) of the profiles.

    A cell's score is ``L - a - b * d[:, 0]`` (its composition less its
    row's at the lowest concentration) on the axis ``d[:, -1] - d[:, 0]``,
    so a row of scale ``b`` scores ``b * expected``.
    """
    if b is None:
        b = np.ones(L.shape[1])
    axis = d[:, -1] - d[:, 0]
    norm = float(axis @ axis)
    if norm == 0:
        raise ValueError("The lowest and highest concentrations have the "
                         "same composition; there is no axis to score.")
    S = L - a[:, :, None] - b[None, :, None] * d[:, None, [0]]
    score = np.einsum("grc,g->rc", S, axis) / norm
    expected = ((d - d[:, [0]]).T @ axis) / norm
    return score, expected


def _fit_profiles(L, n_iter=20, tol=1e-8):
    """
    Robust fit of ``L[g, r, c] ~ a[g, r] + b[r] * d[g, c]``; returns
    ``a``, ``d`` and ``b``.

    Starts from a median polish (b = 1). The titrant's effect on
    composition grows with time, so rows differ in scale ``b``; an additive
    row effect alone leaves a row with ``b != 1`` shifted by
    ``(1 - b)`` times its median concentration's profile. Each pass takes
    every row's scale from its own scores (``_row_scale``; rows without a
    high-concentration tube keep 1), then updates ``a`` and ``d`` by
    medians.
    """
    a, d = _median_polish(L)
    b = np.ones(L.shape[1])
    for _ in range(n_iter):
        score, expected = _project(L, a, d, b)
        new = np.array([_row_scale(score[r], expected, default=np.nan)
                        for r in range(L.shape[1])])
        known = np.isfinite(new)
        if known.any():
            new[known] /= np.median(new[known])
        new[~known] = 1.0
        new = np.maximum(new, 1e-3)
        done = np.max(np.abs(new - b)) < tol
        b = new
        a = np.nanmedian(L - b[None, :, None] * d[:, None, :], axis=2)
        d = np.nanmedian((L - a[:, :, None]) / b[None, :, None], axis=1)
        if done:
            break
    return a, d, b


def _test_rows(score, expected, conc, r_idx, flag_z):
    """
    Order and level tests of every tube within its row (module docstring,
    step 4). Returns per-tube ``deviation``, ``order_violation``,
    ``flagged`` and the panel's ``sigma``.
    """
    n = len(score)
    deviation = np.zeros(n)
    dev_sd = np.ones(n)
    for r in np.unique(r_idx):
        idx = np.flatnonzero(r_idx == r)
        deviation[idx], dev_sd[idx] = deviations(score[idx], expected[idx])
    std = deviation / dev_sd
    sigma = _MAD_TO_SD * float(np.median(np.abs(std - np.median(std))))
    sigma = max(sigma, np.finfo(float).tiny)

    violation = np.zeros(n)
    flagged = np.zeros(n, dtype=bool)
    for r in np.unique(r_idx):
        idx = np.flatnonzero(r_idx == r)
        live = list(idx[np.argsort(conc[idx], kind="stable")])
        while len(live) >= 3:
            v = order_violations(score[live])
            dv, sd = deviations(score[live], expected[live])
            violation[live] = v
            deviation[live] = dv
            dev_sd[live] = sd
            oz, dz = v / sigma, dv / (sd * sigma)
            over = np.maximum(np.abs(oz), np.abs(dz)) > flag_z
            if not over.any():
                break
            # An adjacent pair out of order has equal order violations; the
            # level test says which one is off.
            worst = int(np.argmax(np.where(over, np.hypot(oz, dz), -1.0)))
            flagged[live[worst]] = True
            live.pop(worst)

    return {"deviation": deviation, "deviation_sd": dev_sd,
            "sigma": np.full(n, sigma),
            "order_violation": violation, "flagged": flagged}


def _score_panel(panel, row_cols, min_counts, min_genotypes, flag_z,
                 max_rounds=5):
    """Score and flag the tubes of one panel; see the module docstring."""

    P = panel.pivot_table(index="genotype", columns="_tube", values="counts",
                          aggfunc="sum")
    tubes = (panel.drop_duplicates("_tube")
             .set_index("_tube")
             .loc[P.columns, row_cols + ["titrant_conc", "sample"]])

    P = P[(P.fillna(0) >= min_counts).all(axis=1)]
    if len(P) < min_genotypes:
        raise ValueError(
            f"Only {len(P)} genotypes have at least {min_counts} reads in "
            f"every tube of panel {panel['_panel'].iloc[0]}; need "
            f"{min_genotypes}. Lower min_counts or drop near-empty tubes.")

    logp = np.log(P.to_numpy(dtype=float))
    logp -= logp.mean(axis=0, keepdims=True)

    row_id = tubes[row_cols].astype(str).agg("|".join, axis=1).to_numpy()
    rows, r_idx = np.unique(row_id, return_inverse=True)
    concs, c_idx = np.unique(tubes["titrant_conc"].to_numpy(dtype=float),
                             return_inverse=True)
    if len(concs) < 2:
        raise ValueError(f"Panel {panel['_panel'].iloc[0]} has only one "
                         "titrant concentration; there is no axis to score.")

    G = logp.shape[0]
    L = np.full((G, len(rows), len(concs)), np.nan)
    L[:, r_idx, c_idx] = logp
    out = tubes.reset_index(drop=True).copy()
    out["n_genotypes"] = G
    conc = out["titrant_conc"].to_numpy(dtype=float)

    # Fit the profiles, test the rows, refit without the flagged tubes, and
    # test every tube again, until the flags stop changing. A bad tube
    # otherwise sets its own row's scale and shifts the row.
    flagged = np.zeros(len(out), dtype=bool)
    for _ in range(max_rounds):
        Lfit = L.copy()
        Lfit[:, r_idx[flagged], c_idx[flagged]] = np.nan
        try:
            a, d, b = _fit_profiles(Lfit)
            grid_score, expected_c = _project(L, a, d, b)
        except ValueError as err:
            raise ValueError(f"Panel {panel['_panel'].iloc[0]}: {err}") from None
        score = grid_score[r_idx, c_idx]
        expected = expected_c[c_idx]
        result = _test_rows(score, expected, conc, r_idx, flag_z)
        same = np.array_equal(result["flagged"], flagged)
        flagged = result["flagged"]
        if same:
            break

    out["score"] = score
    out["expected"] = expected
    for k, v in result.items():
        out[k] = v
    sigma = out["sigma"].iloc[0]
    out["order_z"] = out["order_violation"] / sigma
    out["deviation_z"] = out["deviation"] / (sigma * out["deviation_sd"])
    out["row_scale"] = np.nan
    for r in np.unique(r_idx):
        keep = (r_idx == r) & ~flagged
        out.loc[r_idx == r, "row_scale"] = _row_scale(score[keep], expected[keep])

    return out


def score_tubes(growth_df, min_counts=1, min_genotypes=10, flag_z=4.0):
    """
    Apparent-titrant score of every tube from its genotype composition.

    Parameters
    ----------
    growth_df : pandas.DataFrame
        Long-form counts, one row per genotype and tube: ``genotype``,
        ``counts``, ``titrant_conc`` and ``replicate``, plus whichever of
        ``library``, ``condition_pre``, ``condition_sel``, ``titrant_name``
        (the panel), ``t_pre``, ``t_sel`` (the row) and ``sample`` (the tube's
        name) are present. The growth table ``tfs-process-counts`` writes has
        all of them.
    min_counts : int
        A genotype is used only if it has at least this many reads in every
        tube of its panel.
    min_genotypes : int
        Fewest such genotypes a panel may have.
    flag_z : float
        Flag a tube when its order violation or deviation exceeds this many
        noise SDs. On the 2026-10 dev data no tube outside the one swapped
        row exceeded 2.7.

    Returns
    -------
    pandas.DataFrame
        One row per tube: the panel and row keys, ``titrant_conc``,
        ``sample``, ``n_genotypes``, ``score`` (0 = the lowest
        concentration's composition, 1 = the highest's), ``expected`` (the
        score of the label's own profile), ``deviation``, ``sigma``,
        ``order_violation``, ``flagged``, ``order_z``, ``deviation_z`` and
        ``row_scale`` (the row's scale from its unflagged tubes).

    Raises
    ------
    ValueError
        If a required column is missing, a panel has two tubes at one row
        and concentration, has a single concentration, or keeps fewer than
        ``min_genotypes`` genotypes.
    """
    check_columns(growth_df, required_columns=["genotype", "counts",
                                               "titrant_conc", "replicate"])
    df = growth_df[growth_df["genotype"].astype(str) != _UNKNOWN].copy()

    panel_cols = [c for c in PANEL_COLUMNS if c in df.columns]
    row_cols = [c for c in ROW_COLUMNS if c in df.columns]
    tube_cols = panel_cols + row_cols + ["titrant_conc"]

    df["_panel"] = (df[panel_cols].astype(str).agg("|".join, axis=1)
                    if panel_cols else "all")
    df["_tube"] = df[tube_cols].astype(str).agg("|".join, axis=1)
    if "sample" in df.columns:
        n_samples = df.groupby("_tube")["sample"].nunique()
        dup = n_samples[n_samples > 1]
        if len(dup):
            raise ValueError(
                "More than one tube (sample) shares a row and concentration: "
                f"{list(dup.index[:5])}. Each replicate x time row needs one "
                "tube per concentration.")
    else:
        df["sample"] = df["_tube"]

    parts = []
    for _, panel in df.groupby("_panel", sort=True):
        scored = _score_panel(panel, row_cols, min_counts, min_genotypes, flag_z)
        for c in panel_cols:
            scored[c] = panel[c].iloc[0]
        parts.append(scored)

    out = pd.concat(parts, ignore_index=True)
    front = panel_cols + row_cols + ["titrant_conc", "sample"]
    out = out[front + [c for c in out.columns if c not in front]]
    return out.sort_values(front[:-1], ignore_index=True)


def score_table(scores):
    """Scores pivoted to rows (panel x row keys) by titrant concentration."""
    keys = [c for c in PANEL_COLUMNS + ROW_COLUMNS if c in scores.columns]
    return scores.pivot_table(index=keys, columns="titrant_conc",
                              values="score")
