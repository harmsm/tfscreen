"""
A shift shared by every genotype, added to posterior draws of X.

The two-stage fit (``planning/two-stage-fit.md``) pins every condition's
growth baseline k at a stage-1 estimate. Its uncertainty, measured by a
stage-1 jackknife, moves every non-wt genotype's X by nearly the same
amount (on the full-size simulation one standard deviation moved X by
-0.032 to -0.038, with a genotype-to-genotype SD of 0.005), while wt is
fixed by the X gauge. Treated as independent width per genotype it would
widen comparisons between genotypes that it barely touches. Added to the
joint posterior draws as one standard-normal multiple per draw, shared by
every row, it widens each genotype's own interval by the right amount
while differences between genotypes keep it only where they should (an
epistasis on X, X_ab - X_a - X_b + X_wt, keeps one copy, since wt does not
move).

The shift table has columns ``genotype``, ``titrant_name``,
``titrant_conc`` and ``shift`` (one standard deviation's signed change in
X), written by ``tfscreen.tfmodel.two_stage.shift_table``.
"""

import numpy as np
import pandas as pd

SHIFT_KEY = ["genotype", "titrant_name", "titrant_conc"]


def read_shift_table(shift):
    """
    The shift table, from a path or a DataFrame.

    Parameters
    ----------
    shift : str or pandas.DataFrame
        Path to a CSV, or a DataFrame, with ``genotype``, ``titrant_name``,
        ``titrant_conc`` and ``shift``.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        On a missing column or a key given twice.
    """
    df = pd.read_csv(shift) if isinstance(shift, str) else shift.copy()
    missing = [c for c in SHIFT_KEY + ["shift"] if c not in df.columns]
    if missing:
        raise ValueError(f"Shared-shift table is missing column(s) {missing}.")
    df = df[SHIFT_KEY + ["shift"]].copy()
    df["genotype"] = df["genotype"].astype(str)
    df["titrant_name"] = df["titrant_name"].astype(str)
    df["titrant_conc"] = df["titrant_conc"].astype(float)
    dup = df.duplicated(SHIFT_KEY)
    if dup.any():
        raise ValueError("Shared-shift table gives a row more than once: "
                         f"{df.loc[dup, SHIFT_KEY].values.tolist()[:5]}")
    return df


def row_shifts(calc_df, shift):
    """
    The shift for each row of a prediction table, aligned to its rows.

    wt's shift is 0 whatever the table says (the X gauge fixes wt).

    Parameters
    ----------
    calc_df : pandas.DataFrame
        Prediction rows with ``genotype``, ``titrant_name`` and
        ``titrant_conc``.
    shift : str or pandas.DataFrame
        The shift table (see ``read_shift_table``).

    Returns
    -------
    numpy.ndarray
        One shift per row of ``calc_df``.

    Raises
    ------
    ValueError
        If a non-wt row has no shift.
    """
    table = read_shift_table(shift)
    rows = calc_df[SHIFT_KEY].copy()
    rows["genotype"] = rows["genotype"].astype(str)
    rows["titrant_name"] = rows["titrant_name"].astype(str)
    rows["titrant_conc"] = rows["titrant_conc"].astype(float)
    merged = rows.merge(table, on=SHIFT_KEY, how="left")
    is_wt = merged["genotype"] == "wt"
    lacking = merged["shift"].isna() & ~is_wt
    if lacking.any():
        bad = merged.loc[lacking, SHIFT_KEY].drop_duplicates()
        raise ValueError(
            f"Shared-shift table has no shift for {len(bad)} predicted "
            f"row(s), e.g. {bad.values.tolist()[:5]}. It must cover every "
            "non-wt genotype and concentration predicted.")
    # a copy: under pandas 3's copy-on-write to_numpy returns a read-only
    # view, which the wt assignment below cannot write into
    out = merged["shift"].to_numpy(dtype=float, copy=True)
    out[is_wt.to_numpy()] = 0.0
    return out


def apply_shared_shift(samples, calc_df, shift, seed=0):
    """
    Add one shared standard-normal multiple of the shift to each draw.

    Parameters
    ----------
    samples : numpy.ndarray
        Posterior draws, shape (num_draws, num_rows), rows aligned to
        ``calc_df``.
    calc_df : pandas.DataFrame
        The rows (``genotype``, ``titrant_name``, ``titrant_conc``).
    shift : str or pandas.DataFrame
        The shift table.
    seed : int, optional
        Seed for the per-draw multiples. Default 0.

    Returns
    -------
    numpy.ndarray
        ``samples + z[:, None] * shift[None, :]``, with one ``z`` per draw.

    Raises
    ------
    ValueError
        With a single draw (a MAP point has no draws to share a shift
        across) or a row count that does not match ``calc_df``.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(calc_df):
        raise ValueError(f"samples have shape {samples.shape}; expected "
                         f"(num_draws, {len(calc_df)}).")
    if samples.shape[0] < 2:
        raise ValueError("A shared shift needs posterior draws; this "
                         "posterior has one (a MAP point). Run "
                         "tfs-sample-posterior first.")
    per_row = row_shifts(calc_df, shift)
    z = np.random.default_rng(seed).standard_normal(samples.shape[0])
    return samples + z[:, None] * per_row[None, :]
