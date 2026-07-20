"""
CLI for calculating second-order epistasis from a long-form observable table.
"""

import pandas as pd

from tfscreen.analysis.extract_epistasis import (
    extract_epistasis as _extract_epistasis,
)
from tfscreen.util import resolve_obs_columns
from tfscreen.util.cli import generalized_main


def _diagnose_empty_output(df, group_by, exclude=()):
    """
    Explain why no mutant cycles were found and, when possible, suggest a fix.

    The most common cause is a table with one row per genotype *per condition*
    (e.g. per titrant_conc) run without ``--group_by``. In that case
    every genotype is non-unique within the single implicit group, so
    ``mutant_cycle_pivot`` drops all rows as duplicates and returns nothing. This
    inspects the input and, if that is what happened, names the column(s) that
    would resolve it.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw input table (with a 'genotype' column).
    group_by : list of str or None
        The grouping columns that were used, if any.
    exclude : iterable of str, optional
        Columns that must never be suggested as condition selectors -- namely the
        observable and its error column, whose values legitimately vary within a
        genotype and would otherwise be flagged as spurious candidates.

    Returns
    -------
    str or None
        A hint to print, or None if no specific cause could be identified.
    """
    if "genotype" not in df.columns:
        return None

    used = list(group_by) if group_by is not None else []
    exclude = set(exclude)

    # Recreate the grouping the pivot used, and check for non-unique genotypes
    # within any group -- that is the duplicate-drop trap.
    if used:
        groups = (g for _, g in df.groupby(used))
    else:
        groups = (df,)
    has_dupes = any(not g["genotype"].is_unique for g in groups)
    if not has_dupes:
        return None

    # Candidate condition columns: those whose value varies within a genotype
    # (and that are not already being used as selectors).
    candidates = [
        c for c in df.columns
        if c != "genotype" and c not in used and c not in exclude
        and (df.groupby("genotype")[c].nunique(dropna=False) > 1).any()
    ]

    lines = [
        "Every genotype appears more than once within a condition group, so all "
        "rows were dropped as duplicates before any cycle could be built."
    ]
    if candidates:
        # Prefer a single column that alone makes genotypes unique; otherwise
        # fall back to suggesting the full set.
        single = [
            c for c in candidates
            if not df.duplicated(subset=["genotype", c]).any()
        ]
        hint_cols = single if single else candidates
        lines.append(
            "This usually means the table has one row per genotype *per "
            "condition*, and the condition column was not passed."
        )
        lines.append(f"  Columns that vary within a genotype: {candidates}")
        lines.append(
            f"  Try: --group_by {' '.join(hint_cols)}"
        )
    else:
        lines.append(
            "Check the input for genuinely duplicated genotype rows within a "
            "condition."
        )
    return "\n".join(lines)


def extract_epistasis(data_file,
                      y_obs=None,
                      out_prefix="tfs_epistasis",
                      y_std=None,
                      group_by=None,
                      scale="add",
                      keep_extra=False,
                      logit_eps=1e-9):
    """
    Calculate second-order epistasis for pairs of mutations.

    Reads a long-form CSV (one row per genotype, with genotypes in the format
    'MUT1/MUT2') and, for every double mutant, builds a mutant cycle from its
    two single-mutant parents and the wildtype. Epistasis is calculated on the
    requested observable and written, one row per double mutant, to
    {out_prefix}.csv.

    Parameters
    ----------
    data_file : str
        Path to the input CSV. Must contain a 'genotype' column, the column
        named by y_obs, and (if given) the columns named by y_std and
        group_by.
    y_obs : str or None, optional
        Name of the column holding the observable for which epistasis is
        calculated (e.g. 'fitness', 'dG'). If None (default) and the input has a
        'q0.5' column (as written by tfs-predict-theta), 'q0.5' is used.
    out_prefix : str, optional
        Prefix for the output CSV file, written to {out_prefix}.csv.
        Default 'tfs_epistasis'.
    y_std : str or None, optional
        Name of the column holding the standard error of y_obs. If given, the
        epistasis error (ep_std) is propagated and written. If None (default)
        and both 'q0.841' and 'q0.159' are present, the standard error is taken
        as (q0.841 - q0.159) / 2.
    group_by : list of str or None, optional
        One or more column names that define a unique experimental condition.
        Epistasis is calculated independently within each condition. If omitted,
        the whole table is treated as a single condition.
    scale : {"add", "mult", "logit"}, optional
        Epistatic scale. "add" (default): (Y11 - Y10) - (Y01 - Y00).
        "mult": (Y11 / Y10) / (Y01 / Y00). "logit": additive epistasis of
        logit(Y) — requires an observable in (0, 1), such as theta/occupancy.
    keep_extra : bool, optional
        If True, retain all columns from the input CSV in the output. If False
        (default), keep only the identifier columns and the calculated epistasis
        values.
    logit_eps : float, optional
        Only used when scale="logit". Observations are clamped to
        [logit_eps, 1 - logit_eps] before the transform. Default 1e-9.
    """
    print(f"Reading {data_file}...", flush=True)
    df = pd.read_csv(data_file)

    # Fill in y_obs/y_std defaults from quantile columns (q0.5 for the point
    # estimate, (q0.841 - q0.159)/2 for the std) when not given explicitly.
    df, y_obs, y_std = resolve_obs_columns(df, y_obs=y_obs, y_std=y_std)

    # Fail fast on missing columns rather than deep inside the pivot.
    required = ["genotype", y_obs]
    if y_std is not None:
        required.append(y_std)
    if group_by is not None:
        required.extend(group_by)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{data_file}' is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    result = _extract_epistasis(df,
                                y_obs=y_obs,
                                y_std=y_std,
                                group_by=group_by,
                                scale=scale,
                                keep_extra=keep_extra,
                                logit_eps=logit_eps)

    out_file = f"{out_prefix}.csv"
    if result.empty:
        print("Warning: no valid mutant cycles found; writing empty output.",
              flush=True)
        exclude = [c for c in (y_obs, y_std) if c is not None]
        hint = _diagnose_empty_output(df, group_by, exclude=exclude)
        if hint is not None:
            print(hint, flush=True)

    result.to_csv(out_file, index=False)
    print(f"Wrote {len(result)} rows to {out_file}", flush=True)


def main():
    generalized_main(extract_epistasis,
                     manual_arg_types={"y_obs": str,
                                       "y_std": str,
                                       "group_by": str,
                                       "scale": str},
                     manual_arg_nargs={"group_by": "+"})


if __name__ == "__main__":
    main()
