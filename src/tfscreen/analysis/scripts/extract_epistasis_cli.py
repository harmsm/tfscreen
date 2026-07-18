"""
CLI for calculating second-order epistasis from a long-form observable table.
"""

import pandas as pd

from tfscreen.analysis.extract_epistasis import (
    extract_epistasis as _extract_epistasis,
)
from tfscreen.util.cli import generalized_main


def extract_epistasis(data_file,
                      y_obs,
                      out_prefix="tfs_epistasis",
                      y_std=None,
                      condition_selector=None,
                      scale="add",
                      keep_extra=False):
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
        condition_selector.
    y_obs : str
        Name of the column holding the observable for which epistasis is
        calculated (e.g. 'fitness', 'dG').
    out_prefix : str, optional
        Prefix for the output CSV file, written to {out_prefix}.csv.
        Default 'tfs_epistasis'.
    y_std : str or None, optional
        Name of the column holding the standard error of y_obs. If given, the
        epistasis error (ep_std) is propagated and written.
    condition_selector : list of str or None, optional
        One or more column names that define a unique experimental condition.
        Epistasis is calculated independently within each condition. If omitted,
        the whole table is treated as a single condition.
    scale : {"add", "mult"}, optional
        Epistatic scale. "add" (default): (Y11 - Y10) - (Y01 - Y00).
        "mult": (Y11 / Y10) / (Y01 / Y00).
    keep_extra : bool, optional
        If True, retain all columns from the input CSV in the output. If False
        (default), keep only the identifier columns and the calculated epistasis
        values.
    """
    print(f"Reading {data_file}...", flush=True)
    df = pd.read_csv(data_file)

    # Fail fast on missing columns rather than deep inside the pivot.
    required = ["genotype", y_obs]
    if y_std is not None:
        required.append(y_std)
    if condition_selector is not None:
        required.extend(condition_selector)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{data_file}' is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    result = _extract_epistasis(df,
                                y_obs=y_obs,
                                y_std=y_std,
                                condition_selector=condition_selector,
                                scale=scale,
                                keep_extra=keep_extra)

    out_file = f"{out_prefix}.csv"
    if result.empty:
        print("Warning: no valid mutant cycles found; writing empty output.",
              flush=True)

    result.to_csv(out_file, index=False)
    print(f"Wrote {len(result)} rows to {out_file}", flush=True)


def main():
    generalized_main(extract_epistasis,
                     manual_arg_types={"y_std": str,
                                       "condition_selector": str,
                                       "scale": str},
                     manual_arg_nargs={"condition_selector": "+"})


if __name__ == "__main__":
    main()
