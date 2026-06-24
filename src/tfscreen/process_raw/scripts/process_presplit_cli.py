from tfscreen.process_raw import counts_to_lncfu
from tfscreen.process_raw._counts_io import _prep_sample_df, _aggregate_counts
from tfscreen.util.cli import generalized_main
from tfscreen.util.dataframe import check_columns, get_scaled_cfu

from typing import Union

import pandas as pd


def process_presplit(
    sample_df: Union[pd.DataFrame, str],
    counts_csv_path: str,
    output_file: str,
    counts_glob_prefix: str = "counts",
    min_genotype_obs: int = 10,
    pseudocount: int = 1,
    verbose: bool = True
):
    """
    Convert per-sample genotype count CSVs into a presplit ln_cfu DataFrame.

    Reads sample metadata from sample_df, matches each sample to a counts
    CSV file in counts_csv_path, aggregates all counts, converts to ln(CFU)
    via counts_to_lncfu, and writes a presplit CSV to output_file.  The
    presplit CSV contains only the columns required by tfs-configure-model
    (replicate, condition_pre, genotype, ln_cfu, ln_cfu_std).

    Parameters
    ----------
    sample_df : str or pandas.DataFrame
        Path to (or pre-loaded) sample metadata CSV.  Must contain a unique
        'sample' column (used as the index) plus 'replicate', 'condition_pre',
        'sample_cfu', and 'sample_cfu_std' columns.  An optional 'library'
        column groups genotypes for the minimum-observation filter; defaults
        to 'default' when absent.
    counts_csv_path : str
        Directory containing per-sample count CSV files.  Each file must
        match the glob pattern ``{counts_glob_prefix}*{sample}*.csv``.
    output_file : str
        Path to write the output presplit CSV (passed to tfs-configure-model
        as the presplit_df argument).
    counts_glob_prefix : str, optional
        Prefix used when globbing for count files in counts_csv_path.
        Default 'counts'.
    min_genotype_obs : int, optional
        Minimum total count across samples for a genotype to be retained.
        Genotypes below this threshold are dropped before the ln_cfu
        calculation.  Default 10.
    pseudocount : int, optional
        Pseudocount added to each genotype count before the log transform,
        to avoid log(0).  Default 1.
    verbose : bool, optional
        If True, print a summary of matched samples and file paths.
        Default True.
    """

    # After this call, sample_df will be indexed by sample name and have
    # a column 'obs_file' that points to the csv file to read.
    sample_df = _prep_sample_df(sample_df,
                                counts_csv_path,
                                counts_glob_prefix,
                                verbose)

    # Require the caller to supply the presplit-specific metadata columns.
    check_columns(sample_df, required_columns=["sample_cfu", "sample_cfu_std",
                                               "replicate", "condition_pre"])

    # counts_to_lncfu requires a 'library' column for grouping; default when absent.
    if "library" not in sample_df.columns:
        sample_df = sample_df.copy()
        sample_df["library"] = "default"

    counts_df = _aggregate_counts(sample_df)

    ln_cfu_df = counts_to_lncfu(sample_df,
                                 counts_df,
                                 min_genotype_obs=min_genotype_obs,
                                 pseudocount=pseudocount)

    # counts_to_lncfu outputs ln_cfu_var; compute ln_cfu_std from it.
    ln_cfu_df = get_scaled_cfu(ln_cfu_df, need_columns=["ln_cfu", "ln_cfu_std"])

    # Keep only the columns expected by _read_presplit_df.
    presplit_df = ln_cfu_df[["replicate", "condition_pre", "genotype",
                              "ln_cfu", "ln_cfu_std"]].copy()

    presplit_df = presplit_df.sort_values(
        by=["replicate", "condition_pre", "genotype"],
        ignore_index=True
    )

    presplit_df.to_csv(output_file, index=False)


def main():
    return generalized_main(process_presplit,
                            manual_arg_types={"sample_df": str,
                                              "counts_csv_path": str,
                                              "output_file": str,
                                              "min_genotype_obs": int,
                                              "pseudocount": int})


if __name__ == "__main__":
    main()
