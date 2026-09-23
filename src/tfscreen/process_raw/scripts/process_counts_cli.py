from tfscreen.process_raw import counts_to_lncfu
from tfscreen.process_raw.counts_to_lncfu import get_sample_ln_cfu
from tfscreen.process_raw._counts_io import _prep_sample_df, _aggregate_counts
from tfscreen.util.cli import generalized_main

from typing import Union

import pandas as pd


def process_counts(
    sample_df: Union[pd.DataFrame, str],
    counts_csv_path: str,
    output_file: str,
    counts_glob_prefix: str="counts",
    min_genotype_obs: int=10,
    pseudocount: int=1,
    verbose: bool = True):
    """
    Convert per-sample genotype count CSVs into a single ln_cfu DataFrame.

    Reads sample metadata from sample_df, matches each sample to a counts
    CSV file in counts_csv_path, aggregates all counts, converts to ln(CFU)
    via counts_to_lncfu, and writes the result to output_file.

    Parameters
    ----------
    sample_df : str or pandas.DataFrame
        Path to (or pre-loaded) sample metadata CSV.  Must contain a unique
        'sample' column (used as the index) plus the total CFU in each
        sample tube and its uncertainty, as 'sample_ln_cfu' and
        'sample_ln_cfu_std' (or 'sample_ln_cfu_var'), or as 'sample_cfu'
        and 'sample_cfu_std' (or 'sample_cfu_var'). See counts_to_lncfu.
    counts_csv_path : str
        Directory containing per-sample count CSV files.  Each file must
        match the glob pattern ``{counts_glob_prefix}*{sample}*.csv``.
    output_file : str
        Path to write the output ln_cfu CSV (passed directly to
        tfs-fit-model as the growth data).
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

    # Check/infer sample_ln_cfu and sample_ln_cfu_std before reading counts.
    sample_df = get_sample_ln_cfu(sample_df)

    # This will be a single dataframe holding all counts for all samples, with
    # a 'sample' column that can be indexed back to to counts.
    counts_df = _aggregate_counts(sample_df)

    # Infer cfu per genotype
    ln_cfu_df = counts_to_lncfu(sample_df,
                                counts_df,
                                min_genotype_obs=min_genotype_obs,
                                pseudocount=pseudocount)

    # Write outputs
    ln_cfu_df.to_csv(output_file,index=False)


def main():
    return generalized_main(process_counts)