import tfscreen

from tfscreen.process_raw import counts_to_lncfu
from tfscreen.util.cli import generalized_main
from tfscreen.util.dataframe import check_columns
    
import glob
import os
from typing import Union, Tuple

import pandas as pd
from tqdm.auto import tqdm


def _prep_sample_df(
    sample_df: Union[pd.DataFrame, str],
    counts_csv_path: str,
    counts_glob_prefix: str = "obs",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Validate input paths and sample_df, matching samples to CSV files.

    This function reads a sample dataframe and cross-references it with files
    in a specified directory. It ensures that exactly one observation CSV file
    exists for each sample ID. If validation passes, it adds an 'obs_file'
    column to the dataframe with the full path to each matched file.

    Parameters
    ----------
    sample_df : pandas.DataFrame or str
        The sample dataframe or a path to it. Must contain a 'sample' column.
    counts_csv_path : str
        The path to the directory containing the observation CSV files.
    counts_glob_prefix : str, default "obs"
        The prefix used for globbing to find observation files.
    verbose : bool, default True
        If True, print a summary of matched samples and files.

    Returns
    -------
    pandas.DataFrame
        The validated and updated sample dataframe, indexed by 'sample' and
        containing an 'obs_file' column.

    Raises
    ------
    FileNotFoundError
        If `counts_csv_path` is not a valid directory.
    ValueError
        If the 'sample' column in `sample_df` is not unique, or if any sample
        is missing a corresponding file or has multiple ambiguous file matches.
    """

    # Load sample_df, making sure the sample_df index is set to the sample 
    # column. 
    sample_df = tfscreen.util.io.read_dataframe(sample_df,index_column="sample")

    # Make sure that the sample_df is indexed by sample 
    if not sample_df.index.name == "sample":
        raise ValueError("sample_df must be indexed by 'sample'")

    # Make sure the sample index is unique
    if not sample_df.index.is_unique:
        dups = (sample_df.index[sample_df.index.duplicated(keep=False)]
                .unique().tolist())
        raise ValueError(f"samples must be unique. Duplicates: {dups}")

    # Make sure counts_csv_path is sane
    if not os.path.isdir(counts_csv_path):
        raise FileNotFoundError (
            f"counts_csv_path '{counts_csv_path}' is not a directory."
        )

    # Make sure that exactly one expected csv file is in the counts_csv_path for
    # each sample.
    error_messages = []
    all_files_found = {}
    for s in sample_df.index:
        
        # The glob pattern looks for files containing the sample ID.
        # e.g., 'path/to/csvs/obs_sample1_data.csv'
        file_pattern = os.path.join(counts_csv_path, f"{counts_glob_prefix}*{s}*.csv")
        files_found = glob.glob(file_pattern)

        # No files were found for the sample.
        if len(files_found) == 0:
            error_messages.append(
                f"  - MISSING: No files found for sample '{s}'."
            )
        # More than one file was found for the sample.
        elif len(files_found) > 1:
            error_messages.append(
                f"  - AMBIGUOUS: {len(files_found)} files found for sample '{s}'. Expected 1. (Found: {files_found})"
            )
        # Correct number of files
        else:
            all_files_found[s] = files_found[0]

    # Load the obs files into the sample dataframe
    sample_df["obs_file"] = sample_df.index.map(all_files_found)

    # If any errors, combine and raise
    if error_messages:
        header = f"File validation failed with {len(error_messages)} error(s):"
        full_error_message = "\n".join([header] + error_messages)
        raise ValueError(full_error_message)

    # Print out some information so the user gets a visual indication of 
    # what the processor saw
    if verbose:
    
        msg = [f"Sample dataframe has {len(sample_df)} unique samples with",
               f"matched observation csv files.\n"]
        print(" ".join(msg))
        for sample in all_files_found:
            print(f"{sample:40s} : {all_files_found[sample]}")
        print()
        
    return sample_df    
    
def _aggregate_counts(
    sample_df: Union[pd.DataFrame, str],
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Builds a combined dataframe holding all counts for all samples. 

    Parameters
    ----------
    sample_df : pandas.DataFrame or str
        The sample dataframe or a path to it. Must contain a 'sample' column.
    Returns
    -------
    pandas.DataFrame
        Long-form counts DataFrame with columns 'sample', 'genotype', and
        'counts', concatenated across all samples.
    """

    counts_dfs = []
    for s in tqdm(sample_df.index):

        # Parse the input csv
        df = pd.read_csv(sample_df.loc[s,"obs_file"])
        df["sample"] = s
        df = df[["sample","genotype","counts"]]

        counts_dfs.append(df)

    counts_df = pd.concat(counts_dfs,ignore_index=True)

    return counts_df

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
        'sample' column (used as the index) plus 'sample_cfu' and
        'sample_cfu_std' columns giving the total CFU and its uncertainty for
        each sample tube.
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

    # Require the caller to supply sample_cfu and sample_cfu_std directly.
    check_columns(sample_df, required_columns=["sample_cfu", "sample_cfu_std"])

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