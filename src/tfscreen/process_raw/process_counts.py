import tfscreen

from tfscreen.process_raw import (
    od600_to_cfu,
    counts_to_lncfu
)
from tfscreen.util import (
    generalized_main,
    check_columns
)
    
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
    sample_df = tfscreen.util.read_dataframe(sample_df,index_column="sample")

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
        The validated sample dataframe, which includes the 'obs_file' column
        pointing to the raw data for each sample.
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

def _infer_sample_cfu(sample_df,od600_calibration_data):

    # Make sure the sample dataframe has all required columns
    check_columns(sample_df,required_columns=["od600"])
    
    # Extract sample cfu from od600
    cfu, cfu_std, detectable = od600_to_cfu(sample_df["od600"],
                                            od600_calibration_data)
    sample_df["sample_cfu"] = cfu
    sample_df["sample_cfu_std"] = cfu_std
    sample_df["sample_cfu_detectable"] = detectable

    return sample_df

def process_counts(
    sample_df: Union[pd.DataFrame, str],
    counts_csv_path: str,
    od600_calibration_data: Union[str,dict],
    output_file: str,
    counts_glob_prefix: str="counts",
    min_genotype_obs: int=10,
    pseudocount: int=1,
    verbose: bool = True):

    # After this call, sample_df will be indexed by sample name and have 
    # a column 'obs_file' that points to the csv file to read. 
    sample_df = _prep_sample_df(sample_df,
                                counts_csv_path,
                                counts_glob_prefix,
                                verbose)
    
    # This will be a single dataframe holding all counts for all samples, with
    # a 'sample' column that can be indexed back to to counts. 
    counts_df = _aggregate_counts(sample_df)

    # Assign sample cfu/mL given the od600 for each sample
    sample_df = _infer_sample_cfu(sample_df,od600_calibration_data)

    # Infer cfu per genotype
    ln_cfu_df = counts_to_lncfu(sample_df,
                                counts_df,
                                min_genotype_obs=min_genotype_obs,
                                pseudocount=pseudocount)
    
    # Write outputs
    ln_cfu_df.to_csv(output_file,index=False)


def main():
    return generalized_main(process_counts)