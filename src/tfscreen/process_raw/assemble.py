import tfscreen

import itertools
import glob
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _prep_to_read(
    sample_df: Union[pd.DataFrame, str],
    obs_csv_path: str,
    expected_suffixes: Optional[list] = None,
    obs_glob_prefix: str = "obs",
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
    obs_csv_path : str
        The path to the directory containing the observation CSV files.
    expected_suffixes : list, optional
        This parameter is checked for but not used in the current implementation.
    obs_glob_prefix : str, default "obs"
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
        If `obs_csv_path` is not a valid directory.
    ValueError
        If the 'sample' column in `sample_df` is not unique, or if any sample
        is missing a corresponding file or has multiple ambiguous file matches.
    """

    # Load sample_df, making sure the sample_df index is set to the sample 
    # column. 
    sample_df = tfscreen.util.read_dataframe(sample_df,index_column="sample")

    # This sets what `condition_{suffix}` columns are checked
    if expected_suffixes is None:
        expected_suffixes = ["pre","sel"]

    # Make sure obs_csv_path is sane
    if not os.path.isdir(obs_csv_path):
        raise FileNotFoundError (
            f"obs_csv_path '{obs_csv_path}' is not a directory."
        )

    # Make sure that the sample_df is indexed by sample 
    if not sample_df.index.name == "sample":
        raise ValueError("sample_df must be indexed by 'sample'")

    # Make sure the sample index is unique
    if not sample_df.index.is_unique:
        dups = (sample_df.index[sample_df.index.duplicated(keep=False)]
                .unique().tolist())
        raise ValueError(f"samples must be unique. Duplicates: {dups}")

    # Make sure that exactly one expected csv file is in the obs_csv_path for
    # each sample.
    error_messages = []
    all_files_found = {}
    for s in sample_df.index:
        
        # The glob pattern looks for files containing the sample ID.
        # e.g., 'path/to/csvs/obs_sample1_data.csv'
        file_pattern = os.path.join(obs_csv_path, f"{obs_glob_prefix}*{s}*.csv")
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
        
        # samples
        msg = [f"Sample dataframe has {len(sample_df)} unique samples with",
               f"matched observation csv files.\n"]
        print(" ".join(msg))
        for sample in all_files_found:
            print(f"{sample:40s} : {all_files_found[sample]}")
        print()
        
    return sample_df    
    
    
def reads_to_muts(
    sample_df: Union[pd.DataFrame, str],
    obs_csv_path: str,
    ref_seq: str,
    output_directory: str,
    ref_start_resid: int = 1,
    batch_size: int = 10000,
    prep_to_read_kwargs: Optional[dict] = None
) -> pd.DataFrame:
    """
    Process raw sequence count files into mutation count files.

    This function orchestrates the entire workflow of converting raw sequence
    data into summarized mutation (genotype) counts. It iterates through a
    sample sheet, finds the corresponding raw data file for each sample,
    translates all DNA sequences into amino acid mutations relative to a
    reference, aggregates the counts for each unique mutation set, and writes
    the results to a new CSV file for each sample.

    Parameters
    ----------
    sample_df : pandas.DataFrame or str
        The sample dataframe or a path to it. Must contain a 'sample' column.
    obs_csv_path : str
        Path to the directory containing the raw sequence/count CSV files.
    ref_seq : str
        The reference amino acid sequence for mutation calling.
    output_directory : str
        Path to the directory where the output translated CSV files will be saved.
    ref_start_resid : int, default 1
        The residue number corresponding to the first amino acid in `ref_seq`.
    batch_size : int, default 10000
        Number of sequences to process at a time. Larger batches are faster
        but use more memory.
    prep_to_read_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the `_prep_to_read`
        validation function.

    Returns
    -------
    pandas.DataFrame
        The validated sample dataframe, which includes the 'obs_file' column
        pointing to the raw data for each sample.

    Raises
    ------
    FileExistsError
        If `output_directory` exists but is not a directory.
    FileNotFoundError, ValueError
        Propagated from the `_prep_to_read` validation function.
    """

    if prep_to_read_kwargs is None:
        prep_to_read_kwargs = {}

    # After this call, sample_df will be indexed by sample name and have 
    # a column 'obs_file' that points to the csv file to read
    sample_df = _prep_to_read(sample_df,
                              obs_csv_path,
                              **prep_to_read_kwargs)

    # Deal with output directory
    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            raise FileExistsError (
                f"output directory '{output_directory}' exists and is not a directory."
            )
    else:
        os.makedirs(output_directory)
    
    for s in tqdm(sample_df.index):

        # Parse the input csv
        df = pd.read_csv(sample_df.loc[s,"obs_file"],
                         header=None,
                         names=["sequence","counts"])
        df["counts"] = df["counts"].astype(float)
        df = df[~pd.isna(df["counts"])]
        df["counts"] = df["counts"].astype(int)

        # Translate 
        counts = _translate_and_count(df["sequence"].to_numpy(),
                                      df["counts"].to_numpy(),
                                      ref_seq,
                                      ref_start_resid=ref_start_resid,
                                      batch_size=batch_size)

        # Create a dataframe with the results
        counts_df = pd.DataFrame({
            "sample":[s for _ in counts],
            "genotype":["/".join(c) if c != "wt" else "wt" for c in counts],
            "counts":counts.values()
        })

        counts_df.to_csv(os.path.join(output_directory,f"trans_{s}.csv"),
                         index=False)
        
    
    return sample_df
    

