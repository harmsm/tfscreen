import tfscreen

import itertools
import glob
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _build_ambiguous_codon_table(codon_table: Optional[dict] = None) -> dict:
    """Build a codon table that resolves ambiguity from 'N' nucleotides.

    This function expands a standard codon table to include all 125 possible
    codons containing 'A', 'C', 'G', 'T', or 'N'. If all possible resolutions
    of an ambiguous codon (e.g., 'CTN') map to a single amino acid ('L'),
    that mapping is added. If they map to multiple amino acids, the codon
    is mapped to 'X' (unknown).

    Parameters
    ----------
    codon_table : dict, optional
        A dictionary mapping 3-letter DNA codons to single-letter amino acids.
        If not provided, a default table is loaded from `tfscreen.data`.

    Returns
    -------
    dict
        A new, expanded codon table containing mappings for ambiguous codons.
    """
    if codon_table is None:
        codon_table = {k.upper(): v for k, v in tfscreen.data.CODON_TO_AA.items()}

    ambiguous_table = codon_table.copy()
    
    bases = 'ACGT'
    for codon_tuple in itertools.product('ACGTN', repeat=3):
        codon = "".join(codon_tuple)
        if codon in ambiguous_table:
            continue
            
        possible_codons = [""]
        for base in codon:
            if base == 'N':
                possible_codons = [p + b for p in possible_codons for b in bases]
            else:
                possible_codons = [p + base for p in possible_codons]
        
        # FIX: Only consider translations for codons present in the base table.
        valid_translations = [codon_table[c] for c in possible_codons if c in codon_table]
        
        # If 'N' resolves to codons not in the table, treat as 'X'.
        if not valid_translations:
            translated_aas = {'X'}
        else:
            translated_aas = set(valid_translations)

        if len(translated_aas) == 1:
            ambiguous_table[codon] = translated_aas.pop()
        else:
            ambiguous_table[codon] = 'X'
            
    return ambiguous_table


def _setup_tables(codon_table: Optional[dict]) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """
    Create lookup tables for fast, vectorized sequence translation.

    This function pre-computes all necessary mappings to convert nucleotide
    sequences into amino acid sequences using direct NumPy array indexing
    instead of slower dictionary lookups.

    Parameters
    ----------
    codon_table : dict, optional
        An optional codon table to use. If None, a default table including
        ambiguous codons will be generated.

    Returns
    -------
    nucl_to_int : dict
        Mapping from nucleotide character ('A', 'C', 'G', 'T', 'N') to integer (0-4).
    codon_to_aa_int : numpy.ndarray
        A 5x5x5 array for fast lookup. Indexing with three nucleotide integers
        (e.g., `codon_to_aa_int[0, 1, 2]`) returns the integer for the
        corresponding amino acid.
    int_to_aa : numpy.ndarray
        An array mapping amino acid integers back to their single-letter strings.
    aa_to_int : dict
        Mapping from amino acid character to its integer representation.
    """

    codon_table = _build_ambiguous_codon_table(codon_table)
        
    # Map DNA bases to integers 0-4
    nucl_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Create an array of all possible amino acid outputs
    int_to_aa = np.array(list(sorted(set(codon_table.values()))))
    
    # Map those amino acids to integers
    aa_to_int = {aa: i for i, aa in enumerate(int_to_aa)}
    
    # Create a 5x5x5 lookup table, initialized to the 'X' value
    codon_to_aa_int = np.full((5, 5, 5), fill_value=aa_to_int['X'], dtype=np.int8)

    # Populate the lookup table from the codon dictionary
    for codon, aa in codon_table.items():
        i, j, k = nucl_to_int[codon[0]], nucl_to_int[codon[1]], nucl_to_int[codon[2]]
        codon_to_aa_int[i, j, k] = aa_to_int[aa]

    return nucl_to_int, codon_to_aa_int, int_to_aa, aa_to_int


def _translate_and_count(
    sequences: np.ndarray,
    counts: np.ndarray,
    ref_seq: str,
    ref_start_resid: int = 1,
    codon_table: Optional[dict] = None,
    batch_size: int = 1000
) -> dict[Union[str, tuple[str, ...]], int]:
    """Translate DNA sequences, identify mutations, and aggregate their counts.

    This function performs a high-performance, batch-wise translation of
    nucleotide sequences to amino acid sequences. It identifies all mutations
    relative to a reference amino acid sequence and returns a dictionary
    mapping each unique set of mutations (genotype) to its total count.

    Parameters
    ----------
    sequences : numpy.ndarray
        A 1D array of nucleotide strings.
    counts : numpy.ndarray
        A 1D array of integer counts corresponding to each sequence.
    ref_seq : str
        The reference amino acid sequence to which mutations will be compared.
    ref_start_resid : int, default 1
        The residue number of the first amino acid in `ref_sequence`.
    codon_table : dict, optional
        A dictionary mapping 3-letter codons to single-letter amino acids.
        If None, a default table will be generated.
    batch_size : int, default 1000
        The number of sequences to process in each vectorized batch.

    Returns
    -------
    dict
        A dictionary where keys are either the string "wt" or a tuple of
        mutation strings (e.g., ('A10G', 'P25L')) and values are the summed
        integer counts.
    """

    # Construct lookup tables. 
    nucl_to_int, codon_to_aa_int, int_to_aa, aa_to_int = _setup_tables(codon_table)

    # Convert reference sequence to ints
    ref_chars = np.array(ref_seq,dtype="U").flatten().view("U1")
    ref_ints = np.vectorize(aa_to_int.get)(ref_chars)
    ref_resid = np.arange(ref_start_resid,
                          ref_start_resid+len(ref_ints),
                          dtype=int)

    # Cast inputs
    sequences = np.array(sequences,dtype="U").flatten()
    counts = np.asarray(counts,dtype=int).flatten()

    # Sanity check
    if len(sequences) != len(counts):
        raise ValueError(
            f"sequences and counts must be the same length. "
            f"sequences: {len(sequences)}, counts: {len(counts)}"
        )

    final_seq_counts = {}
    for i in tqdm(range(0,len(sequences),batch_size)):

        # Grab a batch of sequences
        batch_seqs = sequences[i:(i+batch_size)]
        batch_counts = counts[i:(i+batch_size)]

        # Convert array of strings to a 2D array of single characters
        seq_chars = np.array(batch_seqs).view('U1').reshape(len(batch_seqs), -1)

        # Map the character array to an integer array
        seq_ints = np.vectorize(nucl_to_int.get)(seq_chars)

        # Reshape into codons: (num_sequences, num_codons, 3)
        num_sequences = seq_ints.shape[0]
        codon_ints = seq_ints.reshape(num_sequences, -1, 3)

        # Extract the integers for each codon position
        p1 = codon_ints[:, :, 0]
        p2 = codon_ints[:, :, 1]
        p3 = codon_ints[:, :, 2]
    
        # Perform all lookups at once
        translated_aa_ints = codon_to_aa_int[p1, p2, p3]

        # Mask checks which sites are different than reference
        not_ref_mask = translated_aa_ints != ref_ints[np.newaxis,:]

        for j in range(translated_aa_ints.shape[0]):

            if np.sum(not_ref_mask[j]) == 0:
                mut_key = "wt"
            else:
                mut_wt_aa = ref_chars[not_ref_mask[j]]
                mut_resid = ref_resid[not_ref_mask[j]]
                mut_aa = int_to_aa[translated_aa_ints[j,not_ref_mask[j]]]

                mut_key = tuple([f"{wt}{res}{mut}" for wt, res, mut in zip(mut_wt_aa,
                                                                        mut_resid,
                                                                        mut_aa)])
            if mut_key not in final_seq_counts:
                final_seq_counts[mut_key] = 0
            final_seq_counts[mut_key] += batch_counts[j]

    return final_seq_counts

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
    

