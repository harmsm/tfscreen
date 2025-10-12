from .fastq_to_calls import FastqToCalls
from tfscreen.genetics import (
    LibraryManager,
    set_categorical_genotype
)

from tqdm.auto import tqdm
import pyfastx
import numpy as np
import pandas as pd

import os
from collections import Counter
import itertools
from typing import Optional, Tuple, Iterable, Union


def _process_paired_fastq(f1_fastq: str, f2_fastq: str, fc: FastqToCalls, max_num_reads: Optional[int]) -> Tuple[Counter, Counter]:
    """
    Process paired-end FASTQ files and produce sequence and message counts.

    This reads the two paired FASTQ files in parallel, converts bases to
    numeric representations using the provided :class:`FastqToCalls` instance,
    applies quality filtering and calling logic via ``fc.call_read_pair``,
    and accumulates counters for successful sequence calls and processing
    messages.

    Parameters
    ----------
    f1_fastq : str
        Path to the first (read 1) FASTQ file. Can be gzipped.
    f2_fastq : str
        Path to the second (read 2) FASTQ file. Must be the mate file for
        ``f1_fastq``.
    fc : FastqToCalls
        A configured :class:`FastqToCalls` instance that performs base-to-number
        mapping and the read-pair calling logic.
    max_num_reads : int or None
        If provided, only the first ``max_num_reads`` read pairs are processed.
        If ``None``, process the entire files.

    Returns
    -------
    sequences : collections.Counter
        Counter mapping called sequence strings (genotypes) to observed counts.
    messages : collections.Counter
        Counter mapping processing outcome messages (strings) to counts.

    Notes
    -----
    The function uses :mod:`pyfastx` to stream FASTQ records. If either input
    file is empty the function returns empty counters. Quality scores are
    converted from ASCII-33 (Phred+33) before calling ``fc.call_read_pair``.
    """
    
    # This will hold results
    sequences = Counter()
    messages = Counter()

    # Create a vectorized lookup table for base-to-number conversion
    # This is much faster than a dictionary lookup inside the loop.
    base_map = np.zeros(256, dtype=np.uint8)
    for base, num in fc.base_to_number.items():
        base_map[ord(base)] = num

    # Handle empty input files gracefully before pyfastx tries to parse them.
    if os.path.getsize(f1_fastq) == 0:
        return sequences, messages

    # create pyfastx objects
    f1 = pyfastx.Fastx(f1_fastq)
    f2 = pyfastx.Fastx(f2_fastq)

    # Build iterator that will go through both fastq files
    read_iterator = zip(f1, f2)

    # Set the iterator to stop at max_num_reads
    if max_num_reads is not None:
        read_iterator = itertools.islice(read_iterator, max_num_reads)
    
    # Wrap the zip iterator with tqdm (unknown length, but shows stuff is 
    # happening)
    pbar = tqdm(read_iterator, total=None, desc=os.path.basename(f1_fastq))

    # Iterate over files
    for read1, read2 in pbar:

        # Get names, sequences, and quality scores
        name1, seq1, qual1 = read1
        name2, seq2, qual2 = read2

        # Standard FASTQ headers (e.g., from Illumina) have read pair info
        # after a space. We check that the core ID matches.
        if name1.split()[0] != name2.split()[0]:
            messages["fail, read id mismatch"] += 1
            continue

        # Get numpy array representations of reads
        f1_array = np.array([fc.base_to_number[b] for b in seq1],dtype=np.uint8)
        f2_array = np.array([fc.base_to_number[b] for b in seq2],dtype=np.uint8)
        
        # Get numpy array representations of quality scores
        f1_q = np.frombuffer(qual1.encode('ascii'), dtype=np.uint8) - 33
        f2_q = np.frombuffer(qual2.encode('ascii'), dtype=np.uint8) - 33

        # call sequence
        seq_call, msg = fc.call_read_pair(f1_array, f2_array, f1_q, f2_q)

        # Record results
        messages[msg] += 1
        if seq_call is not None:
            sequences[seq_call] += 1

    return sequences, messages

def _create_stats_df(messages: Counter) -> pd.DataFrame:
    """
    Convert a messages counter into a summarized pandas DataFrame.

    Parameters
    ----------
    messages : collections.Counter
        Counter mapping processing messages (strings) to counts.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['success', 'result', 'counts', 'fraction'] where
        'success' is a boolean indicating whether the message started with
        'pass', 'result' is the remainder of the message, 'counts' is the raw
        count, and 'fraction' is the fraction of total messages.
    """
    
    if not messages:
        return pd.DataFrame({
            'success': pd.Series(dtype=bool),
            'result': pd.Series(dtype=object),
            'counts': pd.Series(dtype='int64'),
            'fraction': pd.Series(dtype=float)
        })

    df = pd.DataFrame(messages.items(), columns=['message', 'counts'])

    # Split message into 'success' and 'result'
    parts = df['message'].str.split(',', n=1, expand=True)
    df['success'] = (parts[0] == 'pass')
    df['result'] = parts[1].str.strip()

    df['fraction'] = df['counts'] / df['counts'].sum()

    return df[['success', 'result', 'counts', 'fraction']]

def _create_counts_df(sequences: Counter, expected_genotypes: Iterable[str]) -> pd.DataFrame:
    """
    Build a counts DataFrame for expected genotypes from a sequence counter.

    Parameters
    ----------
    sequences : collections.Counter
        Counter mapping genotype strings to observed counts.
    expected_genotypes : iterable of str
        List/iterable of expected genotype strings to include in the
        output DataFrame. Missing genotypes will have a count of zero.

    Returns
    -------
    pandas.DataFrame
        DataFrame with categorical 'genotype' column and 'counts', standardized
        and sorted by ``set_categorical_genotype``.
    """

    if not expected_genotypes:
         return pd.DataFrame({"genotype": [], "counts": []}).astype({"genotype": object, "counts": int})

    out_dict = {"genotype":[],
                "counts":[]}
    for g in expected_genotypes:
        out_dict["genotype"].append(g)
        out_dict["counts"].append(sequences[g])
    
    df = pd.DataFrame(out_dict)
    df = set_categorical_genotype(df,standardize=True,sort=True)

    return df


def process_fastq(f1_fastq: str,
                  f2_fastq: str,
                  out_dir: str,
                  run_config: Union[str, dict, LibraryManager],
                  phred_cutoff: int = 10,
                  min_read_length: int = 50,
                  allowed_num_flank_diffs: int = 1,
                  allowed_diff_from_expected: int = 2,
                  max_num_reads: Optional[int] = None) -> None:
    """
    Count the protein genotypes observed in a pair of fastq files. 

    Parameters
    ----------
    f1_fastq, f2_fastq : str, str
        fastq/fastq.gz with paired-end reads
    out_dir : str
        output directory
    run_config : str, dict, LibraryManager
        input file/dict to initialize a LibraryManager or a pre-initialized
        library instance. This defines the expected library. See the docstring
        for the `LibraryManager` class for more details on the inputs. 
    phred_cutoff : int
        assign any base with phred < cutoff to "N"
    min_read_length : int, default=50
        toss reads where the final readable sequence is less than this long
    allowed_num_flank_diffs : int, default=1
        allow up to allowed_num_diffs differences between the read and 
        expected_5p and expected_3p when looking for them.
    allowed_diff_from_expected : int, default=2
        allow up to allowed_diff_from_expected differences between the read and 
        the expected library sequences when calling the sequence
    max_num_reads : int, optional
        only read the first max_num_reads. if None, read whole files.
    
    Returns
    -------
    None
        Writes out two CSV files to ``out_dir``: a stats file and a counts file.
    """

    # Make output directory if it does not already exist
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise FileExistsError(f"out_dir '{out_dir}' already exists and is not a directory.")
    else:
        os.makedirs(out_dir)

    # Initialize library manager
    if isinstance(run_config,LibraryManager):
        lm = run_config
    else:
        lm = LibraryManager(run_config)

    # Set up a FastqToCalls object for calling read pairs
    fc = FastqToCalls(lm,
                      phred_cutoff=phred_cutoff,
                      min_read_length=min_read_length,
                      allowed_num_flank_diffs=allowed_num_flank_diffs,
                      allowed_diff_from_expected=allowed_diff_from_expected)
        
    # Extract sequence calls with statistics about outcomes
    sequences, messages = _process_paired_fastq(f1_fastq,f2_fastq,fc,max_num_reads)

    # summarize and write out counts
    stats_df = _create_stats_df(messages)
    stats_file = os.path.join(out_dir,f"stats_{os.path.basename(f1_fastq)}.csv")
    stats_df.to_csv(stats_file,index=False)

    # summarize and write out counts    
    counts_df = _create_counts_df(sequences,fc.all_expected_genotypes)
    counts_file = os.path.join(out_dir,f"counts_{os.path.basename(f1_fastq)}.csv")
    counts_df.to_csv(counts_file,index=False)



