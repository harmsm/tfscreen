from .fastq_to_counts import FastqToCounts
from tfscreen.util.cli import generalized_main
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
from typing import Optional, Tuple, Iterable, Union, List
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)

def _process_reads_chunk(chunk: List[Tuple], ftc_instance: 'FastqToCounts') -> Tuple[Counter, Counter]:
    """
    Worker function to process a chunk of read pairs. This will be executed in a
    separate process. It returns a counter with all fwd/rev read pairs seen 
    and a counter with error messages. 
    """

    # Counters to hold results
    fwd_rev_pairs = Counter()
    messages = Counter()

    # Go through pairs of pyfastx reads
    for read1, read2 in chunk:

        # Extract information
        name1, seq1, qual1 = read1
        name2, seq2, qual2 = read2

        # Check for id mismatch
        if name1.split()[0] != name2.split()[0]:
            messages["fail, read id mismatch"] += 1
            continue

        # Get sequences as integer arrays
        f1_array = ftc_instance.fast_base_to_number[np.frombuffer(seq1.encode('ascii'),dtype=np.uint8)]
        f2_array = ftc_instance.fast_base_to_number[np.frombuffer(seq2.encode('ascii'),dtype=np.uint8)]
        
        # Get quality scores as integers
        f1_q = np.frombuffer(qual1.encode('ascii'), dtype=np.uint8) - 33
        f2_q = np.frombuffer(qual2.encode('ascii'), dtype=np.uint8) - 33

        # Call pair and update counters
        pairs, msg = ftc_instance.build_call_pair(f1_array, f2_array, f1_q, f2_q)

        # Record pairs identified or, if that failed, the message returned. 
        if pairs is not None:
            fwd_rev_pairs[pairs] += 1
        else:
            messages[msg] += 1
        
    return fwd_rev_pairs, messages

def _process_pairs_chunk(fwd_rev_chunk: List,
                         fwd_rev_counter: Counter,
                         ftc_instance: FastqToCounts) -> Tuple[Counter,Counter]:
    """
    Worker function to process a chunk of fwd/rev pairs, generating final
    protein sequence calls. This is designed to be executed as its own process.
    It returns a counter with all sequences seen, as well as messages from the
    caller.
    """

    # Counters to hold results
    sequences = Counter()
    messages = Counter()

    # Grab fwd_wins, and rev_wins pair
    for pair in fwd_rev_chunk:

        # Call sequence from int array versions of fwd_wins and rev_wins
        seq, msg = ftc_instance.reconcile_reads(np.frombuffer(pair[0],dtype=np.uint8),
                                                np.frombuffer(pair[1],dtype=np.uint8))

        # Get the number of times we saw this pair 
        num_seen = fwd_rev_counter[pair]

        # Update counters with the number of times we saw this in the first 
        # pass to identify fwd/rev
        if seq is not None:
            sequences[seq] += num_seen
        messages[msg] += num_seen

    return sequences, messages

def _process_paired_fastq(f1_fastq: str,
                          f2_fastq: str,
                          ftc_instance: FastqToCounts,
                          max_num_reads: Optional[int],
                          chunk_size: int = 1000,
                          num_workers: int | None = None) -> Tuple[Counter, Counter]:
    """
    Process paired-end FASTQ files and produce sequence and message counts.

    This reads the two paired FASTQ files in parallel, processes, applies 
    quality filters, and makes final genotype calls using `ftc_instance`, and
    accumulates counters for successful sequence calls and processing
    messages.

    Parameters
    ----------
    f1_fastq : str
        Path to the first (read 1) FASTQ file. Can be gzipped.
    f2_fastq : str
        Path to the second (read 2) FASTQ file. Must be the mate file for
        ``f1_fastq``.
    ftc_instance : FastqToCounts
        A configured :class:`FastqToCounts` instance that performs base-to-number
        mapping and the read-pair calling logic.
    max_num_reads : int or None
        If provided, only the first ``max_num_reads`` read pairs are processed.
        If ``None``, process the entire files.
    chunk_size : int
        break the reads into chunks of chunk_size reads for processing by their
        own workier
    num_workers : int or None
        number of workers to use. If not specified, set to os.cup_count() - 1

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
    
    # Figure out the number of workers if not specified
    if num_workers is None:
        num_workers = os.cpu_count() - 1

    # Counters to hold final results
    total_sequences = Counter()
    total_messages = Counter()

    # Handle empty input files gracefully before pyfastx tries to parse them.
    if os.path.getsize(f1_fastq) == 0:
        return total_sequences, total_messages

    # create pyfastx objects
    f1 = pyfastx.Fastx(f1_fastq)
    f2 = pyfastx.Fastx(f2_fastq)

    # Build iterator that will go through both fastq files
    read_iterator = zip(f1, f2)

    # Set the iterator to stop at max_num_reads if requested
    if max_num_reads is not None:
        read_iterator = itertools.islice(read_iterator, max_num_reads)
    
    # Create a pool for running analyses in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # Read through the fastq files, building chunks of reads and submitting
        # them to the pool so we can identify F/R and clean up the reads.
        reads_futures = []
        reads_chunk = []
        pbar_read_fastq = tqdm(read_iterator,desc="Reading fastq file", total=max_num_reads)
        for read_pair in pbar_read_fastq:
            reads_chunk.append(read_pair)
            if len(reads_chunk) == chunk_size:
                reads_futures.append(executor.submit(_process_reads_chunk,
                                                     reads_chunk,
                                                     ftc_instance))
                reads_chunk = []
        
        # Submit the last, leftover chunk. 
        if reads_chunk:
            reads_futures.append(executor.submit(_process_reads_chunk,
                                                 reads_chunk,
                                                 ftc_instance))

        # Aggregate final fwd/rev pairs as they complete
        fwd_rev_counter = Counter()
        pbar_fwd_rev = tqdm(as_completed(reads_futures),
                            total=len(reads_futures),
                            desc="Finding F/R pairs")
        for future in pbar_fwd_rev:
            pairs, messages = future.result()
            fwd_rev_counter.update(pairs)
            total_messages.update(messages)

        # Submit chunks of fwd/rev pairs to workers for genotype calling
        call_futures = []
        all_pairs = list(fwd_rev_counter.keys())
        chunk_size = int(np.ceil(len(all_pairs)/num_workers))
        for i in range(0,len(all_pairs),chunk_size):
            call_futures.append(executor.submit(_process_pairs_chunk,
                                                all_pairs[i:(i+chunk_size)],
                                                fwd_rev_counter,
                                                ftc_instance))
        
        # Aggregate final calls as they complete
        pbar_process = tqdm(as_completed(call_futures),
                            total=len(call_futures),
                            desc="Calling genotypes")
        for future in pbar_process:
            seq_calls, messages = future.result()
            total_sequences.update(seq_calls)
            total_messages.update(messages)

    return total_sequences, total_messages


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

def _create_counts_df(sequences: Counter,
                     expected_genotypes: Iterable[str],
                     messages: Optional[Counter] = None,
                     unknown_genotype_label: str = "__unknown__") -> pd.DataFrame:
    """
    Build a counts DataFrame for expected genotypes from a sequence counter.

    Parameters
    ----------
    sequences : collections.Counter
        Counter mapping genotype strings to observed counts.
    expected_genotypes : iterable of str
        List/iterable of expected genotype strings to include in the
        output DataFrame. Missing genotypes will have a count of zero.
    messages : collections.Counter, optional
        Counter mapping processing messages (strings) to counts. If provided,
        messages indicating valid but unknown/ambiguous genotypes will be
        aggregated into `unknown_genotype_label`.
    unknown_genotype_label : str, default "__unknown__"
        Label to use for the aggregated unknown genotype counts.

    Returns
    -------
    pandas.DataFrame
        DataFrame with categorical 'genotype' column and 'counts', standardized
        and sorted by ``set_categorical_genotype``.
    """

    if not expected_genotypes:
         return pd.DataFrame({"genotype": [], "counts": []}).astype({"genotype": object, "counts": int})

    # Make expected genotypes unique
    expected_genotypes = list(set(expected_genotypes))

    out_dict = {"genotype":[],
                "counts":[]}
    for g in expected_genotypes:
        out_dict["genotype"].append(g)
        out_dict["counts"].append(sequences[g])
    
    # Aggregrate unknown genotypes from messages. This looks for any 
    # failure mode that occurs AFTER the reads are successfully oriented 
    # and trimmed (e.g. they look like valid payload but are not in the 
    # library or are too ambiguous to call uniquely). 
    if messages is not None:

        unknown_messages = [
            "fail, F/R agree but their sequence is not in the expected library",
            "fail, F/R agree but match more than one expected sequence",
            "fail, F/R disagree and neither sequence is expected",
            "fail, F/R disagree and F is not expected and R is ambiguous",
            "fail, F/R disagree and R is not expected and F is ambiguous",
            "fail, F/R match different sequences in expected",
        ]
        num_unknown = 0
        for m in unknown_messages:
            num_unknown += messages[m]
        
        out_dict["genotype"].append(unknown_genotype_label)
        out_dict["counts"].append(num_unknown)

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
                  print_raw_seq: bool = False,
                  max_num_reads: Optional[int] = None,
                  chunk_size: int = 10000,
                  num_workers: int | None = None) -> None:
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
    print_raw_seq : bool, optional
        if True, print the foward and reverse sequences to standard out.
        These are written with random 10 letter prefixes in the order 
        they are encountered.  This only writes out correctly oriented
        sequences, but does not alter the sequences in any way. Setting
        this to True with a high number of allowed_num_flank_diffs reveals
        sequences that are somewhat close to expected and can be used for
        troubleshooting. The sequences include the 5' and 3' flank regions.
    max_num_reads : int, optional
        only read the first max_num_reads. if None, read whole files.
    chunk_size : int
        break the reads into chunks of chunk_size reads for processing by their
        own workier
    num_workers : int or None
        number of workers to use. If not specified, set to os.cup_count() - 1
    
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

    # Set up a FastqToCounts object for calling read pairs
    ftc_instance = FastqToCounts(lm,
                      phred_cutoff=phred_cutoff,
                      min_read_length=min_read_length,
                      allowed_num_flank_diffs=allowed_num_flank_diffs,
                      allowed_diff_from_expected=allowed_diff_from_expected,
                      print_raw_seq=print_raw_seq)
        
    # Extract sequence calls with statistics about outcomes
    sequences, messages = _process_paired_fastq(f1_fastq,f2_fastq,
                                                ftc_instance,
                                                max_num_reads,
                                                chunk_size,num_workers)
    
    # summarize and write out counts
    stats_df = _create_stats_df(messages)
    stats_file = os.path.join(out_dir,f"stats_{os.path.basename(f1_fastq)}.csv")
    stats_df.to_csv(stats_file,index=False)

    # summarize and write out counts    
    counts_df = _create_counts_df(sequences,
                                  ftc_instance.all_expected_genotypes,
                                  messages=messages)
    counts_file = os.path.join(out_dir,f"counts_{os.path.basename(f1_fastq)}.csv")
    counts_df.to_csv(counts_file,index=False)

def main():
    return generalized_main(process_fastq)
