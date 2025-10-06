from tfscreen.data import (
    codon_to_aa,
    complement_dict
)

import numpy as np
from tqdm.auto import tqdm

import os
import gzip

class RunConfiguration:

    def __init__(self,
                 expected_5p,
                 expected_3p,
                 expected_length,
                 phred_cutoff,
                 min_read_length=50,
                 allowed_num_diffs=0):

        self.allowed_num_diffs = allowed_num_diffs

        self.expected_5p = expected_5p
        self.expected_3p = expected_3p
        self.expected_5p_int = self.str_to_int(self.expected_5p)
        self.expected_3p_int = self.str_to_int(self.expected_3p)
        self.expected_5p_size = len(expected_5p)
        self.expected_3p_size = len(expected_3p)
    
        self.expected_length = expected_length
        self.min_read_length = min_read_length
        self.phred_cutoff = phred_cutoff

        self._initialize_converters()

    def _initialize_converters(self):

        number_to_base = "ACGT-N"
        base_to_number = dict([(b,i) for i, b in enumerate(number_to_base)])
        base_to_number.update([(b.lower(),i) for i, b in enumerate(number_to_base)])

        # base <-> integer
        self.number_to_base = number_to_base
        self.base_to_number = base_to_number

        # load codon table, converting to uppercase
        self.codons = {}
        for k in codon_to_aa:
            self.codons[k.upper()] = codon_to_aa[k]

        # Load base complements, converting to uppercase
        self.complement_dict = {}
        for base in complement_dict:
            self.complement_dict[base.upper()] = complement_dict[base].upper()

        # Build numpy array for fast integer reverse complement lookup
        complement_int = {}
        for k in complement_dict:
            complement_int[base_to_number[k]] = base_to_number[complement_dict[k]]

        keys = list(complement_int.keys())
        keys.sort()
        complement_int = np.array([complement_int[k]
                                        for k in keys],dtype=np.uint8)

        self.complement_dict = complement_dict
        self.complement_int = complement_int

        self.ambig_base_number = self.base_to_number["N"]
        self.gap_base_number = self.base_to_number["-"]


    def rev_comp(self,sequence):
        """
        Return reverse complement of a sequence.
        """

        return "".join([self.complement_dict[s] for s in sequence[::-1]])

    def rev_comp_int(self,sequence):
        """
        Return reverse complement of sequence as integers.
        """

        return self.complement_int[sequence][::-1]

    def str_to_int(self,some_str):
        """
        Convert string to numpy integer array representation.
        """
        return np.array([self.base_to_number[b] for b in some_str],dtype=np.uint8)

    def int_to_str(self,int_array):
        """
        Comvert numpy integer array into string representation.
        """
        return "".join([self.number_to_base[i] for i in int_array])


    def translate(self,some_str):
        """
        Translate a string sequence using the codons dictionary.
        """

        aa_list = []
        for i in range(0,len(some_str),3):
            try:
                aa_list.append(self.codons[some_str[i:(i+3)]])
            except KeyError:
                aa_list.append("X")

        return "".join(aa_list)


def _strict_search_array_for_seq(search_in,search_for):
    """
    Find sequence in an array using NumPy.

    Parameters
    ----------
    search_in : numpy.ndarray
        1D numpy array to search IN
    search_for : numpy.ndarray
        1D numpy array to search FOR

    Output
    ------
    int : 
        If nothing is found, return -1. If one or more matches are found, return
        the index *after* the end of the first match.

    Examples
    --------
        1101010 10 would return 3
        1000000 10 would return 2
        0000000 10 would return -1
    
    Notes
    -----
    https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """

    # Store sizes of input array and sequence
    Na, Nseq = search_in.size, search_for.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (search_in[np.arange(Na-Nseq+1)[:,None] + r_seq] == search_for).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        matches = np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
        return matches[search_for.size - 1] + 1

    # No match
    return -1

def _find_orientation_strict(f1_array,f2_array,rc):
    """
    """

    if _strict_search_array_for_seq(f1_array,rc.expected_5p_int) >= 0:
        fwd = f1_array
        rev = rc.rev_comp_int(f2_array)
    else:
        fwd = f2_array
        rev = rc.rev_comp_int(f1_array)

    # search_array_for_seq returns index *after* match
    idx0 = _strict_search_array_for_seq(fwd,rc.expected_5p_int)
    idx1 = _strict_search_array_for_seq(rev,rc.expected_3p_int)
    idx1 = idx1 - rc.expected_3p_size

    if idx0 < 0 or idx1 < 0:
        return None, None, None, None, "could not find F and/or R"

    return fwd, rev, idx0, idx1, None

def _fuzzy_search_array_for_seq(search_in,search_for):
    """
    Fuzzy match of sequences in numpy array. In case of equivalent matches,
    returns the first it encounters.

    Parameters
    ----------
    search_in : numpy.ndarray
        1D numpy array to search IN. This must be longer than search_for. 
    search_for : numpy.ndarray
        1D numpy array to search FOR

    Returns
    -------
    int :
        number of seq differences for best match
    int :
        index of start of match
    """

    # indexer is an array like this: [[0,1,2],[1,2,3],[2,3,4],[3,4,5]] if we
    # are searching for something of length three in an array that is length 5.
    search_length = search_for.shape[0]
    num_rows = search_in.shape[0] - search_length + 1
    indexer = np.expand_dims(np.arange(search_length),0) + \
              np.expand_dims(np.arange(num_rows),0).T

    # Score will be a vector with the number of exact matches for each possible
    # position of the search array on the sequence
    score = np.sum(search_in[indexer] == search_for,axis=1)
    match_index = np.argmax(score)
    match_score = score[match_index]

    return len(search_for) - match_score, match_index

def _find_orientation_fuzzy(f1_array,f2_array,rc):
    
    f1_score, f1_idx = _fuzzy_search_array_for_seq(f1_array,rc.expected_5p_int)
    f2_score, f2_idx = _fuzzy_search_array_for_seq(f2_array,rc.expected_5p_int)
    
    if np.min([f1_score,f2_score]) > rc.allowed_num_diffs:
        return None, None, None, None, "could not match F search"

    if f1_score < f2_score:
        idx0 = f1_idx + rc.expected_5p_size
        fwd = f1_array
        rev = rc.rev_comp_int(f2_array)
    else:
        idx0 = f2_idx + rc.expected_5p_size
        fwd = f2_array
        rev = rc.rev_comp_int(f1_array) 

    idx1_score, idx1 = _fuzzy_search_array_for_seq(rev,rc.expected_3p_int)
    
    if idx1_score > rc.allowed_num_diffs:
        return None, None, None, None, "could not match R search"

    return fwd, rev, idx0, idx1, None

def _process_read_pair(f1_array,f2_array,f1_q,f2_q,rc):
    
    # Update base ambiguity based on phred_cutoff
    f1_array[f1_q < rc.phred_cutoff] = rc.ambig_base_number
    f2_array[f2_q < rc.phred_cutoff] = rc.ambig_base_number

    # # Look for flank to identify forward/reverse
    if rc.allowed_num_diffs > 0:
        fwd, rev, idx0, idx1, msg = _find_orientation_fuzzy(f1_array,f2_array,rc)
    else:
        fwd, rev, idx0, idx1, msg = _find_orientation_strict(f1_array,f2_array,rc)

    # fwd will be None if this failed. Return the message.
    if fwd is None:
        return None, msg

    final_seq = np.ones(rc.expected_length,dtype=np.uint8)*rc.ambig_base_number

    fwd_seq = fwd[idx0:]
    fwd_seq = fwd_seq[:rc.expected_length]
    if len(fwd_seq) < rc.min_read_length:
        return None, "F sequence too short"

    rev_seq = rev[:idx1]
    if len(rev_seq) > rc.expected_length:
        rev_seq = rev_seq[-rc.expected_length:]
    if len(rev_seq) < rc.min_read_length:
        return None, "R sequence too short"

    rev_good = rev_seq != rc.ambig_base_number
    rev_load_mask = np.zeros(rc.expected_length,dtype=bool)
    rev_load_mask[-idx1:] = rev_good

    # load fwd seq, then rev seq. rev seq is masked so it does
    # not bring in ambiguous <-- 
    final_seq[:fwd_seq.shape[0]] = fwd_seq
    final_seq[rev_load_mask] = rev_seq[rev_good]
    
    final_seq = rc.int_to_str(final_seq)

    return final_seq, "passed"

def count_seqs(f1_fastq,
               f2_fastq,
               out_dir,
               expected_5p,
               expected_3p,
               expected_length,
               phred_cutoff=10,
               min_read_length=50,
               allowed_num_diffs=4,
               max_num_reads=None):
    """
    Extract, align, and parse paired-end next-generation sequencing reads. 

    Parameters
    ----------
    f1_fastq, f2_fastq : str, str
        fastq/fastq.gz with paired-end reads
    out_dir : str
        output directory
    phred_cutoff : int
        assign any base with phred < cutoff to "N"
    expected_5p : str
        expected sequence just 5' of the region to be sequenced (fwd direction)
    expected_3p : str
        expected sequence just 3' of the region to be sequenced (fwd direction)
    expected_length : int
        expected length of gene between (but not including) expected 5p and 3p 
        bits.
    min_read_length : int, default=50
        toss reads where the final readable sequence is less than this long
    allowed_num_diffs : int, default=3
        allow up to allowed_num_diffs differences between the read and 
        expected_5p and expected_3p when looking for them.
    max_num_reads : int, optional
        only read the first max_num_reads. if None, read whole files.
    """

    # Make output directory if it does not already exist
    if os.path.exist(out_dir):
        if not os.path.isdir(out_dir):
            raise FileExistsError(
                f"out_dir '{out_dir}' already exists and is not a directory."
            )
    else:
        os.makedirs(out_dir)

    # Figure out what kind of file this is. compression_ratio is hack to make
    # the status bar give approximately correct results for gzipped fastq
    # files.
    if f1_fastq[-3:] == ".gz":
        open_function = gzip.open
        compression_ratio = 6
    else:
        open_function = open
        compression_ratio = 1

    # make empty barcode dict
    barcode_dict = {}

    # Set up a configuration object that we will use to control the processing
    rc = RunConfiguration(expected_5p=expected_5p,
                          expected_3p=expected_3p,
                          expected_length=expected_length,
                          phred_cutoff=phred_cutoff,
                          min_read_length=min_read_length,
                          allowed_num_diffs=allowed_num_diffs)

    # Dictionary to hold result types
    stats = {"could not find barcode start":0,
             "could not match F search":0,
             "could not match R search":0,  
             "could not find F and/or R":0,
             "F sequence too short":0,
             "R sequence too short":0,
             "passed":0}

    # Read through the files. counter keeps track of the line type (0-3),
    # total_counter keeps track of how many lines have been read.
    total_counter = 0
    counter = 0

    with tqdm(total=os.path.getsize(f1_fastq)) as pbar:
        with open_function(f1_fastq,"rb") as f1, open_function(f2_fastq,"rb") as f2:
            for f1_line in f1:
                f2_line = f2.readline()

                pbar.update(len(f1_line)//compression_ratio)

                # If line 0, get first chunk of read id and use as key
                if counter == 0:
                    f1_id = f1_line.decode().split()[0]
                    f2_id = f2_line.decode().split()[0]

                # sequence
                elif counter == 1:
                    f1_array = np.array([rc.base_to_number[b.upper()]
                                         for b in f1_line.decode()[:-1]],
                                        dtype=np.uint8)
                    f2_array = np.array([rc.base_to_number[b.upper()]
                                         for b in f2_line.decode()[:-1]],
                                        dtype=np.uint8)
                # phred
                elif counter == 3:
                    f1_q = np.array([ord(q)-33 for q in f1_line.decode()[:-1]],dtype=int)
                    f2_q = np.array([ord(q)-33 for q in f2_line.decode()[:-1]],dtype=int)

                # Update counter
                counter += 1

                # We've reached end of a read; process it
                if counter == 4:

                    counter = 0

                    # Read ids did not match -- do not process
                    if f1_id != f2_id:
                        stats["read id mismatch"] += 1
                        continue

                    final_seq, msg = _process_read_pair(f1_array,f2_array,
                                                        f1_q,f2_q,rc)
                    
                    if final_seq:
                        if final_seq not in barcode_dict:
                            barcode_dict[final_seq] = 0
                        barcode_dict[final_seq] += 1

                    # Record message
                    stats[msg] += 1

                    # Total number of reads
                    total_counter += 1
                    if max_num_reads is not None:
                        if total_counter >= max_num_reads:
                            break


    # Write out stats
    stats_file = os.path.join(out_dir,
                              f"stats_{os.path.split(f1_fastq)[-1]}.csv")
    f  = open(stats_file,"w")
    for k in stats:
        f.write(f"{k},{stats[k]}\n")
    f.close()

    # Write out obs
    obs_file = os.path.join(out_dir,
                            f"obs_{os.path.split(f1_fastq)[-1]}.csv")
    f  = open(obs_file,"w")
    for k in barcode_dict:
        f.write(f"{k},{barcode_dict[k]}\n")
    f.close()









