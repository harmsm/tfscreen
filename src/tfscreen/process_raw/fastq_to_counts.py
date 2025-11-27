from tfscreen.data import (
    COMPLEMENT_DICT
)

from tfscreen.genetics import LibraryManager
from tfscreen.util.check import (
    check_number,
)
from tfscreen.util import (
    strict_array_search,
    fuzzy_array_search
)

import numpy as np
import numba
import pybktree
from typing import Optional, Tuple

from itertools import takewhile
import random, string, sys

@numba.jit(nopython=True, fastmath=True)
def _hamming_dist(s1, s2):
    """
    Calculate the Hamming distance between two 1D numpy arrays of the same 
    length.
    """
    dist = 0
    for i in range(s1.shape[0]):
        if s1[i] != s2[i]:
            dist += 1
    return dist


class FastqToCounts:
    """
    Processes paired-end sequencing reads to call genotypes at the amino acid
    level.

    This class provides a complete pipeline to take raw, paired-end FASTQ
    reads (as NumPy arrays) and converts them into a final genotype call based
    on a defined genetic library. The process involves:
    1.  Applying a PHRED quality score cutoff to mask low-quality bases.
    2.  Identifying 5' and 3' flanking sequences, allowing for mismatches.
    3.  Orienting the forward and reverse reads.
    4.  Building consensus sequences from the overlapping reads.
    5.  Using a hierarchical set of rules to reconcile any disagreements.
    6.  Matching the final, reconciled sequence against the expected library.

    Attributes
    ----------
    phred_cutoff : int
        Bases with a quality score below this are treated as ambiguous.
    min_read_length : int
        Minimum required length of the payload after flank trimming.
    allowed_num_flank_diffs : int
        Maximum mismatches allowed when finding flanking sequences.
    allowed_diff_from_expected : int
        Maximum mismatches allowed when matching a read to the library.
    expected_5p : str
        The expected 5' flanking sequence as a string.
    expected_3p : str
        The expected 3' flanking sequence as a string.
    expected_length : int
        The expected length of the payload sequence between the flanks.
    all_expected_genotypes : list
        A list of all expected genotypes (protein level. wildtype will be `wt`,
        mutants like H29A, H29A/M98I, etc.) 
    """

    def __init__(self,
                 lm: LibraryManager,
                 phred_cutoff: int = 10,
                 min_read_length: int = 50,
                 allowed_num_flank_diffs: int = 1,
                 allowed_diff_from_expected: int = 2,
                 print_raw_seq: bool = False) -> None:
        """
        Initializes the FastqToCounts processor.

        Parameters
        ----------
        lm : LibraryManager
            An initialized `LibraryManager` instance that contains the run
            configuration (flanks, expected length) and the library of
            expected DNA/protein sequences.
        phred_cutoff : int, optional
            Bases with a PHRED score below this cutoff will be treated as
            ambiguous ('N'), by default 10.
        min_read_length : int, optional
            The minimum required length of a read's payload after flanks
            have been identified and removed, by default 50.
        allowed_num_flank_diffs : int, optional
            The maximum number of mismatches (Hamming distance) allowed when
            searching for flanking sequences. If 0, an exact match is
            required. By default 1.
        allowed_diff_from_expected : int, optional
            The maximum number of mismatches (Hamming distance) allowed when
            comparing a reconciled read sequence to the sequences in the
            expected library, by default 2.
        print_raw_seq : bool, optional
            if True, print the foward and reverse sequences to standard out.
            These are written with random 10 letter prefixes in the order 
            they are encountered.  This only writes out correctly oriented
            sequences, but does not alter the sequences in any way. Setting
            this to True with a high number of allowed_num_flank_diffs reveals
            sequences that are somewhat close to expected and can be used for
            troubleshooting. The sequences include the 5' and 3' flank regions.
        """

        # LibraryManager instance
        if not isinstance(lm,LibraryManager):
            raise ValueError("`lm` must be a LibraryManager instance")

        # Load arguments
        self._lm = lm
        self.phred_cutoff = check_number(phred_cutoff,
                                         param_name="phred_cutoff",
                                         cast_type=int,
                                         min_allowed=1)
        self.min_read_length = check_number(min_read_length,
                                            param_name="min_read_length",
                                            cast_type=int,
                                            min_allowed=0)
        self.allowed_num_flank_diffs = check_number(allowed_num_flank_diffs,
                                                    param_name="allowed_num_flank_diffs",
                                                    cast_type=int,
                                                    min_allowed=0)
        self.allowed_diff_from_expected = check_number(allowed_diff_from_expected,
                                                       param_name="allowed_diff_from_expected",
                                                       cast_type=int,
                                                       min_allowed=0)

        self.print_raw_seq = print_raw_seq
        
        # the LibraryManager yaml parser does not require/check for expected 
        # 5p and 3p flanks. (This is because it can be used for simulation 
        # input as well as experimental input.) Check for these here. 
        if "expected_5p" not in self._lm.run_config:
            raise ValueError(
                "run configuration must define `expected_5p`."
            )
        
        if "expected_3p" not in self._lm.run_config:
            raise ValueError(
                "run configuration must define `expected_3p`."
            )
        
        self.expected_5p = self._lm.run_config["expected_5p"]
        self.expected_3p = self._lm.run_config["expected_3p"]

        # initialize inference
        self._initialize_converters()
        self._initialize_expected_seq()
        
    def _initialize_converters(self):
        """
        Initialize converters that go between base <--> integer, reverse
        complement, and let us look up things like ambiguous base number.
        """

        number_to_base = "ACGT-N"
        base_to_number = dict([(b, i) for i, b in enumerate(number_to_base)])
        base_to_number.update([(b.lower(), i) for i, b in enumerate(number_to_base)])

        # Create a vectorized lookup table for base-to-number conversion
        fast_base_to_number = np.zeros(256, dtype=np.uint8)
        for base, num in base_to_number.items():
            fast_base_to_number[ord(base)] = num

        self.number_to_base = number_to_base
        self.base_to_number = base_to_number
        self.fast_base_to_number = fast_base_to_number

        # Create a full-size lookup table. Initialize it so that any base
        # without a defined complement is its own complement (e.g., N -> N).
        complement_int = np.arange(len(self.number_to_base), dtype=np.uint8)

        # Fill in the known complements from the source dictionary
        for base, comp_base in COMPLEMENT_DICT.items():
            base_num = self.base_to_number[base.lower()]
            comp_num = self.base_to_number[comp_base.lower()]
            if base_num is not None and comp_num is not None:
                complement_int[base_num] = comp_num
        
        self.complement_int = complement_int
        
        self.ambig_base_number = self.base_to_number["N"]
        self.gap_base_number = self.base_to_number["-"]

    def _initialize_expected_seq(self):
        """
        Build: 
         + integer representations of flanks/expected length
         + _bytes_to_aa (for looking up amino acid genotype with a bytes
           representation of a dna sequence)
         + _search_tree (for looking for reads in the expected library).
        """

        # Define overall features of the read -- flanks and length
        self.expected_5p_int = np.array([self.base_to_number[b]
                                         for b in self.expected_5p],
                                         dtype=np.uint8)
        self.expected_3p_int = np.array([self.base_to_number[b]
                                         for b in self.expected_3p],
                                         dtype=np.uint8)
        
        self.expected_5p_size = len(self.expected_5p)
        self.expected_3p_size = len(self.expected_3p)
        self.expected_length = self._lm.expected_length

        # Get dna and amino acid sequences specified by library. These come out
        # as dictionaries with sub-libraries as keys and sequences as values. 
        # dna_lib will look like:
        #    {"single-1":["atcga...","atccg...","atcgg...",...],...}. 
        # aa_lib will look like:
        #    {"single-1":["H29A","","H29A/M98I",...],...}
        # The lists will be in order relative to one another. wildtype will 
        # appear as '""' in the aa_lib. 
        dna_lib, aa_lib = self._lm.get_libraries()

        # Build a list of all unique (dna,aa) tuples. 
        all_sequences = set()
        for key in dna_lib:
            all_sequences.update(list(zip(dna_lib[key],aa_lib[key])))

        all_dna_as_ints = []
        all_dna_as_bytes = []

        # Build two dictionaries. bytes_to_num uses the bytes representation of
        # the dna sequence to look up the number of amino acid mutations in the
        # genotype. bytes_to_aa looks up the amino acid genotype. 
        self._bytes_to_aa = {}
        self.all_expected_genotypes = []
        for dna, prot in all_sequences:

            dna_as_ints = np.array([self.base_to_number[b] for b in dna],
                                   dtype=np.uint8)
            all_dna_as_ints.append(dna_as_ints)

            dna_as_bytes = dna_as_ints.tobytes()
            all_dna_as_bytes.append(dna_as_bytes)

            if prot == "":
                aa = "wt"
            else: 
                aa = prot
    
            if dna_as_bytes in self._bytes_to_aa:
                if self._bytes_to_aa[dna_as_bytes] != aa:
                    raise ValueError(
                        f"DNA sequence '{dna}' is mapped to multiple different "
                        f"genotypes ('{self._bytes_to_aa[dna_as_bytes]}' and "
                        f"'{aa}'). Please correct the library."
                    )

            self._bytes_to_aa[dna_as_bytes] = aa
            self.all_expected_genotypes.append(aa)
        
        # Build the search tree we will use to see if a read matches a seq
        # in the library (Hamming distance). This should take a read as an 
        # integer array as its search query. 
        self._search_tree = pybktree.BKTree(_hamming_dist, all_dna_as_ints)

                
    def _find_orientation_strict(self, f1_array, f2_array):
        """
        Identifies the forward and reverse reads based on the flanking
        sequences. Does not allow any mismatches.
        """
        # Look for the start of the 5p flank in f1_array
        idx0 = strict_array_search(f1_array, self.expected_5p_int)

        if idx0 >= 0:
            fwd = f1_array
            rev = self.complement_int[f2_array][::-1]
        else:
            idx0 = strict_array_search(f2_array, self.expected_5p_int)
            if idx0 < 0:
                return None, None, "fail, could not find expected 5p flank"
            fwd = f2_array
            rev = self.complement_int[f1_array][::-1]

        # Look for the start of the 3p flank in the reverse read
        idx1 = strict_array_search(rev, self.expected_3p_int)
        if idx1 < 0:
            return None, None, "fail, could not find expected 3p flank"

        # The forward sequence starts after the 5' flank. The reverse is 
        # everything from the start to the beginning of the 3' flank.
        fwd_seq = fwd[idx0 + self.expected_5p_size:]
        rev_seq = rev[:idx1]

        return fwd_seq, rev_seq, None
    
    def _find_orientation_fuzzy(self,f1_array,f2_array):
        """
        Identify the forward and reverse reads by looking for matching 5p and 3p
        flanking sequences. Allow up to allowed_num_flank_diffs mismatches.
        """
        
        f1_score, f1_idx = fuzzy_array_search(f1_array,self.expected_5p_int)
        f2_score, f2_idx = fuzzy_array_search(f2_array,self.expected_5p_int)
        
        if np.min([f1_score,f2_score]) > self.allowed_num_flank_diffs:
            return None, None, "fail, could not find expected 5p flank"

        if f1_score < f2_score:
            idx0 = f1_idx + self.expected_5p_size
            fwd = f1_array
            rev = self.complement_int[f2_array][::-1]
        else:
            idx0 = f2_idx + self.expected_5p_size
            fwd = f2_array
            rev = self.complement_int[f1_array][::-1]

        idx1_score, idx1 = fuzzy_array_search(rev,self.expected_3p_int)
        
        if idx1_score > self.allowed_num_flank_diffs:
            return None, None, "fail, could not find expected 3p flank"
       
        if self.print_raw_seq:
            
            L = self.expected_length + self.expected_5p_size + self.expected_3p_size
            mapper = np.asarray(list(self.number_to_base),dtype=object)

            fwd_out = np.full(L,self.ambig_base_number,dtype=np.uint8)
            tmp_F = fwd[(idx0 - self.expected_5p_size):]
            tmp_F = tmp_F[:L]
            fwd_out[:len(tmp_F)] = tmp_F
            
            fwd_str_seq = "".join(list(mapper[fwd_out]))
          
            rev_out = np.full(L,self.ambig_base_number,dtype=np.uint8)
            tmp_R = rev[:(idx1 + self.expected_3p_size)] 
            if len(tmp_R) > L:
                tmp_R = tmp_R[-L:] 
            rev_out[-len(tmp_R):] = tmp_R

            rev_str_seq = "".join(list(mapper[rev_out]))
        
            prefix = "".join([random.choice(string.ascii_letters) for _ in range(10)])

            sys.stdout.write(f">fwd_{prefix}\n{fwd_str_seq}\n")
            sys.stdout.write(f">rev_{prefix}\n{rev_str_seq}\n")
            
         
        # The forward sequence starts after the 5' flank. The reverse is 
        # everything from the start to the beginning of the 3' flank.
        fwd_seq = fwd[idx0:]
        rev_seq = rev[:idx1]

        return fwd_seq, rev_seq, None


    def _build_call_pair(self,fwd_seq,rev_seq):
        """
        Build two arrays of base calls: one where the fwd_seq wins at shared
        unambiguous sites and one where the rev_seq at shared unambiguous
        sites. 
        """

        # For both fwd and rev: 1) build a seq, which has the base sequence of 
        # the read with somewhere between min_read_length and expected_length. 
        # 2) build a load_mask, which has length expected_length and is 1 at
        # each position where the read has an unambiguous base call and 0 
        # elsewhere. 
        fwd_seq = fwd_seq[:self.expected_length]
        fwd_good = fwd_seq != self.ambig_base_number
        fwd_load_mask = np.zeros(self.expected_length,dtype=bool)
        fwd_load_mask[:len(fwd_good)] = fwd_good

        if len(rev_seq) > self.expected_length:
            rev_seq = rev_seq[-self.expected_length:]
        rev_good = rev_seq != self.ambig_base_number
        rev_load_mask = np.zeros(self.expected_length,dtype=bool)
        rev_load_mask[-len(rev_seq):] = rev_good

        # Build an array where fwd trumps rev at unambiguous sites. Load rev 
        # then fwd. 
        fwd_wins = np.full(self.expected_length,
                           self.ambig_base_number,
                           dtype=np.uint8)
        fwd_wins[rev_load_mask] = rev_seq[rev_good]
        fwd_wins[fwd_load_mask] = fwd_seq[fwd_good]

        # Build an array where rev trumps fwd at unambiguous sites
        rev_wins = np.full(self.expected_length,
                           self.ambig_base_number,
                           dtype=np.uint8)
        rev_wins[fwd_load_mask] = fwd_seq[fwd_good]
        rev_wins[rev_load_mask] = rev_seq[rev_good]

        return fwd_wins, rev_wins

    def build_call_pair(self,
                        f1_array: np.ndarray,
                        f2_array: np.ndarray,
                        f1_q: np.ndarray,
                        f2_q: np.ndarray) -> Tuple[Optional[str], str]:
        """
        Processes a single pair of raw reads to determine their orientation. 

        This function outputs two arrays: fwd_wins (assuming fwd is right at 
        all disagreeing, non-ambiguous positions) and rev_wins (assuming rev 
        is right at all disagreeing, non-ambiguous positions). Unambiguous sites
        on rev that are ambiguous on fwd are copied into fwd_wins; likewise for
        rev_wins. fwd_wins and rev_wins can be reconciled in subsequent steps
        (see .reconcile_reads). Both arrays are represented as bytes so they 
        can be used as keys in Counters(). 

        Parameters
        ----------
        f1_array : numpy.ndarray
            1D array of `uint8` representing the DNA sequence of the first read
            in the pair.
        f2_array : numpy.ndarray
            1D array of `uint8` representing the DNA sequence of the second read
            in the pair.
        f1_q : numpy.ndarray
            1D array of PHRED quality scores for the first read, same length
            as `f1_array`.
        f2_q : numpy.ndarray
            1D array of PHRED quality scores for the second read, same length
            as `f2_array`.

        Returns
        -------
        tuple[str or None, str or None]
            A tuple containing two elements:
            1. (fwd_wins,rev_wins) (if successful) or None (if failed)
            2. None (if successful) or str (if failed)
        """
        
        # Set base ambiguity based on phred_cutoff
        f1_array[f1_q < self.phred_cutoff] = self.ambig_base_number
        f2_array[f2_q < self.phred_cutoff] = self.ambig_base_number

        # Identify forward and reverse based on expected flanks. 
        if self.allowed_num_flank_diffs > 0:
            fwd_seq, rev_seq, msg = self._find_orientation_fuzzy(f1_array,
                                                                 f2_array)
        else:
            fwd_seq, rev_seq, msg = self._find_orientation_strict(f1_array,
                                                                  f2_array)

        # fwd will be None if this failed. Return the message.
        if fwd_seq is None:
            return None, msg
        
        # Check lengths of reads
        if len(fwd_seq) < self.min_read_length:
            return None, "fail, F sequence too short"
        if len(rev_seq) < self.min_read_length:
            return None, "fail, R sequence too short"

        # Build arrays assuming either forward or reverse is right at all 
        # unambiguous sites. These could either agree completely or have 
        # disagreements. If they disagree, they disagree at sites that passed 
        # phred_cutoff
        fwd_wins, rev_wins = self._build_call_pair(fwd_seq, rev_seq)

        call_pair = (fwd_wins.tobytes(),rev_wins.tobytes())
        return call_pair, None

    def _search_expected_lib(self,seq,max_diffs):
        """
        Search the expected sequences seq, allowing up to max_diffs differences.
        Return a list of tuples corresponding to the nearest 'shell' of matches.
        (If the closest match has 0 diffs, return 1 sequence; if the closest 
        match has 1 diff, return all 1 diff sequences, etc.)
        """

        # We can't query the bktree with something that is not the same length
        # as the input sequences. 
        if len(seq) != self.expected_length:
            return [], -1

        # -- pybktree --
        # # Look for matches in the search tree. 
        matches = self._search_tree.find(seq,max_diffs)
        if not matches:
            return [], -1

        # This grabs only the hits that match the minimum distance seen in the
        # search. Using takewhile, we only iterate until m[0] != min_dist. 
        min_dist = matches[0][0]
        shell_matches = [m[1] for m in takewhile(lambda m: m[0] == min_dist, matches)]
        return shell_matches, min_dist

    def _rr_perfect_agreement(self,fwd_wins,rev_wins):
        """
        Check whether F and R match exactly.
        """
        
        # If fwd and rev match exactly. 
        if np.array_equal(fwd_wins, rev_wins):

            matches, _ = self._search_expected_lib(fwd_wins,
                                                   self.allowed_diff_from_expected)

            if len(matches) == 0:
                return False, None, "fail, F/R agree but their sequence is not in the expected library"
            elif len(matches) == 1:
                return False, matches[0], "pass, F/R agree exactly" 
            else:
                return False, None, "fail, F/R agree but match more than one expected sequence"

        return True, None, None
        
    def _rr_one_sided_match(self,matches_f,matches_r):
        """
        Check to see if F or R matches by the other does not.
        """
        
        # F does not match anything in the expected library
        if len(matches_f) == 0:

            if len(matches_r) == 0:
                return False, None, "fail, F/R disagree and neither sequence is expected"
            elif len(matches_r) == 1:
                return False, matches_r[0], "pass, F/R disagree but R is expected"
            else:
                return False, None, "fail, F/R disagree and F is not expected and R is ambiguous"
            
        # R does not match anything in the expected library
        if len(matches_r) == 0:
            
            if len(matches_f) == 1:
                return False, matches_f[0], "pass, F/R disagree but F is expected"
            else:
                return False, None, "fail, F/R disagree and R is not expected and F is ambiguous"

        return True, None, None

    def _rr_one_closer_match(self,matches_f,matches_r,match_dist_f,match_dist_r):
        """
        Check to see if F or R has a closer match.
        """

        # If F has a unique match closer than R...
        if match_dist_f < match_dist_r:
            if len(matches_f) == 1:
                return False, matches_f[0], "pass, F/R disagree but F is expected"
            
        # If R has a unique match closer than F...
        if match_dist_r < match_dist_f:
            if len(matches_r) == 1:
                return False, matches_r[0], "pass, F/R disagree but R is expected"

        return True, None, None
            
    def _rr_unique_intersection(self,matches_f,matches_r):
        """
        Check to see if f and r matches share exactly one unique sequence.
        """
        
        set_f = {m.tobytes() for m in matches_f}
        set_r = {m.tobytes() for m in matches_r}
        intersect = set_f.intersection(set_r)
        if len(intersect) == 1:
            seq_int = np.frombuffer(intersect.pop(), dtype=np.uint8)
            return False, seq_int, "pass, F/R have a unique shared expected sequence"
    
        return True, None, None
        
    def _reconcile_reads(self,fwd_wins,rev_wins):
        """
        Infer a single sequence from the forward wins and reverse wins arrays
        using a hierarchial set of rules. If we make a sequence call, it is 
        guaranteed to be in the input library. If the information from the two
        reads conflict or the sequence is not in the expected library, this
        returns None. 
        """
    
        # ----------------------------------------------------------------------
        # Handle simple case of perfect agreement first
        
        keep_going, seq, msg = self._rr_perfect_agreement(fwd_wins,rev_wins)
        if not keep_going: return seq, msg

        # ----------------------------------------------------------------------
        # If we get here, fwd and reverse do not match exactly. Look for each 
        # in the expected library.
        
        matches_f, match_dist_f = self._search_expected_lib(fwd_wins,
                                                            self.allowed_diff_from_expected)
        matches_r, match_dist_r = self._search_expected_lib(rev_wins,
                                                            self.allowed_diff_from_expected)

        # ----------------------------------------------------------------------
        # Check to see if one read matches and the other does not
        
        keep_going, seq, msg = self._rr_one_sided_match(matches_f,matches_r)
        if not keep_going: return seq, msg

        # ----------------------------------------------------------------------
        # If we get here, both F and R match something in the expected library.
        # Check to see if one is clearly better. 

        keep_going, seq, msg = self._rr_one_closer_match(matches_f,matches_r,
                                                         match_dist_f,match_dist_r)
        if not keep_going: return seq, msg
        
        # ----------------------------------------------------------------------
        # If we get here, both F and R match something in the expected library
        # and one is not obviously better than the other. Look for a single
        # shared sequence in both sets of matches. If this is present, take that
        # as our sequence. 

        keep_going, seq, msg = self._rr_unique_intersection(matches_f,matches_r)
        if not keep_going: return seq, msg
        
        # ----------------------------------------------------------------------
        # If we get here, both F and R match something in the expected library
        # within but do not point to a single shared sequence. Give up.
    
        return None, "fail, F/R match different sequences in expected"

    def reconcile_reads(self,fwd_wins,rev_wins):
        """
        Reconcile the sequences in `fwd_wins` and `rev_wins`. 
        
        This function assumes these sequences are the same length, on the 
        same strand (e.g., 5'->3'), and represented as integers. This function
        uses a set of hierarchical rules to decide on the correct sequence,
        ultimately returning either a sequence from the expected set of
        sequences defined in the LibraryManager or None.

        Parameters
        ----------
        fwd_wins: np.ndarray
            numpy integer arrays holding the DNA sequence (as integers) if the
            forward read is correct at all unambiguous sites. This assumes any
            ambiguous sites on forward have been replaced by unambiguous sites
            on reverse (if present)
        rev_wins: np.ndarray
            numpy integer arrays holding the DNA sequence (as integers) if the
            reverse read is correct at all unambiguous sites. This assumes any
            ambiguous sites on reverse have been replaced by unambiguous sites
            on forward (if present)

        Returns
        -------
        str, None:
            protein genotype of the sequence (if identified) or None
        str:
            string message describing the result of the analysis
        """

        # Reconcile the forward and reverse reads        
        seq, msg = self._reconcile_reads(fwd_wins,rev_wins)
        if seq is None:
            return None, msg

        # If we get here, we have a real sequence. Look up amino acid mutations
        # by fast bytes lookup and return final message
        final_seq = self._bytes_to_aa[seq.tobytes()]

        return final_seq, msg



