import pytest
import numpy as np
from numpy.testing import assert_array_equal

# LibraryManager is imported to be used as a 'spec' for the mock object.
# This ensures the mock has the same public interface as the real class.
from tfscreen.genetics import LibraryManager
from tfscreen.process_raw.fastq_to_calls import FastqToCalls


# ------------------------
# Pytest Fixtures
# ------------------------

@pytest.fixture(scope="module")
def fx_mock_library_manager(module_mocker):
    """
    Creates a mock LibraryManager instance with predefined data.
    This avoids file I/O and decouples the tests from the
    LibraryManager's implementation.
    The 'module_mocker' is used for a module-scoped fixture.
    """
    # Create a mock object that mimics the LibraryManager's interface
    mock_lm = module_mocker.create_autospec(LibraryManager, instance=True)

    # Configure the attributes that FastqToCalls will access
    mock_lm.run_config = {
        "expected_5p": "GATTACA",
        "expected_3p": "TACATAG"
    }
    mock_lm.expected_length = 10

    # Configure the return value of the get_libraries() method
    dna_lib = {
        "test_lib": [
            "AAAAAAAAAA",  # wt
            "CAAAAAAAAA",  # mut1
            "TTTTTTTTTT",  # mut2
            "TAAAAAAAAA",  # mut_close
        ]
    }
    aa_lib = {
        "test_lib": [
            "",            # wt
            "A1C",         # mut1
            "polyT",       # mut2
            "A2T",         # mut_close
        ]
    }
    mock_lm.get_libraries.return_value = (dna_lib, aa_lib)

    return mock_lm


@pytest.fixture
def fx_fastq_to_calls_strict(fx_mock_library_manager):
    """
    Creates a 'strict' FastqToCalls instance using the mock
    LibraryManager.
    """
    return FastqToCalls(lm=fx_mock_library_manager,
                        allowed_num_flank_diffs=0)


@pytest.fixture
def fx_fastq_to_calls_fuzzy(fx_mock_library_manager):
    """
    Creates a 'fuzzy' FastqToCalls instance using the mock
    LibraryManager.
    """
    return FastqToCalls(lm=fx_mock_library_manager,
                        allowed_num_flank_diffs=1)


# ------------------------
# test _initialize_converters
# ------------------------

def test_initialize_converters(fx_fastq_to_calls_strict):
    """
    Tests that the converters and lookup tables are initialized correctly.
    This method is called by __init__, so we test its effects by
    inspecting a fully initialized object.
    """
    ftc = fx_fastq_to_calls_strict

    # Check the simple base/number mappings
    assert ftc.number_to_base == "ACGT-N"
    assert ftc.ambig_base_number == 5
    assert ftc.gap_base_number == 4

    # Check that the base_to_number dictionary is correct
    assert ftc.base_to_number['A'] == 0
    assert ftc.base_to_number['c'] == 1  # Check lowercase
    assert ftc.base_to_number['G'] == 2
    assert ftc.base_to_number['t'] == 3  # Check lowercase
    assert ftc.base_to_number['-'] == 4
    assert ftc.base_to_number['N'] == 5

    # Check the integer-based complement lookup table
    b2n = ftc.base_to_number
    expected_complement = np.array([b2n['T'], b2n['G'], b2n['C'], b2n['A'],
                                    b2n['-'], b2n['N']], dtype=np.uint8)
    assert_array_equal(ftc.complement_int, expected_complement)

#------------------------
# test _initialize_expected_seq
#------------------------

def test_initialize_expected_seq(fx_fastq_to_calls_strict):
    """
    Tests that the expected sequences, lookup tables, and search tree
    are initialized correctly based on the mock LibraryManager.
    """
    ftc = fx_fastq_to_calls_strict

    # 1. Verify flank sequences and lengths
    # These are derived from the 'run_config' of the mock LibraryManager.
    expected_5p = np.array([ftc.base_to_number[b] for b in "GATTACA"],
                             dtype=np.uint8)
    expected_3p = np.array([ftc.base_to_number[b] for b in "TACATAG"],
                             dtype=np.uint8)

    assert_array_equal(ftc.expected_5p_int, expected_5p)
    assert_array_equal(ftc.expected_3p_int, expected_3p)
    assert ftc.expected_5p_size == 7
    assert ftc.expected_3p_size == 7
    assert ftc.expected_length == 10

    # 2. Verify the bytes-to-amino-acid lookup dictionary
    # The mock library has 4 unique sequences.
    assert len(ftc._bytes_to_aa) == 4

    # Check the wild-type entry (protein name "" -> "wt")
    wt_dna_int = np.array([ftc.base_to_number[b] for b in "AAAAAAAAAA"],
                            dtype=np.uint8)
    assert ftc._bytes_to_aa[wt_dna_int.tobytes()] == "wt"

    # Check a mutant entry
    mut1_dna_int = np.array([ftc.base_to_number[b] for b in "CAAAAAAAAA"],
                             dtype=np.uint8)
    assert ftc._bytes_to_aa[mut1_dna_int.tobytes()] == "A1C"

    # 3. Verify the BK-Tree for sequence searching
    # We test the tree by using it to find known members of the library.
    
    # An exact match for the wild-type should return one result with dist 0.
    wt_search_result = ftc._search_tree.find(wt_dna_int, 0)
    assert len(wt_search_result) == 1
    assert wt_search_result[0][0] == 0  # Distance is 0
    assert_array_equal(wt_search_result[0][1], wt_dna_int)

    # A search for a sequence with one difference from 'mut1' should find it.
    query_seq = np.array([ftc.base_to_number[b] for b in "TAAAAAAAAA"],
                           dtype=np.uint8)
    mut_close_search_result = ftc._search_tree.find(query_seq, 0)
    assert len(mut_close_search_result) == 1
    assert mut_close_search_result[0][0] == 0  # Distance is 0
    assert_array_equal(mut_close_search_result[0][1], query_seq)


# ------------------------
# test _find_orientation_strict
# ------------------------

# A small helper to make test data generation more readable
def _dna_to_int(dna_str: str, ftc: FastqToCalls) -> np.ndarray:
    """Converts a DNA string to a NumPy integer array using the instance's map."""
    return np.array([ftc.base_to_number[b] for b in dna_str], dtype=np.uint8)

def _rev_comp_str(dna_str: str, ftc: FastqToCalls) -> str:
    """Helper to get the reverse complement string using the instance's tables."""
    int_arr = _dna_to_int(dna_str, ftc)
    rc_int_arr = ftc.complement_int[int_arr][::-1]
    return "".join(ftc.number_to_base[i] for i in rc_int_arr)


def test_find_orientation_strict_f1_is_forward(fx_fastq_to_calls_strict):
    """
    Tests the standard case where f1 contains the 5' flank and is the
    forward read.
    """
    ftc = fx_fastq_to_calls_strict
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"  # Same length as expected_length

    # f1 is the forward read: 5'-flank | payload
    f1_read = _dna_to_int(f5 + payload, ftc)

    # f2 is the reverse read: rev_comp(payload | 3'-flank)
    f2_read = _dna_to_int(_rev_comp_str(payload + f3, ftc),ftc)

    # Run the method
    fwd_seq, rev_seq, msg = ftc._find_orientation_strict(f1_read, f2_read)

    # Assertions
    expected_seq = _dna_to_int(payload, ftc)
    assert msg is None
    # After orientation, BOTH sequences should match the forward payload
    assert_array_equal(fwd_seq, expected_seq)
    assert_array_equal(rev_seq, expected_seq)


def test_find_orientation_strict_f2_is_forward(fx_fastq_to_calls_strict):
    """
    Tests the case where f2 contains the 5' flank and is the forward read.
    """
    ftc = fx_fastq_to_calls_strict
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"

    # f1 is the forward read: 5'-flank | payload
    f2_read = _dna_to_int(f5 + payload, ftc)

    # f2 is the reverse read: rev_comp(payload | 3'-flank)
    f1_read = _dna_to_int(_rev_comp_str(payload + f3, ftc),ftc)

    # Run the method
    fwd_seq, rev_seq, msg = ftc._find_orientation_strict(f1_read, f2_read)

    # Assertions
    expected_seq = _dna_to_int(payload, ftc)
    assert msg is None
    assert_array_equal(fwd_seq, expected_seq)
    assert_array_equal(rev_seq, expected_seq)


def test_find_orientation_strict_no_5p_flank(fx_fastq_to_calls_strict):
    """
    Tests failure when the 5' flank is missing from both reads.
    (This test was correct and does not need changes).
    """
    ftc = fx_fastq_to_calls_strict
    payload = "ACGTACGTAC"
    f1_read = _dna_to_int("AAAAAAAAAA" + payload, ftc)
    f2_read = _dna_to_int("CCCCCCCCCC" + payload, ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_strict(f1_read, f2_read)

    assert fwd_seq is None
    assert rev_seq is None
    assert msg == "fail, could not find expected 5p flank"


def test_find_orientation_strict_no_3p_flank(fx_fastq_to_calls_strict):
    """
    Tests failure when the 3' flank is missing after orientation.
    (This test was correct and does not need changes).
    """
    ftc = fx_fastq_to_calls_strict
    f5 = ftc.expected_5p
    payload = "ACGTACGTAC"

    f1_read = _dna_to_int(f5 + payload, ftc)
    f2_read = _dna_to_int("GGGGGGGGGGGGGGGGGGGG", ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_strict(f1_read, f2_read)

    assert fwd_seq is None
    assert rev_seq is None
    assert msg == "fail, could not find expected 3p flank"


def test_find_orientation_strict_with_extra_dna(fx_fastq_to_calls_strict):
    """
    Tests that slicing is correct when reads have extra sequence data.
    """
    ftc = fx_fastq_to_calls_strict
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"

        # f1 is the forward read: 5'-flank | payload
    f1_read = _dna_to_int("TTT" + f5 + payload + "AAA", ftc)

    # f2 is the reverse read: rev_comp(payload | 3'-flank)
    f2_read = _dna_to_int(_rev_comp_str("CCC" + payload + f3, ftc),ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_strict(f1_read, f2_read)

    # The returned sequences should be correctly sliced
    # fwd_seq is everything after the 5' flank
    expected_fwd = _dna_to_int(payload + "AAA", ftc)
    # rev_seq is the payload oriented to the fwd strand
    expected_rev = _dna_to_int("CCC" + payload, ftc)
    
    assert msg is None
    assert_array_equal(fwd_seq, expected_fwd)
    assert_array_equal(rev_seq, expected_rev)

# ------------------------
# test _find_orientation_fuzzy
# ------------------------

def test_find_orientation_fuzzy_f1_fwd_5p_mismatch(fx_fastq_to_calls_fuzzy):
    """
    Tests success when f1 is forward and its 5' flank has one mismatch.
    """
    ftc = fx_fastq_to_calls_fuzzy
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"

    # Introduce one mismatch into the 5' flank
    f5_mut = "N" + f5[1:]
    f1_read = _dna_to_int(f5_mut + payload, ftc)
    f2_read = _dna_to_int(_rev_comp_str(payload + f3, ftc), ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_fuzzy(f1_read, f2_read)

    expected_payload = _dna_to_int(payload, ftc)
    assert msg is None
    # This assumes the slicing bug fwd[idx0:] is fixed
    assert_array_equal(fwd_seq, expected_payload)
    assert_array_equal(rev_seq, expected_payload)


def test_find_orientation_fuzzy_f1_fwd_3p_mismatch(fx_fastq_to_calls_fuzzy):
    """
    Tests success when the 3' flank has one mismatch.
    """
    ftc = fx_fastq_to_calls_fuzzy
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"

    # Introduce one mismatch into the 3' flank
    f3_mut = "N" + f3[1:]
    f1_read = _dna_to_int(f5 + payload, ftc)
    f2_read = _dna_to_int(_rev_comp_str(payload + f3_mut, ftc), ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_fuzzy(f1_read, f2_read)

    expected_payload = _dna_to_int(payload, ftc)
    assert msg is None
    assert_array_equal(fwd_seq, expected_payload)
    assert_array_equal(rev_seq, expected_payload)


def test_find_orientation_fuzzy_f1_is_better_match(fx_fastq_to_calls_fuzzy):
    """
    Tests that the read with the better flank score is chosen as forward.
    """
    ftc = fx_fastq_to_calls_fuzzy
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"

    # f1's flank has 1 mismatch (score=1)
    f5_mut1 = "N" + f5[1:]
    f1_read = _dna_to_int(f5_mut1 + payload, ftc)

    # f2's flank has 2 mismatches (score=2)
    f5_mut2 = "NN" + f5[2:]
    f2_read = _dna_to_int(f5_mut2 + payload, ftc) # Not a real reverse read, just for testing orientation

    fwd_seq, rev_seq, msg = ftc._find_orientation_fuzzy(f1_read, f2_read)

    # We expect f1 to be chosen as forward, but finding the 3' flank will fail.
    # The test confirms orientation logic by checking the failure message.
    assert fwd_seq is None
    assert msg == "fail, could not find expected 3p flank"


def test_find_orientation_fuzzy_fail_5p_too_divergent(fx_fastq_to_calls_fuzzy):
    """
    Tests failure when the best 5' flank match exceeds allowed_num_flank_diffs.
    """
    ftc = fx_fastq_to_calls_fuzzy # Allows 1 diff
    f5 = ftc.expected_5p
    payload = "ACGTACGTAC"

    # Introduce 2 mismatches, which is > allowed
    f5_mut = "NN" + f5[2:]
    f1_read = _dna_to_int(f5_mut + payload, ftc)
    f2_read = _dna_to_int(f5_mut + payload, ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_fuzzy(f1_read, f2_read)
    assert fwd_seq is None
    assert msg == "fail, could not find expected 5p flank"


def test_find_orientation_fuzzy_fail_3p_too_divergent(fx_fastq_to_calls_fuzzy):
    """
    Tests failure when the 3' flank match exceeds allowed_num_flank_diffs.
    """
    ftc = fx_fastq_to_calls_fuzzy # Allows 1 diff
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "ACGTACGTAC"
    
    # 5' flank is perfect
    f1_read = _dna_to_int(f5 + payload, ftc)

    # 3' flank has 2 mismatches
    f3_mut = "NN" + f3[2:]
    f2_read = _dna_to_int(_rev_comp_str(payload + f3_mut, ftc), ftc)

    fwd_seq, rev_seq, msg = ftc._find_orientation_fuzzy(f1_read, f2_read)
    assert fwd_seq is None
    assert msg == "fail, could not find expected 3p flank"

import collections

# Helper function for comparing the complex return type of the method
def _compare_search_results(results, expected):
    """
    Compares the list of (distance, np.ndarray) tuples.
    Converts arrays to tuples to make them hashable for comparison.
    """
    # Use Counter to handle order-insensitivity for multiple matches
    results_counter = collections.Counter(
        (dist, tuple(arr)) for dist, arr in results
    )
    expected_counter = collections.Counter(
        (dist, tuple(arr)) for dist, arr in expected
    )
    return results_counter == expected_counter

# ------------------------
# test _search_expected_lib
# ------------------------

def test_search_expected_lib_exact_match(fx_fastq_to_calls_strict):
    """
    Tests that searching for an exact library member returns only that member
    with a distance of 0.
    """
    ftc = fx_fastq_to_calls_strict
    wt_seq_int = _dna_to_int("AAAAAAAAAA", ftc)

    # Search with a generous max_diffs; should still only return the dist=0 match
    matches = ftc._search_expected_lib(wt_seq_int, max_diffs=5)

    expected = [(0, wt_seq_int)]
    assert _compare_search_results(matches, expected)


def test_search_expected_lib_single_closest_match(fx_fastq_to_calls_strict):
    """
    Tests finding a unique closest match that is not an exact match.
    """
    ftc = fx_fastq_to_calls_strict
    mut1_seq_int = _dna_to_int("CAAAAAAAAA", ftc)

    # This query is 1 diff from mut1 ("CA...") and 2 diffs from wt ("AA...")
    query_seq = _dna_to_int("CNAAAAAAAA", ftc)
    matches = ftc._search_expected_lib(query_seq, max_diffs=5)

    # The nearest shell has a distance of 1 and contains only mut1
    expected = [(1, mut1_seq_int)]
    assert _compare_search_results(matches, expected)


def test_search_expected_lib_multiple_equidistant_matches(fx_fastq_to_calls_strict):
    """
    Tests that if multiple library members are in the closest "shell",
    all are returned.
    """
    ftc = fx_fastq_to_calls_strict
    wt_seq_int = _dna_to_int("AAAAAAAAAA", ftc) # Get the wt sequence
    mut1_seq_int = _dna_to_int("CAAAAAAAAA", ftc)
    mut_close_seq_int = _dna_to_int("TAAAAAAAAA", ftc)

    # This query is 1 diff from "CA...", "TA...", and "AA..."
    query_seq = _dna_to_int("NAAAAAAAAA", ftc)
    matches = ftc._search_expected_lib(query_seq, max_diffs=1)

    # FIX: The expected list must include all three equidistant matches.
    expected = [(1, wt_seq_int), (1, mut1_seq_int), (1, mut_close_seq_int)]
    assert _compare_search_results(matches, expected)


def test_search_expected_lib_no_match_in_radius(fx_fastq_to_calls_strict):
    """
    Tests that an empty list is returned if no matches are found within
    the max_diffs radius.
    """
    ftc = fx_fastq_to_calls_strict
    # This query is 2 diffs from the closest match (wt)
    query_seq = _dna_to_int("GGAAAAAAAA", ftc)

    # Search with a max_diffs that is too small
    matches = ftc._search_expected_lib(query_seq, max_diffs=1)

    assert matches == []


def test_search_expected_lib_empty_query(fx_fastq_to_calls_strict):
    """
    Tests that searching with a sequence of incorrect length finds no matches.
    BK-Trees require all items to have the same length for Hamming distance.
    """
    ftc = fx_fastq_to_calls_strict
    # Query has length 9, library has length 10
    query_seq = _dna_to_int("AAAAAAAAA", ftc)

    # This will raise an exception in the BK-Tree, which pybktree handles
    # and returns an empty list.
    matches = ftc._search_expected_lib(query_seq, max_diffs=5)

    assert matches == []

# ------------------------
# test _build_call_pair
# ------------------------

def test_build_call_pair_full_overlap_no_ambiguity(fx_fastq_to_calls_strict):
    """
    Tests the ideal case: reads are full length, overlap completely,
    and have no ambiguous bases.
    """
    ftc = fx_fastq_to_calls_strict
    # fwd and rev disagree at index 1
    fwd_seq = _dna_to_int("ACGTACGTAC", ftc)
    rev_seq = _dna_to_int("AGGTACGTAC", ftc)

    fwd_wins, rev_wins = ftc._build_call_pair(fwd_seq, rev_seq)

    # fwd_wins should match the fwd_seq where they disagree
    assert_array_equal(fwd_wins, fwd_seq)
    # rev_wins should match the rev_seq where they disagree
    assert_array_equal(rev_wins, rev_seq)


def test_build_call_pair_short_reads_partial_overlap(fx_fastq_to_calls_strict):
    """
    Tests that short reads are correctly left-aligned (fwd) and
    right-aligned (rev), and padded with 'N's.
    """
    ftc = fx_fastq_to_calls_strict
    # Reads are length 7, will overlap by 4 bases in the middle
    # Fwd: ACGTACG----
    # Rev: ----CGTACGT
    fwd_seq = _dna_to_int("ACGTACG", ftc)
    rev_seq = _dna_to_int("CGTACGT", ftc)

    fwd_wins, rev_wins = ftc._build_call_pair(fwd_seq, rev_seq)

    # In fwd_wins, the left side comes from fwd, right from rev.
    # The end of rev_seq is "CGT", not "TGT".
    expected_fwd_wins = _dna_to_int("ACGTACGCGT", ftc)

    # In rev_wins, the left side comes from fwd, right from rev.
    expected_rev_wins = _dna_to_int("ACGCGTACGT", ftc)

    assert_array_equal(fwd_wins, expected_fwd_wins)
    assert_array_equal(rev_wins, expected_rev_wins)


def test_build_call_pair_with_ambiguous_bases(fx_fastq_to_calls_strict):
    """
    Tests that an unambiguous base always wins over an ambiguous 'N'.
    """
    ftc = fx_fastq_to_calls_strict
    # fwd is ambiguous at index 1, rev is ambiguous at index 3
    fwd_seq = _dna_to_int("ANGTACGTAC", ftc)
    rev_seq = _dna_to_int("ACGNACGTAC", ftc)

    fwd_wins, rev_wins = ftc._build_call_pair(fwd_seq, rev_seq)

    # The final sequence should have no ambiguity because for each 'N',
    # the other read has a good base call.
    expected_seq = _dna_to_int("ACGTACGTAC", ftc)

    # Both fwd_wins and rev_wins should resolve to the same sequence
    assert_array_equal(fwd_wins, expected_seq)
    assert_array_equal(rev_wins, expected_seq)


def test_build_call_pair_truncates_long_reads(fx_fastq_to_calls_strict):
    """
    Tests that long reads are correctly truncated (fwd from start, rev from end).
    """
    ftc = fx_fastq_to_calls_strict
    # expected_length is 10
    long_fwd = "ACGTACGTAC" + "GGGG" # Keep first 10
    long_rev = "CCCC" + "AGGTACGTAC" # Keep last 10

    fwd_seq = _dna_to_int(long_fwd, ftc)
    rev_seq = _dna_to_int(long_rev, ftc)

    fwd_wins, rev_wins = ftc._build_call_pair(fwd_seq, rev_seq)

    expected_fwd_wins = _dna_to_int("ACGTACGTAC", ftc)
    expected_rev_wins = _dna_to_int("AGGTACGTAC", ftc)

    assert_array_equal(fwd_wins, expected_fwd_wins)
    assert_array_equal(rev_wins, expected_rev_wins)


def test_build_call_pair_conflicting_unambiguous_bases(fx_fastq_to_calls_strict):
    """
    Tests the main case: an unambiguous disagreement between reads.
    """
    ftc = fx_fastq_to_calls_strict
    # Disagreement at index 4 (A vs G)
    fwd_seq = _dna_to_int("ACGTACGTAC", ftc)
    rev_seq = _dna_to_int("ACGTGCGTAC", ftc)

    fwd_wins, rev_wins = ftc._build_call_pair(fwd_seq, rev_seq)

    # fwd_wins should take the 'A' from the fwd_seq
    expected_fwd_wins = _dna_to_int("ACGTACGTAC", ftc)
    # rev_wins should take the 'G' from the rev_seq
    expected_rev_wins = _dna_to_int("ACGTGCGTAC", ftc)

    assert_array_equal(fwd_wins, expected_fwd_wins)
    assert_array_equal(rev_wins, expected_rev_wins)

# ------------------------
# test _rr_perfect_agreement
# ------------------------

def test_rr_perfect_agreement_reads_disagree(fx_fastq_to_calls_strict):
    """
    Tests the primary "pass-through" case where the reads do not agree.
    The method should signal to keep going with reconciliation.
    """
    ftc = fx_fastq_to_calls_strict
    fwd_wins = _dna_to_int("ACGTACGTAC", ftc)
    rev_wins = _dna_to_int("AGGTACGTAC", ftc)

    keep_going, seq, msg = ftc._rr_perfect_agreement(fwd_wins, rev_wins)

    assert keep_going is True
    assert seq is None
    assert msg is None


def test_rr_perfect_agreement_agree_and_unique_match(fx_fastq_to_calls_strict):
    """
    Tests the ideal success case: reads agree and match a unique
    sequence in the library.
    """
    ftc = fx_fastq_to_calls_strict
    # Use the wild-type sequence from our mock library
    wt_seq = _dna_to_int("AAAAAAAAAA", ftc)

    keep_going, seq, msg = ftc._rr_perfect_agreement(wt_seq, wt_seq)

    assert keep_going is False
    assert msg == "pass, F/R agree exactly"
    assert_array_equal(seq, wt_seq)


def test_rr_perfect_agreement_agree_but_no_match(fx_fastq_to_calls_strict):
    """
    Tests the failure case where reads agree on a sequence that is not
    in the expected library.
    """
    ftc = fx_fastq_to_calls_strict
    # This sequence is not in our mock library
    unknown_seq = _dna_to_int("GGGGGGGGGG", ftc)

    # Set a strict diff allowance
    ftc.allowed_diff_from_expected = 0
    keep_going, seq, msg = ftc._rr_perfect_agreement(unknown_seq, unknown_seq)

    assert keep_going is False
    assert seq is None
    assert msg == "fail, F/R agree but their sequence is not in the expected library"


def test_rr_perfect_agreement_agree_but_ambiguous(fx_fastq_to_calls_strict):
    """
    Tests the failure case where the agreed-upon sequence is ambiguously
    close to multiple library members.
    """
    ftc = fx_fastq_to_calls_strict
    # As we saw in a previous test, this sequence is 1 diff away from three
    # library members (wt, mut1, mut_close).
    ambiguous_seq = _dna_to_int("NAAAAAAAAA", ftc)

    # Allow for fuzzy matching to find the multiple hits
    ftc.allowed_diff_from_expected = 1
    keep_going, seq, msg = ftc._rr_perfect_agreement(ambiguous_seq, ambiguous_seq)

    assert keep_going is False
    assert seq is None
    assert msg == "fail, F/R agree but match more than one expected sequence"


# ------------------------
# test _rr_one_sided_match
# ------------------------

@pytest.fixture
def fx_mock_sequences(fx_fastq_to_calls_strict):
    """
    Provides a dictionary of mock DNA sequences as integer arrays for testing.
    Correctly depends on another fixture instead of being called directly.
    """
    ftc = fx_fastq_to_calls_strict
    return {
        "wt": _dna_to_int("AAAAAAAAAA", ftc),
        "mut1": _dna_to_int("CAAAAAAAAA", ftc),
        "mut2": _dna_to_int("TAAAAAAAAA", ftc),
    }

def test_rr_one_sided_match_f_fails_r_passes(fx_fastq_to_calls_strict, fx_mock_sequences):
    
    """Tests case where F has no match but R has a unique match."""
    ftc = fx_fastq_to_calls_strict
    matches_f = []
    matches_r = [(1, fx_mock_sequences["mut1"])]
    keep_going, seq, msg = ftc._rr_one_sided_match(matches_f, matches_r)
    assert keep_going is False
    assert_array_equal(seq, fx_mock_sequences["mut1"])
    assert msg == "pass, F/R disagree but R is expected"

def test_rr_one_sided_match_f_fails_r_ambiguous(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests case where F has no match and R has multiple matches."""
    ftc = fx_fastq_to_calls_strict
    matches_f = []
    matches_r = [(1, fx_mock_sequences["mut1"]), (1, fx_mock_sequences["mut2"])]
    keep_going, seq, msg = ftc._rr_one_sided_match(matches_f, matches_r)
    assert keep_going is False
    assert seq is None
    assert msg == "fail, F/R disagree and F is not expected and R is ambiguous"

def test_rr_one_sided_match_both_fail(fx_fastq_to_calls_strict):
    """Tests case where neither read has a match."""
    ftc = fx_fastq_to_calls_strict
    matches_f, matches_r = [], []
    keep_going, seq, msg = ftc._rr_one_sided_match(matches_f, matches_r)
    assert keep_going is False
    assert seq is None
    assert msg == "fail, F/R disagree and neither sequence is expected"

def test_rr_one_sided_match_both_pass_continues(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests the pass-through case where both reads have matches."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["mut1"])]
    matches_r = [(1, fx_mock_sequences["mut2"])]
    keep_going, seq, msg = ftc._rr_one_sided_match(matches_f, matches_r)
    assert keep_going is True
    assert seq is None
    assert msg is None

# ------------------------
# test _rr_one_closer_match
# ------------------------

def test_rr_one_closer_match_f_is_closer(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests when F has a unique, closer match than R."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["mut1"])] # dist 1
    matches_r = [(2, fx_mock_sequences["mut2"])] # dist 2
    keep_going, seq, msg = ftc._rr_one_closer_match(matches_f, matches_r)
    assert keep_going is False
    assert_array_equal(seq, fx_mock_sequences["mut1"])
    assert msg == "pass, F/R disagree but F is expected"

def test_rr_one_closer_match_f_is_closer_but_ambiguous(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests pass-through when F is closer but has multiple matches."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["mut1"]), (1, fx_mock_sequences["wt"])]
    matches_r = [(2, fx_mock_sequences["mut2"])]
    keep_going, seq, msg = ftc._rr_one_closer_match(matches_f, matches_r)
    assert keep_going is True
    assert seq is None
    assert msg is None

def test_rr_one_closer_match_distances_equal(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests pass-through when match distances are equal."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["mut1"])]
    matches_r = [(1, fx_mock_sequences["mut2"])]
    keep_going, seq, msg = ftc._rr_one_closer_match(matches_f, matches_r)
    assert keep_going is True
    assert seq is None
    assert msg is None

# ------------------------
# test _rr_unique_intersection
# ------------------------

def test_rr_unique_intersection_success(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests finding a single shared sequence between two ambiguous match sets."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["wt"]), (1, fx_mock_sequences["mut1"])]
    matches_r = [(1, fx_mock_sequences["mut2"]), (1, fx_mock_sequences["mut1"])] # mut1 is shared
    keep_going, seq, msg = ftc._rr_unique_intersection(matches_f, matches_r)
    assert keep_going is False
    assert_array_equal(seq, fx_mock_sequences["mut1"])
    assert msg == "pass, F/R have a unique shared expected sequence"

def test_rr_unique_intersection_no_intersection(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests pass-through when there is no shared sequence."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["wt"])]
    matches_r = [(1, fx_mock_sequences["mut2"])]
    keep_going, seq, msg = ftc._rr_unique_intersection(matches_f, matches_r)
    assert keep_going is True
    assert seq is None
    assert msg is None

def test_rr_unique_intersection_multiple_intersections(fx_fastq_to_calls_strict, fx_mock_sequences):
    """Tests pass-through when there are multiple shared sequences."""
    ftc = fx_fastq_to_calls_strict
    matches_f = [(1, fx_mock_sequences["wt"]), (1, fx_mock_sequences["mut1"])]
    matches_r = [(1, fx_mock_sequences["wt"]), (1, fx_mock_sequences["mut1"])] # Both are shared
    keep_going, seq, msg = ftc._rr_unique_intersection(matches_f, matches_r)
    assert keep_going is True
    assert seq is None
    assert msg is None

# ------------------------
# test _reconcile_reads
# ------------------------

def test_reconcile_reads_rule1_perfect_agreement(fx_fastq_to_calls_strict, fx_mock_sequences):
    """
    Tests that reconciliation succeeds via Rule 1 (_rr_perfect_agreement).
    """
    ftc = fx_fastq_to_calls_strict
    wt_seq = fx_mock_sequences["wt"]

    # Input reads agree perfectly and match a library sequence
    seq, msg = ftc._reconcile_reads(wt_seq, wt_seq)

    assert msg == "pass, F/R agree exactly"
    assert_array_equal(seq, wt_seq)


def test_reconcile_reads_rule2_one_sided_match(fx_fastq_to_calls_strict, fx_mock_sequences):
    """
    Tests that reconciliation succeeds via Rule 2 (_rr_one_sided_match).
    """
    ftc = fx_fastq_to_calls_strict
    ftc.allowed_diff_from_expected = 1

    # fwd_wins is close to mut1; rev_wins is far from everything
    fwd_wins = fx_mock_sequences["mut1"]
    rev_wins = _dna_to_int("GGGGGGGGGG", ftc)

    seq, msg = ftc._reconcile_reads(fwd_wins, rev_wins)

    assert msg == "pass, F/R disagree but F is expected"
    assert_array_equal(seq, fx_mock_sequences["mut1"])


def test_reconcile_reads_rule3_one_closer_match(fx_fastq_to_calls_strict, fx_mock_sequences):
    """
    Tests that reconciliation succeeds via Rule 3 (_rr_one_closer_match).
    """
    ftc = fx_fastq_to_calls_strict
    ftc.allowed_diff_from_expected = 2

    # fwd_wins is 1 diff from mut1
    fwd_wins = _dna_to_int("CNAAAAAAAA", ftc)
    # rev_wins is 2 diffs from wt
    rev_wins = _dna_to_int("GGAAAAAAAA", ftc)

    seq, msg = ftc._reconcile_reads(fwd_wins, rev_wins)

    assert msg == "pass, F/R disagree but F is expected"
    assert_array_equal(seq, fx_mock_sequences["mut1"])


def test_reconcile_reads_rule4_unique_intersection(fx_fastq_to_calls_strict, fx_mock_sequences, mocker):
    """
    Tests that reconciliation succeeds via Rule 4 (_rr_unique_intersection).
    """
    ftc = fx_fastq_to_calls_strict
    ftc.allowed_diff_from_expected = 1
    
    # Mock the search results to test the intersection logic directly
    matches_f = [(1, fx_mock_sequences["wt"]), (1, fx_mock_sequences["mut1"])]
    matches_r = [(1, fx_mock_sequences["mut2"]), (1, fx_mock_sequences["mut1"])]
    
    # When _search_expected_lib is called, return matches_f then matches_r
    mocker.patch.object(ftc, '_search_expected_lib', side_effect=[matches_f, matches_r])

    # Inputs don't matter as much since we are mocking the search results
    fwd_wins = _dna_to_int("ACGT", ftc) # Dummy input
    rev_wins = _dna_to_int("TGCA", ftc) # Dummy input

    seq, msg = ftc._reconcile_reads(fwd_wins, rev_wins)

    assert msg == "pass, F/R have a unique shared expected sequence"
    assert_array_equal(seq, fx_mock_sequences["mut1"])


def test_reconcile_reads_final_failure(fx_fastq_to_calls_strict, fx_mock_sequences, mocker):
    """
    Tests that reconciliation fails when all rules are exhausted.
    """
    ftc = fx_fastq_to_calls_strict
    ftc.allowed_diff_from_expected = 1

    # Mock search results that will fail all rules:
    # - Not perfect agreement (dummy inputs differ)
    # - Not one-sided (both have matches)
    # - Not one-closer (distances are equal)
    # - Not a unique intersection (no intersection)
    matches_f = [(1, fx_mock_sequences["wt"])]
    matches_r = [(1, fx_mock_sequences["mut1"])]
    mocker.patch.object(ftc, '_search_expected_lib', side_effect=[matches_f, matches_r])

    fwd_wins = _dna_to_int("ACGT", ftc) # Dummy input
    rev_wins = _dna_to_int("TGCA", ftc) # Dummy input

    seq, msg = ftc._reconcile_reads(fwd_wins, rev_wins)

    assert seq is None
    assert msg == "fail, F/R match different sequences in expected"

# ------------------------
# test call_read_pair
# ------------------------

def test_call_read_pair_ideal_success_strict(fx_fastq_to_calls_strict):
    """
    Tests a perfect, high-quality read pair that matches a library
    entry exactly, using the strict orientation finder.
    """
    ftc = fx_fastq_to_calls_strict
    # FIX: Set a read length appropriate for the test data
    ftc.min_read_length = 10
    
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "AAAAAAAAAA" # Wild-type

    f1_read = _dna_to_int(f5 + payload, ftc)
    f2_read = _dna_to_int(_rev_comp_str(payload + f3, ftc), ftc)
    
    # High quality scores for all bases
    q_scores = np.full_like(f1_read, 100)

    final_seq, msg = ftc.call_read_pair(f1_read, f2_read, q_scores, q_scores)

    assert final_seq == "wt"
    assert msg == "pass, F/R agree exactly"


def test_call_read_pair_ideal_success_fuzzy(fx_fastq_to_calls_fuzzy):
    """
    Tests a high-quality read pair with a mismatch in a flank,
    using the fuzzy orientation finder.
    """
    ftc = fx_fastq_to_calls_fuzzy
    # FIX: Set a read length appropriate for the test data
    ftc.min_read_length = 10

    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    payload = "CAAAAAAAAA" # mut1

    # Introduce one mismatch in the 5' flank
    f5_mut = "N" + f5[1:]
    f1_read = _dna_to_int(f5_mut + payload, ftc)
    f2_read = _dna_to_int(_rev_comp_str(payload + f3, ftc), ftc)
    q_scores = np.full_like(f1_read, 100)

    final_seq, msg = ftc.call_read_pair(f1_read, f2_read, q_scores, q_scores)

    assert final_seq == "A1C"
    assert msg == "pass, F/R agree exactly"


def test_call_read_pair_phred_cutoff_resolves_conflict(fx_fastq_to_calls_strict):
    """
    Tests that a low Q-score correctly turns a base into 'N', allowing
    the other read's high-quality base to win reconciliation.
    """
    ftc = fx_fastq_to_calls_strict
    # FIX: Set a read length appropriate for the test data
    ftc.min_read_length = 10
    ftc.phred_cutoff = 30
    
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    
    # f1 has a 'G' instead of 'A', but f2 has the correct 'A'
    payload_f1 = "AGAAAAAAAA"
    payload_f2 = "AAAAAAAAAA"

    f1_read = _dna_to_int(f5 + payload_f1, ftc)
    f2_read = _dna_to_int(_rev_comp_str(payload_f2 + f3, ftc), ftc)
    
    # Q-score for the mismatch in f1 is LOW (5), but high everywhere else
    q_scores_f1 = np.full_like(f1_read, 100)
    q_scores_f1[len(f5) + 1] = 5

    q_scores_f2 = np.full_like(f2_read, 100)

    final_seq, msg = ftc.call_read_pair(f1_read, f2_read, q_scores_f1, q_scores_f2)

    assert final_seq == "wt"
    assert msg == "pass, F/R agree exactly"


def test_call_read_pair_fails_on_orientation(fx_fastq_to_calls_strict):
    """
    Tests that the method fails correctly if flanks cannot be found.
    """
    ftc = fx_fastq_to_calls_strict
    payload = "AAAAAAAAAA"
    # Flanks are junk
    f1_read = _dna_to_int("GGGGGGG" + payload, ftc)
    f2_read = _dna_to_int("CCCCCCC" + payload, ftc)
    q_scores = np.full_like(f1_read, 100)

    final_seq, msg = ftc.call_read_pair(f1_read, f2_read, q_scores, q_scores)

    assert final_seq is None
    assert msg == "fail, could not find expected 5p flank"


def test_call_read_pair_fails_on_min_length(fx_fastq_to_calls_strict):
    """
    Tests that the method fails if the extracted payload is too short.
    """
    ftc = fx_fastq_to_calls_strict
    ftc.min_read_length = 50 # Default, but explicit here
    f5 = ftc.expected_5p
    f3 = ftc.expected_3p
    
    # Payload is much shorter than min_read_length
    short_payload = "ACGT"
    
    f1_read = _dna_to_int(f5 + short_payload, ftc)
    f2_read = _dna_to_int(_rev_comp_str(short_payload + f3, ftc), ftc)
    q_scores = np.full_like(f1_read, 100)

    final_seq, msg = ftc.call_read_pair(f1_read, f2_read, q_scores, q_scores)

    assert final_seq is None
    assert msg == "fail, F sequence too short"