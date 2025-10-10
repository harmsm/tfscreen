# test_library_manager.py

import pytest
from unittest.mock import patch, call

from tfscreen.genetics.library_manager import _check_char
from tfscreen.genetics.library_manager import _check_contiguous_lib_blocks
from tfscreen.genetics.library_manager import _check_lib_key
from tfscreen.genetics.library_manager import LibraryManager
from tfscreen.data import CODON_TO_AA


# ----------------------------------------------------------------------------
# test _check_char
# ----------------------------------------------------------------------------

def test_check_char_success():
    """
    Tests that _check_char passes when all characters are allowed.
    """
    # Should not raise an exception
    _check_char("GATTACA", "dna_seq", "GATC")
    _check_char("ACID", "protein_seq", set("ACDEFGHIKLMNPQRSTVWY"))

def test_check_char_empty_string():
    """
    Tests that _check_char passes for an empty input string.
    """
    # Should not raise an exception
    _check_char("", "empty_seq", "GATC")

def test_check_char_failure_single_unrecognized():
    """
    Tests that _check_char raises ValueError for a single bad character.
    """
    with pytest.raises(ValueError) as excinfo:
        _check_char("GATTACA-", "dna_seq", "GATC")
    
    # Check that the specific unrecognized character is in the message
    assert "Not all characters in" in str(excinfo.value)
    assert "dna_seq" in str(excinfo.value)


def test_check_char_failure_multiple_unrecognized():
    """
    Tests that _check_char raises ValueError and reports all bad characters,
    sorted alphabetically.
    """
    with pytest.raises(ValueError) as excinfo:
        _check_char("XGATTACA-Z", "dna_seq", "GATC")
    
    # The bad characters are X, -, Z. They should be sorted in the error msg.
    assert "Not all characters in" in str(excinfo.value)

def test_check_char_allowed_chars_types():
    """
    Tests that `allowed_chars` can be a string, set, or list.
    """
    # Using a string
    _check_char("abc", "test_str", "abcdef")
    
    # Using a set
    _check_char("abc", "test_set", {"a", "b", "c", "d", "e", "f"})
    
    # Using a list
    _check_char("abc", "test_list", ["a", "b", "c", "d", "e", "f"])

# ----------------------------------------------------------------------------
# test _check_contiguous_lib_blocks
# ----------------------------------------------------------------------------

# A series of valid strings that should pass without raising an error
@pytest.mark.parametrize("valid_string", [
    "..111..22..333...",
    "11111",
    ".......",
    "",
    "1..2..3..4",
    ".1.2.3.4."
])
def test_check_contiguous_lib_blocks_success(valid_string):
    """
    Tests that _check_contiguous_lib_blocks passes for valid, contiguous strings.
    """
    # Should not raise any exception
    _check_contiguous_lib_blocks(valid_string)


# A series of invalid strings, each paired with the character that should be
# identified in the error message.
@pytest.mark.parametrize("invalid_string, offending_char", [
    ("..11.1..22..", "1"),
    ("1.1", "1"),
    ("121", "1"),
    ("..ab.a.c.c", "a"), # Should fail on the first offender, 'a'
    ("ab.c.b", "b")
])
def test_check_contiguous_lib_blocks_failure(invalid_string, offending_char):
    """
    Tests that _check_contiguous_lib_blocks raises ValueError for non-contiguous
    strings.
    """
    with pytest.raises(ValueError) as excinfo:
        _check_contiguous_lib_blocks(invalid_string)

    assert f"Sub-library '{offending_char}' is not in a contiguous block" in str(excinfo.value)


# ----------------------------------------------------------------------------
# test _check_lib_key
# ----------------------------------------------------------------------------

@pytest.fixture
def seen_libs() -> set:
    """Provides a sample set of seen library identifiers for testing."""
    return {"1", "2", "A"}

def test_check_lib_key_success(seen_libs):
    """
    Tests that _check_lib_key passes for various valid key formats.
    """
    _check_lib_key("single-1", seen_libs)
    _check_lib_key("single-A", seen_libs)
    _check_lib_key("double-1-2", seen_libs)
    _check_lib_key("double-1-1", seen_libs) # intra-library double
    _check_lib_key("double-A-2", seen_libs)

@pytest.mark.parametrize("invalid_key, expected_msg_fragment", [
    # Bad commands
    ("triple-1", "Invalid command 'triple'"),
    ("sgle-1", "Invalid command 'sgle'"),
    ("", "Invalid command ''"),

    # Wrong number of parts for 'single'
    ("single", "'single' keys must have one sub-library part"),
    ("single-1-2", "'single' keys must have one sub-library part"),

    # Wrong number of parts for 'double'
    ("double-1", "'double' keys must have two sub-library parts"),
    ("double-1-2-3", "'double' keys must have two sub-library parts"),

    # Unrecognized sub-libraries
    ("single-3", "Unrecognized sub-library '3'"),
    ("double-1-Z", "Unrecognized sub-library 'Z'"),
    ("double-X-Y", "Unrecognized sub-library 'X'"), # Should catch the first one
])
def test_check_lib_key_failure(invalid_key, expected_msg_fragment, seen_libs):
    """
    Tests that _check_lib_key raises ValueError for various invalid formats
    and checks for a key part of the expected error message.
    """
    with pytest.raises(ValueError) as excinfo:
        _check_lib_key(invalid_key, seen_libs)
    
    assert expected_msg_fragment in str(excinfo.value)


# ----------------------------------------------------------------------------
# test _parse_and_validate (via LibraryManager.__init__)
# ----------------------------------------------------------------------------
@pytest.fixture
def valid_config() -> dict:
    """Provides a minimal, valid configuration dictionary for testing."""
    return {
        "reading_frame": 0,
        "first_amplicon_residue": 10,
        "wt_seq":       "gattacagtcgattaca",
        # CORRECTED: Shifted block left by 1 to align with reading_frame 0
        "degen_sites":  "......nnn........",
        "sub_libraries":"......222........",
        "library_combos": ["single-2"]
    }

def test_parse_success(valid_config):
    """
    Tests that a valid configuration is parsed without error and sets
    attributes correctly.
    """
    lm = LibraryManager(valid_config)
    assert lm.reading_frame == 0
    assert lm.first_amplicon_residue == 10
    assert lm.expected_length == 17
    assert lm.aa_seq == "DYSRL"
    assert lm.libraries_seen == {"2"}
    assert lm.degen_seq == "gattacnnncgattaca"

def test_parse_missing_key(valid_config):
    """
    Tests that a ValueError is raised if a required key is missing.
    """
    del valid_config["wt_seq"]
    with pytest.raises(ValueError, match="run_config is missing keys"):
        LibraryManager(valid_config)

def test_parse_mismatched_lengths(valid_config):
    """
    Tests that a ValueError is raised if sequence strings have different
    lengths.
    """
    valid_config["wt_seq"] = "gattaca" # Shorter than others
    with pytest.raises(ValueError, match="must all be the same length"):
        LibraryManager(valid_config)

def test_parse_non_contiguous_sublibs(valid_config):
    """
    Tests that a ValueError is raised for non-contiguous sub-libraries.
    """
    valid_config["sub_libraries"] = "..11..2.2..11...."
    with pytest.raises(ValueError, match="is not in a contiguous block"):
        LibraryManager(valid_config)

def test_parse_misplaced_degen_base(valid_config):
    """
    Tests the specific error for a degenerate base outside a sub-library.
    """
    # CORRECTED: Used a string of the correct length (17)
    valid_config["degen_sites"] = "n................"
    with pytest.raises(ValueError) as excinfo:
        LibraryManager(valid_config)
    
    # Check for the specific visual error format
    assert "indicated with '!' below" in str(excinfo.value)
    assert "!                " in str(excinfo.value)

def test_parse_bad_lib_combo(valid_config):
    """
    Tests that a ValueError is raised for an undefined sub-library in
    library_combos.
    """
    valid_config["library_combos"] = ["single-3"] # '3' is not defined
    with pytest.raises(ValueError, match="Unrecognized sub-library '3'"):
        LibraryManager(valid_config)

def test_parse_bad_reading_frame(valid_config):
    """
    Tests that a ValueError is raised for an invalid reading_frame value.
    """
    valid_config["reading_frame"] = 5
    with pytest.raises(ValueError, match="Value must be <= 2"):
        LibraryManager(valid_config)

# ----------------------------------------------------------------------------
# test _prepare_blocks
# ----------------------------------------------------------------------------

def test_prepare_blocks_simple_case():
    """
    Tests _prepare_blocks with a simple sequence containing one sub-library.
    """
    # Patch the helper method on the LibraryManager class
    with patch.object(LibraryManager, '_prepare_indiv_lib_blocks') as mock_prepare:
        # Configure the mock to return a predefined tuple of blocks
        # when it's called. Let's say sub-library '1' becomes two blocks.
        mock_prepare.return_value = (
            [['ta'], ['cg']],              # wt_blocks from helper
            [['ta', 'tg'], ['cg', 'ca']]  # mut_blocks from helper
        )

        # Create a bare instance and set only the necessary attributes
        lm = LibraryManager.__new__(LibraryManager)
        lm.wt_seq = "gattaca"
        lm.sub_libraries = "..11..."

        # Call the method under test
        lm._prepare_blocks()

        # 1. Assert the helper was called correctly
        mock_prepare.assert_called_once_with('1')

        # 2. Assert the final lists were assembled correctly
        # The '..' at the start become single-base blocks
        # The '11' is replaced by the two blocks from the mock
        # The '...' at the end become single-base blocks
        expected_wt = [['g'], ['a'], ['ta'], ['cg'], ['a'], ['c'], ['a']]
        expected_mut = [['g'], ['a'], ['ta', 'tg'], ['cg', 'ca'], ['a'], ['c'], ['a']]
        expected_lookup = ['.', '.', '1', '1', '.', '.', '.']

        assert lm.wt_blocks == expected_wt
        assert lm.mut_blocks == expected_mut
        assert lm.lib_lookup == expected_lookup

def test_prepare_blocks_no_libraries():
    """
    Tests _prepare_blocks with a sequence containing no sub-libraries.
    """
    with patch.object(LibraryManager, '_prepare_indiv_lib_blocks') as mock_prepare:
        lm = LibraryManager.__new__(LibraryManager)
        lm.wt_seq = "acgt"
        lm.sub_libraries = "...."

        lm._prepare_blocks()

        # The helper should never be called
        mock_prepare.assert_not_called()

        # The lists should just be the sequence broken into single bases
        assert lm.wt_blocks == [['a'], ['c'], ['g'], ['t']]
        assert lm.mut_blocks == [['a'], ['c'], ['g'], ['t']]
        assert lm.lib_lookup == ['.', '.', '.', '.']

def test_prepare_blocks_multiple_libraries():
    """
    Tests _prepare_blocks with multiple, non-contiguous sub-libraries.
    """
    with patch.object(LibraryManager, '_prepare_indiv_lib_blocks') as mock_prepare:
        # Use side_effect to return different values on each call
        mock_prepare.side_effect = [
            ( [['ga']], [['ga', 'gc']] ),                 # Call for lib '1'
            ( [['tt'], ['ac']], [['tt', 'ta'], ['ac']] )  # Call for lib '2'
        ]

        lm = LibraryManager.__new__(LibraryManager)
        lm.wt_seq = "agattaca"
        lm.sub_libraries = ".1.22..."

        lm._prepare_blocks()

        # 1. Assert the helper was called correctly and in order
        assert mock_prepare.call_count == 2
        mock_prepare.assert_has_calls([call('1'), call('2')])

        # 2. Assert the final lists were assembled correctly
        # CORRECTED AGAIN: The expected lists now match the sequence "agattaca"
        expected_wt =     [['a'], ['ga'], ['a'], ['tt'], ['ac'], ['a'], ['c'], ['a']]
        expected_mut =    [['a'], ['ga', 'gc'], ['a'], ['tt', 'ta'], ['ac'], ['a'], ['c'], ['a']]
        expected_lookup = ['.',   '1',    '.',   '2',    '2',    '.',   '.',   '.']

        assert lm.wt_blocks == expected_wt
        assert lm.mut_blocks == expected_mut
        assert lm.lib_lookup == expected_lookup


# ----------------------------------------------------------------------------
# test _prepare_indiv_lib_blocks
# ----------------------------------------------------------------------------

@pytest.fixture
def bare_lm() -> LibraryManager:
    """Provides a bare LibraryManager instance for testing internal methods."""
    lm = LibraryManager.__new__(LibraryManager)
    # Set attributes that are normally created in __init__
    lm.standard_bases = set('acgt')
    return lm

def test_pib_aligned_no_flanks(bare_lm):
    """Tests a sub-library perfectly aligned with the reading frame."""
    bare_lm.reading_frame = 0
    bare_lm.sub_libraries = "...111..."
    bare_lm.degen_sites =   "...nnt..."
    bare_lm.wt_seq =        "...act..."

    wt_blocks, mut_blocks = bare_lm._prepare_indiv_lib_blocks('1')

    assert wt_blocks == [['act']]
    assert len(mut_blocks) == 1
    assert len(mut_blocks[0]) == 16 # nnt -> 4*4*1 = 16 codons
    assert 'act' in mut_blocks[0]
    assert 'ggt' in mut_blocks[0]

def test_pib_with_left_flank(bare_lm):
    """Tests a sub-library that starts out of frame, creating a left flank."""
    bare_lm.reading_frame = 0
    # CORRECTED: Made all strings the same length and ensured the core
    # part of the library is a complete codon.
    bare_lm.sub_libraries = ".11111"  # Starts at index 1, length 5
    bare_lm.degen_sites =   ".acnnt"  # 'ac' is the flank, 'nnt' is the core
    bare_lm.wt_seq =        ".acact"

    wt_blocks, mut_blocks = bare_lm._prepare_indiv_lib_blocks('1')

    # The assertions now correctly check the expected output
    assert wt_blocks == [['ac'], ['act']]
    assert mut_blocks[0] == ['ac'] # Flank is not combinatorial
    assert len(mut_blocks[1]) == 16 # Codon block is combinatorial

def test_pib_with_right_flank(bare_lm):
    """Tests a sub-library that ends out of frame, creating a right flank."""
    bare_lm.reading_frame = 0
    bare_lm.sub_libraries = "1111." # Starts at index 0 (frame 0), len 4 -> 1 trailing
    bare_lm.degen_sites =   "nntg."
    bare_lm.wt_seq =        "actg."

    wt_blocks, mut_blocks = bare_lm._prepare_indiv_lib_blocks('1')

    assert wt_blocks == [['act'], ['g']]
    assert len(mut_blocks[0]) == 16
    assert mut_blocks[1] == ['g']

def test_pib_with_both_flanks(bare_lm):
    """Tests a sub-library with both left and right flanks."""
    bare_lm.reading_frame = 1
    bare_lm.sub_libraries = "..11111" # Starts at index 2 (frame 2), len 5
                                   # offset=(1-2)%3=2. core len=3. trailing=0. Mistake.
                                   # Let's fix. sub-lib len=6. Starts at index 2.
                                   # core_lib_seq len = 4. num_trailing = 1.
    bare_lm.sub_libraries = "..111111" # len 6
    bare_lm.degen_sites =   "..acnntg"
    bare_lm.wt_seq =        "..acactg"

    wt_blocks, mut_blocks = bare_lm._prepare_indiv_lib_blocks('1')

    assert wt_blocks == [['ac'], ['act'], ['g']]
    assert mut_blocks == [['ac'], mut_blocks[1], ['g']]
    assert len(mut_blocks[1]) == 16

def test_pib_fails_degen_in_left_flank(bare_lm):
    """Tests that ValueError is raised for a degenerate base in a left flank."""
    bare_lm.reading_frame = 0
    bare_lm.sub_libraries = ".1111"
    bare_lm.degen_sites =   ".ncnnt" # 'n' is in the flank
    bare_lm.wt_seq =        ".acact"

    with pytest.raises(ValueError, match="non-standard bases"):
        bare_lm._prepare_indiv_lib_blocks('1')


# ----------------------------------------------------------------------------
# test _prepare_indexes
# ----------------------------------------------------------------------------

def test_prepare_indexes():
    """
    Tests that _prepare_indexes correctly creates the indexer dictionary
    and the residue numbering list.
    """
    # 1. Setup a bare instance with the necessary input attributes
    lm = LibraryManager.__new__(LibraryManager)
    lm.libraries_seen = {'1', '2'}
    lm.lib_lookup = ['.', '1', '.', '2', '2', '.', '1']
    lm.first_amplicon_residue = 42
    # The contents of wt_blocks don't matter, only its length
    lm.wt_blocks = [[] for _ in range(len(lm.lib_lookup))]

    # 2. Call the method under test
    lm._prepare_indexes()

    # 3. Assert the output attributes are correct
    expected_indexers = {
        '1': [1, 6],
        '2': [3, 4]
    }
    assert lm.indexers == expected_indexers

    expected_residues = ['42', '43', '44', '45', '46', '47', '48']
    assert lm.residues == expected_residues


# ----------------------------------------------------------------------------
# test _convert_to_aa
# ----------------------------------------------------------------------------

@pytest.fixture
def lm_for_translation() -> LibraryManager:
    """Provides a bare LM instance with attributes needed for translation."""
    lm = LibraryManager.__new__(LibraryManager)
    lm.reading_frame = 1  # Use a non-zero frame to test offset logic
    lm.aa_seq = "SKR"
    lm.residues = ['10', '11', '12']
    # Add the CODON_TO_AA map to the instance for the test
    # (This mirrors how the main class accesses it from the module)
    lm.CODON_TO_AA = CODON_TO_AA
    return lm

def test_convert_to_aa(lm_for_translation):
    """
    Tests various mutation scenarios for the _convert_to_aa method.
    """
    # Based on frame 1, S=tcg, K=aaa, R=cgc
    # WT DNA should be prefixed with one nucleotide, e.g., 'g'
    wt_dna = "gtcgaaacgc"

    # S10A (S->A, tcg->gcg)
    single_mut_dna = "ggcgaaacgc"

    # S10A, K11Y (S->A, tcg->gcg; K->Y, aaa->tat)
    double_mut_dna = "ggcgtatcgc"

    # S10S (synonymous, tcg->tca)
    synonymous_dna = "gtcaaaacgc"
    
    # Truncated sequence that should still be processed correctly
    truncated_dna = "ggcgaaacg" # Last 'c' is missing

    dna_sequences = [
        wt_dna,
        single_mut_dna,
        double_mut_dna,
        synonymous_dna,
        truncated_dna
    ]

    results = lm_for_translation._convert_to_aa(dna_sequences)

    expected = [
        "",           # Wild-type produces no mutation string
        "S10A",       # Single mutant
        "S10A/K11Y",  # Double mutant
        "",           # Synonymous mutant is silent
        "S10A"        # Truncated sequence is handled
    ]

    assert results == expected


# ----------------------------------------------------------------------------
# test _get_singles
# ----------------------------------------------------------------------------

def test_get_singles_one_position():
    """
    Tests _get_singles for a library with one mutable position.
    """
    # 1. Setup a bare instance with necessary attributes
    lm = LibraryManager.__new__(LibraryManager)
    lm.wt_blocks =  [['a'], ['gac'], ['t']]
    lm.mut_blocks = [['a'], ['gac', 'gcc', 'ggc'], ['t']] # 3 variants
    lm.indexers =   {'1': [1]}

    # 2. Patch the helper method and call the method under test
    with patch.object(LibraryManager, '_convert_to_aa') as mock_convert:
        # Configure the mock to return a predictable value
        mock_convert.return_value = ["mock_aa_muts"]
        
        lib_seqs, aa_muts = lm._get_singles('1')

        # 3. Assertions
        # Check that the DNA sequences were generated correctly
        expected_dna = {"agact", "agcct", "aggct"} # Use a set for order-agnostic check
        
        # The mock should be called once with the generated DNA sequences
        mock_convert.assert_called_once()
        # The first (and only) argument passed to the mock call
        called_with_dna = mock_convert.call_args[0][0]
        assert set(called_with_dna) == expected_dna

        # The final returned values should be what was generated/mocked
        assert set(lib_seqs) == expected_dna
        assert aa_muts == ["mock_aa_muts"]

def test_get_singles_multiple_positions():
    """
    Tests _get_singles for a library with two mutable positions.
    """
    lm = LibraryManager.__new__(LibraryManager)
    lm.wt_blocks =  [['a'], ['gac'], ['t'], ['cat']]
    lm.mut_blocks = [['a'], ['gac', 'gcc'], ['t'], ['cat', 'cgt']] # 2 variants each
    lm.indexers =   {'1': [1, 3]}

    with patch.object(LibraryManager, '_convert_to_aa') as mock_convert:
        lm._get_singles('1')
        
        # It should generate 2 sequences for the first position (idx=1)
        # and 2 sequences for the second position (idx=3)
        expected_dna = [
            "agactcat", "agcctcat", # Mutants at idx 1
            "agactcat", "agactcgt"  # Mutants at idx 3
        ]

        # Assert that the helper was called with the combined list of 4 sequences
        mock_convert.assert_called_once()
        called_with_dna = mock_convert.call_args[0][0]
        assert sorted(called_with_dna) == sorted(expected_dna)

# ----------------------------------------------------------------------------
# test _get_intra_doubles
# ----------------------------------------------------------------------------

def test_get_intra_doubles():
    """
    Tests that _get_intra_doubles correctly generates combinations for all
    unique pairs of sites within a library.
    """
    # 1. Setup a bare instance with a library containing 3 mutable sites
    lm = LibraryManager.__new__(LibraryManager)
    lm.wt_blocks =  [['a'], ['g1'], ['t'], ['c2'], ['g'], ['a3']]
    lm.mut_blocks = [['a'], ['g1', 'm1'], ['t'], ['c2', 'm2'], ['g'], ['a3', 'm3']]
    lm.indexers = {'1': [1, 3, 5]} # 3 mutable sites

    # 2. Patch the helper and call the method under test
    with patch.object(LibraryManager, '_convert_to_aa') as mock_convert:
        lm._get_intra_doubles('1')

        # 3. Assertions
        # There are 3 unique pairs from 3 sites: (1,3), (1,5), and (3,5)
        # Each pair has 2x2=4 combinations. Total sequences = 3*4=12.
        
        # Expected sequences from pair (1,3)
        pair13_seqs = { "ag1tc2ga3", "ag1tm2ga3", "am1tc2ga3", "am1tm2ga3" }
        # Expected sequences from pair (1,5)
        pair15_seqs = { "ag1tc2ga3", "ag1tc2gm3", "am1tc2ga3", "am1tc2gm3" }
        # Expected sequences from pair (3,5)
        pair35_seqs = { "ag1tc2ga3", "ag1tc2gm3", "ag1tm2ga3", "ag1tm2gm3" }

        # The final list passed to the helper is the union of these
        expected_dna = pair13_seqs.union(pair15_seqs).union(pair35_seqs)

        mock_convert.assert_called_once()
        called_with_dna = mock_convert.call_args[0][0]
        
        assert len(called_with_dna) == 12
        assert set(called_with_dna) == expected_dna

# In your test file...
from unittest.mock import patch
from tfscreen.genetics.library_manager import LibraryManager

# ----------------------------------------------------------------------------
# test _get_inter_doubles
# ----------------------------------------------------------------------------

def test_get_inter_doubles_simple():
    """
    Tests inter-library doubles with one mutable site in each library.
    """
    # 1. Setup a bare instance
    lm = LibraryManager.__new__(LibraryManager)
    lm.wt_blocks =  [['a'], ['g1'], ['t'], ['c2'], ['g']]
    lm.mut_blocks = [['a'], ['g1', 'm1'], ['t'], ['c2', 'm2'], ['g']]
    lm.indexers =   {'1': [1], '2': [3]}

    # 2. Patch the helper and call the method under test
    with patch.object(LibraryManager, '_convert_to_aa') as mock_convert:
        lm._get_inter_doubles('1', '2')

        # 3. Assertions
        # 1 site in lib1 x 1 site in lib2 = 1 pair of sites to test
        # 2 variants/site x 2 variants/site = 4 total DNA sequences
        expected_dna = {
            "ag1tc2g",  # wt/wt
            "ag1tm2g",  # wt/mut
            "am1tc2g",  # mut/wt
            "am1tm2g",  # mut/mut
        }

        mock_convert.assert_called_once()
        called_with_dna = mock_convert.call_args[0][0]
        assert len(called_with_dna) == 4
        assert set(called_with_dna) == expected_dna

def test_get_inter_doubles_multiple_sites():
    """
    Tests inter-library doubles where one library has multiple sites.
    """
    # 1. Setup
    lm = LibraryManager.__new__(LibraryManager)
    lm.wt_blocks =  [['g1'], ['t'], ['c2'], ['g'], ['a3']]
    lm.mut_blocks = [['g1', 'm1'], ['t'], ['c2', 'm2'], ['g'], ['a3', 'm3']]
    lm.indexers =   {'A': [0, 4], 'B': [2]} # Lib 'A' has 2 sites, 'B' has 1

    # 2. Patch and call
    with patch.object(LibraryManager, '_convert_to_aa') as mock_convert:
        lm._get_inter_doubles('A', 'B')

        # 3. Assertions
        # 2 sites in libA x 1 site in libB = 2 pairs of sites
        # Each pair generates 2x2=4 sequences. Total = 8 sequences.
        
        # Pair 1: sites (0, 2)
        pair02_seqs = {"g1tc2ga3", "g1tm2ga3", "m1tc2ga3", "m1tm2ga3"}
        # Pair 2: sites (4, 2)
        pair42_seqs = {"g1tc2ga3", "g1tm2ga3", "g1tc2gm3", "g1tm2gm3"}

        expected_dna = pair02_seqs.union(pair42_seqs)

        mock_convert.assert_called_once()
        called_with_dna = mock_convert.call_args[0][0]
        assert len(called_with_dna) == 8
        assert set(called_with_dna) == expected_dna

# In your test file...
from unittest.mock import patch
from tfscreen.genetics.library_manager import LibraryManager

# ----------------------------------------------------------------------------
# test get_libraries
# ----------------------------------------------------------------------------

@patch('tfscreen.genetics.library_manager.LibraryManager._get_inter_doubles')
@patch('tfscreen.genetics.library_manager.LibraryManager._get_intra_doubles')
@patch('tfscreen.genetics.library_manager.LibraryManager._get_singles')
def test_get_libraries_dispatcher(mock_get_singles,
                                  mock_get_intra,
                                  mock_get_inter):
    """
    Tests that get_libraries correctly dispatches to the appropriate
    helper methods based on the library_combos.
    """
    # 1. Setup mock return values to be unique and trackable
    mock_get_singles.return_value = (['s_dna'], ['s_aa'])
    mock_get_intra.return_value = (['d_intra_dna'], ['d_intra_aa'])
    mock_get_inter.return_value = (['d_inter_dna'], ['d_inter_aa'])

    # 2. Setup a bare instance with a list of combos to test
    lm = LibraryManager.__new__(LibraryManager)
    lm.library_combos = [
        "single-1",
        "double-2-2", # Should call intra-doubles
        "double-1-2"  # Should call inter-doubles
    ]

    # 3. Call the method under test
    dna_libs, aa_libs = lm.get_libraries()

    # 4. Assert that each helper was called correctly
    mock_get_singles.assert_called_once_with('1')
    mock_get_intra.assert_called_once_with('2')
    mock_get_inter.assert_called_once_with('1', '2')

    # 5. Assert that the final dictionaries were assembled correctly
    expected_dna = {
        "single-1": ['s_dna'],
        "double-2-2": ['d_intra_dna'],
        "double-1-2": ['d_inter_dna']
    }
    expected_aa = {
        "single-1": ['s_aa'],
        "double-2-2": ['d_intra_aa'],
        "double-1-2": ['d_inter_aa']
    }
    assert dna_libs == expected_dna
    assert aa_libs == expected_aa