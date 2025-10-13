
import pytest
import pandas as pd
import numpy as np

import tfscreen
from tfscreen.process_raw.assemble import _setup_tables
from tfscreen.process_raw.assemble import _build_ambiguous_codon_table
from tfscreen.process_raw.assemble import _translate_and_count
from tfscreen.process_raw.assemble import _prep_to_read
from tfscreen.process_raw.assemble import reads_to_muts


import os

# ----------------------------------------------------------------------------
# Shared Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def test_data_path(tmp_path):
    """
    Creates a temporary directory structure for tests that read from or
    write to the file system.
    """
    root_dir = tmp_path / "test_data"
    root_dir.mkdir()
    
    # Create subdirectories for inputs and outputs
    obs_dir = root_dir / "obs_csvs"
    obs_dir.mkdir()
    output_dir = root_dir / "output"
    output_dir.mkdir()

    # Create mock observation files for samples A and B
    file_a_content = "ATGAAG,10\nATGGGG,5\n" # M1M (wt), M1G
    (obs_dir / "obs_sampleA_data.csv").write_text(file_a_content)

    file_b_content = "ATGAAGACG,8\nATGAAATAG,2\n" # MKT (wt), MKS
    (obs_dir / "obs_sampleB_reads.csv").write_text(file_b_content)

    return {"root": root_dir, "obs": obs_dir, "output": output_dir}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provides a simple sample DataFrame for tests."""
    return pd.DataFrame({"sample": ["sampleA", "sampleB"]})


@pytest.fixture
def ref_seq() -> str:
    """Provides a reference amino acid sequence for translation tests."""
    return "MKT" # Met-Lys-Thr



# ----------------------------------------------------------------------------
# test _build_ambiguous_codon_table
# ----------------------------------------------------------------------------

def test_build_ambiguous_codon_table_logic():
    """
    Tests the core logic of resolving ambiguous codons using a minimal table.
    """
    # A simple table where AAN -> K, but GGN -> X (Gly/Ala)
    custom_table = {
        "AAA": "K", "AAG": "K", # AAN -> K
        "GGA": "G", "GGC": "G",
        "GCA": "A", "GCC": "A",
        "TTT": "F" # Original codon to ensure it's preserved
    }

    result = _build_ambiguous_codon_table(codon_table=custom_table)

    # --- Assertions ---
    # Preserves original codons
    assert result["TTT"] == "F"
    
    # Correctly resolves an unambiguous 'N'
    assert result["AAN"] == "K"
    
    # Correctly resolves an ambiguous 'N' to 'X'
    assert result["GGN"] == "G"
    
    # Correctly resolves a highly ambiguous codon to 'X'
    assert result["NNN"] == "X"

def test_build_ambiguous_codon_table_uses_default(mocker):
    """
    Tests that the function correctly falls back to the default codon table
    from the tfscreen package when no table is provided.
    """
    # Mock the default data source to control the test environment
    mock_default_table = {"AAA": "K"}
    mocker.patch.dict(tfscreen.data.CODON_TO_AA, mock_default_table, clear=True)

    result = _build_ambiguous_codon_table()

    # Check that the result is based on our mocked default table
    assert result["AAA"] == "K"
    assert result["AAN"] == "K" # Should be resolved from the default


# ----------------------------------------------------------------------------
# test _setup_tables
# ----------------------------------------------------------------------------

def test_setup_tables(mocker):
    """
    Tests that the function correctly constructs all four lookup tables.
    
    This test mocks the _build_ambiguous_codon_table helper to provide a
    simple, controlled input codon table.
    """
    # 1. ARRANGE: Define a simple codon table and mock the helper
    mock_codon_table = {"ACG": "T", "AAA": "K", "NNN": "X"}
    mocker.patch(
        "tfscreen.process_raw.reads_to_muts._build_ambiguous_codon_table",
        return_value=mock_codon_table
    )

    # 2. ACT: Call the function
    nucl_to_int, codon_to_aa_int, int_to_aa, aa_to_int = _setup_tables(codon_table=None)

    # 3. ASSERT: Check each of the four returned tables
    
    # -- Check nucleotide -> integer map --
    assert nucl_to_int == {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    # -- Check amino acid <-> integer maps --
    # Amino acids should be sorted alphabetically
    expected_aas = np.array(['K', 'T', 'X'])
    np.testing.assert_array_equal(int_to_aa, expected_aas)
    assert aa_to_int == {'K': 0, 'T': 1, 'X': 2}
    
    # -- Check the main 3D codon lookup table --
    assert codon_to_aa_int.shape == (5, 5, 5)
    assert codon_to_aa_int.dtype == np.int8
    
    # Verify a few specific entries in the 3D array
    # The default fill value should be the integer for 'X'
    assert codon_to_aa_int[4, 4, 3] == aa_to_int['X'] 
    
    # Check our specific mocked codons
    # ACG -> T
    assert codon_to_aa_int[nucl_to_int['A'], nucl_to_int['C'], nucl_to_int['G']] == aa_to_int['T']
    # AAA -> K
    assert codon_to_aa_int[nucl_to_int['A'], nucl_to_int['A'], nucl_to_int['A']] == aa_to_int['K']
    # NNN -> X
    assert codon_to_aa_int[nucl_to_int['N'], nucl_to_int['N'], nucl_to_int['N']] == aa_to_int['X']


# ----------------------------------------------------------------------------
# test _translate_and_count
# ----------------------------------------------------------------------------

def test_translate_and_count_logic(mocker):
    """
    Tests the core translation, mutation calling, and counting logic.
    
    Mocks `_setup_tables` to provide a minimal, predictable translation scheme.
    Verifies that wt, single, and multi-mutant sequences are correctly
    identified and their counts aggregated.
    """
    # 1. ARRANGE: Define a simple translation world and mock the setup function
    
    # Define a minimal set of lookup tables
    mock_nucl_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    mock_int_to_aa = np.array(['G', 'K', 'M', 'S', 'T', 'X'])
    mock_aa_to_int = {aa: i for i, aa in enumerate(mock_int_to_aa)}
    
    mock_codon_lookup = np.full((5, 5, 5), mock_aa_to_int['X'], dtype=np.int8)
    mock_codon_lookup[0,3,2] = mock_aa_to_int['M'] # ATG -> M
    mock_codon_lookup[0,0,2] = mock_aa_to_int['K'] # AAG -> K
    mock_codon_lookup[0,2,2] = mock_aa_to_int['S'] # AGG -> S
    mock_codon_lookup[2,2,2] = mock_aa_to_int['G'] # GGG -> G
    mock_codon_lookup[3,2,0] = mock_aa_to_int['T'] # TGA -> T (not stop for this test)

    mocker.patch(
        "tfscreen.process_raw.reads_to_muts._setup_tables",
        return_value=(mock_nucl_to_int, mock_codon_lookup, mock_int_to_aa, mock_aa_to_int)
    )

    # Define test data: wt, single mut, multi mut, and a duplicate wt
    ref_seq = "MKT"
    sequences = np.array([
        "ATGAAGTGA", # wt (MKT)
        "ATGGGGTGA", # single mutant (K2G)
        "ATGAAGAGG", # single mutant (T3S)
        "ATGGGGAGG", # double mutant (K2G, T3S)
        "ATGAAGTGA", # another wt
    ])
    counts = np.array([10, 5, 8, 2, 3])

    # 2. ACT: Call the function (using a small batch size to test the loop)
    result = _translate_and_count(sequences, counts, ref_seq, batch_size=2)

    # 3. ASSERT: Check that counts were correctly aggregated
    expected_counts = {
        "wt": 13,                # 10 + 3
        ("K2G",): 5,
        ("T3S",): 8,
        ("K2G", "T3S"): 2,
    }
    
    assert result == expected_counts

def test_translate_and_count_length_mismatch():
    """
    Tests that the function raises a ValueError if sequences and counts
    arrays have different lengths.
    """
    sequences = np.array(["SEQ1", "SEQ2"])
    counts = np.array([10]) # Mismatched length
    ref_seq = "M"

    with pytest.raises(ValueError, match="must be the same length"):
        _translate_and_count(sequences, counts, ref_seq)

# ----------------------------------------------------------------------------
# test _prep_to_read
# ----------------------------------------------------------------------------

def test_prep_to_read_success(sample_df, test_data_path):
    """
    Tests the success case where all samples in the dataframe have exactly
    one corresponding CSV file in the observation path.
    """
    obs_path = test_data_path["obs"]
    
    # ACT: Run the function
    result_df = _prep_to_read(sample_df, obs_path, verbose=False)
    
    # ASSERT
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.index.name == "sample"
    assert "obs_file" in result_df.columns
    
    # Check that the correct files were matched
    assert result_df.loc["sampleA", "obs_file"].endswith("obs_sampleA_data.csv")
    assert result_df.loc["sampleB", "obs_file"].endswith("obs_sampleB_reads.csv")
    assert os.path.exists(result_df.loc["sampleA", "obs_file"])

def test_prep_to_read_missing_file(sample_df, test_data_path):
    """
    Tests that a ValueError is raised if a sample is missing a file.
    """
    obs_path = test_data_path["obs"]
    
    # Add a sample that doesn't have a corresponding file
    bad_sample_df = pd.concat([sample_df, pd.DataFrame([{"sample": "sampleC"}])])
    
    with pytest.raises(ValueError, match="MISSING: No files found for sample 'sampleC'"):
        _prep_to_read(bad_sample_df, obs_path, verbose=False)

def test_prep_to_read_ambiguous_files(sample_df, test_data_path):
    """
    Tests that a ValueError is raised if a sample matches multiple files.
    """
    obs_path = test_data_path["obs"]
    
    # Create an extra file that also matches "sampleA"
    (obs_path / "obs_another_file_for_sampleA.csv").touch()
    
    with pytest.raises(ValueError, match="AMBIGUOUS: 2 files found for sample 'sampleA'"):
        _prep_to_read(sample_df, obs_path, verbose=False)

def test_prep_to_read_bad_path(sample_df):
    """
    Tests that a FileNotFoundError is raised for a non-existent directory.
    """
    bad_path = "./non_existent_directory/"
    with pytest.raises(FileNotFoundError):
        _prep_to_read(sample_df, bad_path)
        
def test_prep_to_read_duplicate_samples(test_data_path):
    """
    Tests that a ValueError is raised if the sample_df contains duplicate samples.
    """
    obs_path = test_data_path["obs"]
    duplicate_sample_df = pd.DataFrame({"sample": ["sampleA", "sampleA"]})
    
    with pytest.raises(ValueError, match="samples must be unique"):
        _prep_to_read(duplicate_sample_df, obs_path)


# ----------------------------------------------------------------------------
# test reads_to_muts
# ----------------------------------------------------------------------------

def test_reads_to_muts_end_to_end(mocker, sample_df, ref_seq, test_data_path):
    """
    Tests the full pipeline from reading sample sheet to writing output files.
    
    This integration test mocks the core _translate_and_count function to
    provide a predictable output, then verifies that the final CSV files
    are written correctly to the specified output directory.
    """
    # 1. ARRANGE: Get paths from fixtures and mock the main helper
    obs_path = test_data_path["obs"]
    output_path = test_data_path["output"]
    
    # Define a simple, predictable output for the translation function
    mock_counts = {"wt": 15, ("M1G",): 5}
    mocker.patch(
        "tfscreen.process_raw.reads_to_muts._translate_and_count",
        return_value=mock_counts
    )
    
    # 2. ACT: Run the main function
    reads_to_muts(
        sample_df=sample_df,
        obs_csv_path=obs_path,
        ref_seq=ref_seq,
        output_directory=output_path,
        prep_to_read_kwargs={"verbose": False} # Suppress printout for tests
    )
    
    # 3. ASSERT: Check that the output files were created and have correct content
    
    # Check that a file was created for each sample
    expected_file_A = output_path / "trans_sampleA.csv"
    expected_file_B = output_path / "trans_sampleB.csv"
    assert os.path.exists(expected_file_A)
    assert os.path.exists(expected_file_B)
    
    # Read one of the output files back in and verify its contents
    result_df = pd.read_csv(expected_file_A)
    
    # Construct the expected dataframe from the mock output
    expected_df = pd.DataFrame({
        "sample": ["sampleA", "sampleA"],
        "genotype": ["wt", "M1G"],
        "counts": [15, 5]
    })
    
    # Use pandas testing utility for robust comparison
    pd.testing.assert_frame_equal(
        result_df.sort_values("genotype").reset_index(drop=True),
        expected_df.sort_values("genotype").reset_index(drop=True)
    )

def test_reads_to_muts_output_is_file(sample_df, ref_seq, test_data_path):
    """
    Tests that a FileExistsError is raised if the output_directory path
    points to an existing file.
    """
    obs_path = test_data_path["obs"]
    
    # Create a file where the output directory should go
    bad_output_path = test_data_path["root"] / "output_is_a_file.txt"
    bad_output_path.touch()
    
    with pytest.raises(FileExistsError):
        reads_to_muts(
            sample_df=sample_df,
            obs_csv_path=obs_path,
            ref_seq=ref_seq,
            output_directory=bad_output_path,
            prep_to_read_kwargs={"verbose": False}
        )