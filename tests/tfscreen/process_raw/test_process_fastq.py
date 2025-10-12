import pytest
from unittest.mock import MagicMock, patch
from collections import Counter
import pandas as pd
import pandas.testing as pd_testing


# Import the functions to be tested
from tfscreen.process_raw.process_fastq import (
    _process_paired_fastq,
    _create_stats_df,
    _create_counts_df,
    process_fastq,
    LibraryManager
)

# Helper function to create mock FASTQ files
def create_fastq_file(path, reads):
    """Writes a list of tuples (name, seq, qual) to a FASTQ file."""
    with open(path, "w") as f:
        for name, seq, qual in reads:
            f.write(f"@{name}\n")
            f.write(f"{seq}\n")
            f.write("+\n")
            f.write(f"{qual}\n")

@pytest.fixture
def mock_fc():
    """Create a mock FastqToCalls object."""
    fc = MagicMock()
    fc.base_to_number = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    # Configure a default return value for the mock method
    fc.call_read_pair.return_value = ("CALLED_SEQ_1", "pass, found wildtype")
    return fc

# -----------------------------------------------------------------------------
# _process_paired_fastq
# -----------------------------------------------------------------------------

def test_process_paired_fastq_happy_path(tmp_path, mock_fc):
    """Test standard processing of two matching read pairs."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"

    reads1 = [
        ("READ1 header", "ACGT", "!!!!"),
        ("READ2 header", "TGCA", "!!!!"),
    ]
    reads2 = [
        ("READ1 header", "ACGT", "!!!!"),
        ("READ2 header", "TGCA", "!!!!"),
    ]
    create_fastq_file(r1_path, reads1)
    create_fastq_file(r2_path, reads2)

    # Define different return values for sequential calls
    mock_fc.call_read_pair.side_effect = [
        ("CALLED_SEQ_1", "pass, found wildtype"),
        ("CALLED_SEQ_2", "pass, found variant"),
    ]

    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_fc, None)

    assert sequences == Counter({"CALLED_SEQ_1": 1, "CALLED_SEQ_2": 1})
    assert messages == Counter({"pass, found wildtype": 1, "pass, found variant": 1})
    assert mock_fc.call_read_pair.call_count == 2

def test_process_paired_fastq_id_mismatch(tmp_path, mock_fc):
    """Test that a read ID mismatch is caught and logged."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"

    reads1 = [("READ1", "ACGT", "!!!!")]
    reads2 = [("READ_MISMATCH", "ACGT", "!!!!")]
    create_fastq_file(r1_path, reads1)
    create_fastq_file(r2_path, reads2)

    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_fc, None)

    assert sequences == Counter()
    assert messages == Counter({"fail, read id mismatch": 1})
    mock_fc.call_read_pair.assert_not_called()

def test_process_paired_fastq_max_reads(tmp_path, mock_fc):
    """Test that `max_num_reads` correctly limits processing."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"

    reads1 = [("READ1", "A", "!"), ("READ2", "C", "!")]
    reads2 = [("READ1", "A", "!"), ("READ2", "C", "!")]
    create_fastq_file(r1_path, reads1)
    create_fastq_file(r2_path, reads2)

    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_fc, max_num_reads=1)

    assert sequences == Counter({"CALLED_SEQ_1": 1})
    assert messages == Counter({"pass, found wildtype": 1})
    mock_fc.call_read_pair.assert_called_once()

def test_process_paired_fastq_no_seq_call(tmp_path, mock_fc):
    """Test case where a message is returned but the sequence call is None."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"
    
    reads1 = [("READ1", "ACGT", "!!!!")]
    reads2 = [("READ1", "ACGT", "!!!!")]
    create_fastq_file(r1_path, reads1)
    create_fastq_file(r2_path, reads2)

    mock_fc.call_read_pair.return_value = (None, "fail, low quality")
    
    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_fc, None)

    assert sequences == Counter()
    assert messages == Counter({"fail, low quality": 1})
    mock_fc.call_read_pair.assert_called_once()
    
def test_process_paired_fastq_empty_files(tmp_path, mock_fc):
    """Test that the function handles empty FASTQ files gracefully."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"
    
    # Create empty files
    r1_path.touch()
    r2_path.touch()
    
    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_fc, None)

    assert sequences == Counter()
    assert messages == Counter()
    mock_fc.call_read_pair.assert_not_called()

# -----------------------------------------------------------------------------
# _create_stats_df
# -----------------------------------------------------------------------------

def test_create_stats_df_happy_path():
    """Test standard conversion of a messages counter to a stats DataFrame."""
    messages = Counter({
        "pass, found wildtype": 150,
        "fail, low quality": 40,
        "pass, found variant": 10
    })
    
    expected_df = pd.DataFrame({
        'success': [True, False, True],
        'result': ['found wildtype', 'low quality', 'found variant'],
        'counts': [150, 40, 10],
        'fraction': [0.75, 0.20, 0.05]
    })

    result_df = _create_stats_df(messages)
    
    # Sort for consistent comparison
    expected_df = expected_df.sort_values(by="result").reset_index(drop=True)
    result_df = result_df.sort_values(by="result").reset_index(drop=True)

    pd_testing.assert_frame_equal(result_df, expected_df)

def test_create_stats_df_empty_input():
    """Test that an empty counter produces an empty DataFrame with correct columns."""
    messages = Counter()
    
    expected_df = pd.DataFrame({
        'success': pd.Series(dtype=bool),
        'result': pd.Series(dtype=object),
        'counts': pd.Series(dtype='int64'),
        'fraction': pd.Series(dtype=float)
    })

    result_df = _create_stats_df(messages)
    pd_testing.assert_frame_equal(result_df, expected_df)


# -----------------------------------------------------------------------------
# _create_counts_df
# -----------------------------------------------------------------------------

@patch('tfscreen.process_raw.process_fastq.set_categorical_genotype')
def test_create_counts_df_happy_path(mock_set_categorical):
    """Test standard conversion of a sequence counter to a counts DataFrame."""
    mock_set_categorical.side_effect = lambda df, standardize, sort: df

    sequences = Counter({"WT": 500, "A1G": 100, "UNEXPECTED": 50})
    expected_genotypes = ["WT", "A1G", "G2C"]
    
    expected_df_arg = pd.DataFrame({
        "genotype": ["WT", "A1G", "G2C"],
        "counts": [500, 100, 0]
    })

    result_df = _create_counts_df(sequences, expected_genotypes)

    ### FIX: Properly assert DataFrame equality on the mock's arguments ###
    mock_set_categorical.assert_called_once()
    call_args, call_kwargs = mock_set_categorical.call_args
    pd_testing.assert_frame_equal(call_args[0], expected_df_arg)
    assert call_kwargs == {"standardize": True, "sort": True}
    
    pd_testing.assert_frame_equal(result_df, expected_df_arg)

@patch('tfscreen.process_raw.process_fastq.set_categorical_genotype')
def test_create_counts_df_empty_sequences(mock_set_categorical):
    """Test with an empty sequence counter, ensuring all expected genotypes are 0."""
    mock_set_categorical.side_effect = lambda df, standardize, sort: df
    
    sequences = Counter()
    expected_genotypes = ["WT", "A1G"]
    
    expected_df = pd.DataFrame({
        "genotype": ["WT", "A1G"],
        "counts": [0, 0]
    })

    result_df = _create_counts_df(sequences, expected_genotypes)
    pd_testing.assert_frame_equal(result_df, expected_df)

@patch('tfscreen.process_raw.process_fastq.set_categorical_genotype')
def test_create_counts_df_empty_expected(mock_set_categorical):
    """Test with an empty list of expected genotypes, producing an empty DataFrame."""
    mock_set_categorical.side_effect = lambda df, standardize, sort: df

    sequences = Counter({"WT": 100})
    expected_genotypes = []
    
    expected_df = pd.DataFrame(
        {"genotype": [], "counts": []}
    ).astype({"genotype": object, "counts": int})
    
    result_df = _create_counts_df(sequences, expected_genotypes)
    pd_testing.assert_frame_equal(result_df, expected_df)

# -----------------------------------------------------------------------------
# process_fastq
# -----------------------------------------------------------------------------

