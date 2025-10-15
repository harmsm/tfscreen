import pytest
from unittest.mock import MagicMock, patch, call, create_autospec
from collections import Counter
import pandas as pd
import pandas.testing as pd_testing
import numpy as np
import os

# Import the functions and classes to be tested
from tfscreen.process_raw.process_fastq import (
    _process_reads_chunk,
    _process_pairs_chunk,
    _process_paired_fastq,
    _create_stats_df,
    _create_counts_df,
    process_fastq,
    LibraryManager,
    FastqToCounts
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
def mock_ftc():
    """
    Create a mock FastqToCounts object that is auto-specced.
    This ensures the mock has the same interface as the real class.
    """
    ftc = create_autospec(FastqToCounts, instance=True)
    
    base_to_number = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4, 'a':0, 'c':1, 'g':2, 't':3, 'n':4}
    fast_base_to_number = np.zeros(256, dtype=np.uint8)
    for base, num in base_to_number.items():
        fast_base_to_number[ord(base)] = num
    ftc.fast_base_to_number = fast_base_to_number
    
    ftc.build_call_pair.return_value = ((b'fwd_bytes', b'rev_bytes'), None)
    ftc.reconcile_reads.return_value = ("SUCCESS_GENOTYPE", "pass, some message")
    ftc.all_expected_genotypes = ["wt", "A1C"]

    return ftc

# -----------------------------------------------------------------------------
# _process_reads_chunk
# -----------------------------------------------------------------------------

def test_process_reads_chunk_happy_path(mock_ftc):
    """Tests that a valid chunk of reads is processed correctly."""
    mock_read1 = ("read1", "ACGT", "!!!!")
    mock_read2 = ("read2", "TGCA", "!!!!")
    chunk = [(mock_read1, mock_read1), (mock_read2, mock_read2)]

    mock_ftc.build_call_pair.side_effect = [
        ((b'pair1', b'pair1'), None),
        ((b'pair2', b'pair2'), None)
    ]

    fwd_rev_pairs, messages = _process_reads_chunk(chunk, mock_ftc)

    assert fwd_rev_pairs == Counter({(b'pair1', b'pair1'): 1, (b'pair2', b'pair2'): 1})
    assert messages == Counter()
    assert mock_ftc.build_call_pair.call_count == 2

def test_process_reads_chunk_id_mismatch(mock_ftc):
    """Tests that a read ID mismatch is correctly caught and logged."""
    chunk = [
        (("read1", "A", "!"), ("read1_mismatch", "T", "!"))
    ]
    
    fwd_rev_pairs, messages = _process_reads_chunk(chunk, mock_ftc)

    assert fwd_rev_pairs == Counter()
    assert messages == Counter({"fail, read id mismatch": 1})
    mock_ftc.build_call_pair.assert_not_called()

def test_process_reads_chunk_call_failure(mock_ftc):
    """Tests that a failure from build_call_pair is correctly logged."""
    chunk = [(("read1", "A", "!"), ("read1", "T", "!"))]
    mock_ftc.build_call_pair.return_value = (None, "fail, flank not found")

    fwd_rev_pairs, messages = _process_reads_chunk(chunk, mock_ftc)

    assert fwd_rev_pairs == Counter()
    assert messages == Counter({"fail, flank not found": 1})
    mock_ftc.build_call_pair.assert_called_once()

# -----------------------------------------------------------------------------
# _process_pairs_chunk
# -----------------------------------------------------------------------------

def test_process_pairs_chunk_happy_path(mock_ftc):
    """Tests that a valid chunk of read pairs is reconciled correctly."""
    pair1 = (b'fwd1', b'rev1')
    pair2 = (b'fwd2', b'rev2')
    fwd_rev_chunk = [pair1, pair2]
    fwd_rev_counter = Counter({pair1: 10, pair2: 5})

    mock_ftc.reconcile_reads.side_effect = [
        ("GENOTYPE_1", "pass, success"),
        ("GENOTYPE_2", "pass, success")
    ]
    
    sequences, messages = _process_pairs_chunk(fwd_rev_chunk, fwd_rev_counter, mock_ftc)

    assert sequences == Counter({"GENOTYPE_1": 10, "GENOTYPE_2": 5})
    assert messages == Counter({"pass, success": 15})
    assert mock_ftc.reconcile_reads.call_count == 2

def test_process_pairs_chunk_reconcile_failure(mock_ftc):
    """Tests that a failure from reconcile_reads is correctly logged."""
    pair1 = (b'fwd1', b'rev1')
    fwd_rev_chunk = [pair1]
    fwd_rev_counter = Counter({pair1: 7})
    
    mock_ftc.reconcile_reads.return_value = (None, "fail, ambiguous")

    sequences, messages = _process_pairs_chunk(fwd_rev_chunk, fwd_rev_counter, mock_ftc)

    assert sequences == Counter()
    assert messages == Counter({"fail, ambiguous": 7})
    mock_ftc.reconcile_reads.assert_called_once()

# -----------------------------------------------------------------------------
# _process_paired_fastq
# -----------------------------------------------------------------------------

@patch('tfscreen.process_raw.process_fastq.as_completed')
@patch('tfscreen.process_raw.process_fastq.ProcessPoolExecutor')
def test_process_paired_fastq_orchestration(mock_executor_cls, mock_as_completed, tmp_path, mock_ftc):
    """
    Tests the orchestration logic of _process_paired_fastq by mocking the
    ProcessPoolExecutor to avoid pickling errors and actual subprocesses.
    """
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"
    create_fastq_file(r1_path, [("r1", "A", "!") for _ in range(5)])
    create_fastq_file(r2_path, [("r1", "T", "!") for _ in range(5)])

    mock_executor = MagicMock()
    mock_executor_cls.return_value.__enter__.return_value = mock_executor
    
    mock_future_reads1 = MagicMock()
    mock_future_reads1.result.return_value = (Counter({(b'p1',b'r1'): 2}), Counter({"msg1": 2}))
    mock_future_reads2 = MagicMock()
    mock_future_reads2.result.return_value = (Counter({(b'p2',b'r2'): 3}), Counter({"msg2": 3}))
    
    mock_future_pairs1 = MagicMock()
    mock_future_pairs1.result.return_value = (Counter({"GENO1": 2}), Counter({"msg3": 2}))
    mock_future_pairs2 = MagicMock()
    mock_future_pairs2.result.return_value = (Counter({"GENO2": 3}), Counter({"msg4": 3}))

    mock_as_completed.side_effect = [
        [mock_future_reads1, mock_future_reads2],
        [mock_future_pairs1, mock_future_pairs2]
    ]
    # **FIX:** Provide 4 futures to match the 4 expected submit calls.
    mock_executor.submit.side_effect = [
        mock_future_reads1, mock_future_reads2,
        mock_future_pairs1, mock_future_pairs2
    ]

    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_ftc, None, chunk_size=3, num_workers=2)

    assert sequences == Counter({"GENO1": 2, "GENO2": 3})
    assert messages == Counter({"msg1": 2, "msg2": 3, "msg3": 2, "msg4": 3})
    # **FIX:** Assert 4 calls: 2 for reads chunks, 2 for pairs chunks.
    assert mock_executor.submit.call_count == 4

def test_process_paired_fastq_empty_files(tmp_path, mock_ftc):
    """Test that the function handles empty FASTQ files gracefully."""
    r1_path = tmp_path / "test_R1.fastq"
    r2_path = tmp_path / "test_R2.fastq"
    r1_path.touch()
    r2_path.touch()
    
    sequences, messages = _process_paired_fastq(str(r1_path), str(r2_path), mock_ftc, None)

    assert sequences == Counter()
    assert messages == Counter()

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

    sequences = Counter({"wt": 500, "A1C": 100, "UNEXPECTED": 50})
    expected_genotypes = ["wt", "A1C", "G2C"]
    
    expected_df = pd.DataFrame({
        "genotype": ["wt", "A1C", "G2C"],
        "counts": [500, 100, 0]
    })

    result_df = _create_counts_df(sequences, expected_genotypes)

    expected_df = expected_df.sort_values(by="genotype").reset_index(drop=True)
    result_df = result_df.sort_values(by="genotype").reset_index(drop=True)
    
    pd_testing.assert_frame_equal(result_df, expected_df)

@patch('tfscreen.process_raw.process_fastq.set_categorical_genotype')
def test_create_counts_df_empty_sequences(mock_set_categorical):
    """Test with an empty sequence counter, ensuring all expected genotypes are 0."""
    mock_set_categorical.side_effect = lambda df, standardize, sort: df
    
    sequences = Counter()
    expected_genotypes = ["wt", "A1C"]
    
    expected_df = pd.DataFrame({
        "genotype": ["wt", "A1C"],
        "counts": [0, 0]
    })

    result_df = _create_counts_df(sequences, expected_genotypes)
    
    expected_df = expected_df.sort_values(by="genotype").reset_index(drop=True)
    result_df = result_df.sort_values(by="genotype").reset_index(drop=True)

    pd_testing.assert_frame_equal(result_df, expected_df)

@patch('tfscreen.process_raw.process_fastq.set_categorical_genotype')
def test_create_counts_df_empty_expected(mock_set_categorical):
    """Test with an empty list of expected genotypes, producing an empty DataFrame."""
    mock_set_categorical.side_effect = lambda df, standardize, sort: df

    sequences = Counter({"wt": 100})
    expected_genotypes = []
    
    expected_df = pd.DataFrame(
        {"genotype": [], "counts": []}
    ).astype({"genotype": object, "counts": int})
    
    result_df = _create_counts_df(sequences, expected_genotypes)
    pd_testing.assert_frame_equal(result_df, expected_df)

# -----------------------------------------------------------------------------
# process_fastq
# -----------------------------------------------------------------------------

@patch('tfscreen.process_raw.process_fastq._create_counts_df')
@patch('tfscreen.process_raw.process_fastq._create_stats_df')
@patch('tfscreen.process_raw.process_fastq._process_paired_fastq')
@patch('tfscreen.process_raw.process_fastq.FastqToCounts')
def test_process_fastq_happy_path_with_instance(mock_ftc_cls, mock_process_paired, mock_create_stats, mock_create_counts, tmp_path):
    """
    Tests the main `process_fastq` function by passing a mock LibraryManager
    instance directly, which avoids the isinstance() TypeError.
    """
    out_dir = tmp_path / "output"
    mock_process_paired.return_value = (Counter({"seq": 1}), Counter({"msg": 1}))
    mock_stats_df = pd.DataFrame({'stats': [1]})
    mock_counts_df = pd.DataFrame({'counts': [1]})
    mock_create_stats.return_value = mock_stats_df
    mock_create_counts.return_value = mock_counts_df
    
    # Create a mock LibraryManager instance to pass in
    mock_lm_instance = create_autospec(LibraryManager, instance=True)
    mock_ftc_instance = mock_ftc_cls.return_value
    mock_ftc_instance.all_expected_genotypes = ["wt"]

    # Run the function with the mock instance
    process_fastq(
        f1_fastq="r1.fq", f2_fastq="r2.fq", out_dir=str(out_dir),
        run_config=mock_lm_instance, phred_cutoff=20, num_workers=4
    )
    
    assert os.path.isdir(out_dir)
    # The LibraryManager constructor should NOT be called
    # (The class is not patched, so we can't assert on it directly,
    # but we know the `else` block was skipped)
    
    mock_ftc_cls.assert_called_once_with(
        mock_lm_instance, # Check that our instance was passed
        phred_cutoff=20,
        min_read_length=50,
        allowed_num_flank_diffs=1,
        allowed_diff_from_expected=2
    )
    
    mock_create_counts.assert_called_once_with(Counter({"seq": 1}), ["wt"])

def test_process_fastq_raises_error_for_file_as_dir(tmp_path):
    """
    Tests that FileExistsError is raised if out_dir exists but is a file.
    """
    out_file = tmp_path / "output"
    out_file.touch()

    with pytest.raises(FileExistsError):
        process_fastq("r1.fq", "r2.fq", out_dir=str(out_file), run_config={})