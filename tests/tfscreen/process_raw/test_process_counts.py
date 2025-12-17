import pytest
import pandas as pd
import pandas.testing as pd_testing
from unittest.mock import patch, call

# Import the functions to be tested from the specified module
from tfscreen.process_raw.process_counts import (
    _prep_sample_df,
    _aggregate_counts,
    _infer_sample_cfu,
    process_counts,
)

# ------------------------
# Pytest Fixtures
# ------------------------

@pytest.fixture
def fx_sample_df() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        "sample": ["s1", "s2"],
        "od600": [0.5, 0.8]
    })

@pytest.fixture
def fx_counts_dir(tmp_path):
    """Creates a temporary directory with dummy count CSV files."""
    counts_path = tmp_path / "counts_dir"
    counts_path.mkdir()
    
    # Create valid files
    (counts_path / "counts_s1_file.csv").touch()
    (counts_path / "counts_s2_file.csv").touch()
    
    # Create an extra file to test ambiguous matching
    (counts_path / "counts_s2_ambiguous.csv").touch()

    return counts_path

# ------------------------
# Test _prep_sample_df
# ------------------------

@patch('glob.glob')
@patch('os.path.isdir', return_value=True)
@patch('tfscreen.util.io.read_dataframe')
def test_prep_sample_df_happy_path(mock_read_df, mock_isdir, mock_glob, fx_sample_df):
    """Tests successful validation and file matching."""
    mock_read_df.return_value = fx_sample_df.set_index("sample")
    
    # Configure glob to return a unique file for each sample
    mock_glob.side_effect = [
        ["/path/to/counts_s1_file.csv"],
        ["/path/to/counts_s2_file.csv"]
    ]

    result_df = _prep_sample_df(
        sample_df="dummy_path", 
        counts_csv_path="/path/to",
        counts_glob_prefix="counts",
        verbose=False # Suppress print output for this test
    )

    # Assertions
    mock_read_df.assert_called_once_with("dummy_path", index_column="sample")
    assert "obs_file" in result_df.columns
    assert result_df.loc["s1", "obs_file"] == "/path/to/counts_s1_file.csv"
    assert result_df.index.name == "sample"

@patch('builtins.print')
@patch('glob.glob')
@patch('os.path.isdir', return_value=True)
@patch('tfscreen.util.io.read_dataframe')
def test_prep_sample_df_verbose_output(mock_read_df, mock_isdir, mock_glob, mock_print, fx_sample_df):
    """Tests that verbose=True produces print output."""
    mock_read_df.return_value = fx_sample_df.set_index("sample")
    mock_glob.side_effect = [["f1"], ["f2"]]
    
    _prep_sample_df(sample_df="dummy", counts_csv_path="dummy", verbose=True)
    
    # Check that print was called with summary information
    assert mock_print.call_count > 0
    assert "2 unique samples" in mock_print.call_args_list[0][0][0]

def test_prep_sample_df_dir_not_found(fx_sample_df):
    """Tests that FileNotFoundError is raised for an invalid directory."""
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with patch('os.path.isdir', return_value=False):
            with pytest.raises(FileNotFoundError, match="is not a directory"):
                _prep_sample_df(fx_sample_df, "invalid_path")

def test_prep_sample_df_duplicate_samples(fx_sample_df):
    """Tests that ValueError is raised for non-unique samples."""
    dup_df = pd.concat([fx_sample_df, fx_sample_df.iloc[[0]]]).set_index("sample")
    with patch('tfscreen.util.io.read_dataframe', return_value=dup_df):
        with pytest.raises(ValueError, match="samples must be unique"):
            _prep_sample_df("dummy_path", "dummy_dir")

def test_prep_sample_df_missing_file(fx_sample_df, fx_counts_dir):
    """Tests that ValueError is raised when a sample's file is missing."""
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with pytest.raises(ValueError, match="MISSING: No files found for sample 's1'"):
             # s1 file does not have the default 'obs' prefix
            _prep_sample_df(fx_sample_df, str(fx_counts_dir), counts_glob_prefix="obs")

def test_prep_sample_df_ambiguous_files(fx_sample_df, fx_counts_dir):
    """Tests that ValueError is raised when a sample has multiple file matches."""
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with pytest.raises(ValueError, match="AMBIGUOUS: 2 files found for sample 's2'"):
            _prep_sample_df(fx_sample_df, str(fx_counts_dir), counts_glob_prefix="counts")

def test_prep_sample_df_bad_index(fx_sample_df):
    """Tests that ValueError is raised if 'sample' is not the index."""
    # Return a DataFrame where 'sample' is just a column
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df):
        with pytest.raises(ValueError, match="sample_df must be indexed by 'sample'"):
            _prep_sample_df(fx_sample_df, "dummy_dir")


# ------------------------
# Test _aggregate_counts
# ------------------------

@patch('pandas.read_csv')
def test_aggregate_counts(mock_read_csv):
    """Tests the aggregation of multiple count files into a single DataFrame."""
    # Setup input sample_df with file paths
    sample_df = pd.DataFrame({
        "obs_file": ["/path/s1.csv", "/path/s2.csv"]
    }, index=pd.Index(["s1", "s2"], name="sample"))

    # Mock DataFrames to be returned by pd.read_csv
    df1 = pd.DataFrame({"genotype": ["wt", "A1C"], "counts": [100, 10]})
    df2 = pd.DataFrame({"genotype": ["wt", "G2T"], "counts": [200, 20]})
    mock_read_csv.side_effect = [df1, df2]

    result_df = _aggregate_counts(sample_df)

    # Expected result is the concatenation of df1 and df2 with a 'sample' column
    expected_df = pd.DataFrame({
        "sample": ["s1", "s1", "s2", "s2"],
        "genotype": ["wt", "A1C", "wt", "G2T"],
        "counts": [100, 10, 200, 20]
    })
    
    # Assertions
    pd_testing.assert_frame_equal(result_df, expected_df)
    mock_read_csv.assert_has_calls([call("/path/s1.csv"), call("/path/s2.csv")])

# ------------------------
# Test _infer_sample_cfu
# ------------------------

@patch('tfscreen.process_raw.process_counts.od600_to_cfu')
def test_infer_sample_cfu(mock_od600_to_cfu, fx_sample_df):
    """Tests that CFU columns are correctly added to the sample DataFrame."""
    # Configure mock return values
    mock_cfu = pd.Series([1e8, 1.6e8], index=fx_sample_df.index)
    mock_cfu_std = pd.Series([1e7, 1.6e7], index=fx_sample_df.index)
    mock_detectable = pd.Series([True, True], index=fx_sample_df.index)
    mock_od600_to_cfu.return_value = (mock_cfu, mock_cfu_std, mock_detectable)

    result_df = _infer_sample_cfu(fx_sample_df, "dummy_calib_data")

    # Assertions
    mock_od600_to_cfu.assert_called_once()
    pd_testing.assert_series_equal(result_df["od600"], mock_od600_to_cfu.call_args[0][0])
    
    assert "sample_cfu" in result_df.columns
    assert "sample_cfu_std" in result_df.columns
    assert "sample_cfu_detectable" in result_df.columns
    assert result_df["sample_cfu"].iloc[0] == 1e8

def test_infer_sample_cfu_missing_column():
    """Tests that ValueError is raised if the 'od600' column is missing."""
    bad_df = pd.DataFrame({"sample": ["s1"]})
    with pytest.raises(ValueError, match="Not all required columns seen"):
        _infer_sample_cfu(bad_df, "dummy_calib_data")


# ------------------------
# Test process_counts
# ------------------------

@patch('pandas.DataFrame.to_csv')
@patch('tfscreen.process_raw.process_counts.counts_to_lncfu')
@patch('tfscreen.process_raw.process_counts._aggregate_counts')
@patch('tfscreen.process_raw.process_counts._prep_sample_df')
@patch('tfscreen.process_raw.process_counts._infer_sample_cfu')
def test_process_counts_orchestration(
    mock_infer, mock_prep, mock_agg, mock_lncfu, mock_to_csv, fx_sample_df
):
    """
    Tests the main `process_counts` function by mocking its dependencies
    to verify the overall workflow and data handoffs.
    """
    # Create mock DataFrames for each step in the pipeline
    df_step1 = pd.DataFrame({"s": ["s1"], "inferred": [True]})
    df_step2 = pd.DataFrame({"s": ["s1"], "prepped": [True]})
    df_step3 = pd.DataFrame({"s": ["s1"], "aggregated": [True]})
    df_final = pd.DataFrame({"s": ["s1"], "final_lncfu": [True]})

    # Configure mocks to return the staged DataFrames
    mock_infer.return_value = df_step1
    mock_prep.return_value = df_step2
    mock_agg.return_value = df_step3
    mock_lncfu.return_value = df_final

    # Execute the main function
    process_counts(
        sample_df=fx_sample_df,
        counts_csv_path="/counts/path",
        od600_calibration_data="calib.csv",
        output_file="/out/final.csv",
        counts_glob_prefix="test_prefix",
        min_genotype_obs=20,
        pseudocount=2,
        verbose=False
    )

    # Assert that each step was called once with the correct arguments
    # 1. _prep_sample_df is called with the original input df
    mock_prep.assert_called_once_with(
        fx_sample_df, "/counts/path", "test_prefix", False
    )

    # 2. _aggregate_counts is called with the output of _prep_sample_df (df_step2)
    mock_agg.assert_called_once_with(df_step2)

    # 3. _infer_sample_cfu is called with the output of _prep_sample_df (df_step2)
    mock_infer.assert_called_once_with(df_step2, "calib.csv")

    # 4. counts_to_lncfu is called with the output of _infer_sample_cfu (df_step1) 
    #    and the output of _aggregate_counts (df_step3)
    mock_lncfu.assert_called_once_with(
        df_step1, df_step3, min_genotype_obs=20, pseudocount=2
    )
    # Assert that the final output was saved
    mock_to_csv.assert_called_once_with("/out/final.csv", index=False)