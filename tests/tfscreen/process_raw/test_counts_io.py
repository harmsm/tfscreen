import pytest
import pandas as pd
import pandas.testing as pd_testing
from unittest.mock import patch, call

from tfscreen.process_raw._counts_io import _prep_sample_df, _aggregate_counts


# ------------------------
# Fixtures
# ------------------------

@pytest.fixture
def fx_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sample": ["s1", "s2"],
        "sample_cfu": [1e8, 1.6e8],
        "sample_cfu_std": [1e7, 1.6e7],
    })


@pytest.fixture
def fx_counts_dir(tmp_path):
    counts_path = tmp_path / "counts_dir"
    counts_path.mkdir()
    (counts_path / "counts_s1_file.csv").touch()
    (counts_path / "counts_s2_file.csv").touch()
    (counts_path / "counts_s2_ambiguous.csv").touch()
    return counts_path


# ------------------------
# Tests for _prep_sample_df
# ------------------------

@patch('glob.glob')
@patch('os.path.isdir', return_value=True)
@patch('tfscreen.util.io.read_dataframe')
def test_prep_sample_df_happy_path(mock_read_df, mock_isdir, mock_glob, fx_sample_df):
    mock_read_df.return_value = fx_sample_df.set_index("sample")
    mock_glob.side_effect = [
        ["/path/to/counts_s1_file.csv"],
        ["/path/to/counts_s2_file.csv"]
    ]

    result_df = _prep_sample_df(
        sample_df="dummy_path",
        counts_csv_path="/path/to",
        counts_glob_prefix="counts",
        verbose=False
    )

    mock_read_df.assert_called_once_with("dummy_path", index_column="sample")
    assert "obs_file" in result_df.columns
    assert result_df.loc["s1", "obs_file"] == "/path/to/counts_s1_file.csv"
    assert result_df.index.name == "sample"


@patch('builtins.print')
@patch('glob.glob')
@patch('os.path.isdir', return_value=True)
@patch('tfscreen.util.io.read_dataframe')
def test_prep_sample_df_verbose_output(mock_read_df, mock_isdir, mock_glob, mock_print, fx_sample_df):
    mock_read_df.return_value = fx_sample_df.set_index("sample")
    mock_glob.side_effect = [["f1"], ["f2"]]

    _prep_sample_df(sample_df="dummy", counts_csv_path="dummy", verbose=True)

    assert mock_print.call_count > 0
    assert "2 unique samples" in mock_print.call_args_list[0][0][0]


def test_prep_sample_df_dir_not_found(fx_sample_df):
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with patch('os.path.isdir', return_value=False):
            with pytest.raises(FileNotFoundError, match="is not a directory"):
                _prep_sample_df(fx_sample_df, "invalid_path")


def test_prep_sample_df_duplicate_samples(fx_sample_df):
    dup_df = pd.concat([fx_sample_df, fx_sample_df.iloc[[0]]]).set_index("sample")
    with patch('tfscreen.util.io.read_dataframe', return_value=dup_df):
        with pytest.raises(ValueError, match="samples must be unique"):
            _prep_sample_df("dummy_path", "dummy_dir")


def test_prep_sample_df_missing_file(fx_sample_df, fx_counts_dir):
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with pytest.raises(ValueError, match="MISSING: No files found for sample 's1'"):
            _prep_sample_df(fx_sample_df, str(fx_counts_dir), counts_glob_prefix="obs")


def test_prep_sample_df_ambiguous_files(fx_sample_df, fx_counts_dir):
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df.set_index("sample")):
        with pytest.raises(ValueError, match="AMBIGUOUS: 2 files found for sample 's2'"):
            _prep_sample_df(fx_sample_df, str(fx_counts_dir), counts_glob_prefix="counts")


def test_prep_sample_df_bad_index(fx_sample_df):
    with patch('tfscreen.util.io.read_dataframe', return_value=fx_sample_df):
        with pytest.raises(ValueError, match="sample_df must be indexed by 'sample'"):
            _prep_sample_df(fx_sample_df, "dummy_dir")


# ------------------------
# Tests for _aggregate_counts
# ------------------------

@patch('pandas.read_csv')
def test_aggregate_counts(mock_read_csv):
    sample_df = pd.DataFrame({
        "obs_file": ["/path/s1.csv", "/path/s2.csv"]
    }, index=pd.Index(["s1", "s2"], name="sample"))

    df1 = pd.DataFrame({"genotype": ["wt", "A1C"], "counts": [100, 10]})
    df2 = pd.DataFrame({"genotype": ["wt", "G2T"], "counts": [200, 20]})
    mock_read_csv.side_effect = [df1, df2]

    result_df = _aggregate_counts(sample_df)

    expected_df = pd.DataFrame({
        "sample": ["s1", "s1", "s2", "s2"],
        "genotype": ["wt", "A1C", "wt", "G2T"],
        "counts": [100, 10, 200, 20]
    })

    pd_testing.assert_frame_equal(result_df, expected_df)
    mock_read_csv.assert_has_calls([call("/path/s1.csv"), call("/path/s2.csv")])
