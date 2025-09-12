from tfscreen.util import read_dataframe

import pytest
import pandas as pd

# --- Fixtures ---

@pytest.fixture
def sample_df():
    """Provides a basic, clean DataFrame for tests."""
    return pd.DataFrame({
        "id": ["A", "B", "C"],
        "value": [10, 20, 30]
    })

@pytest.fixture
def df_with_spurious_index():
    """Provides a DataFrame with a classic 'Unnamed: 0' from to_csv."""
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "id": ["A", "B", "C"],
        "value": [10, 20, 30]
    })

# --- Test Cases ---

def test_read_from_dataframe_source(sample_df):
    """
    Tests that passing a DataFrame returns a copy, not the same object.
    """
    processed_df = read_dataframe(sample_df)
    assert isinstance(processed_df, pd.DataFrame)
    assert processed_df is not sample_df
    pd.testing.assert_frame_equal(processed_df, sample_df)

def test_read_from_invalid_source():
    """
    Tests that a TypeError is raised for unsupported input types.
    """
    with pytest.raises(TypeError, match="must be a file path .* or pandas DataFrame"):
        read_dataframe([1, 2, 3])

def test_read_from_csv(tmp_path, sample_df):
    """
    Tests successful reading from a CSV file.
    """
    csv_path = tmp_path / "test.csv"
    sample_df.to_csv(csv_path, index=False)
    
    loaded_df = read_dataframe(str(csv_path))
    pd.testing.assert_frame_equal(loaded_df, sample_df)

def test_read_from_tsv(tmp_path, sample_df):
    """
    Tests successful reading from a TSV file. (NEW)
    """
    tsv_path = tmp_path / "test.tsv"
    sample_df.to_csv(tsv_path, sep="\t", index=False)
    
    loaded_df = read_dataframe(str(tsv_path))
    pd.testing.assert_frame_equal(loaded_df, sample_df)

def test_read_from_file_with_no_extension(tmp_path, sample_df):
    """
    Tests the fallback reader for files without a recognized extension. (NEW)
    """
    fallback_path = tmp_path / "datafile"
    sample_df.to_csv(fallback_path, index=False) # Save as CSV
    
    # This should trigger the `sep=None` fallback in read_csv
    loaded_df = read_dataframe(str(fallback_path))
    pd.testing.assert_frame_equal(loaded_df, sample_df)

def test_read_from_excel(tmp_path, sample_df):
    """
    Tests successful reading from an Excel (.xlsx) file.
    Requires the `openpyxl` package.
    """
    excel_path = tmp_path / "test.xlsx"
    sample_df.to_excel(excel_path, index=False)

    loaded_df = read_dataframe(str(excel_path))
    pd.testing.assert_frame_equal(loaded_df, sample_df)

def test_read_file_not_found():
    """
    Tests that a ValueError is raised for a non-existent file path.
    """
    with pytest.raises(ValueError, match="File not found at path"):
        read_dataframe("non_existent_file.csv")

def test_read_raises_ioerror_on_parse_error(tmp_path, monkeypatch):
    """
    Tests that a generic pandas read error is caught and re-raised as IOError. (NEW)
    """
    # Define a mock function that will replace pd.read_csv and raise an error
    def mock_read_with_error(*args, **kwargs):
        raise ValueError("Mock parsing error")

    # Use monkeypatch to replace the real pd.read_csv with our mock
    monkeypatch.setattr(pd, "read_csv", mock_read_with_error)

    bad_file_path = tmp_path / "bad.csv"
    bad_file_path.write_text("this content doesn't matter, the read is mocked")

    with pytest.raises(IOError, match="Mock parsing error"):
        read_dataframe(str(bad_file_path))

def test_drops_spurious_unnamed_col_by_default(df_with_spurious_index, sample_df):
    """
    Tests the default behavior of finding and dropping a spurious index.
    """
    processed_df = read_dataframe(df_with_spurious_index)
    pd.testing.assert_frame_equal(processed_df, sample_df)

def test_keeps_non_spurious_unnamed_col():
    """
    Tests that an 'Unnamed: 0' column is KEPT if it's not a default index.
    """
    df = pd.DataFrame({"Unnamed: 0": ["x", "y", "z"], "value": [1, 2, 3]})
    processed_df = read_dataframe(df.copy())
    pd.testing.assert_frame_equal(processed_df, df)

def test_set_existing_column_as_index(sample_df):
    """
    Tests setting an existing column as the DataFrame index.
    """
    processed_df = read_dataframe(sample_df.copy(), index_column="id")
    assert processed_df.index.name == "id"
    assert "id" not in processed_df.columns
    assert list(processed_df.index) == ["A", "B", "C"]

def test_rename_unnamed_col_to_index(df_with_spurious_index):
    """
    Tests the 'magic' renaming of 'Unnamed: 0' to a requested index_column.
    """
    with pytest.warns(UserWarning, match="Renaming column 'Unnamed: 0' to 'sample_id'"):
        processed_df = read_dataframe(df_with_spurious_index, index_column="sample_id")
    
    assert processed_df.index.name == "sample_id"
    assert "sample_id" not in processed_df.columns
    assert list(processed_df.index) == [0, 1, 2]

def test_raises_error_if_index_col_not_found(sample_df):
    """
    Tests that a ValueError is raised if the requested index can't be found.
    """
    with pytest.raises(ValueError, match="Column 'non_existent_col' not found"):
        read_dataframe(sample_df.copy(), index_column="non_existent_col")
        
def test_no_change_if_already_indexed(sample_df):
    """
    Tests that nothing happens if the requested index is already set.
    """
    indexed_df = sample_df.set_index("id")
    processed_df = read_dataframe(indexed_df.copy(), index_column="id")
    
    pd.testing.assert_frame_equal(processed_df, indexed_df)