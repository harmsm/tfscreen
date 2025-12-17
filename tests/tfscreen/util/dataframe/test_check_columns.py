import pytest
import pandas as pd
from tfscreen.util import check_columns

def test_all_columns_present():
    """
    Tests that no error is raised when all required columns exist.
    """
    df = pd.DataFrame({"col_A": [1], "col_B": [2], "col_C": [3]})
    
    # Case 1: Required columns are a subset of DataFrame columns
    try:
        check_columns(df, required_columns=["col_A", "col_C"])
    except ValueError:
        pytest.fail("check_columns raised ValueError unexpectedly.")

    # Case 2: Required columns are an exact match for DataFrame columns
    try:
        check_columns(df, required_columns=["col_A", "col_B", "col_C"])
    except ValueError:
        pytest.fail("check_columns raised ValueError unexpectedly.")

def test_missing_columns_raises_error():
    """
    Tests that a ValueError is raised when required columns are missing.
    """
    df = pd.DataFrame({"col_A": [1], "col_B": [2]})
    
    # This now matches the actual error message from your function
    expected_error_text = "Not all required columns seen"

    # Test for a single missing column
    with pytest.raises(ValueError, match=expected_error_text):
        check_columns(df, required_columns=["col_A", "col_Z"])

    # Test for multiple missing columns
    with pytest.raises(ValueError, match=expected_error_text):
        check_columns(df, required_columns=["col_X", "col_Y", "col_Z"])

def test_error_message_contents():
    """
    Tests that the error message correctly lists the missing columns.
    """
    df = pd.DataFrame({"col_A": [1]})
    required = ["col_A", "col_B", "col_C"]
    
    with pytest.raises(ValueError) as exc_info:
        check_columns(df, required_columns=required)
    
    error_message = str(exc_info.value)
    assert "col_B" in error_message
    assert "col_C" in error_message
    assert "col_A" not in error_message

def test_empty_required_list():
    """
    Tests the edge case where the list of required columns is empty.
    The function should not raise an error.
    """
    df = pd.DataFrame({"col_A": [1]})
    try:
        check_columns(df, required_columns=[])
    except ValueError:
        pytest.fail("check_columns raised ValueError for an empty required_columns list.")

def test_empty_dataframe():
    """
    Tests the edge case of checking an empty DataFrame for required columns.
    A ValueError should be raised.
    """
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        check_columns(df, required_columns=["col_A"])