# test_util.py

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Import the function to be tested
from tfscreen.util import df_to_arrays


@pytest.fixture
def base_df():
    """Provides a standard, unsorted DataFrame for testing."""
    data = {
        "strain": ["B", "A", "A", "B", "A"],
        "condition": ["Y", "X", "X", "Y", "X"],
        "OD": [0.15, 0.1, 0.2, 0.35, 0.4],
        "fluorescence": [12, 10, 22, 25, 45],
        "notes": ["n1", "n2", "n3", "n4", "n5"]
    }
    return pd.DataFrame(data)


def test_basic_functionality_single_pivot(base_df):
    """
    Tests the primary functionality with a single pivot column, unsorted input,
    and unequal group sizes.
    """
    row_ids, out = df_to_arrays(base_df, pivot_on="strain")
    assert isinstance(row_ids, pd.Index)
    assert row_ids.name == "strain"
    assert row_ids.tolist() == ["A", "B"]
    assert set(out.keys()) == {"OD", "fluorescence"}
    assert out["OD"].shape == (2, 3)
    expected_od = np.array([[0.1, 0.2, 0.4], [0.15, 0.35, np.nan]])
    np.testing.assert_allclose(out["OD"], expected_od, equal_nan=True)


def test_multi_column_pivot(base_df):
    """
    Tests functionality when pivot_on is a list of columns.
    """
    pivot_cols = ["strain", "condition"]
    row_ids, out = df_to_arrays(base_df, pivot_on=pivot_cols)
    assert isinstance(row_ids, pd.MultiIndex)
    expected_index = pd.MultiIndex.from_tuples(
        [("A", "X"), ("B", "Y")], names=pivot_cols
    )
    assert row_ids.equals(expected_index)


def test_no_numeric_columns():
    """
    Tests behavior when the input DataFrame has no numeric columns to process.
    """
    df = pd.DataFrame({"strain": ["A", "A", "B"], "notes": ["n1", "n2", "n3"]})
    row_ids, out = df_to_arrays(df, pivot_on="strain")
    assert row_ids.tolist() == ["A", "B"]
    assert isinstance(out, dict) and not out


def test_empty_dataframe():
    """
    Tests the edge case of an empty DataFrame input.
    """
    df = pd.DataFrame({"strain": [], "OD": []})
    row_ids, out = df_to_arrays(df, pivot_on="strain")
    assert row_ids.empty
    assert isinstance(out, dict) and not out


def test_no_side_effects(base_df):
    """
    Ensures the original input DataFrame is not modified by the function.
    """
    df_original = base_df.copy()
    df_to_arrays(base_df, pivot_on="strain")
    assert_frame_equal(base_df, df_original)

# --- NEW TESTS START HERE ---

def test_all_nan_column_is_handled():
    """
    Tests that a numeric column containing only NaNs is correctly
    re-created as a NaN array in the output.
    """
    data = {
        "group": ["A", "A", "B", "B", "B"],
        "values": [10, 20, 30, 40, 50],
        "pred": [np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(data)

    _, out = df_to_arrays(df, pivot_on="group")

    # 1. Check that 'pred' is a key in the output dictionary
    assert set(out.keys()) == {"values", "pred"}

    # 2. Check that the shape of the recreated 'pred' array is correct
    assert out["values"].shape == (2, 3)
    assert out["pred"].shape == (2, 3)

    # 3. Check that the 'pred' array contains only NaNs
    assert np.all(np.isnan(out["pred"]))


def test_all_numeric_columns_are_nan():
    """
    Tests the edge case where ALL numeric columns are entirely NaN.
    """
    data = {
        "group": ["A", "A", "B"],
        "nan_col_1": [np.nan, np.nan, np.nan],
        "nan_col_2": [np.nan, np.nan, np.nan],
        "notes": ["x", "y", "z"]
    }
    df = pd.DataFrame(data)

    _, out = df_to_arrays(df, pivot_on="group")

    # 1. Check that both NaN columns are present in the output
    assert set(out.keys()) == {"nan_col_1", "nan_col_2"}

    # 2. Check that the shapes are correct (2 groups, max 2 members)
    assert out["nan_col_1"].shape == (2, 2)
    assert out["nan_col_2"].shape == (2, 2)

    # 3. Check that the arrays contain only NaNs
    assert np.all(np.isnan(out["nan_col_1"]))
    assert np.all(np.isnan(out["nan_col_2"]))

def test_empty_dataframe_multi_pivot():
    """
    Tests empty dataframe with multiple pivot columns.
    """
    df = pd.DataFrame({"strain": [], "condition": [], "val": []})
    row_ids, out = df_to_arrays(df, pivot_on=["strain", "condition"])
    assert row_ids.empty
    assert isinstance(row_ids, pd.MultiIndex)
    assert row_ids.names == ["strain", "condition"]
    assert isinstance(out, dict) and not out