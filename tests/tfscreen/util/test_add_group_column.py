import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pandas.api.types import is_integer_dtype

# Import the function as requested
from tfscreen.util.add_group_columns import add_group_columns

# --- Fixtures ---

@pytest.fixture
def target_df():
    """A sample DataFrame to which columns will be added."""
    data = {
        'cond': ['A', 'A', 'B', 'B', 'C', 'C'],
        'rep':  [1, 2, 1, 2, 1, 2],
        'val':  [10, 11, 20, 21, 30, 31]
    }
    return pd.DataFrame(data)

@pytest.fixture
def existing_df():
    """A sample DataFrame defining an existing group mapping."""
    data = {
        'cond': ['B', 'A'],  # Note: C-order is A, then B
        'rep':  [1, 1],
        'genotype': [1, 0]  # This is the group_name column
    }
    return pd.DataFrame(data)

# --- Test Cases ---

class TestAddGroupColumns:

    def test_new_group_creation(self, target_df):
        """Tests the function when existing_df is None."""
        group_cols = ['cond', 'rep']
        group_name = 'test_group'
        
        # Make a copy to check that the original is not modified
        target_df_copy = target_df.copy()
        
        result_df = add_group_columns(target_df, group_cols, group_name)

        # Check that the original df is unmodified
        assert_frame_equal(target_df, target_df_copy)

        # Check for new columns
        assert f"{group_name}" in result_df.columns
        assert f"{group_name}_tuple" in result_df.columns

        # Check types
        assert is_integer_dtype(result_df[group_name])
        assert isinstance(result_df[f"{group_name}_tuple"].dtype,pd.CategoricalDtype)

        # Check C-order sorting for the integer index
        # ('A', 1) -> 0, ('A', 2) -> 1, ('B', 1) -> 2, etc.
        expected_idx = pd.Series([0, 1, 2, 3, 4, 5], name=group_name)
        pd.testing.assert_series_equal(result_df[group_name], expected_idx)

        # Check tuple column
        expected_tuples = pd.Categorical(
            [('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)]
        )
        pd.testing.assert_series_equal(
            result_df[f"{group_name}_tuple"], 
            pd.Series(expected_tuples, name=f"{group_name}_tuple")
        )

    def test_with_existing_df_merge_and_filter(self, target_df, existing_df):
        """Tests merging from existing_df and filtering non-matching rows."""
        group_cols = ['cond', 'rep']
        group_name = 'genotype' # Matches the column in existing_df
        
        result_df = add_group_columns(target_df, 
                                      group_cols, 
                                      group_name, 
                                      existing_df)

        # Check that new columns are present
        assert group_name in result_df.columns
        assert f"{group_name}_tuple" in result_df.columns

        # CRITICAL: Check that only matching rows are kept
        # The target_df had 6 rows, but existing_df only defines
        # ('A', 1) and ('B', 1). Only 2 rows should remain.
        assert len(result_df) == 2
        
        # Check that the group_name column is integer (not float after merge)
        assert is_integer_dtype(result_df[group_name])

        # Check remaining data
        expected_data = {
            'cond': ['A', 'B'],
            'rep':  [1, 1],
            'val':  [10, 20],
            'genotype': [0, 1], # Merged from existing_df
            'genotype_tuple': pd.Categorical([('A', 1), ('B', 1)])
        }
        expected_df = pd.DataFrame(expected_data)
        
        # Sort for reliable comparison
        result_df = result_df.sort_values(by='cond').reset_index(drop=True)
        
        assert_frame_equal(result_df, expected_df)

    def test_existing_df_raises_missing_group_col(self, target_df):
        """Tests error if existing_df misses a column from group_cols."""
        group_cols = ['cond', 'rep']
        group_name = 'genotype'
        
        # 'rep' is missing
        bad_existing_df = pd.DataFrame({'cond': ['A'], 'genotype': [0]})
        
        with pytest.raises(ValueError, match="existing_df does not have all"):
            add_group_columns(target_df, 
                              group_cols, 
                              group_name, 
                              bad_existing_df)

    def test_existing_df_raises_missing_group_name_col(self, target_df):
        """Tests error if existing_df misses the group_name column."""
        group_cols = ['cond', 'rep']
        group_name = 'genotype'
        
        # 'genotype' is missing
        bad_existing_df = pd.DataFrame({'cond': ['A'], 'rep': [1]})
        
        with pytest.raises(ValueError, match="existing_df does not have all"):
            add_group_columns(target_df, 
                              group_cols, 
                              group_name, 
                              bad_existing_df)