import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

import tfscreen

# Import the function to be tested
from tfscreen.genetics import get_single_with_wt



# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def base_df_for_singles() -> pd.DataFrame:
    """Provides a DataFrame with wt, single, and double mutants across conditions."""
    data = {
        'genotype': ['wt', 'A10G', 'P25L', 'wt', 'A10C', 'G50R/A10G'],
        'condition': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
        'fitness': [1.0, 0.8, 0.5, 0.9, 0.7, 0.2],
        'extra_data': [100]*6
    }
    return pd.DataFrame(data)

# --- Unit Tests --------------------------------------------------------------

# ----------------------------------------------
# tests for get_single_with_wt
# ----------------------------------------------

class TestGetSingleWithWT:

    def test_happy_path(self, base_df_for_singles):
        """
        GIVEN a valid dataframe with wt, singles, and doubles
        WHEN get_single_with_wt is called
        THEN it should return only singles and new site-specific wt rows.
        """
        # ACT
        result_df = get_single_with_wt(base_df_for_singles, condition_selector=['condition'])
        
        # ASSERT
        # Check that original 'wt' and the double mutant are gone
        assert 'wt' not in result_df['genotype'].values
        assert 'G50R/A10G' not in result_df['genotype'].values
        
        # Check that new "no-mutation" rows were created for each site/condition
        assert 'A10A' in result_df['genotype'].values
        assert 'P25P' in result_df['genotype'].values
        
        # There should be 2 original singles + (2 sites * 2 conditions) = 6 rows total
        assert len(result_df) == 7
        
        # Check that data was copied correctly. The 'A10A' row for condition 'X'
        # should have the fitness of the original 'wt' row for condition 'X'.
        a10a_x_fitness = result_df.query("genotype == 'A10A' and condition == 'X'")['fitness'].iloc[0]
        assert np.isclose(a10a_x_fitness, 1.0)

        # Check that the 'P25P' row for condition 'Y' has the correct fitness
        p25p_y_fitness = result_df.query("genotype == 'P25P' and condition == 'Y'")['fitness'].iloc[0]
        assert np.isclose(p25p_y_fitness, 0.9)
        
        # Check that other columns are preserved
        assert 'extra_data' in result_df.columns

    def test_calls_expand_genotype_columns_if_needed(self, base_df_for_singles, mocker):
        """
        GIVEN a dataframe without pre-expanded columns
        WHEN get_single_with_wt is called
        THEN it should call the expansion helper function.
        """
        # ARRANGE
        mock_expand = mocker.patch("tfscreen.genetics.expand_genotype_columns", 
                                   side_effect=tfscreen.genetics.expand_genotype_columns)
        
        # ACT
        get_single_with_wt(base_df_for_singles, condition_selector=['condition'])
        
        # ASSERT
        mock_expand.assert_called_once()

    # In test_does_not_call_expand_if_already_expanded...
    def test_does_not_call_expand_if_already_expanded(self, base_df_for_singles, mocker):
        # ARRANGE
        # First, create the real expanded DataFrame *before* patching.
        expanded_df = tfscreen.genetics.expand_genotype_columns(base_df_for_singles)
        
        # NOW, patch the function so we can check if it's called again.
        mock_expand = mocker.patch("tfscreen.genetics.expand_genotype_columns")
        
        # ACT
        get_single_with_wt(expanded_df, condition_selector=['condition'])
        
        # ASSERT
        mock_expand.assert_not_called()

    def test_handles_no_mutants(self, base_df_for_singles):
        """
        GIVEN a dataframe with only wildtype rows
        WHEN get_single_with_wt is called
        THEN it should return an empty dataframe.
        """
        # ARRANGE
        wt_only_df = base_df_for_singles[base_df_for_singles['genotype'] == 'wt']
        
        # ACT
        result_df = get_single_with_wt(wt_only_df, condition_selector=['condition'])
        
        # ASSERT
        assert result_df.empty

    def test_handles_no_wt(self, base_df_for_singles):
        """
        GIVEN a dataframe with only mutant rows (no 'wt')
        WHEN get_single_with_wt is called
        THEN it should return only the single mutants, as no template exists.
        """
        # ARRANGE
        mutants_only_df = base_df_for_singles[base_df_for_singles['genotype'] != 'wt']

        # ACT
        result_df = get_single_with_wt(mutants_only_df, condition_selector=['condition'])
        
        # ASSERT
        # Should contain the single mutants but not the double
        assert 'A10G' in result_df['genotype'].values
        assert 'G50R/A10G' not in result_df['genotype'].values
        
        # No "no-mutation" rows should have been created
        assert 'A10A' not in result_df['genotype'].values
        assert len(result_df) == 3 # Only the two single mutants
        
    def test_handles_empty_input(self):
        """
        GIVEN an empty dataframe
        WHEN get_single_with_wt is called
        THEN it should return an empty dataframe.
        """
        # FIX: Explicitly set the dtype to 'object' for the empty column
        #      to correctly simulate an empty dataframe of strings.
        empty_df = pd.DataFrame({'genotype': pd.Series(dtype='object'), 
                                'condition': pd.Series(dtype='object')})
        result_df = get_single_with_wt(empty_df, condition_selector=['condition'])
        assert result_df.empty