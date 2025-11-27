import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Import the function to be tested
from tfscreen.genetics import expand_genotype_columns

# --- Test Fixtures -----------------------------------------------------------

@pytest.fixture
def mixed_genotypes() -> list:
    """Provides a comprehensive list of genotype strings for testing."""
    return [
        "wt",
        "A15G",
        "A15G/P42Q",
        "V10L/I20M/L30F",
        "bad_format",
        None,
        "C50R/bad_again"
    ]

# --- Unit Tests --------------------------------------------------------------

class TestExpandGenotypeColumns:

    @pytest.mark.parametrize("input_type", [list, pd.Series, np.array])
    def test_input_types(self, input_type, mixed_genotypes):
        """
        GIVEN different iterable input types (list, Series, array)
        WHEN expand_genotype_columns is called
        THEN it should return a valid DataFrame with correct parsing.
        """
        # ARRANGE
        genotype_input = input_type(mixed_genotypes)
        
        # ACT
        result_df = expand_genotype_columns(genotype_input)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        # Check a known value: the third genotype should have 2 mutations
        assert result_df.loc[2, "num_muts"] == 2
        assert result_df.loc[2, "wt_aa_2"] == "P"

    def test_dataframe_input_preserves_columns_and_order(self):
        """
        GIVEN a DataFrame as input with extra columns
        WHEN expand_genotype_columns is called
        THEN it should preserve the extra columns and insert new ones correctly.
        """
        # ARRANGE
        input_df = pd.DataFrame({
            "id": [101, 102],
            "genotype": ["A15G", "P42Q/L60R"],
            "extra_data": ["data1", "data2"]
        })
        
        # ACT
        result_df = expand_genotype_columns(input_df)
        
        # ASSERT
        # Check that original columns are preserved
        assert "id" in result_df.columns
        assert "extra_data" in result_df.columns
        assert result_df.loc[0, "id"] == 101
        
        # Check that new columns are inserted directly after 'genotype'
        expected_cols = [
            'id', 'genotype', 'wt_aa_1', 'resid_1', 'mut_aa_1',
            'wt_aa_2', 'resid_2', 'mut_aa_2', 'num_muts', 'extra_data'
        ]
        assert result_df.columns.tolist() == expected_cols

    def test_parsing_correctness(self, mixed_genotypes):
        """
        GIVEN a list of varied genotype strings
        WHEN expand_genotype_columns is called
        THEN it should correctly parse each component into the right column.
        """
        # ACT
        result_df = expand_genotype_columns(mixed_genotypes)
        
        # ASSERT
        # Check wt (row 0)
        assert result_df.loc[0, "num_muts"] == 0
        assert pd.isna(result_df.loc[0, "wt_aa_1"])
        
        # Check single mutant (row 1)
        assert result_df.loc[1, "num_muts"] == 1
        assert result_df.loc[1, "wt_aa_1"] == "A"
        assert result_df.loc[1, "resid_1"] == 15
        assert result_df.loc[1, "mut_aa_1"] == "G"
        assert pd.isna(result_df.loc[1, "wt_aa_2"])
        
        # Check double mutant (row 2)
        assert result_df.loc[2, "num_muts"] == 2
        assert result_df.loc