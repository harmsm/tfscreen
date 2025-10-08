import pytest
import pandas as pd
import numpy as np

# Import the functions to be tested
from tfscreen.analysis.extract_epistasis import (
    mutant_cycle_pivot,
    extract_epistasis
)

# Assuming this utility exists from your previous code
from tfscreen.genetics import standardize_genotypes

# Mock the utility function so we don't need the actual file
# Mock the utility function so we don't need the actual file
@pytest.fixture(autouse=True)
def mock_standardize_genotypes(mocker):
    mocker.patch("tfscreen.genetics.standardize_genotypes", side_effect=standardize_genotypes)

# --- Test Fixtures -----------------------------------------------------------

@pytest.fixture
def base_df_for_epistasis() -> pd.DataFrame:
    """
    Provides a complex DataFrame to test multiple epistasis scenarios.
    - Condition 1: A complete, valid mutant cycle.
    - Condition 2: A cycle missing a single mutant ('M40T').
    - Condition 3: A group of mutants missing a wildtype ('wt').
    - Condition 4: A group with a duplicated genotype ('Y75F').
    - Condition 5: A group containing a higher-order (triple) mutant.
    - Condition 6: A cycle with a reversed double mutant ('P25L/A10G').
    """
    data = {
        "condition":  [1, 1, 1, 1,   
                       2, 2, 2,   
                       3, 3, 3,   
                       4, 4, 4,   
                       5, 5, 5, 5,
                       6, 6, 6, 6],
        "genotype":   ["wt", "A10G", "P25L", "A10G/P25L", 
                       "wt", "V30A", "V30A/M40T", 
                       "G55S", "L60R", "G55S/L60R", 
                       "wt", "Y75F", "Y75F", 
                       "wt", "R80K", "S90A", "R80K/S90A/T100C",
                       "wt", "A10G", "P25L", "P25L/A10G"],
        "fitness":    [1.0, 0.8, 0.5, 0.3, 
                       1.0, 0.7, 0.2,
                       0.9, 0.6, 0.1,
                       1.0, 0.4, 0.4,
                       1.0, 0.8, 0.7, 0.1,
                       1.0, 0.9, 0.6, 0.4],
        "error":      [0.05, 0.04, 0.03, 0.06,
                       0.05, 0.02, 0.04,
                       0.01, 0.01, 0.02,
                       0.05, 0.03, 0.03,
                       0.05, 0.01, 0.01, 0.01,
                       0.05, 0.01, 0.01, 0.02],
        "extra_col":  "data"
    }
    return pd.DataFrame(data)
# --- `mutant_cycle_pivot` Tests -----------------------------------------------

class TestMutantCyclePivot:
    def test_happy_path(self, base_df_for_epistasis):
        """
        GIVEN a df with a complete mutant cycle
        WHEN mutant_cycle_pivot is called
        THEN it should return a single row with correctly pivoted data.
        """
        # ARRANGE
        df = base_df_for_epistasis[base_df_for_epistasis["condition"] == 1]
        
        # ACT
        result = mutant_cycle_pivot(df, ["fitness", "error"], "condition",)
        
        # ASSERT
        assert len(result) == 1
        assert result.loc[0, "genotype"] == "A10G/P25L"
        assert result.loc[0, "00_fitness"] == 1.0  # wt
        assert result.loc[0, "01_fitness"] == 0.8  # A10G
        assert result.loc[0, "10_fitness"] == 0.5  # P25L
        assert result.loc[0, "11_fitness"] == 0.3  # A10G/P25L
        assert result.loc[0, "01_error"] == 0.04

    def test_standardizes_genotypes(self, base_df_for_epistasis):
        """
        GIVEN a df with a 'P25L/A10G' genotype
        WHEN mutant_cycle_pivot is called
        THEN it should standardize it to 'A10G/P25L' with m0='A10G' and m1='P25L'.
        """
        df = base_df_for_epistasis[base_df_for_epistasis["condition"] == 6]
        result = mutant_cycle_pivot(df, ["fitness"],"condition")
        
        assert len(result) == 1
        assert result.loc[0, "genotype"] == "A10G/P25L"
        assert result.loc[0, "m0"] == "A10G"
        assert result.loc[0, "m1"] == "P25L"
        assert result.loc[0, "11_fitness"] == 0.4

    def test_handles_missing_single_mutant(self, base_df_for_epistasis):
        """
        GIVEN a df with a cycle missing a single mutant
        WHEN mutant_cycle_pivot is called
        THEN the corresponding columns in the output should be NaN.
        """
        df = base_df_for_epistasis[base_df_for_epistasis["condition"] == 2]
        result = mutant_cycle_pivot(df, ["fitness"],"condition")
        
        assert len(result) == 1
        assert result.loc[0, "genotype"] == "V30A/M40T"
        assert result.loc[0, "01_fitness"] == 0.7  # V30A is present
        assert pd.isna(result.loc[0, "10_fitness"]) # M40T is missing

    def test_skips_group_missing_wt(self, base_df_for_epistasis, capsys):
        """
        GIVEN a group with no 'wt' entry
        WHEN mutant_cycle_pivot is called
        THEN that group should be skipped and not appear in the output.
        """
        df = base_df_for_epistasis[base_df_for_epistasis["condition"].isin([1, 3])]
        result = mutant_cycle_pivot(df, ["fitness"], "condition",verbose=True)
        
        assert len(result) == 1
        assert result.loc[0, "condition"] == 1
        
        captured = capsys.readouterr()
        assert "No 'wt' entry for group '3'. Skipping." in captured.out

    def test_skips_group_with_duplicates(self, base_df_for_epistasis, capsys):
        """
        GIVEN a group with duplicate genotypes
        WHEN mutant_cycle_pivot is called
        THEN that group should be skipped.
        """
        df = base_df_for_epistasis[base_df_for_epistasis["condition"].isin([1, 4])]
        result = mutant_cycle_pivot(df, ["fitness"], "condition", verbose=True)
        
        assert len(result) == 1
        assert result.loc[0, "condition"] == 1
        
        captured = capsys.readouterr()
        assert "Duplicate genotypes found for group '4'" in captured.out
        
    def test_filters_higher_order_mutants(self, base_df_for_epistasis):
        """
        GIVEN a group with a triple mutant
        WHEN mutant_cycle_pivot is called
        THEN no cycle is generated for it, and it is ignored.
        """
        df = base_df_for_epistasis[base_df_for_epistasis["condition"] == 5]
        result = mutant_cycle_pivot(df, ["fitness"],"condition")
        assert result.empty # No double mutants in this group

    def test_returns_empty_for_empty_input(self):
        """Tests that an empty input DataFrame results in an empty output."""
        result = mutant_cycle_pivot(pd.DataFrame(),  ["f"],"c",)
        assert result.empty
        assert isinstance(result, pd.DataFrame)
        
    def test_returns_empty_if_no_valid_cycles(self, base_df_for_epistasis):
        """Tests that an empty DataFrame is returned if no groups are valid."""
        df = base_df_for_epistasis[base_df_for_epistasis["condition"] == 3] # Only has a group missing wt
        result = mutant_cycle_pivot(df,  ["fitness"],"condition")
        assert result.empty

# --- `extract_epistasis` Tests ------------------------------------------------

class TestExtractEpistasis:

    def test_additive_epistasis(self, base_df_for_epistasis):
        """Tests correct calculation of additive epistasis and error."""
        # ARRANGE
        # E = (Y11 - Y10) - (Y01 - Y00) = (0.3 - 0.5) - (0.8 - 1.0) = -0.2 - (-0.2) = 0.0
        # E_std = sqrt(0.06^2 + 0.03^2 + 0.04^2 + 0.05^2) = sqrt(0.0086) = 0.0927
        
        # ACT
        result = extract_epistasis(base_df_for_epistasis, "fitness", "error", "condition","add")
        
        # ASSERT
        cycle_1 = result[result["condition"] == 1].iloc[0]
        assert np.isclose(cycle_1["ep_obs"], 0.0)
        assert np.isclose(cycle_1["ep_std"], 0.092736, atol=1e-5)
    
    def test_multiplicative_epistasis(self, base_df_for_epistasis):
        """Tests correct calculation of multiplicative epistasis and error."""
        # ARRANGE
        # E = (Y11/Y10) / (Y01/Y00) = (0.3/0.5) / (0.8/1.0) = 0.6 / 0.8 = 0.75
        # E_std uses relative error propagation
        
        # ACT
        result = extract_epistasis(base_df_for_epistasis, "fitness", "error", condition_selector= "condition",scale="mult")
        
        # ASSERT
        cycle_1 = result[result["condition"] == 1].iloc[0]
        assert np.isclose(cycle_1["ep_obs"], 0.75)
        
        # Check propagated error
        rel_err_sq = (0.06/0.3)**2 + (0.03/0.5)**2 + (0.04/0.8)**2 + (0.05/1.0)**2
        expected_std = 0.75 * np.sqrt(rel_err_sq) # 0.165
        assert np.isclose(cycle_1["ep_std"], expected_std, atol=1e-5)

    def test_without_std_dev(self, base_df_for_epistasis):
        """Tests that the function runs without a y_std column."""
        result = extract_epistasis(base_df_for_epistasis, "fitness", condition_selector="condition") 
        assert "ep_std" not in result.columns
        assert np.isclose(result[result["condition"] == 1].iloc[0]["ep_obs"], 0.0)

    def test_propagates_nan_from_missing_mutant(self, base_df_for_epistasis):
        """Tests that if a cycle is incomplete, ep_obs and ep_std are NaN."""
        result = extract_epistasis(base_df_for_epistasis, "fitness", "error",condition_selector="condition")
        cycle_2 = result[result["condition"] == 2].iloc[0] # The V30A/M40T cycle
        
        assert pd.isna(cycle_2["ep_obs"])
        assert pd.isna(cycle_2["ep_std"])

    def test_raises_on_invalid_scale(self, base_df_for_epistasis):
        """Tests that an invalid `scale` argument raises a ValueError."""
        with pytest.raises(ValueError, match="scale should be"):
            extract_epistasis(base_df_for_epistasis, "fitness", condition_selector="condition",scale="invalid")
            
    def test_keep_extra_columns(self, base_df_for_epistasis):
        """Tests the `keep_extra` flag."""
        # keep_extra = True should preserve 'extra_col'
        result_true = extract_epistasis(base_df_for_epistasis, "fitness", condition_selector="condition", keep_extra=True)
        assert "extra_col" in result_true.columns
        
        # keep_extra = False (default) should drop 'extra_col'
        result_false = extract_epistasis(base_df_for_epistasis, "fitness",condition_selector="condition")
        assert "extra_col" not in result_false.columns