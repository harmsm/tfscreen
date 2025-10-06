import pytest
import pandas as pd
import numpy as np

# Import the functions to be tested
from tfscreen.process_raw.mut_quality_control import (
    _get_expected_geno,
    mut_quality_control
)

# ----------------------------------------------------------------------------
# Fixtures for these tests
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_config() -> dict:
    """Provides a mock library configuration dictionary."""
    return {
        "aa_sequence": "MKT",
        "mutated_sites": {1: "NNK", 2: "NNK"},
        "degen_codon": "NNK",
        "seq_starts_at": 1,
        "library_mixture": {"libA": 1.0}
    }

@pytest.fixture
def trans_df() -> pd.DataFrame:
    """
    Provides a mock translated dataframe with wt, single, double, and
    triple mutants. Also includes an 'unexpected' single mutant.
    """
    return pd.DataFrame({
        "genotype": ["wt", "A1B", "A1B/C2D", "X3Y", "E4F/G5H/I6J"],
        "counts":   [100,  50,    20,        10,    5]
    })


# ----------------------------------------------------------------------------
# test _get_expected_geno
# ----------------------------------------------------------------------------

def test_get_expected_geno(mocker, mock_config):
    """
    Tests that _get_expected_geno correctly calls its dependencies and
    returns a unique array of genotypes.
    """
    # ARRANGE: Mock the two external, trusted functions
    mocker.patch(
        "tfscreen.process_raw.mut_quality_control.read_yaml",
        return_value=mock_config
    )
    mock_return_df = pd.DataFrame({"genotype": ["wt", "A1B", "C2D", "A1B"]})
    mock_generate = mocker.patch(
        "tfscreen.process_raw.mut_quality_control.generate_libraries",
        return_value=mock_return_df
    )
    
    # ACT: Call the function
    result = _get_expected_geno(mock_config)
    
    # ASSERT
    mock_generate.assert_called_once()
    # The result should be the unique genotypes from the mocked return
    expected = np.array(["wt", "A1B", "C2D"])
    np.testing.assert_array_equal(np.sort(result), np.sort(expected))

# ----------------------------------------------------------------------------
# test mut_quality
# ----------------------------------------------------------------------------

def test_mut_quality_simple_filter(trans_df):
    """
    Tests the default filtering mode, which only uses max_allowed_muts.
    """
    # ACT: Filter for a max of 2 mutations
    filtered_df, funnel_df = mut_quality_control(trans_df, max_allowed_muts=2)

    # ASSERT: Check the filtered data
    assert len(filtered_df) == 4 # wt, A1B, A1B/C2D, X3Y
    assert "E4F/G5H/I6J" not in filtered_df["genotype"].values
    
    # ASSERT: Check the funnel dataframe
    assert funnel_df.shape[0] == 2
    assert funnel_df.loc[0, "step"] == "input"
    assert funnel_df.loc[0, "counts"] == 185 # 100+50+20+10+5
    assert funnel_df.loc[1, "step"] == "few_enough"
    assert funnel_df.loc[1, "counts"] == 180 # 185 - 5

def test_mut_quality_expected_filter(mocker, trans_df, mock_config):
    """
    Tests the advanced filtering mode using a lib_config to generate an
    expected set of genotypes.
    """
    # ARRANGE: Mock the helper to return a specific set of expected genotypes
    mock_expected = np.array(["wt", "A1B", "A1B/C2D"])
    mocker.patch(
        "tfscreen.process_raw.mut_quality_control._get_expected_geno",
        return_value=mock_expected
    )

    # ACT: Run the filter with the config
    filtered_df, funnel_df = mut_quality_control(trans_df, lib_config=mock_config)

    # ASSERT: Check the filtered data
    # Should contain only genotypes present in both trans_df and mock_expected
    assert len(filtered_df) == 3
    assert "X3Y" not in filtered_df["genotype"].values # Not in expected
    
    # ASSERT: Check the funnel dataframe
    assert funnel_df.shape[0] == 3
    assert funnel_df.loc[2, "step"] == "in_expected"
    assert funnel_df.loc[2, "counts"] == 170 # 100 (wt) + 50 (A1B) + 20 (A1B/C2D)

def test_mut_quality_with_spiked(mocker, trans_df, mock_config):
    """
    Tests that spiked_expected genotypes are correctly added to the filter.
    """
    # ARRANGE: Mock the helper and define a spiked-in control
    mock_expected = np.array(["wt", "A1B"])
    mocker.patch(
        "tfscreen.process_raw.mut_quality_control._get_expected_geno",
        return_value=mock_expected
    )
    spiked = ["X3Y"] # This is in trans_df but not the mock_expected set

    # ACT: Run with lib_config and the spiked_expected list
    filtered_df, funnel_df = mut_quality_control(trans_df, 
                                         lib_config=mock_config,
                                         spiked_expected=spiked)
    
    # ASSERT: Check that the spiked genotype was kept
    assert len(filtered_df) == 3
    assert "X3Y" in filtered_df["genotype"].values
    assert funnel_df.loc[2, "counts"] == 160 # 100 (wt) + 50 (A1B) + 10 (X3Y)