import pytest
import pandas as pd
import numpy as np

# Import the function to be tested
from tfscreen.util.combine_mutation_effects import combine_mutation_effects

# ----------------------------------------------------------------------------
# Fixtures for the tests
# ----------------------------------------------------------------------------

@pytest.fixture
def genotypes() -> list[str]:
    """A standard list of genotypes for testing."""
    return ["wt", "A1B", "C2D", "A1B/C2D"]

@pytest.fixture
def effects_series() -> pd.Series:
    """A pandas Series mapping single mutations to a scalar effect."""
    return pd.Series({"A1B": -1.5, "C2D": 0.5})

@pytest.fixture
def effects_df() -> pd.DataFrame:
    """A pandas DataFrame mapping single mutations to vector effects."""
    return pd.DataFrame({
        "ddG_spec1": [1.0, -0.5],
        "ddG_spec2": [-2.0, 2.0]
    }, index=["A1B", "C2D"])


# ----------------------------------------------------------------------------
# test combine_mutation_effects
# ----------------------------------------------------------------------------

def test_combine_effects_with_series_sum(genotypes, effects_series):
    """
    Tests the default behavior (summation) with a Series of effects.
    """
    result = combine_mutation_effects(genotypes, effects_series, mut_combine_fcn="sum")

    expected = pd.Series({
        "wt": 0.0,
        "A1B": -1.5,
        "C2D": 0.5,
        "A1B/C2D": -1.0, # -1.5 + 0.5
    })
    
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

def test_combine_effects_with_dataframe_mean(genotypes, effects_df):
    """
    Tests using a string aggregator ("mean") with a DataFrame of effects.
    """
    result = combine_mutation_effects(genotypes, effects_df, mut_combine_fcn="mean")

    # Build the expected DataFrame
    expected = effects_df.copy()
    expected.loc["A1B/C2D"] = effects_df.loc[["A1B", "C2D"]].mean()
    expected.loc["wt"] = 0.0
    
    pd.testing.assert_frame_equal(result.sort_index(), expected.sort_index())

def test_combine_effects_with_callable(genotypes, effects_series):
    """
    Tests using a custom callable function to combine effects.
    """
    # A custom function that sums the effects and adds 10
    custom_fcn = lambda x: x.sum() + 10
    
    result = combine_mutation_effects(genotypes, effects_series, mut_combine_fcn=custom_fcn)
    
    expected = pd.Series({
        "wt": 0.0,
        "A1B": 8.5,
        "C2D": 10.5,
        "A1B/C2D": 9.0, # -1.5 + 0.5 + 10
    })
    
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

@pytest.mark.parametrize("setup_args, error, match", [
    # Case 1: single_mutant_effects is the wrong type
    (lambda g, e_s: (g, [1, 2, 3]), TypeError, "must be a pandas DataFrame or Series"),
    # Case 2: mut_combine_fcn is the wrong type
    (lambda g, e_s: (g, e_s, 123), TypeError, "must be a string or a callable"),
    # Case 3: A mutation is missing from the effects map
    (lambda g, e_s: (g + ["E3F"], e_s), ValueError, "Mutations are missing from the effects map"),
    # Case 4: mut_combine_fcn is an unrecognized string
    (lambda g, e_s: (g, e_s, "bad_function"), ValueError, "is not a recognized aggregator"),
])
def test_combine_effects_raises_errors(genotypes, effects_series, setup_args, error, match):
    """
    Tests that the function raises the correct errors for invalid inputs.
    """
    args = setup_args(genotypes, effects_series)
    with pytest.raises(error, match=match):
        combine_mutation_effects(*args)

@pytest.mark.parametrize("genotype_input, expected_len", [
    ([], 0), # Empty list of genotypes
    (["wt"], 1), # Only wild-type
])
def test_combine_effects_edge_cases(effects_series, genotype_input, expected_len):
    """
    Tests edge cases like empty or wt-only genotype lists.
    """
    result = combine_mutation_effects(genotype_input, effects_series)
    
    assert len(result) == expected_len
    if "wt" in genotype_input:
        assert result.loc["wt"] == 0.0