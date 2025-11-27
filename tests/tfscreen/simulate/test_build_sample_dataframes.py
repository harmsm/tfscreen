
import pytest
import pandas as pd
import pandas.testing as pdt

# Import the function from the specified path
from tfscreen.simulate.build_sample_dataframes import build_sample_dataframes

@pytest.fixture
def sample_condition_blocks():
    """Provides a standard list of condition blocks for testing."""
    return [
        {
            "library": "libA",
            "titrant_name": "iptg",
            "titrant_conc": [0, 10],
            "condition_pre": "preA",
            "t_pre": 30,
            "condition_sel": "selA",
            "t_sel": [60, 120]
        },
        {
            "library": "libB",
            "titrant_name": "none",
            "titrant_conc": [0],
            "condition_pre": "preB",
            "t_pre": 20,
            "condition_sel": "selB",
            "t_sel": [90]
        }
    ]

def test_happy_path(sample_condition_blocks):
    """
    Tests the main success path with multiple condition blocks.
    """
    # --- Run the function ---
    result_df = build_sample_dataframes(sample_condition_blocks)

    # --- Define the expected output ---
    # Block 1 expands to 2 concs * 2 t_sels = 4 rows
    # Block 2 expands to 1 conc * 1 t_sel = 1 row
    # Total = 5 rows
    expected_data = [
        # Replicate, library, pre, sel, titrant, conc, t_sel, t_pre
        [1, "libA", "preA", "selA", "iptg", 0, 60, 30],
        [1, "libA", "preA", "selA", "iptg", 0, 120, 30],
        [1, "libA", "preA", "selA", "iptg", 10, 60, 30],
        [1, "libA", "preA", "selA", "iptg", 10, 120, 30],
        [1, "libB", "preB", "selB", "none", 0, 90, 20]
    ]
    
    columns = [
        "replicate", "library", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "t_sel", "t_pre"
    ]

    expected_df = pd.DataFrame(expected_data, columns=columns)
    
    # The function includes sorting, so we must make sure our expected
    # dataframe matches that sort order.
    sort_columns = [
        "replicate", "library", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "t_sel"
    ]
    expected_df = expected_df.sort_values(sort_columns).reset_index(drop=True)

    # Reorder result columns to match expected for robust comparison
    result_df = result_df[expected_df.columns]

    # --- Assert ---
    pdt.assert_frame_equal(result_df, expected_df)

def test_non_default_replicate(sample_condition_blocks):
    """
    Tests that a non-default replicate number is correctly applied.
    """
    replicate_num = 5
    result_df = build_sample_dataframes(sample_condition_blocks, replicate=replicate_num)

    # Assert that all values in the 'replicate' column are the new number
    assert (result_df["replicate"] == replicate_num).all()
    assert result_df.shape[0] == 5 # Check total rows is still correct

def test_raises_error_on_empty_list():
    """
    Tests that a ValueError is raised for an empty condition_blocks list.
    """
    with pytest.raises(ValueError, match="condition_blocks must be a non-empty list"):
        build_sample_dataframes(condition_blocks=[])

@pytest.mark.parametrize("invalid_input", [
    {"a": 1},  # A dictionary, not a list
    "not_a_list", # A string
    123 # An integer
])
def test_raises_error_on_invalid_container_type(invalid_input):
    """
    Tests that a ValueError is raised if condition_blocks is not a list.
    """
    with pytest.raises(ValueError, match="condition_blocks must be a non-empty list"):
        build_sample_dataframes(condition_blocks=invalid_input)

def test_raises_error_on_invalid_item_type():
    """
    Tests that a ValueError is raised if a list item is not a dictionary.
    """
    invalid_blocks = [
        {"library": "libA", "titrant_conc": [0], "t_sel": [60]},
        "not_a_dictionary"
    ]
    with pytest.raises(ValueError, match="All items in condition_blocks must be dictionaries"):
        build_sample_dataframes(condition_blocks=invalid_blocks)

def test_single_condition_block():
    """
    Tests the function with just a single block in the input list.
    """
    single_block = [{
        "library": "single", "titrant_name": "x", "titrant_conc": [1, 2],
        "condition_pre": "pre", "t_pre": 10, "condition_sel": "sel", "t_sel": [100]
    }]

    result_df = build_sample_dataframes(single_block)

    # Expected output should have 2 * 1 = 2 rows
    assert result_df.shape[0] == 2
    assert result_df["library"].unique() == ["single"]