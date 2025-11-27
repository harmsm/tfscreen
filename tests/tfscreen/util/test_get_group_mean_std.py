# tests/test_util.py

import pytest
import numpy as np

# Import the function to be tested
from tfscreen.util import get_group_mean_std

def test_basic_functionality():
    """
    Tests the primary functionality with a mix of group sizes.
    """
    values = np.array([10, 12,    20, 23, 28,    35])
    groups = np.array([0,  0,     1,  1,  1,     2])

    # Expected means:
    # Group 0: (10 + 12) / 2 = 11.0
    # Group 1: (20 + 23 + 28) / 3 = 23.666...
    # Group 2: 35 / 1 = 35.0
    expected_means = np.array([11.0, 11.0, 23.666667, 23.666667, 23.666667, 35.0])

    # Expected sample standard deviations:
    # Group 0: np.std([10, 12], ddof=1) = 1.414...
    # Group 1: np.std([20, 23, 28], ddof=1) = 4.041...
    # Group 2: std of a single point is 0
    expected_stds = np.array([1.41421356, 1.41421356, 4.04145188, 4.04145188, 4.04145188, 0.0])

    means, stds = get_group_mean_std(values, groups)

    # Check shapes
    assert means.shape == values.shape
    assert stds.shape == values.shape

    # Check values
    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(stds, expected_stds)


def test_input_types_and_conversion():
    """
    Tests that the function handles list inputs and float arrays of whole numbers.
    """
    # Test with Python lists as input
    values_list = [1, 2, 5, 6]
    groups_list = [0, 0, 1, 1]
    means, stds = get_group_mean_std(values_list, groups_list)
    np.testing.assert_allclose(means, [1.5, 1.5, 5.5, 5.5])
    np.testing.assert_allclose(stds, [0.70710678, 0.70710678, 0.70710678, 0.70710678])

    # Test with a float array for groups that can be safely cast
    values = np.array([1, 2, 5, 6])
    groups_float = np.array([0.0, 0.0, 1.0, 1.0])
    means_f, stds_f = get_group_mean_std(values, groups_float)
    np.testing.assert_allclose(means_f, [1.5, 1.5, 5.5, 5.5])


def test_edge_cases():
    """
    Tests behavior with empty inputs, skipped groups, and single-member groups.
    """
    # 1. Empty array inputs
    means, stds = get_group_mean_std([], [])
    assert means.shape == (0,)
    assert stds.shape == (0,)

    # 2. Skipped group numbers (groups are 0 and 2, group 1 is missing)
    values = np.array([1, 2, 10, 12])
    groups = np.array([0, 0, 2, 2])
    means, stds = get_group_mean_std(values, groups)
    expected_means = np.array([1.5, 1.5, 11.0, 11.0])
    expected_stds = np.array([0.70710678, 0.70710678, 1.41421356, 1.41421356])
    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(stds, expected_stds)
    
    # 3. All groups have only one member
    values = np.array([10, 20, 30])
    groups = np.array([0, 1, 2])
    means, stds = get_group_mean_std(values, groups)
    assert np.all(stds == 0)
    assert np.all(means == values)


@pytest.mark.parametrize(
    "values, groups, match_string",
    [
        (
            [1, 2, 3], [0, 0],
            "values and groups must be arrays of the same shape."
        ),
        (
            [1, 2], [0, -1],
            "group numbers must be non-negative."
        ),
        (
            [1, 2], [0.5, 1.0],
            "groups array must contain integer values."
        )
    ]
)
def test_invalid_inputs(values, groups, match_string):
    """
    Tests that the function correctly raises ValueError for invalid inputs.
    """
    with pytest.raises(ValueError, match=match_string):
        get_group_mean_std(values, groups)