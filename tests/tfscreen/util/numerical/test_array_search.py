import numpy as np
import pytest
from numpy.testing import assert_equal

from tfscreen.util.numerical.array_search import strict_array_search
from tfscreen.util.numerical.array_search import fuzzy_array_search


# ----------------------------------------------------------------------------
# test strict_array_search
# ----------------------------------------------------------------------------

def test_match_at_start():
    """Tests for a match at the beginning of the array."""
    search_in = np.array([10, 20, 30, 40, 50])
    search_for = np.array([10, 20, 30])
    expected_index = 0
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_match_at_end():
    """Tests for a match at the end of the array."""
    search_in = np.array([10, 20, 30, 40, 50])
    search_for = np.array([30, 40, 50])
    expected_index = 2
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_match_in_middle():
    """Tests for a match in the middle of the array."""
    search_in = np.array([10, 20, 30, 40, 50])
    search_for = np.array([20, 30])
    expected_index = 1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_multiple_matches():
    """Tests that the index of the *first* match is returned."""
    search_in = np.array([1, 2, 3, 1, 2, 3, 4])
    search_for = np.array([1, 2, 3])
    expected_index = 0
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_no_match():
    """Tests the case where the sequence is not found."""
    search_in = np.array([1, 2, 3, 4, 5])
    search_for = np.array([3, 5])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_partial_match_no_full_match():
    """Tests for a partial match that doesn't become a full match."""
    search_in = np.array([1, 2, 3, 1, 2, 4, 5])
    search_for = np.array([1, 2, 5])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_identical_arrays():
    """Tests when search_in and search_for are identical."""
    search_in = np.array([5, 6, 7])
    search_for = np.array([5, 6, 7])
    expected_index = 0
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_float_arrays():
    """Tests that the function works with floating-point numbers."""
    search_in = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    search_for = np.array([3.3, 4.4])
    expected_index = 2
    assert_equal(strict_array_search(search_in, search_for), expected_index)

# --- Edge Case Tests ---

def test_search_for_is_longer():
    """Tests the edge case where the search sequence is longer than the array."""
    search_in = np.array([1, 2, 3])
    search_for = np.array([1, 2, 3, 4])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_empty_search_for():
    """Tests the edge case of an empty search sequence."""
    search_in = np.array([1, 2, 3])
    search_for = np.array([])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_empty_search_in():
    """Tests the edge case of searching within an empty array."""
    search_in = np.array([])
    search_for = np.array([1, 2])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_both_arrays_empty():
    """Tests the edge case where both arrays are empty."""
    search_in = np.array([])
    search_for = np.array([])
    expected_index = -1
    assert_equal(strict_array_search(search_in, search_for), expected_index)


def test_search_in_non_contiguous_array():
    """
    Tests that the search works correctly on a non-C-contiguous array view.

    This is a regression test for a bug where as_strided failed on array
    views created by reversed slicing (e.g., array[::-1]). The test creates
    such a view and confirms that the search finds the sequence at the
    correct index, which is only possible if the array is made contiguous
    internally by the function.
    """
    # Create a simple, contiguous base array
    base_array = np.arange(25, dtype=np.uint8)

    # Create a non-contiguous VIEW by slicing with a negative step.
    # This is the exact condition that caused the original bug.
    search_in_non_contiguous = base_array[::-1]

    # The non_contiguous array is now: [24, 23, 22, 21, 20, 19, 18, ...]
    search_for = np.array([20, 19, 18], dtype=np.uint8)

    # We expect to find the sequence starting at index 4
    # (index 0=24, 1=23, 2=22, 3=21, 4=20)
    expected_index = 4

    # This call will FAIL on the old code but PASS on the fixed code
    actual_index = strict_array_search(search_in_non_contiguous, search_for)

    assert_equal(actual_index, expected_index)

# ----------------------------------------------------------------------------
# test fuzzy_array_search
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("search_in, search_for, expected_diffs, expected_idx", [
    # Perfect match at the beginning
    ([1, 2, 3, 9, 9], [1, 2, 3], 0, 0),
    # Perfect match in the middle
    ([9, 9, 1, 2, 3, 9], [1, 2, 3], 0, 2),
    # One difference
    ([1, 2, 9, 4, 5], [1, 2, 3], 1, 0),
    # Two differences
    ([9, 8, 3, 4, 5], [1, 2, 3], 2, 0),
    # Total mismatch
    ([9, 8, 7, 6, 5], [1, 2, 3], 3, 0),
    # Tie in score (should return first index)
    ([1, 2, 9, 1, 2], [1, 2], 0, 0),
])
def test_fuzzy_array_search_found(search_in, search_for, expected_diffs, expected_idx):
    """
    Tests cases where a best match is expected to be found.
    """
    arr_in = np.array(search_in, dtype=np.uint8)
    arr_for = np.array(search_for, dtype=np.uint8)
    num_diffs, match_idx = fuzzy_array_search(arr_in, arr_for)
    assert num_diffs == expected_diffs
    assert match_idx == expected_idx

@pytest.mark.parametrize("search_in, search_for", [
    # search_for is longer than search_in
    ([1, 2], [1, 2, 3]),
    # search_in is empty
    ([], [1, 2, 3]),
    # search_for is empty
    ([1, 2, 3], []),
    # both are empty
    ([], []),
])
def test_fuzzy_array_search_guard_clause(search_in, search_for):
    """
    Tests edge cases that should be caught by the initial guard clause.
    """
    arr_in = np.array(search_in, dtype=np.uint8)
    arr_for = np.array(search_for, dtype=np.uint8)

    # The function should indicate a total mismatch at an invalid index
    expected_diffs = arr_for.size
    expected_idx = -1

    num_diffs, match_idx = fuzzy_array_search(arr_in, arr_for)
    assert num_diffs == expected_diffs
    assert match_idx == expected_idx