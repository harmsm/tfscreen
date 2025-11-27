from tfscreen.util import chunk_by_group

import pytest
import numpy as np


def assert_chunks_equal(result_chunks, expected_chunks):
    """Helper function to compare two lists of numpy arrays for equality."""
    assert len(result_chunks) == len(expected_chunks), "Incorrect number of chunks"
    for result_arr, expected_arr in zip(result_chunks, expected_chunks):
        assert np.array_equal(result_arr, expected_arr), "Chunk contents do not match"

def test_chunk_by_group():
    """
    Tests the chunk_by_group function for correctness, edge cases, and errors.
    """
    # === Test Basic Functionality ===
    # Case where multiple splits are required
    arr = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])  # len 11
    result = chunk_by_group(arr, max_chunk_size=5)
    # Expected chunks: [0,0,0,1,1], [2,2,2,2], [3,3]
    # Corresponding indices: [0,1,2,3,4], [5,6,7,8], [9,10]
    expected = [np.arange(0, 5), np.arange(5, 9), np.arange(9, 11)]
    assert_chunks_equal(result, expected)

    # === Test Edge Cases ===
    # Case where no split is needed (array fits in one chunk)
    arr = np.array([10, 10, 20, 20, 20])
    result = chunk_by_group(arr, max_chunk_size=10)
    expected = [np.arange(0, 5)]
    assert_chunks_equal(result, expected)

    # Case with an empty input array
    arr = np.array([])
    result = chunk_by_group(arr, max_chunk_size=100)
    expected = [np.array([], dtype=int)]
    assert_chunks_equal(result, expected)

    # Case where input is a list instead of a NumPy array
    arr_list = [5, 5, 5, 6, 6]
    result = chunk_by_group(arr_list, max_chunk_size=3)
    # Expected chunks: [5,5,5], [6,6] -> indices [0,1,2], [3,4]
    expected = [np.arange(0, 3), np.arange(3, 5)]
    assert_chunks_equal(result, expected)

    # Corrected: Case where a valid chunk ends exactly at max_chunk_size
    arr = np.array([0, 0, 1, 1, 1])
    # With max_size=3, the input is now valid (largest group is size 3)
    # The first chunk will be [0,0] (size 2), and the next group [1,1,1]
    # would make the chunk size 5, which is > 3. So it splits.
    result = chunk_by_group(arr, max_chunk_size=3)
    # Expected chunks: [0,0], [1,1,1] -> indices [0,1], [2,3,4]
    expected = [np.arange(0, 2), np.arange(2, 5)]
    assert_chunks_equal(result, expected)

    # === Test Error Handling ===
    # Case where max_chunk_size is invalid (zero)
    with pytest.raises(ValueError, match="must be 1 or greater"):
        chunk_by_group(np.array([1, 2, 3]), 0)

    # Case where a single group is larger than max_chunk_size
    with pytest.raises(ValueError, match="A group of identical values is larger"):
        chunk_by_group(np.array([0, 0, 0, 0]), 3)
    
    # Case where max_chunk_size is not a valid integer type
    with pytest.raises(TypeError, match="must be an integer"):
        chunk_by_group(np.array([1, 2, 3]), "not a number")