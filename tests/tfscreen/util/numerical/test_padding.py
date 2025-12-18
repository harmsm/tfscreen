# tests/util/test_vstack_padded.py

import pytest
import numpy as np
from tfscreen.util.numerical.padding import vstack_padded

def test_vstack_padded_different_widths():
    """Tests the primary functionality with arrays of different widths."""
    arr1 = np.ones((2, 3))
    arr2 = np.zeros((3, 2))
    result = vstack_padded([arr1, arr2], fill_value=-1)
    
    expected = np.array([
        [1., 1., 1.],
        [1., 1., 1.],
        [0., 0., -1.],
        [0., 0., -1.],
        [0., 0., -1.]
    ])
    
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, expected)

def test_vstack_padded_same_widths():
    """Tests the optimization path for arrays of the same width."""
    arr1 = np.ones((2, 3))
    arr2 = np.zeros((3, 3))
    result = vstack_padded([arr1, arr2])
    
    expected = np.vstack([arr1, arr2])
    
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, expected)

def test_vstack_padded_empty_list():
    """Tests that an empty list of arrays returns an empty array."""
    result = vstack_padded([])
    assert result.shape == (0, 0)
    
def test_vstack_padded_with_1d_array():
    """Tests that 1D arrays are correctly handled as single-row 2D arrays."""
    arr1 = np.array([1, 2, 3, 4]) # 1D array
    arr2 = np.zeros((2, 3))
    result = vstack_padded([arr1, arr2], fill_value=0)
    
    expected = np.array([
        [1., 2., 3., 4.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])
    
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, expected)

def test_vstack_padded_different_dtypes():
    """Tests that the output dtype is correctly promoted."""
    arr1 = np.ones((2, 2), dtype=np.int32)
    arr2 = np.zeros((2, 3), dtype=np.float64)
    result = vstack_padded([arr1, arr2])
    
    # The output dtype should be float64 to accommodate both
    assert result.dtype == np.float64
    
    expected = np.array([
        [1., 1., 0.],
        [1., 1., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    np.testing.assert_array_equal(result, expected)

def test_vstack_padded_with_nan_fill():
    """Tests using a non-default fill_value like np.nan."""
    arr1 = np.ones((2, 4))
    arr2 = np.ones((2, 2))
    result = vstack_padded([arr1, arr2], fill_value=np.nan)

    expected = np.array([
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., np.nan, np.nan],
        [1., 1., np.nan, np.nan]
    ])
    
    np.testing.assert_array_equal(result, expected)

def test_vstack_padded_raises_type_error():
    """Tests that a TypeError is raised for invalid input types."""
    arr1 = np.ones((2, 2))
    not_an_array = [1, 2, 3] # A plain list
    
    with pytest.raises(TypeError, match="All elements in the input list must be numpy arrays."):
        vstack_padded([arr1, not_an_array])