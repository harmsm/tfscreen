import pytest
import numpy as np

from tfscreen.util.broadcast_args import broadcast_args

def test_broadcast_args_standard_case():
    """Tests broadcasting a scalar and a string to match an array's length."""
    x = np.array([10, 20, 30])
    y = 5
    z = "a"

    x_out, y_out, z_out = broadcast_args(x, y, z)

    np.testing.assert_array_equal(x_out, np.array([10, 20, 30]))
    np.testing.assert_array_equal(y_out, np.array([5, 5, 5]))
    np.testing.assert_array_equal(z_out, np.array(["a", "a", "a"]))

def test_broadcast_args_already_matching_length():
    """Tests that arrays of the same length are returned unchanged."""
    x = [1, 2]
    y = np.array([3, 4])

    x_out, y_out = broadcast_args(x, y)

    np.testing.assert_array_equal(x_out, np.array([1, 2]))
    np.testing.assert_array_equal(y_out, np.array([3, 4]))

def test_broadcast_args_all_scalars():
    """Tests that multiple scalar inputs result in single-element arrays."""
    x, y, z = broadcast_args(10, "b", 3.14)

    np.testing.assert_array_equal(x, np.array([10]))
    np.testing.assert_array_equal(y, np.array(["b"]))
    np.testing.assert_array_equal(z, np.array([3.14]))

def test_broadcast_args_single_array_input():
    """Tests that a single array input is returned as a list of one array."""
    result = broadcast_args([1, 2, 3])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))

def test_broadcast_args_no_args():
    """Tests that calling with no arguments returns an empty list."""
    result = broadcast_args()
    assert result == []

def test_broadcast_args_raises_error_on_incompatible_length():
    """Verifies a ValueError is raised for mismatched array lengths > 1."""
    x = [1, 2]
    y = [10, 20, 30]

    with pytest.raises(ValueError, match="incompatible length 2"):
        broadcast_args(x, y)

def test_broadcast_args_raises_error_with_scalar_and_incompatible_array():
    """Verifies ValueError with a mix of scalar and incompatible arrays."""
    x = 5
    y = [1, 2]
    z = [10, 20, 30]

    with pytest.raises(ValueError, match="incompatible length 2"):
        broadcast_args(x, y, z)

def test_broadcast_args_handles_none():
    """Tests that None is treated as a scalar."""
    x_out, y_out = broadcast_args(None, [1, 2])
    np.testing.assert_array_equal(x_out, np.array([None, None]))
    np.testing.assert_array_equal(y_out, np.array([1, 2]))