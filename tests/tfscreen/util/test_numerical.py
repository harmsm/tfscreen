import pytest
import numpy as np
import pandas as pd

# Assuming the functions to_log and from_log are in a file named `log_conversions.py`
# If they are in the same file, you can remove the import statement.
from tfscreen.util.numerical import (
    to_log,
    from_log
)


# --- Tests for to_log ---

def test_to_log_scalar():
    """Tests to_log with a single scalar value."""
    result = to_log(100)
    assert np.isscalar(result) 
    assert np.isclose(result, np.log(100))


def test_to_log_list():
    """Tests to_log with a list of values."""
    result = to_log([1, 10, 100])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.0, np.log(10), np.log(100)])


def test_to_log_numpy_array():
    """Tests to_log with a NumPy array."""
    v = np.array([10, 20, 30])
    result = to_log(v)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.log(v))


def test_to_log_pandas_series():
    """Tests to_log with a pandas Series."""
    v = pd.Series([10, 20, 30])
    result = to_log(v)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.log(v))


def test_to_log_with_std():
    """Tests to_log with standard deviation propagation."""
    log_v, log_std = to_log(100, v_std=10)
    assert np.isclose(log_v, np.log(100))
    assert np.isclose(log_std, 10 / 100)


def test_to_log_with_var():
    """Tests to_log with variance propagation."""
    log_v, log_var = to_log(100, v_var=25)
    assert np.isclose(log_v, np.log(100))
    assert np.isclose(log_var, 25 / (100**2))


def test_to_log_with_std_and_var():
    """Tests to_log with both std and var propagation."""
    log_v, log_var, log_std = to_log(100, v_var=25, v_std=5)
    assert np.isclose(log_v, np.log(100))
    assert np.isclose(log_var, 25 / (100**2))
    assert np.isclose(log_std, 5 / 100)


def test_to_log_edge_cases():
    """Tests to_log with zero and negative inputs."""
    assert to_log(0) == -np.inf
    assert np.isnan(to_log(-10))

    # Test array with mixed values
    v = np.array([-10, 0, 10])
    result = to_log(v)
    assert np.isnan(result[0])
    assert result[1] == -np.inf
    assert np.isclose(result[2], np.log(10))


def test_to_log_length_mismatch_raises_error():
    """Tests that ValueError is raised for inconsistent lengths."""
    with pytest.raises(ValueError, match="Inconsistent lengths for `v` and `v_var`"):
        to_log([1, 2], v_var=[1])
    with pytest.raises(ValueError, match="Inconsistent lengths for `v` and `v_std`"):
        to_log([1, 2], v_std=[1, 2, 3])


# --- Tests for from_log ---

def test_from_log_scalar():
    """Tests from_log with a single scalar value."""
    result = from_log(np.log(50))
    assert np.isscalar(result) 
    assert np.isclose(result, 50)


def test_from_log_list():
    """Tests from_log with a list of values."""
    v_log = [np.log(10), np.log(20)]
    result = from_log(v_log)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [10, 20])


def test_from_log_numpy_array():
    """Tests from_log with a NumPy array."""
    v_log = np.array([1, 2, 3])
    result = from_log(v_log)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.exp(v_log))


def test_from_log_pandas_series():
    """Tests from_log with a pandas Series."""
    v_log = pd.Series([1, 2, 3])
    result = from_log(v_log)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.exp(v_log.values))


def test_from_log_with_std():
    """Tests from_log with standard deviation propagation."""
    v, v_std = from_log(np.log(100), v_std=0.1)
    assert np.isclose(v, 100)
    assert np.isclose(v_std, 0.1 * 100)


def test_from_log_with_var():
    """Tests from_log with variance propagation."""
    v, v_var = from_log(np.log(50), v_var=0.01)
    assert np.isclose(v, 50)
    assert np.isclose(v_var, 0.01 * (50**2))


def test_from_log_with_std_and_var():
    """Tests from_log with both std and var propagation."""
    v_log = np.log(200)
    v, v_var, v_std = from_log(v_log, v_var=0.02, v_std=0.1)
    assert np.isclose(v, 200)
    assert np.isclose(v_var, 0.02 * (200**2))
    assert np.isclose(v_std, 0.1 * 200)


def test_from_log_length_mismatch_raises_error():
    """Tests that ValueError is raised for inconsistent lengths."""
    with pytest.raises(ValueError, match="Inconsistent lengths for `v` and `v_var`"):
        from_log([1, 2], v_var=[1])
    with pytest.raises(ValueError, match="Inconsistent lengths for `v` and `v_std`"):
        from_log([1, 2, 3], v_std=[1, 2])
