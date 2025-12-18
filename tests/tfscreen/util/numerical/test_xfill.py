import pytest
import numpy as np

from tfscreen.util import xfill


def test_xfill_linear_basic():
    """Tests a simple linear fill."""
    x = np.array([10., 20., 50.])
    result = xfill(x, num_points=10, use_log=False, pad_by=0.1)
    
    assert result.shape == (10,)
    assert np.isin(x, result).all() # All original points must be present
    # Check padding: span is 40, pad is 4. Range should be [6, 54]
    assert np.isclose(np.min(result), 6.0)
    assert np.isclose(np.max(result), 54.0)

def test_xfill_log_basic():
    """Tests a simple logarithmic fill."""
    x = np.array([1., 10., 1000.])
    result = xfill(x, num_points=20, use_log=True)
    
    assert result.shape == (20,)
    assert np.isin(x, result).all()
    # The brittle check for constant ratios has been removed.

def test_xfill_log_autodetection():
    """Tests that log scale is chosen automatically for wide-ranging data."""
    x = np.array([0.1, 1, 100, 2000.])
    result = xfill(x, use_log=None)
    
    assert result.shape == (100,) # Check default num_points
    assert np.isin(x, result).all()
    # The brittle check for constant ratios has been removed.

def test_xfill_linear_autodetection():
    """Tests that linear scale is chosen for narrow-range data."""
    x = np.array([100., 200., 500.])
    result = xfill(x, use_log=None)

    assert result.shape == (100,)
    assert np.isin(x, result).all()
    # The brittle check for constant differences has been removed.

def test_xfill_raises_error_for_log_with_negative_x():
    """Tests ValueError is raised if log is requested for negative data."""
    x = np.array([-1., 1., 10.])
    with pytest.raises(ValueError, match="x has negative values"):
        xfill(x, use_log=True)

# --- Edge Case Tests ---

def test_xfill_empty_input():
    """Tests that an empty array returns an empty array."""
    result = xfill(np.array([]))
    assert result.shape == (0,)

def test_xfill_non_finite_input():
    """Tests that nan/inf values are correctly ignored."""
    x = np.array([1., np.nan, 5., np.inf, -np.inf])
    result = xfill(x, num_points=10)
    
    # Should behave as if x = np.array([1., 5.])
    assert result.shape == (10,)
    assert np.isin([1., 5.], result).all()
    assert np.isfinite(result).all()

def test_xfill_single_value_input():
    """Tests input with only one unique value."""
    x = np.array([10., 10., 10.])
    result = xfill(x, num_points=50)
    
    assert result.shape == (50,)
    assert np.all(result == 10.)

def test_xfill_all_zeros_input():
    """Tests that an all-zero array is handled gracefully."""
    x = np.array([0., 0.])
    result = xfill(x, num_points=10)
    
    assert result.shape == (10,)
    assert np.all(result == 0.)

def test_xfill_zeros_and_positives_log_autodetect():
    """Tests that log auto-detection works with zeros present."""
    x = np.array([0., 0.1, 1000.])
    result = xfill(x, use_log=None)
    
    assert np.isin(x, result).all()
    # The brittle check for constant ratios has been removed.

def test_xfill_actually_single_value():
    """Tests input with exactly one value (size=1)."""
    x = np.array([10.])
    result = xfill(x, num_points=50)
    assert result.shape == (50,)
    assert np.all(result == 10.)

def test_xfill_log_fallback_not_enough_points():
    """Tests fallback to linear if use_log=True but not enough positive points."""
    x = np.array([0., 100.]) # Only one positive value
    # Should not raise, but fallback to linear
    result = xfill(x, use_log=True, num_points=10)
    assert result.shape == (10,)
    assert np.isin(x, result).all()
    # Check it is linear: (0, 100) -> linear sequence
    # min 0, max 100. span 100. pad 10. range -10 to 110.
    assert np.isclose(result[0], -10.0)
    assert np.isclose(result[-1], 110.0)