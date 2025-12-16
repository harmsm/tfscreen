
import numpy as np
import pytest
from tfscreen.plot.helper import get_ax_limits, clean_arrays, subsample_index

def test_get_ax_limits_basic():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])
    
    # Basic usage
    min_val, max_val = get_ax_limits(x, y, pad_by=0.0, percentile=0.0)
    assert min_val == 0
    assert max_val == 10

def test_get_ax_limits_padding():
    x = np.array([0, 10])
    min_val, max_val = get_ax_limits(x, pad_by=0.1, percentile=0.0)
    # span is 10. pad by 0.1 * 10 = 1.
    # min should be -1, max should be 11
    assert np.isclose(min_val, -1.0)
    assert np.isclose(max_val, 11.0)

def test_get_ax_limits_center_on_zero():
    x = np.array([-2, 5])
    min_val, max_val = get_ax_limits(x, center_on_zero=True, pad_by=0.0, percentile=0.0)
    # max abs is 5. min should be -5, max 5.
    assert np.isclose(min_val, -5.0)
    assert np.isclose(max_val, 5.0)

def test_get_ax_limits_percentile():
    # 0 to 100. 
    x = np.linspace(0, 100, 101)
    # percentile 0.1 means cutting off top and bottom 10%? 
    # Actually code is: ax_min, ax_max = np.nanquantile(all_values, [percentile, 1-percentile])
    # if percentile is 0.1, we get 10th and 90th percentile.
    min_val, max_val = get_ax_limits(x, percentile=0.1, pad_by=0.0)
    assert np.isclose(min_val, 10.0)
    assert np.isclose(max_val, 90.0)

def test_clean_arrays():
    x = np.array([1, 2, np.nan, 4])
    y = np.array([1, np.nan, 3, 4])
    z = np.array([1, 2, 3, 4])
    
    cx, cy, cz = clean_arrays(x, y, z)
    
    expected_x = np.array([1, 4])
    expected_y = np.array([1, 4])
    expected_z = np.array([1, 4])
    
    np.testing.assert_array_equal(cx, expected_x)
    np.testing.assert_array_equal(cy, expected_y)
    np.testing.assert_array_equal(cz, expected_z)

def test_subsample_index():
    arr = np.zeros(100)
    
    # Subsample < len
    idx = subsample_index(arr, subsample=10)
    assert len(idx) == 10
    assert len(np.unique(idx)) == 10
    assert np.max(idx) < 100
    
    # Subsample > len
    idx = subsample_index(arr, subsample=200)
    assert len(idx) == 100
    
    # Subsample None
    idx = subsample_index(arr, subsample=None)
    assert len(idx) == 100
