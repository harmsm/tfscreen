from tfscreen.genetics import build_cycles

import pandas as pd
import numpy as np
import pytest

def test_build_cycles():
    """Test the build_cycles function for correctness and edge cases."""

    # --- Test Case 1: Standard case with one valid cycle ---
    g_standard = ["wt", "A10V", "B20C", "A10V/B20C", "D30E"]
    result = build_cycles(g_standard)
    expected = np.array([["wt", "A10V", "B20C", "A10V/B20C"]], dtype=object)
    assert np.array_equal(result, expected)

    # --- Test Case 2: Multiple valid cycles ---
    # The order of cycles should be determined by the sort order of the doubles
    g_multi = ["wt", "A10V", "B20C", "A10V/B20C", "C5F", "C5F/A10V"]
    result = build_cycles(g_multi)
    # A10V/B20C sorts before C5F/A10V
    expected = np.array([["wt", "C5F","A10V", "C5F/A10V"],
                         ["wt", "A10V", "B20C", "A10V/B20C"],
                           ], dtype=object)
    assert np.array_equal(result, expected)

    # --- Test Case 3: No 'wt' present ---
    g_no_wt = ["A10V", "B20C", "A10V/B20C"]
    with pytest.raises(ValueError, match="wt must be in genotypes"):
        build_cycles(g_no_wt)

    # --- Test Case 4: No single mutants ---
    g_no_singles = ["wt", "A10V/B20C", "C5F/D30E"]
    result = build_cycles(g_no_singles)
    assert result.shape == (0, 4)
    assert result.dtype == object

    # --- Test Case 5: No double mutants ---
    g_no_doubles = ["wt", "A10V", "B20C"]
    result = build_cycles(g_no_doubles)
    assert result.shape == (0, 4)
    assert result.dtype == object

    # --- Test Case 6: Double mutant with missing single ---
    # Should find one valid cycle and ignore the one with a missing single
    g_missing_single = ["wt", "A10V", "A10V/B20C", "C5F", "C5F/A10V"]
    result = build_cycles(g_missing_single)
    expected = np.array([["wt", "C5F","A10V", "C5F/A10V"]], dtype=object)
    assert np.array_equal(result, expected)

    # --- Test Case 7: No valid cycles found, even with all components present ---
    g_no_valid_cycles = ["wt", "A10V", "B20C", "C5F/D30E"]
    result = build_cycles(g_no_valid_cycles)
    assert result.shape == (0, 4)
    assert result.dtype == object
    
    # --- Test Case 8: Input as a pandas Series ---
    g_series = pd.Series(["wt", "A10V", "B20C", "A10V/B20C"])
    result = build_cycles(g_series)
    expected = np.array([["wt", "A10V", "B20C", "A10V/B20C"]], dtype=object)
    assert np.array_equal(result, expected)

