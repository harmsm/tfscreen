import pandas as pd
import numpy as np

from tfscreen.util import argsort_genotypes


def test_argsort_genotypes():
    """Test the argsort_genotypes function for correctness and robustness."""

    # === Test Case 1: Basic sorting of wt, singles, and doubles ===
    g_basic = ["A15V/T14Y", "wt", "A15W", "T14Y"]
    expected_basic = ["wt", "T14Y", "A15W", "A15V/T14Y"]

    order = argsort_genotypes(g_basic)
    assert np.array_equal(np.array(g_basic)[order], expected_basic)

    # === Test Case 2: Complex sorting of doubles ===
    # Should sort by site1, then mut1, then site2, then mut2
    g_complex = ["A10C/B20D", "A10C/B10E", "A5F/C30G", "A10D/B5H"]
    expected_complex = ["A5F/C30G", "A10C/B10E", "A10C/B20D", "A10D/B5H"]

    order = argsort_genotypes(g_complex)
    assert np.array_equal(np.array(g_complex)[order], expected_complex)

    # === Test Case 3: Case-insensitivity and placement of 'wt' ===
    g_wt_case = ["A1V", "WT", "C3D"]
    expected_wt_case = ["WT", "A1V", "C3D"]

    order = argsort_genotypes(g_wt_case)
    assert np.array_equal(np.array(g_wt_case)[order], expected_wt_case)

    # === Test Case 4: Edge case - empty list ===
    g_empty = []
    expected_empty = []

    order = argsort_genotypes(g_empty)
    assert np.array_equal(np.array(g_empty)[order], expected_empty)
    assert order.dtype == int # Should be an integer array

    # === Test Case 5: Edge case - only 'wt' ===
    g_only_wt = ["wt"]
    expected_only_wt = ["wt"]

    order = argsort_genotypes(g_only_wt)
    assert np.array_equal(np.array(g_only_wt)[order], expected_only_wt)

    # === Test Case 6: Edge case - no 'wt' ===
    g_no_wt = ["C5A", "A1G"]
    expected_no_wt = ["A1G", "C5A"]

    order = argsort_genotypes(g_no_wt)
    assert np.array_equal(np.array(g_no_wt)[order], expected_no_wt)

    # === Test Case 7: Robustness - malformed strings ===
    # Malformed strings should sort based on their original relative order
    # when their sorting keys are identical.
    g_malformed = ["B10C", "A5", "bad-string", "A12F"]
    expected_malformed = ["A5", "bad-string", "B10C", "A12F"]

    order = argsort_genotypes(g_malformed)
    assert np.array_equal(np.array(g_malformed)[order], expected_malformed)

    # === Test Case 8: Robustness - different input types ===
    # Using the same data as the basic test case
    g_list = ["A15V/T14Y", "wt", "A15W", "T14Y"]
    g_numpy = np.array(g_list)
    g_series = pd.Series(g_list)
    expected_types = ["wt", "T14Y", "A15W", "A15V/T14Y"]

    order_list = argsort_genotypes(g_list)
    assert np.array_equal(np.array(g_list)[order_list], expected_types)

    order_numpy = argsort_genotypes(g_numpy)
    assert np.array_equal(g_numpy[order_numpy], expected_types)

    order_series = argsort_genotypes(g_series)
    assert np.array_equal(g_series.iloc[order_series].values, expected_types)

    # === Test Case 9: Triples and variable number of mutations ===
    g_triples = ["A1B/C2D/E3F", "wt", "G4H/I5J", "K6L"]
    expected_triples = ["wt", "K6L", "G4H/I5J", "A1B/C2D/E3F"]

    order = argsort_genotypes(g_triples)
    assert np.array_equal(np.array(g_triples)[order], expected_triples)