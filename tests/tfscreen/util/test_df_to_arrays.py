from tfscreen.util.df_to_arrays import (
    _count_df_to_arrays,
    _get_ln_cfu,
    df_to_arrays,
)

import pytest
import pandas as pd
import numpy as np


# Mock the external dependency for testing df_to_arrays
def mock_read_dataframe(df, **kwargs):
    return df


@pytest.fixture
def base_df():
    """A pytest fixture for a base 'happy path' DataFrame."""
    return pd.DataFrame({
        "genotype": ["A", "A", "A", "B", "B", "B"],
        "sample":   ["s1", "s1", "s1", "s2", "s2", "s2"],
        "time":     [0, 10, 20, 0, 12, 24],
        "counts":   [10, 20, 30, 40, 50, 60],
        "total_counts_at_time": [100, 200, 300, 400, 500, 600],
        "total_cfu_mL_at_time": [1e6, 2e6, 3e6, 4e6, 5e6, 6e6],
    })

def test_count_df_to_arrays_basic(base_df):
    """Test standard, complete data pivoting."""
    (times, seq_counts, total_counts, total_cfu,
     genotypes, samples) = _count_df_to_arrays(base_df)

    # Check shapes
    assert times.shape == (2, 3)
    assert genotypes.shape == (2,)

    # Check content and order
    assert np.array_equal(genotypes, ["A", "B"])
    assert np.array_equal(samples, ["s1", "s2"])
    expected_times = np.array([[0, 10, 20], [0, 12, 24]])
    assert np.array_equal(times, expected_times)
    expected_seq_counts = np.array([[10, 20, 30], [40, 50, 60]])
    assert np.array_equal(seq_counts, expected_seq_counts)

def test_count_df_to_arrays_missing_data(base_df):
    """Test that missing time points are filled with NaN."""
    # Drop the middle time point for genotype B (original times 0, 12, 24)
    df_missing = base_df.drop(4) 
    
    (times, seq_counts, _, _, _, _) = _count_df_to_arrays(df_missing)

    # The shape is still (2, 3) because genotype 'A' has 3 time points
    assert times.shape == (2, 3)

    # For genotype B (index 1), the ranks are now 0 and 1. The position
    # for rank 2 should be NaN.
    assert np.isnan(times[1, 2])
    assert np.isnan(seq_counts[1, 2])
    
    # The value at rank 1 should be the time 24.0, not NaN.
    assert not np.isnan(times[1, 1])

def test_count_df_to_arrays_unordered_input(base_df):
    """Test that original (genotype, sample) order is preserved."""
    df_shuffled = base_df.iloc[[3, 0, 4, 1, 5, 2]].reset_index(drop=True)
    # The first unique (g,s) pair is ('B','s2'), second is ('A','s1')
    (_, _, _, _, genotypes, samples) = _count_df_to_arrays(df_shuffled)
    assert np.array_equal(genotypes, ["B", "A"])
    assert np.array_equal(samples, ["s2", "s1"])

def test_count_df_to_arrays_missing_columns():
    """Test that a ValueError is raised for missing required columns."""
    df = pd.DataFrame({"genotype": [1], "sample": [2]})
    with pytest.raises(ValueError, match="Missing columns"):
        _count_df_to_arrays(df)


# === Tests for _get_ln_cfu ===

def test_get_ln_cfu_basic_calculation():
    """Test the core math of the CFU calculation."""
    seq_counts = np.array([[99]])
    total_counts = np.array([[999]])
    total_cfu_ml = np.array([[1e4]])
    
    cfu, cfu_var, ln_cfu, ln_cfu_var = _get_ln_cfu(
        seq_counts, total_counts, total_cfu_ml, pseudocount=1
    )
    
    # Expected values
    # f = (99+1)/(999+1) = 100/1000 = 0.1
    # f_var = 0.1*(1-0.1)/1000 = 0.00009
    # cfu = 0.1 * 1e4 = 1000
    # cfu_var = 0.00009 * (1e4)^2 = 9000
    # ln_cfu = ln(1000) = ~6.9077
    # ln_cfu_var = 9000 / 1000^2 = 0.009

    assert np.isclose(cfu[0,0], 1000)
    assert np.isclose(cfu_var[0,0], 9000)
    assert np.isclose(ln_cfu[0,0], 6.907755)
    assert np.isclose(ln_cfu_var[0,0], 0.009)

def test_get_ln_cfu_zero_counts():
    """Test that pseudocount prevents division by zero."""
    seq_counts = np.array([[0]])
    total_counts = np.array([[100]])
    total_cfu_ml = np.array([[1e4]])
    
    _, _, ln_cfu, _ = _get_ln_cfu(
        seq_counts, total_counts, total_cfu_ml, pseudocount=1
    )
    # f = (0+1)/(100+1) = 1/101. CFU > 0, so ln(CFU) should be finite.
    assert np.isfinite(ln_cfu[0,0])


# === Tests for df_to_arrays ===

def test_df_to_arrays_integration(base_df, monkeypatch):
    """Test the main wrapper function end-to-end."""
    
    # Mock the read_dataframe utility to simply return the df
    # Assuming the module is named 'your_module' for the test
    monkeypatch.setattr("tfscreen.util.read_dataframe", mock_read_dataframe)
    
    # CORRECTED: Added a dummy column to prevent IndexError in read_dataframe
    sample_df = pd.DataFrame({"sample": [1, 2]}, index=["s1", "s2"]) 

    out = df_to_arrays(base_df, sample_df, pseudocount=1)

    # Check for correct keys and array shapes
    assert "genotypes" in out
    assert "cfu" in out
    assert "ln_cfu_var" in out
    assert out["genotypes"].shape == (2,)
    assert out["times"].shape == (2, 3)
    assert out["cfu"].shape == (2, 3)
    
    # Spot check a value to confirm calculation pipeline
    # For ('A','s1') at time 0: counts=10, total_counts=100, cfu_ml=1e6
    # With pseudocount=1 and n=2 rows:
    # adj_seq = 11, adj_total = 100 + 2*1 = 102
    # f = 11/102 ~= 0.1078
    # cfu = 0.1078 * 1e6 ~= 107843
    assert np.isclose(out["cfu"][0, 0], 107843.137, rtol=1e-3)