import pytest
import pandas as pd
import numpy as np

from tfscreen.calibration.calibrate import (
    FitSetup,
    _build_fit_setup,
    _get_linear_model_df
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provides a representative sample DataFrame for testing."""
    return pd.DataFrame({
        "replicate":     ["r1", "r1", "r2", "r2"],
        "pre_condition": ["wt", "mut1", "wt", "background"],
        "condition":     ["wt", "mut1", "mut1", "background"],
        "titrant_name":  ["iptg", "lactose", "iptg", "lactose"],
    })


def test_build_fit_setup(mocker, sample_df):
    """
    Tests the complete construction of the FitSetup object. (Corrected)
    """
    # 1. ARRANGE
    # Mock the _get_linear_model_df dependency.
    # THIS IS THE CORRECTED MOCK, now including ('mut1', 'iptg').
    mock_lm_df = pd.DataFrame({
        "b_idx": [0, 1, 2, 0], 
        "m_idx": [3, 4, 5, 0] # N=3 non-bg params, so slopes are offset by 3
    }, index=pd.MultiIndex.from_tuples([
        ("wt", "iptg"), 
        ("mut1", "lactose"), 
        ("mut1", "iptg"), # The previously missing key
        ("background", "lactose")
    ], names=['condition', 'titrant_name']))
    
    # Note: When using in your package, change '__main__' to your module's path
    mocker.patch('tfscreen.calibration.calibrate._get_linear_model_df', return_value=mock_lm_df)

    bg_model_guesses = [0.1, 0.01]  # 2 background parameters per titrant
    lnA0_guess = 16.0

    # 2. ACT
    result = _build_fit_setup(sample_df, bg_model_guesses, lnA0_guess)

    # 3. ASSERT
    assert isinstance(result, FitSetup)

    # --- Assert index arrays for b/m parameters (UPDATED) ---
    # Max m_idx from mock is now 5.
    # pre_keys -> [('wt','iptg'), ('mut1','lactose'), ('wt','iptg'), ('background','lactose')]
    # b_pre_idx -> [0, 1, 0, 0]
    # m_pre_idx -> [3, 4, 3, 0]
    np.testing.assert_array_equal(result.b_pre_idx, [0, 1, 0, 0])
    np.testing.assert_array_equal(result.m_pre_idx, [3, 4, 3, 0])
    # keys -> [('wt','iptg'), ('mut1','lactose'), ('mut1','iptg'), ('background','lactose')]
    # b_idx -> [0, 1, 2, 0]
    # m_idx -> [3, 4, 5, 0]
    np.testing.assert_array_equal(result.b_idx, [0, 1, 2, 0])
    np.testing.assert_array_equal(result.m_idx, [3, 4, 5, 0])

    # --- Assert A0 parameters (UPDATED) ---
    # A0 indices start after max(m_idx), so at 6. Two replicates (r1, r2).
    A0_start = 6
    assert result.A0_df.shape == (2, 2)
    np.testing.assert_array_equal(result.A0_df["A0_idx"], [A0_start, A0_start + 1])
    # Per-row A0_idx mapping: r1 -> 6, r2 -> 7
    np.testing.assert_array_equal(result.A0_idx, [6, 6, 7, 7])

    # --- Assert background model parameters (UPDATED) ---
    # BG indices start after max(A0_idx), so at 8.
    # 2 titrants * 2 bg_params each = 4 total bg params.
    expected_bg_idx = np.array([[8, 9], [10, 11], [8, 9], [10, 11]])
    np.testing.assert_array_equal(result.bg_param_idx, expected_bg_idx)

    # --- Assert background lookup dictionary (UPDATED) ---
    np.testing.assert_array_equal(result.bg_results_lookup["iptg"], [8, 9])
    np.testing.assert_array_equal(result.bg_results_lookup["lactose"], [10, 11])
    
    # --- Assert initial_guesses vector (UPDATED) ---
    # Total params = max(bg_idx) + 1 = 11 + 1 = 12.
    assert len(result.initial_guesses) == 12
    # Check that A0 guesses are populated correctly
    assert result.initial_guesses[6] == lnA0_guess
    assert result.initial_guesses[7] == lnA0_guess
    assert result.initial_guesses[0] == 0.0 # Check another param is zero

    # --- Assert the not_bg boolean mask (no change) ---
    expected_not_bg = np.array([True, True, True, False])
    np.testing.assert_array_equal(result.not_bg, expected_not_bg)