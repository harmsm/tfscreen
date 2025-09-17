import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from tfscreen.calibration.manual_fit import manual_fit

def test_manual_fit(mocker):
    """
    Tests the full workflow of grouping, fitting (mocked), and aggregating.
    """
    # 1. ARRANGE
    # Create a sample DataFrame with 2 replicates for the same condition
    df = pd.DataFrame({
        "replicate":      ["r1", "r1", "r2", "r2"],
        "pre_condition":  ["wt", "wt", "wt", "wt"],
        "condition":      ["mut1", "mut1", "mut1", "mut1"],
        "titrant_name":   ["iptg", "iptg", "iptg", "iptg"],
        "titrant_conc":   [10.0, 10.0, 10.0, 10.0],
        "time":           [0, 8, 0, 8],
        "cfu_per_mL":     [1e6, 1e7, 1e6, 1.5e7],
        "cfu_per_mL_std": [1e5, 1e6, 1e5, 1.5e6],
    })
    bg_model_param = {"iptg": np.array([0.1, 0.01])}

    # Mock the external dependencies
    mocker.patch('tfscreen.calibration.manual_fit.get_background', return_value=np.array([0.1]))
    # Mock get_time0 to just pass through reshaped data
    mocker.patch('tfscreen.calibration.manual_fit.get_time0', side_effect=lambda t, lc, lcv, *a, **kw: (None, t, lc, lcv))

    # Mock get_growth_rates_wls to return different results for each replicate
    mock_fit_r1 = pd.DataFrame({"k_est": [0.5]})
    mock_fit_r2 = pd.DataFrame({"k_est": [1.5]})
    mocker.patch('tfscreen.calibration.manual_fit.get_growth_rates_wls', side_effect=[
        (mock_fit_r1, None), (mock_fit_r2, None)
    ])

    # 2. ACT
    result_df = manual_fit(df, bg_model_param)
    
    # 3. ASSERT
    # The function should have grouped by replicate, giving two groups.
    # The mock returns k_est = 0.5 for r1 and 1.5 for r2.
    # The final aggregation should calculate the mean and sem of these two values.
    expected_mean = np.mean([0.5, 1.5])  # 1.0
    expected_sem = np.std([0.5, 1.5], ddof=1) / np.sqrt(2) # 0.707... / 1.414... = 0.5
    
    expected_df = pd.DataFrame({
        "pre_condition": ["wt"],
        "condition": ["mut1"],
        "titrant_name": ["iptg"],
        "titrant_conc": [10.0],
        "k_est": [expected_mean],
        "k_std": [expected_sem],
    })
    
    # Use pandas testing utility for robust comparison
    assert_frame_equal(result_df, expected_df)