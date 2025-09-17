# tests/test_your_module.py

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Import the function to be tested
from tfscreen.calibration.calibrate import _get_linear_model_df


def test_get_linear_model_df_standard_case():
    """
    Test with a mix of pre/post conditions, background, and duplicates.
    This test is robust to the row order from drop_duplicates().
    """
    # 1. ARRANGE
    data = {
        "pre_condition": ["wt", "wt", "background", "mut1"],
        "condition":     ["wt", "mut1", "mut1", "background"],
        "titrant_name":  ["iptg", "iptg", "iptg", "lactose"],
    }
    df = pd.DataFrame(data)
    
    # 2. ACT
    result_df = _get_linear_model_df(df)

    # 3. ASSERT
    # Separate background and non-background results for easier testing
    bg_df = result_df.loc[result_df.index.get_level_values('condition') == 'background']
    non_bg_df = result_df.loc[result_df.index.get_level_values('condition') != 'background']

    # --- Assertions for Background Conditions ---
    assert not bg_df.empty
    # All b_idx and m_idx for background should be 0
    assert (bg_df['b_idx'] == 0).all()
    assert (bg_df['m_idx'] == 0).all()
    
    # --- Assertions for Non-Background Conditions ---
    num_non_bg = len(non_bg_df)
    assert num_non_bg == 3

    # The b_idx values should be a complete, unique set from 0 to N-1
    expected_b_idx = np.arange(num_non_bg)
    np.testing.assert_array_equal(np.sort(non_bg_df['b_idx'].values), expected_b_idx)
    
    # The m_idx values should be the b_idx values offset by N
    expected_m_idx = expected_b_idx + num_non_bg
    np.testing.assert_array_equal(np.sort(non_bg_df['m_idx'].values), expected_m_idx)

def test_get_linear_model_df_only_background():
    """
    Test the edge case where all conditions are 'background'.
    """
    # 1. ARRANGE
    data = {
        "pre_condition": ["background", "background"],
        "condition":     ["background", "background"],
        "titrant_name":  ["iptg", "lactose"],
    }
    df = pd.DataFrame(data)

    expected_data = {
        ("background", "iptg"):    {"b_idx": 0, "m_idx": 0},
        ("background", "lactose"): {"b_idx": 0, "m_idx": 0},
    }
    expected_df = pd.DataFrame.from_dict(expected_data, orient="index")
    expected_df.index.names = ["condition", "titrant_name"]

    # 2. ACT
    result_df = _get_linear_model_df(df)

    # 3. ASSERT
    assert_frame_equal(
        result_df.sort_index(),
        expected_df.sort_index(),
        check_dtype=False
    )

def test_get_linear_model_df_no_background():
    """
    Test the case with no background conditions.
    """
    # 1. ARRANGE
    data = {"pre_condition": ["wt"], "condition": ["mut1"], "titrant_name": ["iptg"]}
    df = pd.DataFrame(data)
    # Non-background pairs: (wt, iptg), (mut1, iptg) -> N=2
    expected_data = {
        ("wt", "iptg"):   {"b_idx": 0, "m_idx": 2},
        ("mut1", "iptg"): {"b_idx": 1, "m_idx": 3},
    }
    expected_df = pd.DataFrame.from_dict(expected_data, orient="index")
    expected_df.index.names = ["condition", "titrant_name"]

    # 2. ACT
    result_df = _get_linear_model_df(df)

    # 3. ASSERT
    assert_frame_equal(
        result_df.sort_index(),
        expected_df.sort_index(),
        check_dtype=False
    )