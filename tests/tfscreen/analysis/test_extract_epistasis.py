from tfscreen.analysis import extract_epistasis

import pandas as pd
import numpy as np

import pytest


def test_extract_epistasis():
    """Test the extract_epistasis function for correctness and edge cases."""
    # --- Setup: A base DataFrame for most tests ---
    base_data = {
        'genotype':  ['wt', 'wt', 'M1A', 'M1A', 'V2C', 'M1A/V2C'],
        'iptg':      [0.1,  0.2,   0.1,   0.2,   0.2,   0.2],
        'theta_est': [10.0, 12.0,  5.0,   6.0,   9.0,   7.0],
        'theta_std': [1.0,  1.2,   0.5,   0.6,   0.9,   0.7],
        'extra_col': ['a',  'b',   'c',   'd',   'e',   'f']
    }
    df_base = pd.DataFrame(base_data)

    # --- Test 1: Additive scale, single condition ---
    result = extract_epistasis(df_base.copy(), "theta_est", "theta_std",
                               conditions="iptg", scale="add")
    assert result.shape == (2, 18) # 2 conditions * 1 cycle = 2 rows

    # Check iptg=0.2 (valid calculation)
    row_02 = result[result.iptg == 0.2].iloc[0]
    assert row_02["ep_obs"] == pytest.approx(4.0)
    assert row_02["ep_std"] == pytest.approx(np.sqrt(3.1))
    assert row_02["m1_obs"] == 6.0

    # Check iptg=0.1 (calculation with NaN from expansion)
    row_01 = result[result.iptg == 0.1].iloc[0]
    assert pd.isna(row_01["ep_obs"])
    assert pd.isna(row_01["ep_std"])
    assert row_01["wt_obs"] == 10.0

    # --- Test 2: Multiplicative scale, drop_extra_columns=False ---
    # This also tests that the extra column is NOT present, even with the flag.
    # (This confirms current behavior but may indicate a flaw in the function).
    result = extract_epistasis(df_base.copy(), "theta_est", "theta_std",
                               conditions="iptg", scale="mult",
                               drop_extra_columns=False)
    assert "extra_col" not in result.columns
    row_02 = result[result.iptg == 0.2].iloc[0]
    assert row_02["ep_obs"] == pytest.approx((7/9) / (6/12))
    assert row_02["ep_std"] == pytest.approx(abs((7/9)/(6/12)) * np.sqrt(0.04))

    # --- Test 3: Multiplicative scale with zero observable ---
    df_zero = df_base.copy()
    df_zero.loc[1, "theta_est"] = 0 # Set wt obs for iptg=0.2 to zero
    with pytest.warns():
        result = extract_epistasis(df_zero, "theta_est", "theta_std",
                                conditions="iptg", scale="mult")
    row_02 = result[result.iptg == 0.2].iloc[0]
    assert row_02["ep_obs"] == pytest.approx(0.0)
    assert pd.isna(row_02["ep_std"])

    # --- Test 4: No conditions specified ---
    df_no_cond = df_base[df_base.iptg == 0.2].drop(columns=['iptg','extra_col'])
    result = extract_epistasis(df_no_cond, "theta_est", "theta_std")
    assert result.shape == (1, 17) # 1 cycle, no condition cols
    assert result["ep_obs"][0] == pytest.approx(4.0)

    # --- Test 5: No valid cycles in data ---
    df_no_cycles = df_base[df_base["genotype"].isin(["wt", "M1A"])]
    result = extract_epistasis(df_no_cycles, "theta_est", "theta_std")
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    # --- Test 6: Error handling for invalid scale ---
    with pytest.raises(ValueError, match="scale should be 'add' or 'mult'"):
        extract_epistasis(df_base, "theta_est", "theta_std", scale="bad_scale")

    # --- Test 7: Error handling for obs/std in conditions ---
    with pytest.raises(ValueError, match="Observable/std columns cannot"):
        extract_epistasis(df_base, "theta_est", "theta_std",
                          conditions=["iptg", "theta_est"])
