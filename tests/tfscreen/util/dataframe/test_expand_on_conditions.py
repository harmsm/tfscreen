from tfscreen.util import expand_on_conditions

import pandas as pd

def test_expand_on_conditions():
    """Test the expand_on_conditions function."""
    
    # --- Setup a base DataFrame with missing combinations ---
    # Missing: ('wt', 0.2, 2), ('A10V', 0.1, 1)
    data = {
        'genotype':  ['wt',   'wt',   'A10V', 'A10V'],
        'iptg':      [0.1,    0.2,    0.1,    0.2],
        'replicate': [1,      1,      2,      2],
        'theta_est': [10.5,   11.2,   5.3,    5.6],
    }
    df_missing = pd.DataFrame(data)

    # --- Test Case 1: conditions is None ---
    # Should return the original DataFrame unchanged.
    result_none = expand_on_conditions(df_missing.copy(), None)
    pd.testing.assert_frame_equal(result_none, df_missing)

    # --- Test Case 2: conditions is an empty list ---
    # Should also return the original DataFrame unchanged.
    result_empty = expand_on_conditions(df_missing.copy(), [])
    pd.testing.assert_frame_equal(result_empty, df_missing)

    # --- Test Case 3: conditions is a single string ---
    result_single = expand_on_conditions(df_missing.copy(), "iptg")

    # Expected shape: 2 genotypes * 2 iptg conditions = 4 rows
    assert result_single.shape == (4, 4)
    # Original data had 4 non-null values. Expanded has 4. No NaNs expected here.
    assert result_single['theta_est'].notna().sum() == 4
    assert result_single['theta_est'].isna().sum() == 0
    # Check sorting: wt comes before A10V, 0.1 before 0.2
    assert result_single['genotype'].tolist() == ['wt', 'wt', 'A10V', 'A10V']
    assert result_single['iptg'].tolist() == [0.1, 0.2, 0.1, 0.2]


    # --- Test Case 4: conditions is a list of strings (main use case) ---
    result_multi = expand_on_conditions(df_missing.copy(), ["iptg", "replicate"])

    # Expected shape: 2 genotypes * 2 iptg * 2 replicates = 8 rows
    assert result_multi.shape == (8, 4)
    # Original data had 4 non-null values. 8 total rows means 4 NaNs.
    assert result_multi['theta_est'].notna().sum() == 4
    assert result_multi['theta_est'].isna().sum() == 4

    # Check that the previously missing rows now exist with NaN
    missing_row_1 = result_multi[(result_multi['genotype'] == 'wt') & (result_multi['replicate'] == 2)]
    missing_row_2 = result_multi[(result_multi['genotype'] == 'A10V') & (result_multi['replicate'] == 1)]
    assert pd.isna(missing_row_1['theta_est'].iloc[0])
    assert pd.isna(missing_row_2['theta_est'].iloc[0])

    # Check sorting: genotype -> iptg -> replicate
    expected_genotypes = ['wt', 'wt', 'wt', 'wt', 'A10V', 'A10V', 'A10V', 'A10V']
    expected_iptg = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]
    expected_reps = [1, 2, 1, 2, 1, 2, 1, 2]
    assert result_multi['genotype'].tolist() == expected_genotypes
    assert result_multi['iptg'].tolist() == expected_iptg
    assert result_multi['replicate'].tolist() == expected_reps

    # --- Test Case 5: DataFrame with no missing data ---
    # Should just sort the dataframe correctly without adding NaNs.
    data_complete = {
        'genotype':  ['A10V', 'wt'],
        'iptg':      [0.1,    0.1],
        'theta_est': [5.0,    10.0]
    }
    df_complete = pd.DataFrame(data_complete)
    result_complete = expand_on_conditions(df_complete.copy(), "iptg")

    assert result_complete.shape == (2, 3)
    assert result_complete['theta_est'].isna().sum() == 0
    # Check that it sorted 'wt' to be first
    assert result_complete['genotype'].tolist() == ['wt', 'A10V']

