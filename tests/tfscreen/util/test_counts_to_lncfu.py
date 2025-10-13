import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Assume the functions to be tested are in a file located at
# `tfscreen/processing.py` for the purpose of patching dependencies.
from tfscreen.process_raw.counts_to_lncfu import (
    _filter_low_observation_genotypes,
    _impute_missing_genotypes,
    _calculate_frequencies,
    _calculate_concentrations_and_variance,
    counts_to_lncfu
)

# ---- Test Fixtures ----

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture for sample metadata DataFrame."""
    data = {
        'sample': ['s1', 's2', 's3', 's4'],
        'library': ['libA', 'libA', 'libB', 'libB'],
        'cfu_per_mL': [1e8, 1.2e8, 2e7, 2.5e7],
        'cfu_per_mL_std': [1e7, 1.0488088481701516e7, 2e6, 2.2360679774997894e6],
        'extra_col': ['cond1', 'cond2', 'cond1', 'cond2']
    }
    return pd.DataFrame(data).set_index('sample')

@pytest.fixture
def counts_df() -> pd.DataFrame:
    """Fixture for counts DataFrame."""
    data = {
        'sample': ['s1', 's1', 's2', 's3', 's3', 's4'],
        'genotype': ['G1', 'G2', 'G1', 'G2', 'G3', 'G3'],
        'counts': [100, 5, 50, 200, 2, 300]
    }
    # Genotype totals per library:
    # libA: G1 -> 150, G2 -> 5
    # libB: G2 -> 200, G3 -> 302
    return pd.DataFrame(data)

# ---- Tests for Private Functions ----

def test_filter_low_observation_genotypes(counts_df, sample_df):
    """
    Tests that genotypes with counts below the threshold within a library
    are correctly removed.
    """
    df = pd.merge(counts_df, sample_df, left_on='sample', right_index=True)
    # G2 in libA has 5 counts, which is < 10. It should be removed.
    filtered_df = _filter_low_observation_genotypes(df, min_genotype_obs=10)

    # Assert G2 was removed from libA but kept in libB
    assert 'G2' not in filtered_df[filtered_df['library'] == 'libA']['genotype'].unique()
    assert 'G2' in filtered_df[filtered_df['library'] == 'libB']['genotype'].unique()

    # Assert other genotypes were kept
    assert 'G1' in filtered_df[filtered_df['library'] == 'libA']['genotype'].unique()
    assert 'G3' in filtered_df[filtered_df['library'] == 'libB']['genotype'].unique()
    
    # Original df had 6 rows, one was removed (s1, G2)
    assert len(filtered_df) == 5

def test_filter_low_observation_genotypes_empty_input():
    """Tests that an empty DataFrame input results in an empty DataFrame output."""
    empty_df = pd.DataFrame(columns=['library', 'genotype', 'counts'])
    result = _filter_low_observation_genotypes(empty_df, 10)
    assert result.empty
    assert_frame_equal(result, empty_df)

def test_impute_missing_genotypes(sample_df):
    """
    Tests that missing genotypes are correctly added to samples within a library
    with a count of 0, and that existing metadata is preserved.
    """
    # This df simulates data after filtering. libB has genotypes G2 and G3.
    # s3 saw G2, s4 saw G3. We expect imputation for the missing pairs.
    filtered_data = {
        'sample':   ['s1', 's2', 's3',   's4'],
        'library':  ['libA', 'libA', 'libB',   'libB'],
        'genotype': ['G1',   'G1',   'G2',   'G3'],
        'counts':   [100,    50,     200,    300],
        'cfu_per_mL': [1e8, 1.2e8, 2e7, 2.5e7],
        'cfu_per_mL_std': [1e8, 1.2e8, 2e7, 2.5e7],
        'extra_col': ['cond1', 'cond2', 'cond1', 'cond2']
    }
    filtered_df = pd.DataFrame(filtered_data)
    
    imputed_df = _impute_missing_genotypes(filtered_df, sample_df)
    
    # Check that libB now has all genotype combinations for each sample
    libB_df = imputed_df[imputed_df['library'] == 'libB']
    assert set(libB_df[libB_df['sample'] == 's3']['genotype']) == {'G2', 'G3'}
    assert set(libB_df[libB_df['sample'] == 's4']['genotype']) == {'G2', 'G3'}

    # Check the imputed row for s3 (genotype G3)
    s3_g3_row = libB_df[(libB_df['sample'] == 's3') & (libB_df['genotype'] == 'G3')]
    assert s3_g3_row['counts'].iloc[0] == 0
    assert s3_g3_row['cfu_per_mL'].iloc[0] == 2e7  # Metadata preserved
    assert s3_g3_row['extra_col'].iloc[0] == 'cond1' # Extra metadata preserved

    # Check that libA, which needed no imputation, is correct
    libA_df = imputed_df[imputed_df['library'] == 'libA']
    assert len(libA_df) == 2
    
    # Final imputed df should have 2 (libA) + 4 (libB) = 6 rows
    assert len(imputed_df) == 6

def test_impute_missing_genotypes_empty_input(sample_df):
    """Tests that an empty DataFrame input results in an empty DataFrame output."""
    empty_df = pd.DataFrame(columns=['library', 'sample', 'genotype', 'counts'])
    result = _impute_missing_genotypes(empty_df, sample_df)
    assert result.empty

def test_calculate_frequencies():
    """
    Tests calculation of adjusted counts and frequencies. Ensures frequencies
    are calculated on a per-sample basis.
    """
    df = pd.DataFrame({
        'sample': ['s1', 's1', 's2'],
        'counts': [99, 1, 50]
    })
    result = _calculate_frequencies(df.copy(), pseudocount=1)

    # Check adjusted counts
    expected_adj = pd.Series([100, 2, 51], name='adjusted_counts')
    pd.testing.assert_series_equal(result['adjusted_counts'], expected_adj, check_names=False)
    
    # Check frequencies for s1 (total adjusted counts = 100 + 2 = 102)
    s1_freqs = result[result['sample'] == 's1']['frequency']
    np.testing.assert_allclose(s1_freqs, [100/102, 2/102])
    
    # Check sum of frequencies per sample is 1.0
    assert np.allclose(result.groupby('sample')['frequency'].sum(), 1.0)

def test_calculate_concentrations_and_variance():
    """
    Tests calculation of genotype CFU/mL and the propagation of variance.
    """
    df = pd.DataFrame({
        'sample': ['s1'],
        'frequency': [0.9],
        'cfu_per_mL': [1e8],         # Original sample-level data
        'cfu_per_mL_std': [1e7],   # Original sample-level data
        'adjusted_counts': [91]
    })
    # Simulate a second genotype in the sample for total counts calculation
    df_with_total = pd.concat([df, pd.DataFrame({
        'sample': ['s1'], 'adjusted_counts': [11]
    })], ignore_index=True)

    result = _calculate_concentrations_and_variance(df_with_total)
    
    # --- Manually calculate expected values ---
    freq, sample_cfu, sample_var_cfu = 0.9, 1e8, 1e14
    total_adj_counts = 91 + 11

    # Expected genotype concentration
    expected_genotype_cfu = freq * sample_cfu

    # Expected genotype variance
    var_freq = freq * (1 - freq) / total_adj_counts
    rel_var_freq = var_freq / (freq**2)
    rel_var_sample_cfu = sample_var_cfu / (sample_cfu**2)
    expected_genotype_var = (expected_genotype_cfu**2) * (rel_var_freq + rel_var_sample_cfu)

    # --- Assertions (checking the first row of the result) ---
    # FIX: Check the new 'cfu' and 'cfu_var' columns for genotype-specific values
    assert np.isclose(result['cfu'].iloc[0], expected_genotype_cfu)
    assert np.isclose(result['cfu_var'].iloc[0], expected_genotype_var)

    # FIX: Check that the RENAMED columns contain the ORIGINAL sample data
    assert 'ln_cfu' in result.columns
    assert 'ln_cfu_var' in result.columns
    assert np.isclose(result['sample_cfu_per_mL'].iloc[0], sample_cfu)


def test_calculate_concentrations_zero_cfu():
    """
    Tests that cases with zero resulting CFU/mL are handled correctly,
    producing NaN for log-transformed values.
    """
    df = pd.DataFrame({
        'sample': ['s1'],
        'frequency': [0.0],
        'cfu_per_mL': [1e8],
        'cfu_per_mL_std': [1e7],
        'adjusted_counts': [100]
    })
    result = _calculate_concentrations_and_variance(df.copy())
    
    # FIX: Check the new 'cfu' and 'cfu_var' columns
    assert result['cfu'].iloc[0] == 0
    assert result['cfu_var'].iloc[0] == 0
    assert np.isnan(result['ln_cfu'].iloc[0])
    assert np.isnan(result['ln_cfu_var'].iloc[0])

# ---- Tests for Public API Function ----

def test_counts_to_lncfu_full_pipeline(sample_df, counts_df, mocker):
    """
    An integration test for the main function, covering filtering, imputation,
    calculation, sorting, and data type conversion.
    """
    # Mock dependencies imported by the module under test
    mocker.patch('tfscreen.util.counts_to_lncfu.read_dataframe', side_effect=lambda df, **kwargs: df.copy())
    mocker.patch(
        'tfscreen.util.counts_to_lncfu.argsort_genotypes', 
        side_effect=lambda g: np.argsort(g)  # Simple alphabetical sort for testing
    )

    result = counts_to_lncfu(sample_df, counts_df, min_genotype_obs=10, pseudocount=1)

    # Expected shape: G2 from libA is filtered. G2 and G3 are cross-imputed in libB.
    # libA: s1-G1, s2-G1 (2 rows)
    # libB: s3-G2, s3-G3, s4-G2, s4-G3 (4 rows) -> Total 6 rows
    assert len(result) == 6
    
    # Check that extra columns from sample_df are preserved
    assert 'extra_col' in result.columns

    # --- FIX: Check the correctly imputed value. ---
    # The original test checked (s3, G3), which was never removed.
    # The actual imputed row is (s4, G2), because s4 is in libB
    # but had no observations for G2.
    s4_g2_row = result[(result['sample'] == 's4') & (result['genotype'] == 'G2')]
    assert not s4_g2_row.empty
    assert s4_g2_row['counts'].iloc[0] == 0
    assert s4_g2_row['adjusted_counts'].iloc[0] == 1 # Check pseudocount
    
    # Check frequency calculation on a sample with observed and preserved values (s3)
    s3_rows = result[result['sample'] == 's3']
    
    # With a pseudocount of 1, the adjusted counts are 201 and 3.
    # The total adjusted count for sample s3 is 201 + 3 = 204.
    g2_freq = s3_rows[s3_rows['genotype'] == 'G2']['frequency'].iloc[0]
    g3_freq = s3_rows[s3_rows['genotype'] == 'G3']['frequency'].iloc[0]
    
    assert np.isclose(g2_freq, 201 / 204)
    assert np.isclose(g3_freq, 3 / 204)

    # Check sorting (genotype, library, sample)
    expected_genotypes = ['G1', 'G1', 'G2', 'G2', 'G3', 'G3']
    assert result['genotype'].tolist() == expected_genotypes
    # Check categorical type and order
    assert isinstance(result["genotype"].dtype, pd.CategoricalDtype)
    assert result['genotype'].cat.categories.tolist() == ['G1', 'G2', 'G3']

def test_counts_to_lncfu_all_filtered(sample_df, counts_df, mocker):
    """
    Tests that if all genotypes are filtered out, an empty DataFrame is returned.
    """
    mocker.patch('tfscreen.util.read_dataframe', side_effect=lambda df, **kwargs: df.copy())

    # Set min_obs so high that all genotypes are removed
    result = counts_to_lncfu(sample_df, counts_df, min_genotype_obs=1000)
    assert result.empty
