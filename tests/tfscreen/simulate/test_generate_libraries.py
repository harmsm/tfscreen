import pytest
import pandas as pd
from collections import Counter

# Import the functions to be tested
from tfscreen.simulate.generate_libraries import (
    _get_aa_counts_from_codon,
    _parse_site_markers,
    _generate_single_outcomes_library,
    _combine_outcomes,
    generate_libraries
)

# --- Fixtures for Test Data ---

@pytest.fixture
def base_config():
    """Provides a standard configuration dictionary for testing."""
    return {
        'aa_sequence': "ASKV",
        'mutated_sites': "A1K2", 
        'seq_starts_at': 10,
        'internal_doubles': False,
        'degen_codon': 'NNK' # G,T -> K
    }

@pytest.fixture
def simple_aa_counts():
    """A simplified Counter for testing downstream functions."""
    # Real NNK is more complex, this is just for predictable test outcomes
    return Counter({'A': 2, 'C': 1, 'G': 1}) # Total of 4 outcomes


# --- Tests for Helper Functions (Unchanged) ---

@pytest.mark.parametrize("codon, expected", [
    ("ATG", Counter({'M': 1})),
    ("GCN", Counter({'A': 4})), # N -> A,C,G,T
    ("TGY", Counter({'C': 2})), # Y -> C,T
    ("TAN", Counter({'*':2,'Y': 2})),
])
def test_get_aa_counts_from_codon(codon, expected):
    """
    Tests the _get_aa_counts_from_codon function with various inputs.
    """
    result = _get_aa_counts_from_codon(codon)
    assert result == expected

def test_parse_site_markers(base_config):
    """
    Tests the _parse_site_markers function for correct site identification.
    """
    expected_sites = {
        '1': [{'res_num': 11, 'wt_aa': 'S', 'index': 1}],
        '2': [{'res_num': 13, 'wt_aa': 'V', 'index': 3}]
    }
    result = _parse_site_markers(base_config["aa_sequence"],
                            base_config["mutated_sites"],
                            base_config["seq_starts_at"])
    assert dict(result) == expected_sites

def test_generate_single_outcomes_library(simple_aa_counts):
    """
    Tests _generate_single_outcomes_library for correct DataFrame creation.
    """
    sites = [{'res_num': 11, 'wt_aa': 'S', 'index': 1}]
    df = _generate_single_outcomes_library(sites, simple_aa_counts)

    assert df.shape == (3, 4)
    assert 'genotype' in df.columns
    assert 'is_wt' in df.columns
    assert df['is_wt'].sum() == 0
    assert "S11A" in df['genotype'].values

def test_combine_outcomes():
    """
    Tests the core logic of the _combine_outcomes function.
    """
    # Create two simple single-outcome DataFrames
    df1 = pd.DataFrame([
        {'genotype': 'A10G', 'count': 2, 'res_num': 10, 'is_wt': False},
        {'genotype': 'A10A', 'count': 1, 'res_num': 10, 'is_wt': True},
    ])
    df2 = pd.DataFrame([
        {'genotype': 'C20T', 'count': 3, 'res_num': 20, 'is_wt': False},
        {'genotype': 'C20C', 'count': 2, 'res_num': 20, 'is_wt': True},
    ])

    result = _combine_outcomes(df1, df2)
    
    # 2 rows * 2 rows = 4 rows
    assert len(result) == 4
    
    # Test a mutant/mutant combo
    mut_mut = result[result['num_muts'] == 2].iloc[0]
    assert mut_mut['genotype'] == 'A10G/C20T'
    assert mut_mut['count'] == 2 * 3
    
    # Test a mutant/wt combo
    mut_wt = result[result['num_muts'] == 1].iloc[0]
    assert mut_wt['count'] == 2 * 2

    # Test a wt/wt combo
    wt_wt = result[result['num_muts'] == 0].iloc[0]
    assert wt_wt['count'] == 1 * 2

# --- Integration Tests for the Main `generate_libraries` Function (Updated) ---

def test_generate_libraries_cross_library_path(base_config):
    """
    Tests the main `generate_libraries` function for the cross-library case.
    """
    base_config['degen_codon'] = 'GCT' # Always Alanine ('A')
    
    # UPDATED: Unpack the config dictionary into keyword arguments
    df = generate_libraries(**base_config)

    expected_origins = {'single-1', 'single-2', 'double-1-2'}
    assert set(df['library_origin'].unique()) == expected_origins

    wt_single1 = df[(df['library_origin'] == 'single-1') & (df['genotype'] == 'wt')]
    wt_double = df[(df['library_origin'] == 'double-1-2') & (df['genotype'] == 'wt')]
    assert len(wt_single1) == 0
    assert len(wt_double) == 0

    s11a = df[df['genotype'] == 'S11A']
    assert len(s11a) == 1
    assert s11a.iloc[0]['count'] == 1

    double_mut = df[df['genotype'] == 'S11A/V13A']
    assert len(double_mut) == 1

def test_generate_libraries_internal_doubles_path():
    """
    Tests the main `generate_libraries` function for the internal doubles case.
    """
    config = {
        'aa_sequence': "ASKV",
        'mutated_sites': "A1K1",
        'seq_starts_at': 10,
        'internal_doubles': True,
        'degen_codon': 'GCN' # Alanine
    }
    
    # UPDATED: Unpack the config dictionary into keyword arguments
    df = generate_libraries(**config)

    # Check for correct library origins
    expected_origins = {'single-1', 'internal-double-1'}
    assert set(df['library_origin'].unique()) == expected_origins

    # There should be no wt outcomes since neither S nor V is A.
    wt_single = df[(df['library_origin'] == 'single-1') & (df['genotype'] == 'wt')]
    assert len(wt_single) == 0
    wt_double = df[(df['library_origin'] == 'internal-double-1') & (df['genotype'] == 'wt')]
    assert len(wt_double) == 0

    # There should be one double mutant.
    double_mut = df[df['genotype'] == 'S11A/V13A']
    assert len(double_mut) == 1

def test_generate_libraries_no_internal_doubles_possible():
    """
    Tests the edge case where internal doubles are requested but impossible.
    """
    config = {
        'aa_sequence': "ASKV",
        'mutated_sites': "A1KV",
        'seq_starts_at': 10,
        'internal_doubles': True,
        'degen_codon': 'GCN'
    }

    # UPDATED: Unpack the config dictionary into keyword arguments
    df = generate_libraries(**config)
    
    # Only the single-1 library should be present.
    # No 'internal-double-1' because there's only one '1' site.
    assert set(df['library_origin'].unique()) == {'single-1'}