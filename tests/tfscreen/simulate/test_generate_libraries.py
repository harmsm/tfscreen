import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, call

# Import all functions from the target module
from tfscreen.simulate.generate_libraries import (
    _validate_inputs,
    _find_mut_sites,
    _expand_degen_codon,
    _generate_singles,
    _generate_inter_block,
    _generate_intra_block,
    _build_final_df,
    generate_libraries,
)

# --------------------------- Fixtures ---------------------------

@pytest.fixture
def base_config():
    """Provides a basic, valid configuration for tests."""
    return {
        "aa_sequence": "ASKVGM",
        "mutated_sites": "A1KVG2", # Site S->1, M->2
        "degen_codon": "tgg", # Only encodes 'W'
        "seq_starts_at": 10,
        "lib_keys": ["single-1", "single-2", "double-1-2"],
    }

# ----------------------- Helper Function Tests -----------------------

class TestValidateInputs:
    def test_happy_path(self):
        clean_aa, clean_mut, start, keys = _validate_inputs(" A S K ", " a s k ", "1", [" key1 "])
        assert clean_aa == "ASK"
        assert clean_mut == "ASK"
        assert start == 1
        assert keys == ["key1"]

    def test_mismatched_length_raises_error(self):
        with pytest.raises(ValueError, match="must be strings of the same length"):
            _validate_inputs("ASK", "AS", 1, [])

    @pytest.mark.parametrize("bad_start", ["not_a_number", None])
    def test_bad_seq_starts_at_raises_error(self, bad_start):
        with pytest.raises(ValueError, match="must be an integer"):
            _validate_inputs("A", "A", bad_start, [])

def test_find_mut_sites():
    """Tests the user-provided logic where any difference defines a site."""
    clean_aa = "ASKVGM"
    clean_mut = "A1KVG2" # S->1, M->2
    seq_starts_at = 10
    
    wt_aa, sites = _find_mut_sites(clean_aa, clean_mut, seq_starts_at)

    # Note: residue number advances only for mutated sites in this implementation
    expected_wt_aa = {'1': ['S'], '2': ['M']}
    expected_sites = {'1': [11], '2': [15]} # Residue S is at index 1 (10+1), M is index 2 (11+1)

    assert wt_aa == expected_wt_aa
    assert sites == expected_sites

def test_find_mut_sites_no_diff():
    """Tests case with no differences between sequences."""
    wt_aa, sites = _find_mut_sites("ASK", "ASK", 1)
    assert wt_aa == {}
    assert sites == {}

def test_expand_degen_codon():
    """Tests translation of a degenerate codon to amino acids."""
    
    possible_muts = _expand_degen_codon("nnt")
    assert len(possible_muts) == 16 # 4*4*1
    assert "F" in possible_muts
    assert "R" in possible_muts
    assert "M" not in possible_muts # NNT does not encode Met
    assert "*" not in possible_muts # NNT avoids stop codons

def test_generate_singles():
    wt_aa = {'1': ['S', 'K'], '2': ['M']}
    sites = {'1': [11, 12], '2': [15]}
    possible_muts = ['A', 'G']

    lib_genotypes = _generate_singles(wt_aa, sites, possible_muts)
    
    assert "single-1" in lib_genotypes
    assert "single-2" in lib_genotypes
    assert lib_genotypes["single-1"] == ['S11A', 'S11G', 'K12A', 'K12G']
    assert lib_genotypes["single-2"] == ['M15A', 'M15G']

def test_generate_inter_block():
    lib_genotypes = {
        "single-1": ["S11A", "S11G"],
        "single-2": ["M15C", "M15D"]
    }
    doubles = _generate_inter_block('1', '2', lib_genotypes)
    expected = [
        "S11A/M15C", "S11A/M15D",
        "S11G/M15C", "S11G/M15D"
    ]
    assert sorted(doubles) == sorted(expected)

def test_generate_intra_block():
    wt_aa = {'1': ['S', 'K']}
    sites = {'1': [11, 12]}
    possible_muts = ['A', 'G']

    doubles = _generate_intra_block(wt_aa, sites, '1', possible_muts)
    
    # Expected singles: S11 -> [S11A, S11G], K12 -> [K12A, K12G]
    # Expected product should have 2 * 2 = 4 combinations
    assert len(doubles) == 4
    assert "S11A/K12A" in doubles
    assert "S11G/K12G" in doubles

@patch('tfscreen.simulate.generate_libraries.standardize_genotypes', side_effect=lambda x: x)
@patch('tfscreen.simulate.generate_libraries.set_categorical_genotype', side_effect=lambda x: x)
def test_build_final_df(mock_set_cat, mock_standardize):
    lib_genotypes = {
        "single-1": ["S11A", "S11G", "wt"], # "wt" from S11S, for example
        "single-2": ["M15C", "M15C"] # Test degeneracy
    }
    df = _build_final_df(lib_genotypes)
    
    # Check mocks were called
    assert mock_standardize.call_count == 2
    mock_set_cat.assert_called_once()
    
    # Check DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["library_origin", "genotype", "degeneracy","weight"]
    
    # Check values for single-1
    s1_df = df[df["library_origin"] == "single-1"]
    assert len(s1_df) == 3 # S11A, S11G, wt
    
    # Check values for single-2 (degeneracy)
    s2_df = df[df["library_origin"] == "single-2"]
    assert len(s2_df) == 1
    assert s2_df.iloc[0]["genotype"] == "M15C"
    assert s2_df.iloc[0]["degeneracy"] == 2

# ----------------------- Main Function Tests -----------------------

@patch('tfscreen.simulate.generate_libraries.standardize_genotypes', side_effect=lambda g: g)
@patch('tfscreen.simulate.generate_libraries.set_categorical_genotype', side_effect=lambda df: df)
class TestGenerateLibraries:

    def test_full_run_singles_and_doubles(self, mock_set, mock_std, base_config):
        df = generate_libraries(**base_config)
        
        libs = df["library_origin"].unique()
        assert "single-1" in libs
        assert "single-2" in libs
        assert "double-1-2" in libs
        
        # Check single-1 results (S->W at site 11)
        s1_genos = df[df.library_origin == 'single-1'].genotype.values
        assert "S11W" in s1_genos
        
        # Check double-1-2 results (S11W/M15W)
        d12_genos = df[df.library_origin == 'double-1-2'].genotype.values
        assert "S11W/M15W" in d12_genos

    def test_intra_block_doubles(self, mock_set, mock_std, base_config):
        base_config["mutated_sites"] = "AS1V1M" # Block '1' at S and V
        base_config["lib_keys"] = ["double-1-1"]
        
        df = generate_libraries(**base_config)
        
        assert "double-1-1" in df["library_origin"].unique()
        assert "K12W/G14W" in df.genotype.values
        
    def test_unrequested_singles_are_filtered(self, mock_set, mock_std, base_config):
        """Ensures that singles generated for doubles are not in the final output."""
        base_config["lib_keys"] = ["double-1-2"]
        df = generate_libraries(**base_config)
        
        libs = df["library_origin"].unique()
        assert "single-1" not in libs
        assert "single-2" not in libs
        assert "double-1-2" in libs

    @pytest.mark.parametrize("bad_key, err_msg", [
        ("double-1", "could not parse"),
        ("double-1-", "could not match"),
        ("other-1-2", "could not parse"),
        ("single-3", "should be formatted like 'single-x'"), # Invalid because it wasn't pre-generated
    ])
    def test_bad_lib_key_format_raises_error(self, mock_set, mock_std, bad_key, err_msg, base_config):
        base_config["lib_keys"] = [bad_key]
        with pytest.raises(ValueError, match=err_msg):
            generate_libraries(**base_config)

    @pytest.mark.parametrize("bad_key", ["double-1-3", "double-3-1"])
    def test_lib_key_with_nonexistent_block_raises_error(self, mock_set, mock_std, bad_key, base_config):
        base_config["lib_keys"] = [bad_key]
        with pytest.raises(ValueError, match="could not match mutant identifier"):
            generate_libraries(**base_config)