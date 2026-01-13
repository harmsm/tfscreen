
import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass, _setup_batching
from unittest.mock import MagicMock, patch

@pytest.fixture
def dummy_data():
    # Helper to create basic growth and binding DFs
    rows = []
    genotypes = ["wt", "A10G", "V20L"]
    for g in genotypes:
        rows.append({
            "replicate": 1,
            "time": 0.0,
            "genotype": g,
            "ln_cfu": 1.0,
            "ln_cfu_std": 0.1,
            "condition_pre": "pre",
            "condition_sel": "sel",
            "t_pre": 1.0,
            "t_sel": 10.0,
            "titrant_name": "iptg",
            "titrant_conc": 0.0
        })
    growth_df = pd.DataFrame(rows)

    rows = []
    for g in genotypes:
        rows.append({
            "genotype": g,
            "titrant_name": "iptg",
            "titrant_conc": 0.0,
            "theta_obs": 0.5,
            "theta_std": 0.05
        })
    binding_df = pd.DataFrame(rows)
    return growth_df, binding_df

def test_spiked_genotypes_masking(dummy_data):
    """Test that spiked genotypes are correctly masked in GrowthData."""
    growth_df, binding_df = dummy_data
    spiked = ["A10G"]
    
    # We need to mock components to avoid full model initialization errors if any
    # but ModelClass.__init__ calls _initialize_data which does most of the heavy lifting.
    # We can use real data for this unit test since it's small.
    
    gm = ModelClass(growth_df, binding_df, spiked_genotypes=spiked)
    
    # Get genotype labels from the growth tensor manager
    genotype_idx = gm.growth_tm.tensor_dim_names.index("genotype")
    genotype_labels = gm.growth_tm.tensor_dim_labels[genotype_idx]
    
    # Check the congression_mask in the data object
    mask = np.array(gm.data.growth.congression_mask)
    for i, label in enumerate(genotype_labels):
        if label == "A10G":
            assert not mask[i], f"Genotype {label} should be masked (False)"
        else:
            assert mask[i], f"Genotype {label} should NOT be masked (True)"

def test_spiked_genotypes_validation(dummy_data):
    """Test that specifying a missing genotype raises ValueError."""
    growth_df, binding_df = dummy_data
    with pytest.raises(ValueError, match="not found in the growth dataset"):
        ModelClass(growth_df, binding_df, spiked_genotypes=["C30W"])

def test_setup_batching_zero_division():
    """Test the fix for ZeroDivisionError in _setup_batching."""
    # This happens when not_binding_batch_size is 0
    # _setup_batching(genotype_labels, batch_size, num_binding_to_keep)
    
    genotypes = ["wt", "mut1"]
    batch_size = 2
    num_binding_to_keep = 2 # All are binding
    
    # This result should not raise ZeroDivisionError anymore
    # The new signature is (growth_genotypes, binding_genotypes, batch_size=None)
    # Passing the same genotypes for both growth and binding
    result = _setup_batching(genotypes, genotypes, batch_size)
    
    assert "scale_vector" in result
    assert np.all(result["scale_vector"] == 1.0)
    assert len(result["scale_vector"]) == 2

def test_spiked_genotypes_single_string(dummy_data):
    """Test that a single string is handled correctly as spiked_genotypes."""
    growth_df, binding_df = dummy_data
    gm = ModelClass(growth_df, binding_df, spiked_genotypes="A10G")
    
    genotype_idx = gm.growth_tm.tensor_dim_names.index("genotype")
    genotype_labels = gm.growth_tm.tensor_dim_labels[genotype_idx]
    
    mask = np.array(gm.data.growth.congression_mask)
    for i, label in enumerate(genotype_labels):
        if label == "A10G":
            assert not mask[i]
        else:
            assert mask[i]
