import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch

# --- Mock Data Structures ---

# We need to simulate the nested DataClass structure: DataClass -> GrowthData
# Using namedtuples to mimic the flax dataclasses behavior (replace method)

class MockGrowthData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def replace(self, **updates):
        # Create a new instance with updated fields
        new_data = MockGrowthData(**self.__dict__)
        new_data.__dict__.update(updates)
        return new_data

class MockDataClass:
    def __init__(self, growth):
        self.growth = growth
    
    def replace(self, **updates):
        # Only 'growth' is expected to be updated in this context
        new_growth = updates.get("growth", self.growth)
        return MockDataClass(growth=new_growth)

@pytest.fixture
def full_data_setup():
    """
    Creates a 'full' dataset with known values for testing slicing.
    Dimensions: 
      - Genotypes (Total): 10
      - Other dims (Rep, Time, etc.): 1
    """
    total_size = 10
    
    # Create arrays with distinct values for verification
    # Shape: (1, 1, 1, 1, 1, 1, total_size) for most data tensors
    # We will use simple 1D arrays for the mock to verify the slicing logic on the *last* dimension,
    # assuming the real data follows the ellipsis (...) indexing pattern.
    
    scale_vector = jnp.arange(total_size, dtype=float)
    ln_cfu = jnp.arange(total_size, dtype=float) * 10.0
    ln_cfu_std = jnp.arange(total_size, dtype=float) * 0.1
    t_pre = jnp.arange(total_size, dtype=float) + 100.0
    t_sel = jnp.arange(total_size, dtype=float) + 200.0
    
    map_condition_pre = jnp.arange(total_size, dtype=int)
    map_condition_sel = jnp.arange(total_size, dtype=int)
    good_mask = jnp.ones(total_size, dtype=bool)
    
    # Initial batch metadata
    batch_idx = jnp.arange(total_size, dtype=int)
    
    growth = MockGrowthData(
        batch_size=total_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        ln_cfu=ln_cfu,
        ln_cfu_std=ln_cfu_std,
        t_pre=t_pre,
        t_sel=t_sel,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        good_mask=good_mask
    )
    
    return MockDataClass(growth=growth)

# --- Test Cases ---

def test_get_batch_slicing(full_data_setup):
    """
    Tests that get_batch correctly slices data based on the index array.
    """
    full_data = full_data_setup
    
    # INDICES TO SELECT: [2, 5, 8]
    indices = jnp.array([2, 5, 8], dtype=int)
    
    # Run MUT
    batch_data = get_batch(full_data, indices)
    
    # --- 1. Check Metadata Updates ---
    assert batch_data.growth.batch_size == 3
    assert jnp.array_equal(batch_data.growth.batch_idx, indices)
    
    # --- 2. Check Data Slicing ---
    
    # Scale Vector
    expected_scale = full_data.growth.scale_vector[indices]
    assert jnp.array_equal(batch_data.growth.scale_vector, expected_scale)
    
    # ln_cfu (Values: 20.0, 50.0, 80.0)
    expected_ln_cfu = jnp.array([20.0, 50.0, 80.0], dtype=float)
    assert jnp.allclose(batch_data.growth.ln_cfu, expected_ln_cfu)
    
    # t_pre (Values: 102.0, 105.0, 108.0)
    expected_t_pre = jnp.array([102.0, 105.0, 108.0], dtype=float)
    assert jnp.allclose(batch_data.growth.t_pre, expected_t_pre)
    
    # Maps
    assert jnp.array_equal(batch_data.growth.map_condition_pre, indices) # Mapped 1-to-1 in setup

def test_get_batch_ordering(full_data_setup):
    """
    Tests that the returned batch respects the *order* of the provided indices,
    even if they are not sorted.
    """
    full_data = full_data_setup
    
    # INDICES: [8, 0, 5] (Unsorted)
    indices = jnp.array([8, 0, 5], dtype=int)
    
    batch_data = get_batch(full_data, indices)
    
    # Check ln_cfu order: Should be [80.0, 0.0, 50.0]
    expected_ln_cfu = jnp.array([80.0, 0.0, 50.0], dtype=float)
    
    assert jnp.allclose(batch_data.growth.ln_cfu, expected_ln_cfu)
    assert jnp.array_equal(batch_data.growth.batch_idx, indices)

def test_get_batch_full_multidimensional_support():
    """
    Tests get_batch with actual multi-dimensional arrays to ensure the 
    ellipsis (...) slicing works as intended.
    """
    # Shape: (2, 2, 5) -> (Rep, Time, Genotype)
    # We want to slice the last dimension (Genotype)
    
    data_shape = (2, 2, 5)
    full_array = jnp.arange(20).reshape(data_shape) # 0..19
    
    # indices to select from last dim: [1, 3]