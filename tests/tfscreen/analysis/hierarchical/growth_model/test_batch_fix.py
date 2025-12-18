import pytest
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch

class MockGrowthData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def replace(self, **updates):
        new_data = MockGrowthData(**self.__dict__)
        new_data.__dict__.update(updates)
        return new_data

class MockDataClass:
    def __init__(self, growth):
        self.growth = growth
    
    def replace(self, **updates):
        new_growth = updates.get("growth", self.growth)
        return MockDataClass(growth=new_growth)

def test_get_batch_metadata_updates():
    """
    Test that get_batch correctly updates geno_theta_idx and num_genotype.
    """
    total_size = 10
    batch_size = 2
    
    # Original data with training batch size metadata
    growth = MockGrowthData(
        batch_size=batch_size,
        num_genotype=batch_size,
        batch_idx=jnp.arange(batch_size),
        scale_vector=jnp.ones((1, total_size)),
        geno_theta_idx=jnp.arange(batch_size),
        ln_cfu=jnp.zeros((1, 1, 1, 1, 1, 1, total_size)),
        ln_cfu_std=jnp.zeros((1, 1, 1, 1, 1, 1, total_size)),
        t_pre=jnp.zeros((1, 1, 1, 1, 1, 1, total_size)),
        t_sel=jnp.zeros((1, 1, 1, 1, 1, 1, total_size)),
        map_condition_pre=jnp.zeros((1, total_size)),
        map_condition_sel=jnp.zeros((1, total_size)),
        good_mask=jnp.ones((1, total_size), dtype=bool),
        congression_mask=jnp.ones((total_size,), dtype=bool)
    )
    
    full_data = MockDataClass(growth=growth)
    
    # NEW INDICES: full dataset (3 genotypes for example)
    new_indices = jnp.array([0, 1, 2, 3, 4], dtype=int)
    new_batch_size = len(new_indices)
    
    # Run get_batch
    batch_data = get_batch(full_data, new_indices)
    
    # VERIFY
    assert batch_data.growth.batch_size == new_batch_size
    assert batch_data.growth.num_genotype == new_batch_size
    assert jnp.array_equal(batch_data.growth.batch_idx, new_indices)
    assert jnp.array_equal(batch_data.growth.geno_theta_idx, jnp.arange(new_batch_size))
    
    # Verify shape of sliced data (last dim should be new_batch_size)
    assert batch_data.growth.ln_cfu.shape[-1] == new_batch_size
    assert batch_data.growth.congression_mask.shape[-1] == new_batch_size
