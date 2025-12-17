
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass, GrowthData, BindingData

def test_data_class_jit_stability():
    """
    Test that DataClass can be passed to a JIT-compiled function multiple times
    with distinct but identical array fields without triggering a ValueError.
    This ensures that array fields are correctly treated as dynamic Pytree nodes.
    """
    
    # Helper to create a dummy DataClass
    def create_dummy_data():
        # Create minimal dummy GrowthData
        # We use small arrays for everything
        growth = GrowthData(
            batch_size=10,
            batch_idx=jnp.zeros(10, dtype=int),
            scale_vector=jnp.ones(10),
            geno_theta_idx=jnp.arange(10),
            ln_cfu=jnp.zeros((2, 5, 2, 2, 2, 2, 10)), # match dims roughly
            ln_cfu_std=jnp.ones((2, 5, 2, 2, 2, 2, 10)),
            t_pre=jnp.zeros((2, 5, 2, 2, 2, 2, 10)),
            t_sel=jnp.zeros((2, 5, 2, 2, 2, 2, 10)),
            good_mask=jnp.ones((2, 5, 2, 2, 2, 2, 10), dtype=bool),
            num_replicate=2,
            num_time=5,
            num_condition_pre=2,
            num_condition_sel=2,
            num_titrant_name=2,
            num_titrant_conc=2,
            num_genotype=10,
            num_condition=4,
            map_condition_pre=jnp.zeros((2, 5, 2, 2, 2, 2, 10), dtype=int),
            map_condition_sel=jnp.zeros((2, 5, 2, 2, 2, 2, 10), dtype=int),
            titrant_conc=jnp.zeros(2),
            log_titrant_conc=jnp.zeros(2),
            wt_indexes=jnp.array([0]),
            scatter_theta=1
        )

        binding = BindingData(
            batch_size=5,
            batch_idx=jnp.zeros(5, dtype=int),
            scale_vector=jnp.ones(5),
            geno_theta_idx=jnp.arange(5),
            theta_obs=jnp.zeros((2, 2, 10)),
            theta_std=jnp.ones((2, 2, 10)),
            good_mask=jnp.ones((2, 2, 10), dtype=bool),
            num_titrant_name=2,
            num_titrant_conc=2,
            num_genotype=10,
            titrant_conc=jnp.zeros(2),
            log_titrant_conc=jnp.zeros(2),
            scatter_theta=0
        )

        # The problematic field was not_binding_idx.
        # We create a FRESH array object each time this function is called.
        not_binding_idx = jnp.array([5, 6, 7, 8, 9], dtype=int)

        return DataClass(
            num_genotype=10,
            not_binding_idx=not_binding_idx,
            not_binding_batch_size=5,
            num_binding=5,
            growth=growth,
            binding=binding
        )

    # Define a simple function to JIT
    @jax.jit
    def simple_func(data):
        # Access the field to make sure it's traceable
        return jnp.sum(data.not_binding_idx)

    # Create first instance
    data1 = create_dummy_data()
    
    # First call
    res1 = simple_func(data1)
    assert res1 is not None

    # Create second instance (distinct array objects)
    data2 = create_dummy_data()

    # Verify that not_binding_idx are distinct objects but equal values
    assert data1.not_binding_idx is not data2.not_binding_idx
    assert jnp.array_equal(data1.not_binding_idx, data2.not_binding_idx)

    # Second call - this should NOT raise ValueError
    try:
        res2 = simple_func(data2)
        assert res2 is not None
        assert res1 == res2
    except ValueError as e:
        pytest.fail(f"JAX JIT failed on second call with ValueError: {e}")
    except Exception as e:
        pytest.fail(f"JAX JIT failed on second call with unexpected exception: {e}")

if __name__ == "__main__":
    test_data_class_jit_stability()
    print("Test passed!")
