
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tfscreen.tfmodel.data_class import DataClass, GrowthData, BindingData

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
            num_condition_rep=4,
            map_condition_pre=jnp.zeros((2, 5, 2, 2, 2, 2, 10), dtype=int),
            map_condition_sel=jnp.zeros((2, 5, 2, 2, 2, 2, 10), dtype=int),
            titrant_conc=jnp.zeros(2),
            log_titrant_conc=jnp.zeros(2),
            wt_indexes=jnp.array([0]),
            scatter_theta=1,
            ln_cfu0_spiked_mask=jnp.zeros(10, dtype=bool),
            ln_cfu0_wt_mask=jnp.zeros(10, dtype=bool)
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
            batch_idx=jnp.arange(10),
            batch_size=10,
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

def _static_growth_data(mut_idx, features):
    """Small GrowthData with fresh static mutation/structure arrays."""
    shape = (1, 1, 1, 1, 1, 1, 3)
    return GrowthData(
        batch_size=3,
        batch_idx=jnp.arange(3),
        scale_vector=jnp.ones(3),
        geno_theta_idx=jnp.arange(3),
        ln_cfu=jnp.zeros(shape),
        ln_cfu_std=jnp.ones(shape),
        t_pre=jnp.zeros(shape),
        t_sel=jnp.zeros(shape),
        good_mask=jnp.ones(shape, dtype=bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=3,
        num_condition_rep=1,
        map_condition_pre=jnp.zeros(shape, dtype=int),
        map_condition_sel=jnp.zeros(shape, dtype=int),
        titrant_conc=jnp.zeros(1),
        log_titrant_conc=jnp.zeros(1),
        wt_indexes=jnp.array([0]),
        scatter_theta=1,
        ln_cfu0_spiked_mask=jnp.zeros(3, dtype=bool),
        ln_cfu0_wt_mask=jnp.zeros(3, dtype=bool),
        num_mutation=2,
        mut_nnz_mut_idx=np.array(mut_idx, dtype=np.int32),
        mut_nnz_geno_idx=np.array([1, 2], dtype=np.int32),
        num_struct=1,
        struct_names=("H",),
        struct_features=np.array(features, dtype=np.float32),
        struct_n_chains=np.array([2], dtype=np.int32),
    )


def test_static_arrays_survive_a_second_jit_call():
    """
    Two models in one process whose data carry static numpy arrays of the
    same shapes (a second fit): jax compares the cached treedef's static
    fields with ==, which raised "arrays cannot be passed as metadata
    fields" before those fields were wrapped in StaticArray.
    """
    traces = []

    @jax.jit
    def f(data):
        traces.append(1)
        idx = jnp.array(data.mut_nnz_mut_idx)
        n_chains = tuple(int(c) for c in data.struct_n_chains)
        return (data.ln_cfu_std.sum() + idx.sum()
                + jnp.array(data.struct_features).sum() + n_chains[0])

    # Equal contents, new array objects: same compiled function.
    assert f(_static_growth_data([0, 1], [[[1.0]]])) == 3 + 1 + 1 + 2
    assert f(_static_growth_data([0, 1], [[[1.0]]])) == 3 + 1 + 1 + 2
    assert len(traces) == 1

    # Different static contents: retraced with the new values.
    assert f(_static_growth_data([1, 1], [[[5.0]]])) == 3 + 2 + 5 + 2
    assert len(traces) == 2


def test_static_arrays_in_binding_and_sim_data():
    from tfscreen.simulate.sim_data_class import SimData
    from tfscreen.tfmodel.data_class import StaticArray

    binding = BindingData(
        batch_size=2, batch_idx=jnp.arange(2), scale_vector=jnp.ones(2),
        geno_theta_idx=jnp.arange(2), theta_obs=jnp.zeros((1, 1, 2)),
        theta_std=jnp.ones((1, 1, 2)), good_mask=jnp.ones((1, 1, 2), dtype=bool),
        num_titrant_name=1, num_titrant_conc=1, num_genotype=2,
        titrant_conc=jnp.zeros(1), log_titrant_conc=jnp.zeros(1),
        scatter_theta=0, pair_nnz_pair_idx=np.array([0]),
        struct_contact_distances=np.array([[4.0]]))
    assert isinstance(binding.pair_nnz_pair_idx, StaticArray)
    assert isinstance(binding.struct_contact_distances, StaticArray)
    assert binding.mut_nnz_mut_idx is None

    sim = SimData(
        num_titrant_name=1, num_titrant_conc=1, num_genotype=2,
        batch_size=2, scatter_theta=0, titrant_conc=jnp.zeros(1),
        log_titrant_conc=jnp.zeros(1), batch_idx=jnp.arange(2),
        geno_theta_idx=jnp.arange(2), mut_nnz_geno_idx=np.array([1]))
    assert isinstance(sim.mut_nnz_geno_idx, StaticArray)

    @jax.jit
    def g(d):
        return jnp.array(d.pair_nnz_pair_idx).sum() + d.theta_std.sum()

    assert g(binding) == g(binding.replace(
        pair_nnz_pair_idx=np.array([0]))) == 2


if __name__ == "__main__":
    test_data_class_jit_stability()
    print("Test passed!")
