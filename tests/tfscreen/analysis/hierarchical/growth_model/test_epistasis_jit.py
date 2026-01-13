import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass, GrowthData, BindingData
from tfscreen.analysis.hierarchical.growth_model.components.genotype_utils import sample_genotype_parameter

def test_epistasis_jit_stability():
    """
    Verify that epistasis_mode is treated as a static property by JAX,
    allowing for stable JIT compilation across same modes and correct
    re-compilation for different modes.
    """
    
    def create_mock_data(mode):
        num_genotype = 3
        num_mutation = 2
        batch_size = 3
        
        # Consistent shape for map_genotype_to_mutations (padded with 0)
        mapper = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=int)
        
        growth = GrowthData(
            batch_size=batch_size,
            batch_idx=jnp.array([0, 1, 2]),
            scale_vector=jnp.ones(batch_size),
            geno_theta_idx=jnp.array([0, 1, 2]),
            ln_cfu=jnp.zeros((1, 1, 1, 1, 1, 1, batch_size)),
            ln_cfu_std=jnp.ones((1, 1, 1, 1, 1, 1, batch_size)),
            t_pre=jnp.zeros(batch_size),
            t_sel=jnp.zeros(batch_size),
            good_mask=jnp.ones(batch_size, dtype=bool),
            congression_mask=jnp.ones(num_genotype, dtype=bool),
            num_replicate=1, num_time=1, num_condition_pre=1, 
            num_condition_sel=1, num_titrant_name=1, num_titrant_conc=1,
            num_genotype=num_genotype, num_condition=1,
            map_condition_pre=jnp.array([0, 0, 0]),
            map_condition_sel=jnp.array([0, 0, 0]),
            titrant_conc=jnp.array([0.0]),
            log_titrant_conc=jnp.array([0.0]),
            wt_indexes=jnp.array([0]),
            scatter_theta=0,
            epistasis_mode=mode,
            num_mutation=num_mutation,
            map_genotype_to_mutations=mapper,
            genotype_num_mutations=jnp.array([1, 1, 2])
        )
        
        # BindingData not used for this test but needed for DataClass
        binding = BindingData(
            batch_size=1, batch_idx=jnp.array([0]),
            scale_vector=jnp.array([1.0]), geno_theta_idx=jnp.array([0]),
            theta_obs=jnp.array([0.5]), theta_std=jnp.array([0.1]),
            good_mask=jnp.array([True]), num_titrant_name=1,
            num_titrant_conc=1, num_genotype=num_genotype,
            titrant_conc=jnp.array([0.1]), log_titrant_conc=jnp.array([-2.3]),
            scatter_theta=0, epistasis_mode=mode, num_mutation=num_mutation,
            map_genotype_to_mutations=mapper,
            genotype_num_mutations=jnp.array([1, 1, 2])
        )

        return DataClass(
            num_genotype=num_genotype, batch_idx=jnp.array([0, 1, 2]),
            batch_size=batch_size, not_binding_idx=jnp.array([], dtype=int),
            not_binding_batch_size=0, num_binding=0,
            growth=growth, binding=binding
        )

    # Function to JIT: simulates a simplified model sampling
    def model_fn(data):
        def sample_fn(name, size):
            # Deterministic for JIT test
            return jnp.zeros(size)
        
        # Wrap in seed because sample_genotype_parameter uses pyro.sample internally
        return seed(sample_genotype_parameter, 0)("test", data.growth, sample_fn)

    # Wrap in a way we can observe compilation
    compiled_count = 0
    def observable_model_fn(data):
        nonlocal compiled_count
        compiled_count += 1
        return model_fn(data)

    jit_model = jax.jit(observable_model_fn)

    # 1. Compile for "genotype" mode
    data_gen1 = create_mock_data("genotype")
    jit_model(data_gen1)
    print(f"Compilation count after mode='genotype': {compiled_count}")
    
    # 2. Call again with same mode but different data values
    data_gen2 = create_mock_data("genotype")
    new_growth = data_gen2.growth.replace(ln_cfu=jnp.ones_like(data_gen2.growth.ln_cfu))
    data_gen2 = data_gen2.replace(growth=new_growth)
    jit_model(data_gen2)
    print(f"Compilation count after second call (same mode): {compiled_count}")
    
    # 3. Compile for "horseshoe" mode
    data_hs = create_mock_data("horseshoe")
    jit_model(data_hs)
    print(f"Compilation count after mode='horseshoe': {compiled_count}")

    # 4. Compile for "spikeslab" mode
    data_ss = create_mock_data("spikeslab")
    jit_model(data_ss)
    print(f"Compilation count after mode='spikeslab': {compiled_count}")

    # Verify counts
    # Expect: 1 (genotype) + 0 (reuse genotype) + 1 (horseshoe) + 1 (spikeslab) = 3
    if compiled_count == 3:
        print("SUCCESS: JIT stability verified.")
    else:
        print(f"FAILURE: Expected 3 compilations, got {compiled_count}")

if __name__ == "__main__":
    test_epistasis_jit_stability()
