import pytest
import jax.numpy as jnp
import numpy as np
import numpyro.handlers
import numpyro.distributions as dist
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass, GrowthData, BindingData
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch
from tfscreen.analysis.hierarchical.growth_model.components import dk_geno_hierarchical, activity_horseshoe
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

def test_batch_scaling_unbiased():
    """
    Test that get_batch correctly applies the mini-batch scaling factors.
    """
    total_genotypes = 100
    batch_size = 10
    num_binding = 5
    
    # Calculate expected scale factors
    # num_not_binding = 100 - 5 = 95
    # not_binding_batch_size = 10 - 5 = 5
    # scale_factor = 95 / 5 = 19.0
    expected_scale = 19.0
    
    # Create scale vector as it would be in ModelClass.data
    full_scale_vector = jnp.ones(total_genotypes)
    not_binding_idx = jnp.arange(num_binding, total_genotypes)
    full_scale_vector = full_scale_vector.at[not_binding_idx].set(expected_scale)
    
    # Mock data objects
    growth = GrowthData(
        batch_size=total_genotypes,
        batch_idx=jnp.arange(total_genotypes),
        scale_vector=full_scale_vector,
        geno_theta_idx=jnp.arange(total_genotypes),
        ln_cfu=jnp.zeros((1, 1, 1, total_genotypes)),
        ln_cfu_std=jnp.ones((1, 1, 1, total_genotypes)),
        t_pre=jnp.zeros((1, 1, 1, total_genotypes)),
        t_sel=jnp.zeros((1, 1, 1, total_genotypes)),
        good_mask=jnp.ones((1, 1, 1, total_genotypes), dtype=bool),
        congression_mask=jnp.ones(total_genotypes, dtype=bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition=1,
        map_condition_pre=jnp.array([0]),
        map_condition_sel=jnp.array([0]),
        titrant_conc=jnp.array([1.0]),
        log_titrant_conc=jnp.array([0.0]),
        wt_indexes=jnp.array([0]),
        scatter_theta=1
    )
    
    full_data = DataClass(
        num_genotype=total_genotypes,
        batch_idx=jnp.arange(total_genotypes),
        batch_size=total_genotypes,
        not_binding_idx=not_binding_idx,
        not_binding_batch_size=95,
        num_binding=num_binding,
        growth=growth,
        binding=None
    )
    
    # Get a batch
    batch_idx = jnp.array([0, 1, 2, 3, 4, 5, 20, 50, 80, 99])
    batch_data = get_batch(full_data, batch_idx)
    
    # Verify scale_vector
    # First 5 are binding (idx 0-4) -> scale 1.0
    assert jnp.all(batch_data.growth.scale_vector[:5] == 1.0)
    # Remaining are non-binding -> scale 19.0
    assert jnp.all(batch_data.growth.scale_vector[5:] == expected_scale)

def test_component_shape_guards():
    """
    Test that component guides handle full-sized parameter substitution correctly.
    """
    total_genotypes = 100
    batch_size = 10
    batch_idx = jnp.arange(batch_size)
    
    growth = GrowthData(
        batch_size=batch_size,
        batch_idx=batch_idx,
        scale_vector=jnp.ones(batch_size),
        geno_theta_idx=jnp.arange(batch_size),
        ln_cfu=jnp.zeros((1, 1, 1, batch_size)),
        ln_cfu_std=jnp.ones((1, 1, 1, batch_size)),
        t_pre=jnp.zeros((1, 1, 1, batch_size)),
        t_sel=jnp.zeros((1, 1, 1, batch_size)),
        good_mask=jnp.ones((1, 1, 1, batch_size), dtype=bool),
        congression_mask=jnp.ones(batch_size, dtype=bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition=1,
        map_condition_pre=jnp.array([0]),
        map_condition_sel=jnp.array([0]),
        titrant_conc=jnp.array([1.0]),
        log_titrant_conc=jnp.array([0.0]),
        wt_indexes=jnp.array([0]),
        scatter_theta=1
    )
    
    priors = dk_geno_hierarchical.get_priors()
    
    # Create full-sized substitution values (100 genotypes)
    substitutions = {
        "dk_geno_offset": jnp.zeros(total_genotypes)
    }
    
    # This should NOT raise a broadcasting error because of the shape guard
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.substitute(substitute_fn=lambda site: substitutions.get(site["name"])):
            dk_geno_hierarchical.guide("dk_geno", growth, priors)

    # Test activity_horseshoe
    priors_hs = activity_horseshoe.get_priors()
    substitutions_hs = {
        "activity_global_scale": 0.1,
        "activity_local_scale": jnp.ones(total_genotypes),
        "activity_offset": jnp.zeros(total_genotypes)
    }
    
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.substitute(substitute_fn=lambda site: substitutions_hs.get(site["name"])):
            activity_horseshoe.guide("activity", growth, priors_hs)

def test_num_genotype_preserved():
    """
    Test that get_batch preserves the total num_genotype while updating batch_size.
    """
    total_genotypes = 100
    batch_size = 10
    
    growth = GrowthData(
        batch_size=total_genotypes,
        batch_idx=jnp.arange(total_genotypes),
        scale_vector=jnp.ones(total_genotypes),
        geno_theta_idx=jnp.arange(total_genotypes),
        ln_cfu=jnp.zeros((1, 1, 1, total_genotypes)),
        ln_cfu_std=jnp.ones((1, 1, 1, total_genotypes)),
        t_pre=jnp.zeros((1, 1, 1, total_genotypes)),
        t_sel=jnp.zeros((1, 1, 1, total_genotypes)),
        good_mask=jnp.ones((1, 1, 1, total_genotypes), dtype=bool),
        congression_mask=jnp.ones(total_genotypes, dtype=bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition=1,
        map_condition_pre=jnp.array([0]),
        map_condition_sel=jnp.array([0]),
        titrant_conc=jnp.array([1.0]),
        log_titrant_conc=jnp.array([0.0]),
        wt_indexes=jnp.array([0]),
        scatter_theta=1
    )
    
    full_data = DataClass(
        num_genotype=total_genotypes,
        batch_idx=jnp.arange(total_genotypes),
        batch_size=total_genotypes,
        not_binding_idx=jnp.arange(total_genotypes),
        not_binding_batch_size=total_genotypes,
        num_binding=0,
        growth=growth,
        binding=None
    )
    
    batch_idx = jnp.arange(batch_size)
    batch_data = get_batch(full_data, batch_idx)
    
    assert batch_data.growth.batch_size == batch_size
    assert batch_data.growth.num_genotype == total_genotypes
    assert batch_data.num_genotype == total_genotypes
