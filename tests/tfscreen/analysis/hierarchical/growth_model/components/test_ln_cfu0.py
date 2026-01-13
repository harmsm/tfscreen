import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.ln_cfu0 import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_replicate",
    "num_condition_pre",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "map_ln_cfu0",
    "epistasis_mode"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 2 replicates
    - 2 pre-selection conditions
    - 4 genotypes
    - Batch size 4 (1-to-1 mapping to genotypes for simplicity in this test)
    """
    num_replicate = 2
    num_condition_pre = 2
    num_genotype = 4
    batch_size = 4
    
    # 1-to-1 batch mapping for simplicity
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)
    
    scale_vector = jnp.ones(batch_size, dtype=float)
    
    # Legacy field
    map_ln_cfu0 = jnp.arange(batch_size, dtype=jnp.int32) 
    
    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        map_ln_cfu0=map_ln_cfu0,
        epistasis_mode="genotype"
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "ln_cfu0_hyper_loc_loc" in params
    assert params["ln_cfu0_hyper_loc_loc"] == -2.5

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.ln_cfu0_hyper_loc_loc == -2.5

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_ln_cfu0"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check hyperprior guesses
    assert f"{name}_hyper_loc" in guesses
    assert guesses[f"{name}_hyper_loc"] == -2.5
    
    # Check offset guess
    # Expect shape: (num_replicate, num_condition_pre, num_genotype)
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_replicate, 
                      mock_data.num_condition_pre, 
                      mock_data.num_genotype)
    
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical ln_cfu0.
    """
    name = "test_ln_cfu0"
    priors = get_priors()
    
    # 1. Get base guesses (Full genotype shape)
    base_guesses = get_guesses(name, mock_data)
    
    # 2. Prepare batch guesses for substitute
    # define_model samples offsets with shape (R, C, Batch)
    # get_guesses provided shape (R, C, Genotype)
    # We must slice the last dimension using batch_idx
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_offset"]
    batch_offsets = full_offsets[..., mock_data.batch_idx]
    batch_guesses[f"{name}_offset"] = batch_offsets
    
    # Substitute
    substituted_model = substitute(define_model, data=batch_guesses)
    
    # --- 3. Execute Model ---
    final_ln_cfu0 = substituted_model(name=name, 
                                      data=mock_data, 
                                      priors=priors)

    # --- 4. Trace Execution ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 5. Check Deterministic Site ---
    assert name in model_trace
    ln_cfu0_site = model_trace[name]["value"]
    
    # Expected site shape: (R, C_pre, Batch)
    expected_site_shape = (mock_data.num_replicate, 
                           mock_data.num_condition_pre, 
                           mock_data.batch_size)
    assert ln_cfu0_site.shape == expected_site_shape
    
    # --- 6. Check Values ---
    # With 0 offsets, value should be hyper_loc
    hyper_loc = base_guesses[f"{name}_hyper_loc"]
    assert jnp.allclose(ln_cfu0_site, hyper_loc)
    
    # --- 7. Check Final Expanded Shape ---
    # Code expands: ln_cfu0_per_rep_cond_geno[:,None,:,None,None,None,:]
    # Input: (R, C, Batch)
    # Output: (R, 1, C, 1, 1, 1, Batch)
    expected_expanded_shape = (mock_data.num_replicate,
                               1,
                               mock_data.num_condition_pre,
                               1,
                               1,
                               1,
                               mock_data.batch_size)
    
    assert final_ln_cfu0.shape == expected_expanded_shape
    assert jnp.allclose(final_ln_cfu0, hyper_loc)

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_ln_cfu0_guide"
    priors = get_priors()

    # Seed the guide execution
    with seed(rng_seed=0):
        # Trace the guide
        guide_trace = trace(guide).get_trace(
            name=name,
            data=mock_data,
            priors=priors
        )
        
        # Run guide
        final_ln_cfu0 = guide(name=name,
                              data=mock_data,
                              priors=priors)

    # --- 1. Check Parameter Sites ---
    # Offset locs/scales should cover ALL genotypes
    # Shape: (R, C, Genotype)
    assert f"{name}_offset_locs" in guide_trace
    offset_locs = guide_trace[f"{name}_offset_locs"]["value"]
    
    expected_param_shape = (mock_data.num_replicate, 
                            mock_data.num_condition_pre, 
                            mock_data.num_genotype)
    assert offset_locs.shape == expected_param_shape
    
    # --- 2. Check Sample Sites ---
    # Sampled offsets should match the BATCH size
    # Shape: (R, C, Batch)
    assert f"{name}_offset" in guide_trace
    sampled_offsets = guide_trace[f"{name}_offset"]["value"]
    
    expected_sample_shape = (mock_data.num_replicate, 
                             mock_data.num_condition_pre, 
                             mock_data.batch_size)
    assert sampled_offsets.shape == expected_sample_shape

    # --- 3. Check Return Shape ---
    # (R, 1, C, 1, 1, 1, Batch)
    expected_expanded_shape = (mock_data.num_replicate,
                               1,
                               mock_data.num_condition_pre,
                               1,
                               1,
                               1,
                               mock_data.batch_size)
    assert final_ln_cfu0.shape == expected_expanded_shape