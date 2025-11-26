import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
# Adjust import path as necessary for your actual file structure
from tfscreen.analysis.hierarchical.growth_model.components.growth_hierarchical import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition", 
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    
    UPDATED:
    - 3 conditions
    - 2 replicates (just for context, unused by model definition shapes)
    - Total per-parameter items = 3 (one per condition)
    - The maps must now index into [0, 1, 2], not a flattened replicate list.
    """
    num_condition = 3
    num_replicate = 2 # Irrelevant for parameter shapes in new model, but kept for API compliance
    
    # 4 observations mapping into the [0, 1, 2] condition array
    # Note: Indices must be < num_condition
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)
    
    return MockGrowthData(
        num_condition=num_condition,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure and defaults."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "growth_k_hyper_loc_loc" in params
    assert params["growth_k_hyper_loc_loc"] == 0.025

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.growth_k_hyper_loc_loc == 0.025
    assert priors.growth_m_hyper_loc_loc == 0.0

def test_get_guesses(mock_data):
    """
    Tests that get_guesses returns correctly named and shaped guesses.
    
    UPDATED: Checks for shape (num_condition,) instead of (num_cond, num_rep).
    """
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check scalar guesses exist
    assert f"{name}_k_hyper_loc" in guesses
    
    # Check offset guesses
    assert f"{name}_k_offset" in guesses
    
    # The new model has one offset per condition
    expected_shape = (mock_data.num_condition,)
    
    assert guesses[f"{name}_k_offset"].shape == expected_shape
    assert guesses[f"{name}_m_offset"].shape == expected_shape
    
    # Check that offsets are initialized to zeros
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)

def test_define_model_structure_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical case.
    Verifies output shapes and deterministic site registration.
    """
    name = "test_growth"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # 1. Substitute sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # 2. Run the substituted model to get the final return tuple
    return_tuple = substituted_model(name=name, 
                                     data=mock_data, 
                                     priors=priors)
    k_pre, m_pre, k_sel, m_sel = return_tuple
    
    # 3. Trace to inspect internal deterministic sites
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- Check Return Types and Shapes ---
    assert isinstance(return_tuple, tuple)
    assert len(return_tuple) == 4
    
    # The outputs should match the size of the mapping arrays (observations)
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape

    # --- Check the Per-Condition Deterministic Sites ---
    k_name = f"{name}_k"
    m_name = f"{name}_m"
    assert k_name in model_trace
    assert m_name in model_trace
    
    k_per_condition = model_trace[k_name]["value"]
    m_per_condition = model_trace[m_name]["value"]
    
    # UPDATED: Check shape is (num_condition,)
    expected_shape = (mock_data.num_condition,)
    assert k_per_condition.shape == expected_shape
    assert m_per_condition.shape == expected_shape

def test_define_model_calculation_logic(mock_data):
    """
    Tests that the math (loc + offset * scale) and mapping are correct.
    We inject specific non-zero values to ensure data flows correctly.
    """
    name = "test_growth"
    priors = get_priors()
    
    # Create specific test values
    # k = loc + offset * scale
    # let loc = 10, scale = 2
    # offsets = [0, 1, -1]
    # expected k = [10, 12, 8]
    
    custom_guesses = {
        f"{name}_k_hyper_loc": 10.0,
        f"{name}_k_hyper_scale": 2.0,
        f"{name}_m_hyper_loc": 0.0,   # Keep m simple/ignored for this test
        f"{name}_m_hyper_scale": 1.0,
        
        # Explicit offsets for our 3 conditions
        f"{name}_k_offset": jnp.array([0.0, 1.0, -1.0]),
        f"{name}_m_offset": jnp.zeros(mock_data.num_condition)
    }
    
    # Substitute
    substituted_model = substitute(define_model, data=custom_guesses)
    
    # Run
    k_pre, _, k_sel, _ = substituted_model(name=name, 
                                           data=mock_data, 
                                           priors=priors)
    
    # Expected values per condition
    expected_k_per_condition = jnp.array([10.0, 12.0, 8.0])
    
    # Verify k_pre mapping
    # map_condition_pre was [0, 2, 2, 1]
    # Expected: [val[0], val[2], val[2], val[1]] -> [10, 8, 8, 12]
    expected_k_pre = expected_k_per_condition[mock_data.map_condition_pre]
    assert jnp.allclose(k_pre, expected_k_pre)
    
    # Verify k_sel mapping
    # map_condition_sel was [1, 0, 1, 2]
    # Expected: [val[1], val[0], val[1], val[2]] -> [12, 10, 12, 8]
    expected_k_sel = expected_k_per_condition[mock_data.map_condition_sel]
    assert jnp.allclose(k_sel, expected_k_sel)