import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth_hierarchical import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides the fields define_model needs
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
    - 2 conditions
    - 3 replicates
    - Total per-parameter items = 2 * 3 = 6
    - 4 'pre' observations
    - 4 'sel' observations
    """
    num_condition = 2
    num_replicate = 3
    
    # 4 observations mapping into the flattened [0..5] array
    map_condition_pre = jnp.array([0, 2, 4, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([5, 3, 1, 0], dtype=jnp.int32)
    
    return MockGrowthData(
        num_condition=num_condition,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
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
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    assert f"{name}_k_hyper_loc" in guesses
    
    # Check offset guesses
    assert f"{name}_k_offset" in guesses
    expected_shape = (mock_data.num_condition, mock_data.num_replicate)
    assert guesses[f"{name}_k_offset"].shape == expected_shape
    assert guesses[f"{name}_m_offset"].shape == expected_shape
    
    # Check that offsets are zeros
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical case.
    
    This test checks:
    1.  The deterministic sites have the correct flattened shape and values.
    2.  The final returned tuple contains 4 arrays.
    3.  The 4 returned arrays have the correct expanded shapes.
    4.  The 4 returned arrays have the correct, mapped values.
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
    
    # 3. Trace the execution to capture intermediate (deterministic) values
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 1. Check Return Types and Shapes ---
    assert isinstance(return_tuple, tuple)
    assert len(return_tuple) == 4
    
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert m_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert m_sel.shape == mock_data.map_condition_sel.shape

    # --- 2. Check the Per-Condition/Replicate Deterministic Sites ---
    k_name = f"{name}_k"
    m_name = f"{name}_m"
    assert k_name in model_trace
    assert m_name in model_trace
    
    k_per_cond_rep_1d = model_trace[k_name]["value"]
    m_per_cond_rep_1d = model_trace[m_name]["value"]
    
    # Check shape (must be flattened)
    expected_flat_shape = (mock_data.num_condition * mock_data.num_replicate,)
    assert k_per_cond_rep_1d.shape == expected_flat_shape
    assert m_per_cond_rep_1d.shape == expected_flat_shape
    
    # --- 3. Check Values ---
    
    # Because all offsets in 'guesses' are 0, all k/m values should be
    # identical to their hyper_loc guess.
    expected_k_val = guesses[f"{name}_k_hyper_loc"]
    expected_m_val = guesses[f"{name}_m_hyper_loc"]
    
    assert jnp.allclose(k_per_cond_rep_1d, expected_k_val)
    assert jnp.allclose(m_per_cond_rep_1d, expected_m_val)
    
    # --- 4. Check Final Returned (Expanded) Tensors ---
    
    # The final expanded tensors should also be arrays of the hyper_loc
    assert jnp.allclose(k_pre, expected_k_val)
    assert jnp.allclose(m_pre, expected_m_val)
    assert jnp.allclose(k_sel, expected_k_val)
    assert jnp.allclose(m_sel, expected_m_val)
    
    # Spot-check the mapping logic explicitly (though allclose checks it)
    # k_pre[0] should be the value at index map_condition_pre[0]
    assert k_pre[0] == k_per_cond_rep_1d[mock_data.map_condition_pre[0]]