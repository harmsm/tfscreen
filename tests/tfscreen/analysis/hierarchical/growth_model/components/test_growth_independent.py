import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth_independent import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_priors,
    get_guesses

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
    - 2 conditions
    - 3 replicates
    - Total per-parameter items = 2 * 3 = 6
    - 4 'pre' observations
    - 4 'sel' observations
    """
    num_condition = 2
    num_replicate = 3
    
    map_condition_pre = jnp.array([0, 2, 4, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([5, 3, 1, 0], dtype=jnp.int32)
    
    return MockGrowthData(
        num_condition=num_condition,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel
    )


# --- Test Cases ---

def test_get_hyperparameters(mock_data):
    """Tests that get_hyperparameters returns correctly shaped arrays."""
    params = get_hyperparameters(mock_data.num_condition)
    assert isinstance(params, dict)
    
    k_loc = params["growth_k_hyper_loc_loc"]
    assert k_loc.shape == (mock_data.num_condition,)
    assert jnp.allclose(k_loc, 0.025)

def test_get_priors(mock_data):
    """Tests our corrected get_priors function."""
    priors = get_priors(mock_data.num_condition)
    assert isinstance(priors, ModelPriors)
    assert priors.growth_k_hyper_loc_loc.shape == (mock_data.num_condition,)
    assert priors.growth_m_hyper_loc_loc.shape == (mock_data.num_condition,)

def test_get_guesses(mock_data):
    """Tests our corrected get_guesses function."""
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check hyper-parameter guess shapes
    hyper_shape = (mock_data.num_condition, 1)
    assert guesses[f"{name}_k_hyper_loc"].shape == hyper_shape
    assert guesses[f"{name}_m_hyper_loc"].shape == hyper_shape
    
    # Check offset guess shapes
    offset_shape = (mock_data.num_condition, mock_data.num_replicate)
    assert guesses[f"{name}_k_offset"].shape == offset_shape
    assert guesses[f"{name}_m_offset"].shape == offset_shape

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the independent case.
    """
    name = "test_growth_ind"
    
    # Use our fixed helper functions
    priors = get_priors(mock_data.num_condition)
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
    
    # --- 3. Check Values ---
    
    # We must replicate the model's logic exactly to get the expected
    # values. This includes the broadcasting logic.
    
    # Get the guess values
    k_hyper_loc = guesses[f"{name}_k_hyper_loc"]
    k_hyper_scale = guesses[f"{name}_k_hyper_scale"]
    k_offset = guesses[f"{name}_k_offset"]
    
    m_hyper_loc = guesses[f"{name}_m_hyper_loc"]
    m_hyper_scale = guesses[f"{name}_m_hyper_scale"]
    m_offset = guesses[f"{name}_m_offset"]
    
    # The model calculates: (shape_2_1 + shape_2_3 * shape_2_1)
    # This broadcasts to shape (2,3)
    expected_k_dist_2d = k_hyper_loc + k_offset * k_hyper_scale
    expected_m_dist_2d = m_hyper_loc + m_offset * m_hyper_scale
    
    # Ravel just like the model does
    expected_k_vals = expected_k_dist_2d.ravel()
    expected_m_vals = expected_m_dist_2d.ravel()
    
    # Now, the comparison should be correct
    assert jnp.allclose(k_per_cond_rep_1d, expected_k_vals)
    assert jnp.allclose(m_per_cond_rep_1d, expected_m_vals)
    
    # --- 4. Check Final Returned (Expanded) Tensors ---
    
    # Spot-check the mapping logic
    assert k_pre[0] == k_per_cond_rep_1d[mock_data.map_condition_pre[0]]
    assert m_sel[1] == m_per_cond_rep_1d[mock_data.map_condition_sel[1]]