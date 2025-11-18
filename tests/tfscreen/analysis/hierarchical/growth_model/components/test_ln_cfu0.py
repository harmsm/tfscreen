import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.ln_cfu0 import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides the fields define_model needs
MockGrowthData = namedtuple("MockGrowthData", [
    "num_ln_cfu0", 
    "map_ln_cfu0"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 3 ln_cfu0 groups (e.g., 3 genotype/replicate combinations)
    - 5 total observations mapping to those 3 groups
    """
    num_ln_cfu0 = 3
    map_ln_cfu0 = jnp.array([0, 2, 1, 0, 2], dtype=jnp.int32)
    
    return MockGrowthData(
        num_ln_cfu0=num_ln_cfu0,
        map_ln_cfu0=map_ln_cfu0
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
    
    # Check offset guess (the main parameter plate)
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_ln_cfu0,)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical ln_cfu0.
    
    This test checks:
    1.  The deterministic site has the correct per-group shape and values.
    2.  The final returned value has the correct expanded shape and values.
    3.  The mapping logic is correct.
    """
    name = "test_ln_cfu0"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Substitute all sample sites with our guess values
    # No `seed` is needed since all `sample` calls are substituted.
    substituted_model = substitute(define_model, data=guesses)
    
    # --- 1. Get the final return value ---
    final_ln_cfu0 = substituted_model(name=name, 
                                      data=mock_data, 
                                      priors=priors)

    # --- 2. Trace the execution to capture intermediate values ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Per-Group Deterministic Site ---
    assert name in model_trace
    ln_cfu0_per_group = model_trace[name]["value"]
    
    # Check shape
    assert ln_cfu0_per_group.shape == (mock_data.num_ln_cfu0,)
    
    # --- 4. Check Values ---
    
    # Get guess values
    hyper_loc = guesses[f"{name}_hyper_loc"]
    hyper_scale = guesses[f"{name}_hyper_scale"]
    offsets = guesses[f"{name}_offset"] # This is all zeros
    
    # Calculate expected value: loc + 0.0 * scale
    expected_val = hyper_loc + offsets * hyper_scale
    assert jnp.allclose(ln_cfu0_per_group, expected_val)
    
    # Since offsets are zero, all values should just be the hyper_loc
    assert jnp.allclose(ln_cfu0_per_group, hyper_loc)
    
    # --- 5. Check the Final Returned (Expanded) Tensor ---
    
    # The final shape must match the map
    assert final_ln_cfu0.shape == mock_data.map_ln_cfu0.shape
    
    # The final values must also all be the hyper_loc
    assert jnp.allclose(final_ln_cfu0, hyper_loc)
    
    # Spot-check the mapping logic
    # final_ln_cfu0[0] should map to group 0
    assert final_ln_cfu0[0] == ln_cfu0_per_group[mock_data.map_ln_cfu0[0]]
    # final_ln_cfu0[1] should map to group 2
    assert final_ln_cfu0[1] == ln_cfu0_per_group[mock_data.map_ln_cfu0[1]]