import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity_hierarchical import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides the fields define_model needs
MockGrowthData = namedtuple("MockGrowthData", [
    "num_not_wt", 
    "num_genotype", 
    "not_wt_mask", 
    "map_genotype"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 4 total genotypes (1 WT, 3 Mutants)
    - 8 observations
    """
    num_genotype = 4
    num_not_wt = 3
    
    # [WT, Mutant, Mutant, Mutant]
    # The mask is True for mutants
    not_wt_mask = jnp.array([False, True, True, True]) 
    
    # 8 observations mapping back to the 4 genotypes
    map_genotype = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)
    
    return MockGrowthData(
        num_not_wt=num_not_wt,
        num_genotype=num_genotype,
        not_wt_mask=not_wt_mask,
        map_genotype=map_genotype
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "activity_hyper_loc_loc" in params
    assert params["activity_hyper_loc_loc"] == 0.0

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.activity_hyper_loc_loc == 0.0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_activity"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    assert f"{name}_log_hyper_loc" in guesses
    
    # Check offset guess (the main parameter plate for mutants)
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_not_wt,)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for hierarchical activity.
    
    This test checks:
    1.  The deterministic site has the correct per-genotype shape.
    2.  The WT genotype (index 0) is fixed to 1.0.
    3.  The mutant genotypes are calculated correctly (as 1.0, based on
        the guesses).
    4.  The final returned value has the correct expanded shape and values.
    """
    name = "test_activity"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Substitute all sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # --- 1. Get the final return value ---
    final_activity = substituted_model(name=name, 
                                       data=mock_data, 
                                       priors=priors)

    # --- 2. Trace the execution to capture intermediate values ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Per-Genotype Deterministic Site ---
    assert name in model_trace
    activity_per_genotype = model_trace[name]["value"]
    
    # Check shape
    assert activity_per_genotype.shape == (mock_data.num_genotype,)
    
    # --- 4. Check WT Logic (The Core "Branch") ---
    wt_index = jnp.argmin(mock_data.not_wt_mask.astype(int))
    assert activity_per_genotype[wt_index] == 1.0
    
    # --- 5. Check Mutant Logic ---
    
    # Get all mutant values
    mutant_values = activity_per_genotype[mock_data.not_wt_mask]
    
    # Recalculate the expected mutant value by hand using the guesses
    # (All offsets are 0, so log_activity = hyper_loc = 0.0)
    expected_mutant_val = jnp.exp(0.0)
    
    assert jnp.allclose(mutant_values, expected_mutant_val)
    assert jnp.allclose(mutant_values, 1.0)
    
    # --- 6. Check the Final Returned (Expanded) Tensor ---
    
    # The final shape must match the map
    assert final_activity.shape == mock_data.map_genotype.shape
    
    # Since both WT and mutants are 1.0 (based on guesses),
    # the final expanded array should be all 1.0s.
    assert jnp.allclose(final_activity, 1.0)
    
    # Spot-check the mapping
    # final_activity[0] maps to genotype 0 (WT)
    assert final_activity[0] == activity_per_genotype[mock_data.map_genotype[0]]
    # final_activity[1] maps to genotype 1 (Mutant)
    assert final_activity[1] == activity_per_genotype[mock_data.map_genotype[1]]