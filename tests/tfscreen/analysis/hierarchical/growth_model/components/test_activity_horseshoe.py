import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity_horseshoe import (
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
    assert "global_scale_tau_scale" in params
    assert params["global_scale_tau_scale"] == 0.1

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.global_scale_tau_scale == 0.1

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_activity"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check global scale guess
    assert f"{name}_global_scale" in guesses
    assert isinstance(guesses[f"{name}_global_scale"], (float, int))
    
    # Check local scale and offset guesses (plated by num_not_wt)
    expected_shape = (mock_data.num_not_wt,)
    assert f"{name}_local_scale" in guesses
    assert guesses[f"{name}_local_scale"].shape == expected_shape
    assert f"{name}_offset" in guesses
    assert guesses[f"{name}_offset"].shape == expected_shape
    
    # Check that offsets are zeros
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the Horseshoe prior.
    
    This test uses guesses where local_scale and offset are 0,
    which should result in all mutant activities being 1.0.
    """
    name = "test_activity"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Substitute all sample sites with our guess values
    # No `seed` is needed as all 3 `sample` calls are substituted
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
    # global_scale = 0.1, local_scale = [0, 0, 0], offset = [0, 0, 0]
    # effective_scale = 0.1 * 0.0 = 0.0
    # log_activity = 0.0 * 0.0 = 0.0
    # activity = exp(0.0) = 1.0
    expected_mutant_val = jnp.exp(0.0)
    
    assert jnp.allclose(mutant_values, expected_mutant_val)
    assert jnp.allclose(mutant_values, 1.0)
    
    # --- 6. Check the Final Returned (Expanded) Tensor ---
    
    # The final shape must match the map
    assert final_activity.shape == mock_data.map_genotype.shape
    
    # Since both WT and mutants are 1.0 (based on guesses),
    # the final expanded array should be all 1.0s.
    assert jnp.allclose(final_activity, 1.0)