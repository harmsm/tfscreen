import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity_fixed import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides the fields define_model needs
MockGrowthData = namedtuple("MockGrowthData", [
    "num_genotype", 
    "map_genotype"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 4 total genotypes
    - 8 observations
    """
    num_genotype = 4
    # 8 observations mapping back to the 4 genotypes
    map_genotype = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)
    
    return MockGrowthData(
        num_genotype=num_genotype,
        map_genotype=map_genotype
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns an empty dict."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert len(params) == 0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns an empty dict."""
    name = "test_activity"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    assert len(guesses) == 0

def test_get_priors():
    """Tests that get_priors returns a correctly instantiated ModelPriors."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the fixed (1.0) case.
    
    This test checks:
    1.  The deterministic site has the correct per-genotype shape and is all 1.0s.
    2.  The final returned value has the correct expanded shape and is all 1.0s.
    """
    name = "test_activity_fixed"
    # Priors object is empty but still needs to be passed
    priors = get_priors()
    
    # --- 1. Get the final return value ---
    # No `substitute` is needed as there are no pyro.sample calls
    final_activity = define_model(name=name, 
                                  data=mock_data, 
                                  priors=priors)

    # --- 2. Trace the execution to capture intermediate values ---
    model_trace = trace(define_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Per-Genotype Deterministic Site ---
    
    # This is the 'activity_dists' variable before expansion
    assert name in model_trace
    activity_per_genotype = model_trace[name]["value"]
    
    # Check shape
    assert activity_per_genotype.shape == (mock_data.num_genotype,)
    
    # Check values (must be all 1.0)
    assert jnp.all(activity_per_genotype == 1.0)
    
    # --- 4. Check the Final Returned (Expanded) Tensor ---
    
    # The final shape must match the map_genotype
    assert final_activity.shape == mock_data.map_genotype.shape
    
    # The final values must also be all 1.0
    assert jnp.all(final_activity == 1.0)