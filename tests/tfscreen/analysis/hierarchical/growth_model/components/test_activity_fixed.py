import pytest
import jax.numpy as jnp
from numpyro.handlers import trace
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity_fixed import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# Updated mock to include batch_size, which is now used by the code
MockGrowthData = namedtuple("MockGrowthData", [
    "batch_size"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    """
    batch_size = 8
    
    return MockGrowthData(
        batch_size=batch_size
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
    1. The deterministic site has the correct shape (batch_size) and values (1.0).
    2. The final returned value has the correct broadcasted shape.
    """
    name = "test_activity_fixed"
    priors = get_priors()
    
    # --- 1. Get the final return value ---
    final_activity = define_model(name=name, 
                                  data=mock_data, 
                                  priors=priors)

    # --- 2. Trace the execution to capture intermediate values ---
    model_trace = trace(define_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Deterministic Site ---
    assert name in model_trace
    activity_site = model_trace[name]["value"]
    
    # Check shape: Code uses jnp.ones(data.batch_size)
    assert activity_site.shape == (mock_data.batch_size,)
    
    # Check values (must be all 1.0)
    assert jnp.all(activity_site == 1.0)
    
    # --- 4. Check the Final Returned (Expanded) Tensor ---
    # Code broadcasts: activity_dists[None,None,None,None,None,None,:]
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    
    assert final_activity.shape == expected_shape
    assert jnp.all(final_activity == 1.0)

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function.
    
    This test checks:
    1. The guide returns the correct broadcasted shape.
    2. The values are all 1.0.
    """
    name = "test_activity_guide"
    priors = get_priors()

    final_activity = guide(name=name,
                           data=mock_data,
                           priors=priors)

    # Code broadcasts: activity_dists[None,None,None,None,None,None,:]
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)

    assert final_activity.shape == expected_shape
    assert jnp.all(final_activity == 1.0)