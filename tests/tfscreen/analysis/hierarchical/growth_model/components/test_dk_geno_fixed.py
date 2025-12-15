import pytest
import jax.numpy as jnp
from numpyro.handlers import trace
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno_fixed import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# Updated mock to include batch_size
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
    name = "test_dk"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    assert len(guesses) == 0

def test_get_priors():
    """Tests that get_priors returns a correctly instantiated ModelPriors."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the fixed (zero) case.
    
    This test checks:
    1.  The deterministic site has the correct shape (batch_size) and is all zeros.
    2.  The final returned value has the correct expanded shape and is all zeros.
    """
    name = "test_dk_fixed"
    priors = get_priors()
    
    # --- 1. Get the final return value ---
    final_dk_geno = define_model(name=name, 
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
    dk_geno_site = model_trace[name]["value"]
    
    # Check shape: Code uses jnp.zeros(data.batch_size)
    assert dk_geno_site.shape == (mock_data.batch_size,)
    
    # Check values (must be all zero)
    assert jnp.all(dk_geno_site == 0.0)
    
    # --- 4. Check the Final Returned (Expanded) Tensor ---
    # Code broadcasts: dk_geno_per_genotype[None,None,None,None,None,None,:]
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    
    assert final_dk_geno.shape == expected_shape
    assert jnp.all(final_dk_geno == 0.0)

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function.
    
    This test checks:
    1. The guide returns the correct broadcasted shape.
    2. The values are all 0.0.
    """
    name = "test_dk_guide"
    priors = get_priors()

    final_dk_geno = guide(name=name,
                          data=mock_data,
                          priors=priors)

    # Code broadcasts: dk_geno_per_genotype[None,None,None,None,None,None,:]
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)

    assert final_dk_geno.shape == expected_shape
    assert jnp.all(final_dk_geno == 0.0)