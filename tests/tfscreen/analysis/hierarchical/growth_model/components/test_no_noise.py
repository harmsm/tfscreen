import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.no_noise import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---
# (Using a simple tuple as a placeholder since 'data' is not used)
@pytest.fixture
def mock_data():
    """Provides a minimal placeholder for the 'data' argument."""
    return (1, 2) 

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns an empty dict."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert len(params) == 0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns an empty dict."""
    name = "test_no_noise"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    assert len(guesses) == 0

def test_get_priors():
    """Tests that get_priors returns a correctly instantiated empty ModelPriors."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_define_model_pass_through_logic(mock_data):
    """
    Tests that define_model correctly returns the input 'fx_calc'
    and does not add any sample or deterministic sites.
    """
    name = "test_no_noise"
    priors = get_priors()
    
    # Define a test input array
    fx_calc_in = jnp.array([0.1, 0.5, 0.9])
    
    # --- 1. Get the final return value ---
    fx_calc_out = define_model(name=name, 
                               fx_calc=fx_calc_in, 
                               priors=priors)

    # --- 2. Check that input and output are identical ---
    assert fx_calc_out is fx_calc_in # Should be the exact same object
    assert jnp.all(fx_calc_out == fx_calc_in)
    
    # --- 3. Trace the execution ---
    model_trace = trace(define_model).get_trace(
        name=name, 
        fx_calc=fx_calc_in, 
        priors=priors
    )
    
    # --- 4. Check that no sample or deterministic sites were added ---
    # The trace should be empty
    assert len(model_trace) == 0

def test_guide_pass_through_logic(mock_data):
    """
    Tests that guide correctly returns the input 'fx_calc'
    and does not add any sample or deterministic sites.
    """
    name = "test_no_noise_guide"
    priors = get_priors()
    
    # Define a test input array
    fx_calc_in = jnp.array([0.1, 0.5, 0.9])
    
    # --- 1. Get the final return value ---
    fx_calc_out = guide(name=name, 
                        fx_calc=fx_calc_in, 
                        priors=priors)

    # --- 2. Check that input and output are identical ---
    assert fx_calc_out is fx_calc_in
    assert jnp.all(fx_calc_out == fx_calc_in)