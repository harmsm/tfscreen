import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.theta_cat import (
    ModelPriors,
    ThetaParam,
    define_model,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides all fields needed by this module
MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "map_theta",
    "scatter_theta"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 2 titrant names
    - 3 titrant concentrations
    - 4 genotypes
    - Total categorical parameters = 2 * 3 * 4 = 24
    """
    num_titrant_name = 2
    num_titrant_conc = 3
    num_genotype = 4
    
    # A map for 5 observations, indexing into the 24-param array
    map_theta = jnp.array([0, 5, 10, 23, 1], dtype=jnp.int32)
    
    return MockData(
        num_titrant_name=num_titrant_name,
        num_titrant_conc=num_titrant_conc,
        num_genotype=num_genotype,
        map_theta=map_theta,
        scatter_theta=1 # Default to scatter, can be overridden
    )

@pytest.fixture
def model_setup(mock_data):
    """
    Provides a deterministic ThetaParam object for testing run_model.
    
    This fixture runs define_model with guesses (all zero offsets)
    to produce a known, predictable ThetaParam object.
    """
    name = "test_theta_cat"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Run define_model with all sample sites substituted
    substituted_model = substitute(define_model, data=guesses)
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)
    return theta_param

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "logit_theta_hyper_loc_loc" in params
    assert params["logit_theta_hyper_loc_loc"] == 0.0

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.logit_theta_hyper_loc_loc == 0.0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_theta_cat"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check hyperprior guesses
    assert f"{name}_logit_theta_hyper_loc" in guesses
    
    # Check offset guess (the main parameter plate)
    assert f"{name}_logit_theta_offset" in guesses
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert guesses[f"{name}_logit_theta_offset"].shape == expected_shape
    assert jnp.all(guesses[f"{name}_logit_theta_offset"] == 0.0)

def test_define_model_shapes_and_values(mock_data):
    """
    Tests the core logic of define_model.
    - Checks the return type and shape.
    - Checks the deterministic site.
    - Checks the calculated value (should be 0.5 given guesses).
    """
    name = "test_theta_cat"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Substitute all sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # --- 1. Get the final return value ---
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)

    # --- 2. Trace the execution to capture intermediate values ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check Return Type and Shape ---
    assert isinstance(theta_param, ThetaParam)
    
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert theta_param.theta.shape == expected_shape
    
# --- 4. Check the Deterministic Site ---
    deterministic_key = f"{name}_theta"
    assert deterministic_key in model_trace
    
    theta_deterministic = model_trace[deterministic_key]["value"]
    assert theta_deterministic.shape == expected_shape
    assert jnp.all(theta_deterministic == theta_param.theta)
    
    # --- 5. Check Values ---
    # With zero offsets, logit_theta = hyper_loc = 0.0
    # The final value is sigmoid(0.0) = 0.5
    assert jnp.allclose(theta_param.theta, 0.5)

def test_run_model_no_scatter(model_setup, mock_data):
    """
    Tests the 'run_model' logic when scatter_theta is 0.
    It should just return the original parameter tensor.
    """
    # Get the pre-calculated ThetaParam object from the fixture
    theta_param = model_setup
    
    # Override mock_data to set scatter_theta=0
    data = mock_data._replace(scatter_theta=0)
    
    # Run the model
    theta_calc = run_model(theta_param, data)
    
    # --- Check Results ---
    # 1. Should be the *exact same object*
    assert theta_calc is theta_param.theta
    
    # 2. Shape should be the 3D parameter shape
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert theta_calc.shape == expected_shape

def test_run_model_with_scatter(model_setup, mock_data):
    """
    Tests the 'run_model' logic when scatter_theta is 1.
    It should return a 1D scattered tensor.
    """
    # Get the pre-calculated ThetaParam object from the fixture
    theta_param = model_setup # This has all 0.5s
    
    # Use the default mock_data (scatter_theta=1)
    data = mock_data
    
    # Run the model
    theta_calc = run_model(theta_param, data)
    
    # --- Check Results ---
    # 1. Shape should match the map_theta
    assert theta_calc.shape == data.map_theta.shape
    
    # 2. Check values
    # Since theta_param.theta is all 0.5s, the scattered
    # result must also be all 0.5s.
    assert jnp.allclose(theta_calc, 0.5)
    
    # 3. Check indexing logic explicitly
    expected_vals = theta_param.theta.ravel()[data.map_theta]
    assert jnp.allclose(theta_calc, expected_vals)