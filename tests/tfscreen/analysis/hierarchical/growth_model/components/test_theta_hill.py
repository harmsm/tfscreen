import pytest
import jax
import jax.numpy as jnp
from numpyro.handlers import trace, substitute
from collections import namedtuple

# --- Import Module Under Test ---
from tfscreen.analysis.hierarchical.growth_model.components.theta_hill import (
    ModelPriors,
    ThetaParam,
    define_model,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass

# --- Mock Data Fixture ---

@pytest.fixture
def mock_data():
    """
    Creates a mock DataClass with:
    - 2 Titrants
    - 3 Genotypes
    - 4 Concentrations per titrant
    """
    num_titrant = 2
    num_genotype = 3
    num_conc = 4
    
    # 1. Create log_titrant_conc array: shape (titrant, conc, genotype)
    # We just fill it with dummy data, we'll override it in calculation tests if needed
    log_titrant_conc = jnp.zeros((num_titrant, num_conc, num_genotype))
    
    # 2. Create map_theta_group
    # This maps the (T, C, G) data points to the flattened (T, G) parameter vector.
    # The parameters are raveled C-style (last index varies fastest).
    # Index for (t, g) = t * num_genotype + g
    map_theta_group = jnp.zeros((num_titrant, num_conc, num_genotype), dtype=jnp.int32)
    
    for t in range(num_titrant):
        for g in range(num_genotype):
            flat_idx = t * num_genotype + g
            # Assign this index to all concentrations for this t, g pair
            map_theta_group = map_theta_group.at[t, :, g].set(flat_idx)

    # 3. Create map_theta and scatter_theta for the "scatter=1" case
    # Let's pretend we have 5 observations total that map arbitrarily
    num_obs_total = 5
    map_theta = jnp.array([0, 1, 5, 2, 10], dtype=jnp.int32) # Arbitrary indices into flattened calc
    # Note: run_model.ravel()[map_theta]. Since run_model output is size T*C*G = 2*4*3 = 24.
    # Indices must be < 24.
    
    # Create a simple container to mimic DataClass
    # (Using a simple class or namedtuple works, provided it has attributes)
    class MockDataClass:
        def __init__(self):
            self.num_titrant_name = num_titrant
            self.num_genotype = num_genotype
            self.map_theta_group = map_theta_group
            self.log_titrant_conc = log_titrant_conc
            self.map_theta = map_theta
            self.scatter_theta = 0 # Default to 0
            
    return MockDataClass()

# --- Test Cases ---

def test_get_hyperparameters():
    """Check dictionary keys and default values."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert params["theta_logit_low_hyper_loc_loc"] == 2
    # Check a few others to ensure coverage
    assert "theta_log_hill_K_hyper_loc_loc" in params
    assert "theta_log_hill_n_hyper_loc_loc" in params

def test_get_priors():
    """Check ModelPriors object creation."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.theta_logit_low_hyper_loc_loc == 2

def test_get_guesses(mock_data):
    """Check shape and keys of initial guesses."""
    name = "test_hill"
    guesses = get_guesses(name, mock_data)
    
    # Check scalars
    assert f"{name}_logit_low_hyper_loc" in guesses
    
    # Check Offsets
    # Expected shape: (num_titrant, num_genotype)
    expected_shape = (mock_data.num_titrant_name, mock_data.num_genotype)
    
    offset_key = f"{name}_logit_low_offset"
    assert offset_key in guesses
    assert guesses[offset_key].shape == expected_shape
    
    # Verify n_offset exists
    assert f"{name}_log_hill_n_offset" in guesses

def test_define_model_shapes(mock_data):
    """
    Verifies that define_model produces ThetaParam with correct shapes
    and registers deterministic sites.
    """
    name = "test_hill"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # Run model with guesses
    substituted_model = substitute(define_model, data=guesses)
    theta_param = substituted_model(name=name, data=mock_data, priors=priors)
    
    # Check Output Types
    assert isinstance(theta_param, ThetaParam)
    
    # Check Output Shapes: (num_titrant, num_genotype)
    expected_shape = (mock_data.num_titrant_name, mock_data.num_genotype)
    assert theta_param.theta_low.shape == expected_shape
    assert theta_param.theta_high.shape == expected_shape
    assert theta_param.log_hill_K.shape == expected_shape
    assert theta_param.hill_n.shape == expected_shape
    
    # Check Deterministic Sites in Trace
    # We trace to ensure "test_hill_theta_low", etc., are registered
    model_trace = trace(substituted_model).get_trace(name=name, data=mock_data, priors=priors)
    assert f"{name}_theta_low" in model_trace
    assert f"{name}_hill_n" in model_trace
    
    # Check that deterministic value matches the returned object
    assert jnp.allclose(model_trace[f"{name}_hill_n"]["value"], theta_param.hill_n)

def test_run_model_calculation(mock_data):
    """
    Strict calculation test.
    We inject known parameters and concentrations and verify the Hill equation result.
    """
    # Setup inputs
    # Let's look at index [0, 0] (Titrant 0, Genotype 0)
    # We will set K=1.0 (log_K=0), n=1, low=0, high=1
    
    # Create manual ThetaParam (All ones/zeros to start)
    shape = (mock_data.num_titrant_name, mock_data.num_genotype)
    
    theta_low = jnp.zeros(shape)       # Baseline 0
    theta_high = jnp.ones(shape)       # Max 1
    log_hill_K = jnp.zeros(shape)      # K = 1.0 (ln(1)=0)
    hill_n = jnp.ones(shape)           # n = 1
    
    theta_param = ThetaParam(theta_low, theta_high, log_hill_K, hill_n)
    
    # Setup Data
    # We want to test the concentration exactly at K.
    # If [Conc] == K, and n=1, then occupancy = 0.5.
    # Result = low + (high-low)*0.5 = 0.5
    
    # Set conc for (t=0, g=0) to log(1.0) = 0.0
    mock_data.log_titrant_conc = mock_data.log_titrant_conc.at[0, :, 0].set(0.0)
    
    # Set conc for (t=0, g=1) to be very high (saturation)
    # log(1000). Occupancy -> 1.0
    mock_data.log_titrant_conc = mock_data.log_titrant_conc.at[0, :, 1].set(10.0)
    
    # Set conc for (t=1, g=2) to be very low (baseline)
    # log(0.001). Occupancy -> 0.0
    mock_data.log_titrant_conc = mock_data.log_titrant_conc.at[1, :, 2].set(-10.0)

    # --- Run Calculation (Scatter = 0) ---
    mock_data.scatter_theta = 0
    result = run_model(theta_param, mock_data)
    
    # Expected shape: (num_titrant, num_conc, num_genotype)
    assert result.shape == mock_data.log_titrant_conc.shape
    
    # Check t=0, g=0 (At Kd) -> Should be 0.5
    assert jnp.allclose(result[0, :, 0], 0.5)
    
    # Check t=0, g=1 (Saturation) -> Should be 1.0
    assert jnp.allclose(result[0, :, 1], 1.0, atol=1e-3)
    
    # Check t=1, g=2 (Baseline) -> Should be 0.0
    assert jnp.allclose(result[1, :, 2], 0.0, atol=1e-3)

def test_run_model_scatter(mock_data):
    """Test that the scatter functionality reshapes the output correctly."""
    
    shape = (mock_data.num_titrant_name, mock_data.num_genotype)
    # Fill with arbitrary data
    theta_param = ThetaParam(
        theta_low=jnp.zeros(shape),
        theta_high=jnp.ones(shape),
        log_hill_K=jnp.zeros(shape),
        hill_n=jnp.ones(shape)
    )
    
    # Enable scatter
    mock_data.scatter_theta = 1
    
    # Run
    result = run_model(theta_param, mock_data)
    
    # Result size should match size of mock_data.map_theta
    assert result.shape == mock_data.map_theta.shape
    
    # Verify value mapping logic:
    # result[i] should equal full_tensor.ravel()[map_theta[i]]
    # We calculate the full tensor manually first
    mock_data.scatter_theta = 0
    full_tensor = run_model(theta_param, mock_data)
    
    expected_val = full_tensor.ravel()[mock_data.map_theta[0]]
    assert jnp.isclose(result[0], expected_val)