import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed
from collections import namedtuple
from unittest.mock import MagicMock

# --- Module Imports ---
from tfscreen.analysis.hierarchical.growth_model.components.theta_hill import (
    ModelPriors,
    ThetaParam,
    define_model,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# -------------------
# test get_hyperparameters
# -------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    
    assert isinstance(params, dict)
    
    # Check for a few expected keys
    assert "theta_logit_min_hyper_loc_loc" in params
    assert "theta_log_hill_K_hyper_loc_loc" in params
    
    # Check an expected value
    assert params["theta_logit_min_hyper_loc_loc"] == -1
    assert params["theta_log_hill_n_hyper_loc_loc"] == 0.693

# -------------------
# test get_priors
# -------------------

def test_get_priors():
    priors = get_priors()
    
    # Check that it returns the correct dataclass
    assert isinstance(priors, ModelPriors)
    
    # Check that the values match the hyperparameter function
    params = get_hyperparameters()
    assert priors.theta_logit_min_hyper_loc_loc == params["theta_logit_min_hyper_loc_loc"]
    assert priors.theta_log_hill_n_hyper_scale == params["theta_log_hill_n_hyper_scale"]

# -------------------
# test get_guesses
# -------------------

def test_get_guesses():
    # Mock the data object, which just needs to provide shapes
    mock_data = MagicMock()
    mock_data.num_titrant_name = 2
    mock_data.num_genotype = 3
    
    name = "test_theta"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check for hyperprior keys
    assert f"{name}_theta_logit_min_hyper_loc" in guesses
    assert guesses[f"{name}_theta_logit_min_hyper_loc"] == -1
    
    # Check for offset keys and their shapes
    assert f"{name}_logit_min_offset" in guesses
    assert guesses[f"{name}_logit_min_offset"].shape == (2, 3)
    assert f"{name}_log_hill_n_offset" in guesses
    assert guesses[f"{name}_log_hill_n_offset"].shape == (2, 3)

# -------------------
# test run_model
# -------------------

@pytest.fixture
def hill_test_setup():
    """Sets up known parameters and data for testing the Hill equation."""
    
    # Parameters: [titrant_name, genotype] -> shape (1, 2)
    # Genotype 0: min=0.1, max=0.9, K=10, n=2
    # Genotype 1: min=0.2, max=0.8, K=5, n=1
    theta_param = ThetaParam(
        theta_min=jnp.array([[0.1, 0.2]]),
        theta_max=jnp.array([[0.9, 0.8]]),
        hill_K=jnp.array([[10.0, 5.0]]),
        hill_n=jnp.array([[2.0, 1.0]])
    )
    
    # Data tensors: [titrant_name, titrant_conc, genotype] -> shape (1, 3, 2)
    # These tensors will just map the params 1-to-1
    map_theta_group = jnp.array([[[0, 1], [0, 1], [0, 1]]])
    
    # Concentrations to test: 1, 10, 100
    titrant_conc = jnp.array([
        [
            [1.0, 1.0],    # c=1
            [10.0, 10.0],  # c=10
            [100.0, 100.0] # c=100
        ]
    ])
    
    # Mock data object
    MockData = namedtuple("MockData", ["map_theta_group", "titrant_conc", "scatter_theta", "map_theta"])
    
    return theta_param, MockData, map_theta_group, titrant_conc

def test_run_model_scatter_0(hill_test_setup):
    """Tests the Hill calculation with scatter_theta = 0."""
    theta_param, MockData, map_theta_group, titrant_conc = hill_test_setup
    
    data = MockData(
        map_theta_group=map_theta_group,
        titrant_conc=titrant_conc,
        scatter_theta=0,
        map_theta=None # Not used
    )
    
    theta_calc = run_model(theta_param, data)
    
    # --- Check Shape ---
    assert theta_calc.shape == (1, 3, 2)
    
    # --- Check Calculations ---
    # Genotype 0 (min=0.1, max=0.9, K=10, n=2)
    # c=1:   0.1 + (0.8) * (1^2 / (10^2 + 1^2))   = 0.1 + 0.8 * (1/101)   = 0.10792
    # c=10:  0.1 + (0.8) * (10^2 / (10^2 + 10^2)) = 0.1 + 0.8 * (100/200) = 0.5
    # c=100: 0.1 + (0.8) * (100^2 / (10^2 + 100^2)) = 0.1 + 0.8 * (10000/10100) = 0.89207
    
    # Genotype 1 (min=0.2, max=0.8, K=5, n=1)
    # c=1:   0.2 + (0.6) * (1^1 / (5^1 + 1^1))   = 0.2 + 0.6 * (1/6)  = 0.3
    # c=10:  0.2 + (0.6) * (10^1 / (5^1 + 10^1)) = 0.2 + 0.6 * (10/15) = 0.6
    # c=100: 0.2 + (0.6) * (100^1 / (5^1 + 100^1)) = 0.2 + 0.6 * (100/105) = 0.77142
    
    expected_g0 = jnp.array([0.10792079, 0.5, 0.8920792])
    expected_g1 = jnp.array([0.3, 0.6, 0.77142857])
    
    assert jnp.allclose(theta_calc[0, :, 0], expected_g0)
    assert jnp.allclose(theta_calc[0, :, 1], expected_g1)

def test_run_model_scatter_1(hill_test_setup):
    """Tests the Hill calculation with scatter_theta = 1."""
    theta_param, MockData, map_theta_group, titrant_conc = hill_test_setup
    
    # map_theta has shape [rep, time, treat, geno] -> (1, 1, 4, 2)
    # The input theta_calc (from scatter=0) has shape (1, 3, 2)
    # It will be raveled to [g0_c1, g1_c1, g0_c10, g1_c10, g0_c100, g1_c100]
    # Indices:                 0      1        2        3        4        5
    map_theta = jnp.array([
        [
            [
                [0, 1], # Map to c=1
                [2, 3], # Map to c=10
                [4, 5], # Map to c=100
                [0, 1]  # Map to c=1 (again)
            ]
        ]
    ])
    
    data = MockData(
        map_theta_group=map_theta_group,
        titrant_conc=titrant_conc,
        scatter_theta=1,
        map_theta=map_theta
    )
    
    theta_calc_scattered = run_model(theta_param, data)
    
    # Check shape
    assert theta_calc_scattered.shape == (1, 1, 4, 2)
    
    # Check values
    # Get expected values from the scatter_0 test
    expected_g0_c1 = 0.10792079
    expected_g1_c1 = 0.3
    expected_g0_c10 = 0.5
    expected_g1_c10 = 0.6
    
    # Row 0 (c=1)
    assert jnp.allclose(theta_calc_scattered[0, 0, 0, 0], expected_g0_c1)
    assert jnp.allclose(theta_calc_scattered[0, 0, 0, 1], expected_g1_c1)
    
    # Row 1 (c=10)
    assert jnp.allclose(theta_calc_scattered[0, 0, 1, 0], expected_g0_c10)
    assert jnp.allclose(theta_calc_scattered[0, 0, 1, 1], expected_g1_c10)

    # Row 3 (c=1, again)
    assert jnp.allclose(theta_calc_scattered[0, 0, 3, 0], expected_g0_c1)
    assert jnp.allclose(theta_calc_scattered[0, 0, 3, 1], expected_g1_c1)
    

# -------------------
# test define_model
# -------------------

@pytest.fixture
def model_test_setup():
    """Provides mock data and real priors for define_model."""
    
    # Let's use a concrete object instead of a mock
    MockData = namedtuple("MockData", ["num_titrant_name", "num_genotype"])
    data = MockData(num_titrant_name=2, num_genotype=3)
    
    # Use the real priors
    priors = get_priors()
    
    return data, priors

def test_define_model_runs_and_shapes(model_test_setup):
    """Tests that define_model executes and returns correctly shaped params."""
    data, priors = model_test_setup
    rng_key = jax.random.PRNGKey(42)
    
    # Define a runnable function with a seed
    def run():
        # This will print *inside* the numpyro.seed call
        print(f"\n[Inside run] data.num_titrant_name = {data.num_titrant_name}")
        print(f"[Inside run] data.num_genotype = {data.num_genotype}")
        return define_model("test", data, priors)
        
    seeded_fn = seed(run, rng_key)
    
    # Execute the seeded function
    theta_params = seeded_fn()
    
    # Check return type
    assert isinstance(theta_params, ThetaParam)
    
    # --- Debugging ---
    expected_shape = (data.num_titrant_name, data.num_genotype)
    actual_shape = theta_params.theta_min.shape
    
    print(f"\n[Test] Expected shape (titrant, genotype): {expected_shape}")
    print(f"[Test] Actual shape from define_model: {actual_shape}")
    
    # Check parameter shapes
    assert actual_shape == expected_shape