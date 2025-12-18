import pytest
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.theta_cat import (
    ModelPriors,
    ThetaParam,
    define_model,
    guide,
    run_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "map_theta",
    "scatter_theta"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    """
    num_titrant_name = 2
    num_titrant_conc = 3
    num_genotype = 4
    
    batch_size = 2
    batch_idx = jnp.array([1, 3], dtype=jnp.int32)
    scale_vector = jnp.ones(batch_size, dtype=float)
    map_theta = jnp.array([0, 5, 10, 23, 1], dtype=jnp.int32)
    
    return MockData(
        num_titrant_name=num_titrant_name,
        num_titrant_conc=num_titrant_conc,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        map_theta=map_theta,
        scatter_theta=1
    )

@pytest.fixture
def model_setup(mock_data):
    """
    Provides a deterministic ThetaParam object (BATCHED) for testing run_model.
    """
    name = "test_theta_cat"
    priors = get_priors()
    base_guesses = get_guesses(name, mock_data)
    
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]

    substituted_model = substitute(define_model, data=batch_guesses)
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)
    
    # Ensure mu/sigma are present (since define_model returns them)
    assert theta_param.mu is not None
    assert theta_param.sigma is not None
    
    return theta_param

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "logit_theta_hyper_loc_loc" in params

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_theta_cat"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert guesses[f"{name}_logit_theta_offset"].shape == expected_shape

def test_define_model_shapes_and_values(mock_data):
    """
    Tests the core logic of define_model.
    """
    name = "test_theta_cat"
    priors = get_priors()
    
    base_guesses = get_guesses(name, mock_data)
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]
    
    substituted_model = substitute(define_model, data=batch_guesses)
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)

    expected_batch_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_param.theta.shape == expected_batch_shape
    assert theta_param.mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    assert jnp.allclose(theta_param.theta, 0.5)

def test_run_model_no_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=0.
    """
    theta_param = model_setup
    data = mock_data._replace(scatter_theta=0)
    theta_calc = run_model(theta_param, data)
    assert theta_calc is theta_param.theta
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_calc.shape == expected_shape

def test_run_model_with_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=1.
    """
    theta_param = model_setup
    data = mock_data 
    theta_calc = run_model(theta_param, data)
    expected_shape = (1, 1, 1, 1, 
                      mock_data.num_titrant_name, 
                      mock_data.num_titrant_conc, 
                      mock_data.batch_size)
    assert theta_calc.shape == expected_shape

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_theta_cat_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        theta_param = guide(name=name,
                            data=mock_data,
                            priors=priors)

    expected_sample_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_param.theta.shape == expected_sample_shape
    assert theta_param.mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)

def test_population_moments_logic(mock_data):
    """
    Verify that get_population_moments returns the expected tensors.
    """
    name = "test_moments"
    priors = get_priors()
    
    with seed(rng_seed=0):
        theta_param = define_model(name, mock_data, priors)
        
    from tfscreen.analysis.hierarchical.growth_model.components.theta_cat import get_population_moments
    mu, sigma = get_population_moments(theta_param, mock_data)
    
    assert mu.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    assert sigma.shape == (mock_data.num_titrant_name, mock_data.num_titrant_conc, 1)
    # Check sigma is positive
    assert jnp.all(sigma > 0)