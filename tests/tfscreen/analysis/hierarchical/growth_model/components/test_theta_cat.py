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
    - 2 titrant names
    - 3 titrant concentrations
    - 4 genotypes
    - Batch size 2 (Subset of genotypes)
    """
    num_titrant_name = 2
    num_titrant_conc = 3
    num_genotype = 4
    
    # Batching logic: select 2 genotypes out of 4
    batch_size = 2
    batch_idx = jnp.array([1, 3], dtype=jnp.int32)
    scale_vector = jnp.ones(batch_size, dtype=float)
    
    # Legacy map (unused by current run_model code but kept for structure)
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
    
    # Slice guesses to match batch
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    # Shape: (Name, Conc, Genotype) -> (Name, Conc, Batch)
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]

    # Run define_model with substituted batch guesses
    substituted_model = substitute(define_model, data=batch_guesses)
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
    Checks return shape (batched) and values.
    """
    name = "test_theta_cat"
    priors = get_priors()
    
    # 1. Prepare Batched Guesses
    base_guesses = get_guesses(name, mock_data)
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_logit_theta_offset"]
    # Slice last dim using batch_idx
    batch_guesses[f"{name}_logit_theta_offset"] = full_offsets[..., mock_data.batch_idx]
    
    # Substitute
    substituted_model = substitute(define_model, data=batch_guesses)
    
    # --- 2. Execute ---
    theta_param = substituted_model(name=name,
                                    data=mock_data,
                                    priors=priors)

    # --- 3. Trace ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 4. Check Shape (Batched) ---
    expected_batch_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_param.theta.shape == expected_batch_shape
    
    # --- 5. Check Deterministic Site ---
    deterministic_key = f"{name}_theta"
    assert deterministic_key in model_trace
    theta_deterministic = model_trace[deterministic_key]["value"]
    assert theta_deterministic.shape == expected_batch_shape
    
    # --- 6. Check Values ---
    # Zero offsets -> sigmoid(0.0) = 0.5
    assert jnp.allclose(theta_param.theta, 0.5)

def test_run_model_no_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=0.
    Should return the raw batched tensor.
    """
    theta_param = model_setup
    data = mock_data._replace(scatter_theta=0)
    
    theta_calc = run_model(theta_param, data)
    
    # Should match theta_param.theta exactly (object identity)
    assert theta_calc is theta_param.theta
    
    # Shape: (Name, Conc, Batch)
    expected_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert theta_calc.shape == expected_shape

def test_run_model_with_scatter(model_setup, mock_data):
    """
    Tests 'run_model' with scatter_theta=1.
    Should return expanded tensor.
    """
    theta_param = model_setup
    data = mock_data # defaults to scatter=1
    
    theta_calc = run_model(theta_param, data)
    
    # Shape: (1, 1, 1, 1, Name, Conc, Batch)
    expected_shape = (1, 1, 1, 1, 
                      mock_data.num_titrant_name, 
                      mock_data.num_titrant_conc, 
                      mock_data.batch_size)
    
    assert theta_calc.shape == expected_shape
    assert jnp.allclose(theta_calc, 0.5)

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_theta_cat_guide"
    priors = get_priors()

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name=name,
            data=mock_data,
            priors=priors
        )
        theta_param = guide(name=name,
                            data=mock_data,
                            priors=priors)

    # --- 1. Check Parameter Sites (Global Genotype Shape) ---
    assert f"{name}_logit_theta_offset_locs" in guide_trace
    offset_locs = guide_trace[f"{name}_logit_theta_offset_locs"]["value"]
    
    # Guide params store full genotype shape
    expected_param_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.num_genotype
    )
    assert offset_locs.shape == expected_param_shape

    # --- 2. Check Sample Sites (Batched Shape) ---
    assert f"{name}_logit_theta_offset" in guide_trace
    sampled_offsets = guide_trace[f"{name}_logit_theta_offset"]["value"]
    
    # Samples are sliced to batch size
    expected_sample_shape = (
        mock_data.num_titrant_name,
        mock_data.num_titrant_conc,
        mock_data.batch_size
    )
    assert sampled_offsets.shape == expected_sample_shape

    # --- 3. Check Return Shape ---
    assert theta_param.theta.shape == expected_sample_shape