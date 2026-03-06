import pytest
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth.power import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    PowerParams
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_condition_rep", 
    "num_replicate",
    "map_condition_pre",
    "map_condition_sel"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 3 conditions
    - Maps index into these 3 conditions
    """
    num_condition_rep = 3
    num_replicate = 2 
    
    # 4 observations mapping into the [0, 1, 2] condition array
    map_condition_pre = jnp.array([0, 2, 2, 1], dtype=jnp.int32)
    map_condition_sel = jnp.array([1, 0, 1, 2], dtype=jnp.int32)
    
    return MockGrowthData(
        num_condition_rep=num_condition_rep,
        num_replicate=num_replicate,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure and defaults."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "growth_k_hyper_loc_loc" in params
    assert params["growth_k_hyper_loc_loc"] == 0.025
    assert "growth_n_hyper_loc_loc" in params

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.growth_k_hyper_loc_loc == 0.025
    assert priors.growth_n_hyper_loc_loc == 0.0

def test_get_guesses(mock_data):
    """
    Tests that get_guesses returns correctly named and shaped guesses.
    """
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check scalar guesses exist
    assert f"{name}_k_hyper_loc" in guesses
    assert f"{name}_n_hyper_loc" in guesses
    
    # Check offset guesses
    assert f"{name}_k_offset" in guesses
    assert f"{name}_n_offset" in guesses
    
    # The new model has one offset per condition
    expected_shape = (mock_data.num_condition_rep,)
    
    assert guesses[f"{name}_k_offset"].shape == expected_shape
    assert guesses[f"{name}_n_offset"].shape == expected_shape
    
    # Check that offsets are initialized to zeros
    assert jnp.all(guesses[f"{name}_k_offset"] == 0.0)

def test_define_model_structure_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the power case.
    Verifies output shapes and deterministic site registration.
    """
    name = "test_growth"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # 1. Substitute sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # 2. Run the substituted model to get the final return tuple
    params = substituted_model(name=name, 
                                     data=mock_data, 
                                     priors=priors)
    k_pre = params.k_pre
    n_pre = params.n_pre
    k_sel = params.k_sel
    n_sel = params.n_sel
    
    # 3. Trace to inspect internal deterministic sites
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # The outputs should match the size of the mapping arrays (observations)
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert n_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert n_sel.shape == mock_data.map_condition_sel.shape

    # --- Check the Per-Condition Deterministic Sites ---
    k_name = f"{name}_k"
    n_name = f"{name}_n"
    assert k_name in model_trace
    assert n_name in model_trace
    
    k_per_condition = model_trace[k_name]["value"]
    n_per_condition = model_trace[n_name]["value"]
    
    # Check shape is (num_condition_rep,)
    expected_shape = (mock_data.num_condition_rep,)
    assert k_per_condition.shape == expected_shape
    assert n_per_condition.shape == expected_shape

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    name = "test_growth_guide"
    priors = get_priors()

    # Seed the guide execution to handle sampling
    with seed(rng_seed=0):
        # Trace the guide to inspect parameters and samples
        guide_trace = trace(guide).get_trace(
            name=name,
            data=mock_data,
            priors=priors
        )
        
        # Run guide to check return values
        params = guide(name=name,
                             data=mock_data,
                             priors=priors)
    
    k_pre = params.k_pre
    n_pre = params.n_pre
    k_sel = params.k_sel
    n_sel = params.n_sel

    # --- 1. Check Parameter Sites ---
    assert f"{name}_k_hyper_loc_loc" in guide_trace
    assert f"{name}_n_hyper_loc_loc" in guide_trace
    
    # Local params (Condition-specific)
    assert f"{name}_k_offset_locs" in guide_trace
    k_locs = guide_trace[f"{name}_k_offset_locs"]["value"]
    assert k_locs.shape == (mock_data.num_condition_rep,)

    assert f"{name}_n_offset_scales" in guide_trace
    n_scales = guide_trace[f"{name}_n_offset_scales"]["value"]
    assert n_scales.shape == (mock_data.num_condition_rep,)

    # --- 2. Check Sample Sites ---
    assert f"{name}_k_hyper_loc" in guide_trace
    assert f"{name}_n_hyper_loc" in guide_trace
    
    # Local samples
    assert f"{name}_k_offset" in guide_trace
    assert f"{name}_n_offset" in guide_trace
    
    # --- 3. Check Return Shapes ---
    assert k_pre.shape == mock_data.map_condition_pre.shape
    assert n_pre.shape == mock_data.map_condition_pre.shape
    assert k_sel.shape == mock_data.map_condition_sel.shape
    assert n_sel.shape == mock_data.map_condition_sel.shape
