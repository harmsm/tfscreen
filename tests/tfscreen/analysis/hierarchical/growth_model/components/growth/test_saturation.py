import pytest
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.growth.saturation import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    SaturationParams
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
    assert "growth_min_hyper_loc_loc" in params
    assert params["growth_min_hyper_loc_loc"] == 0.025

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.growth_min_hyper_loc_loc == 0.025
    assert priors.growth_max_hyper_loc_loc == 0.025

def test_get_guesses(mock_data):
    """
    Tests that get_guesses returns correctly named and shaped guesses.
    """
    name = "test_growth"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check scalar guesses exist
    assert f"{name}_min_hyper_loc" in guesses
    
    # Check offset guesses
    assert f"{name}_min_offset" in guesses
    
    # The new model has one offset per condition
    expected_shape = (mock_data.num_condition_rep,)
    
    assert guesses[f"{name}_min_offset"].shape == expected_shape
    assert guesses[f"{name}_max_offset"].shape == expected_shape
    
    # Check that offsets are initialized to zeros
    assert jnp.all(guesses[f"{name}_min_offset"] == 0.0)

def test_define_model_structure_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the saturation case.
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
    min_pre = params.min_pre
    max_pre = params.max_pre
    min_sel = params.min_sel
    max_sel = params.max_sel
    
    # 3. Trace to inspect internal deterministic sites
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # The outputs should match the size of the mapping arrays (observations)
    assert min_pre.shape == mock_data.map_condition_pre.shape
    assert max_pre.shape == mock_data.map_condition_pre.shape
    assert min_sel.shape == mock_data.map_condition_sel.shape
    assert max_sel.shape == mock_data.map_condition_sel.shape

    # --- Check the Per-Condition Deterministic Sites ---
    min_name = f"{name}_min"
    max_name = f"{name}_max"
    assert min_name in model_trace
    assert max_name in model_trace
    
    min_per_condition = model_trace[min_name]["value"]
    max_per_condition = model_trace[max_name]["value"]
    
    # Check shape is (num_condition_rep,)
    expected_shape = (mock_data.num_condition_rep,)
    assert min_per_condition.shape == expected_shape
    assert max_per_condition.shape == expected_shape

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
    
    min_pre = params.min_pre
    max_pre = params.max_pre
    min_sel = params.min_sel
    max_sel = params.max_sel

    # --- 1. Check Parameter Sites ---
    assert f"{name}_min_hyper_loc_loc" in guide_trace
    
    # Local params (Condition-specific)
    assert f"{name}_min_offset_locs" in guide_trace
    min_locs = guide_trace[f"{name}_min_offset_locs"]["value"]
    assert min_locs.shape == (mock_data.num_condition_rep,)

    # --- 2. Check Sample Sites ---
    assert f"{name}_min_hyper_loc" in guide_trace
    
    # Local samples
    assert f"{name}_min_offset" in guide_trace
    
    # --- 3. Check Return Shapes ---
    assert min_pre.shape == mock_data.map_condition_pre.shape
    assert max_pre.shape == mock_data.map_condition_pre.shape
    assert min_sel.shape == mock_data.map_condition_sel.shape
    assert max_sel.shape == mock_data.map_condition_sel.shape

# ---------------------------------------------------------------------------
# Pinning tests
# ---------------------------------------------------------------------------

import pytest
from numpyro.handlers import trace, seed
from tfscreen.analysis.hierarchical.growth_model.components.growth.saturation import (
    _PINNABLE_SUFFIXES,
    ModelPriors,
    define_model,
    guide,
    get_priors,
    get_hyperparameters,
)


def test_pinnable_suffixes_includes_all_four_hypers():
    assert set(_PINNABLE_SUFFIXES) == {
        "min_hyper_loc", "min_hyper_scale",
        "max_hyper_loc", "max_hyper_scale",
    }


def test_model_priors_default_pinned_is_empty_dict():
    assert get_priors().pinned == {}


def test_model_priors_accepts_pinned_dict():
    pinned = {"min_hyper_loc": 0.04}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


def test_define_model_unpinned_uses_sample_sites(mock_data):
    name = "g"
    priors = get_priors()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
    for suffix in _PINNABLE_SUFFIXES:
        assert tr[f"{name}_{suffix}"]["type"] == "sample"


def test_define_model_pinned_replaces_with_deterministic(mock_data):
    name = "g"
    pinned = {"min_hyper_loc": 0.04, "max_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
    assert tr[f"{name}_min_hyper_loc"]["type"] == "deterministic"
    assert float(tr[f"{name}_min_hyper_loc"]["value"]) == pytest.approx(0.04)
    assert tr[f"{name}_max_hyper_scale"]["type"] == "deterministic"
    assert tr[f"{name}_min_hyper_scale"]["type"] == "sample"
    assert tr[f"{name}_max_hyper_loc"]["type"] == "sample"


def test_guide_pinned_drops_variational_params(mock_data):
    name = "g"
    pinned = {"min_hyper_loc": 0.04, "max_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
    # Pinned: dropped
    assert f"{name}_min_hyper_loc_loc" not in tr
    assert f"{name}_min_hyper_loc_scale" not in tr
    assert f"{name}_min_hyper_loc" not in tr
    assert f"{name}_max_hyper_scale_loc" not in tr
    assert f"{name}_max_hyper_scale_scale" not in tr
    assert f"{name}_max_hyper_scale" not in tr


def test_model_and_guide_pinned_have_compatible_sample_sites(mock_data):
    name = "g"
    pinned = {"min_hyper_loc": 0.04, "max_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)
    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
