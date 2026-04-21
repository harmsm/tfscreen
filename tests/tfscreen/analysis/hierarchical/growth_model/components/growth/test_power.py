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
    assert "k_hyper_loc_loc" in params
    assert params["k_hyper_loc_loc"] == 0.025
    assert "n_hyper_loc_loc" in params

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.k_hyper_loc_loc == 0.025
    assert priors.n_hyper_loc_loc == 0.0

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

# ---------------------------------------------------------------------------
# Pinning tests
# ---------------------------------------------------------------------------

from tfscreen.analysis.hierarchical.growth_model.components.growth.power import (
    _PINNABLE_SUFFIXES,
)


def test_pinnable_suffixes_includes_all_six_hypers():
    assert set(_PINNABLE_SUFFIXES) == {
        "k_hyper_loc", "k_hyper_scale",
        "m_hyper_loc", "m_hyper_scale",
        "n_hyper_loc", "n_hyper_scale",
    }


def test_model_priors_default_pinned_is_empty_dict():
    priors = get_priors()
    assert priors.pinned == {}


def test_model_priors_accepts_pinned_dict():
    pinned = {"k_hyper_loc": 0.04, "n_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())
    assert priors.pinned == pinned


def test_define_model_unpinned_uses_sample_sites(mock_data):
    name = "g"
    priors = get_priors()
    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)

    for suffix in _PINNABLE_SUFFIXES:
        assert tr[f"{name}_{suffix}"]["type"] == "sample"


def test_define_model_pinned_replaces_sample_with_deterministic(mock_data):
    name = "g"
    pinned = {"k_hyper_loc": 0.040, "n_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)

    assert tr[f"{name}_k_hyper_loc"]["type"] == "deterministic"
    assert float(tr[f"{name}_k_hyper_loc"]["value"]) == pytest.approx(0.040)
    assert tr[f"{name}_n_hyper_scale"]["type"] == "deterministic"
    assert float(tr[f"{name}_n_hyper_scale"]["value"]) == pytest.approx(0.05)
    # Untouched suffixes still sample
    assert tr[f"{name}_m_hyper_loc"]["type"] == "sample"
    assert tr[f"{name}_k_hyper_scale"]["type"] == "sample"


def test_define_model_all_pinned_has_only_offset_sample_sites(mock_data):
    name = "g"
    pinned = {s: 0.0 for s in _PINNABLE_SUFFIXES}
    pinned["n_hyper_loc"] = 0.0  # n must be 1.0 after exp(0)
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)

    sample_sites = [k for k, v in tr.items() if v["type"] == "sample"]
    expected = {f"{name}_k_offset", f"{name}_m_offset", f"{name}_n_offset"}
    assert set(sample_sites) == expected


def test_guide_pinned_drops_variational_params(mock_data):
    name = "g"
    pinned = {"k_hyper_loc": 0.04, "n_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    # Pinned: no variational params and no sample
    assert f"{name}_k_hyper_loc_loc" not in tr
    assert f"{name}_k_hyper_loc_scale" not in tr
    assert f"{name}_k_hyper_loc" not in tr
    assert f"{name}_n_hyper_scale_loc" not in tr
    assert f"{name}_n_hyper_scale_scale" not in tr
    assert f"{name}_n_hyper_scale_loc" not in tr

    # Unpinned: still present
    assert f"{name}_m_hyper_loc_loc" in tr
    assert f"{name}_m_hyper_loc" in tr


def test_model_and_guide_pinned_have_compatible_sample_sites(mock_data):
    """Critical: guide and model must agree on which sample sites exist."""
    name = "g"
    pinned = {"k_hyper_loc": 0.04, "n_hyper_scale": 0.05}
    priors = ModelPriors(pinned=pinned, **get_hyperparameters())

    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(name=name, data=mock_data, priors=priors)
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(name=name, data=mock_data, priors=priors)

    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
