import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.activity_horseshoe import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "num_genotype",
    "batch_size",
    "batch_idx",
    "wt_indexes",
    "scale_vector",
    "map_genotype",
    "num_not_wt", 
    "not_wt_mask",
    "epistasis_mode"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 4 total genotypes (0 is WT, 1-3 are mutants)
    - 8 observations (batch size)
    """
    num_genotype = 4
    batch_size = 8
    
    # Batch indices mapping observations to genotypes
    # [WT, Mut1, Mut2, Mut3, Mut1, WT, Mut2, Mut3]
    batch_idx = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)
    
    # WT is genotype 0
    wt_indexes = jnp.array([0], dtype=jnp.int32)
    
    # Scale vector for the scale handler
    scale_vector = jnp.ones(batch_size, dtype=float)
    
    # Legacy fields
    num_not_wt = 3
    not_wt_mask = jnp.array([False, True, True, True])
    map_genotype = batch_idx # In a batch context, map matches batch_idx
    
    return MockGrowthData(
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        wt_indexes=wt_indexes,
        scale_vector=scale_vector,
        map_genotype=map_genotype,
        num_not_wt=num_not_wt,
        not_wt_mask=not_wt_mask,
        epistasis_mode="genotype"
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "global_scale_tau_scale" in params
    assert params["global_scale_tau_scale"] == 0.1

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.global_scale_tau_scale == 0.1

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_activity"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check global scale guess
    assert f"{name}_global_scale" in guesses
    assert isinstance(guesses[f"{name}_global_scale"], (float, int))
    
    # Check local scale and offset guesses (plated by num_genotype)
    expected_shape = (mock_data.num_genotype,)
    
    assert f"{name}_local_scale" in guesses
    assert guesses[f"{name}_local_scale"].shape == expected_shape
    
    assert f"{name}_offset" in guesses
    assert guesses[f"{name}_offset"].shape == expected_shape
    
    # Check that offsets are zeros
    assert jnp.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the Horseshoe prior.
    """
    name = "test_activity"
    priors = get_priors()
    
    # Get base guesses (genotype-level)
    base_guesses = get_guesses(name, mock_data)
    
    # Construct batch-level guesses for substitute
    # define_model samples 'local_scale' and 'offset' with shape (batch_size,)
    batch_guesses = base_guesses.copy()
    
    # Map genotype guesses to batch indices
    batch_guesses[f"{name}_local_scale"] = base_guesses[f"{name}_local_scale"][mock_data.batch_idx]
    batch_guesses[f"{name}_offset"] = base_guesses[f"{name}_offset"][mock_data.batch_idx]

    # Substitute
    substituted_model = substitute(define_model, data=batch_guesses)
    
    # --- 1. Execute Model ---
    final_activity = substituted_model(name=name, 
                                       data=mock_data, 
                                       priors=priors)

    # --- 2. Trace execution ---
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 3. Check the Deterministic Site ---
    assert name in model_trace
    activity_site = model_trace[name]["value"]
    
    # Check shape: Should match batch_size
    assert activity_site.shape == (mock_data.batch_size,)
    
    # --- 4. Check WT Logic ---
    wt_indices = jnp.where(jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.all(activity_site[wt_indices] == 1.0)
    
    # --- 5. Check Mutant Logic ---
    # With offsets guessed as 0.0, effective scale becomes 0 -> activity 1.0
    mutant_indices = jnp.where(~jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    assert jnp.allclose(activity_site[mutant_indices], 1.0)
    
    # --- 6. Check Final Expanded Shape ---
    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_activity.shape == expected_shape

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes and execution.
    """
    name = "test_activity_guide"
    priors = get_priors()

    # Seed the guide execution because it samples
    with seed(rng_seed=0):
        final_activity = guide(name=name,
                               data=mock_data,
                               priors=priors)

    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_activity.shape == expected_shape
    
    # Basic sanity check on values (should be positive)
    assert jnp.all(final_activity >= 0.0)