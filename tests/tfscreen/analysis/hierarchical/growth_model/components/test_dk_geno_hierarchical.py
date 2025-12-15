import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno_hierarchical import (
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
    "not_wt_mask" 
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
        not_wt_mask=not_wt_mask
    )

# --- Test Cases ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "dk_geno_hyper_loc_loc" in params
    assert params["dk_geno_hyper_loc_loc"] == -3.5

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.dk_geno_hyper_loc_loc == -3.5
    assert priors.dk_geno_hyper_shift_loc == 0.02

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_dk"
    guesses = get_guesses(name, mock_data)
    
    assert isinstance(guesses, dict)
    
    # Check hyperprior guesses
    assert f"{name}_hyper_loc" in guesses
    assert guesses[f"{name}_hyper_loc"] == -3.5
    
    # Check offset guess (the main parameter plate)
    # The code initializes offsets for ALL genotypes (num_genotype)
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_genotype,)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.allclose(guesses[f"{name}_offset"][0], -0.8240460108562919)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model using handlers.
    """
    name = "test_dk"
    priors = get_priors()
    
    # Get base guesses (genotype-level)
    base_guesses = get_guesses(name, mock_data)
    
    # Construct batch-level guesses for substitute
    # define_model samples 'offset' with shape (batch_size,), not (num_genotype,)
    # We must map the genotype guesses to the batch
    batch_guesses = base_guesses.copy()
    
    genotype_offsets = base_guesses[f"{name}_offset"]
    batch_offsets = genotype_offsets[mock_data.batch_idx]
    batch_guesses[f"{name}_offset"] = batch_offsets
    
    # Substitute
    substituted_model = substitute(define_model, data=batch_guesses)
    
    # --- 1. Execute Model ---
    final_dk_geno = substituted_model(name=name, 
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
    dk_geno_site = model_trace[name]["value"]
    
    # Check shape: Should match batch_size
    assert dk_geno_site.shape == (mock_data.batch_size,)
    
    # --- 4. Check WT Logic ---
    # WT indices in batch_idx are 0 and 5
    wt_indices = jnp.where(jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    
    # WT must be exactly 0.0
    assert jnp.all(dk_geno_site[wt_indices] == 0.0)
    
    # --- 5. Check Mutant Logic ---
    
    # Get mutant indices
    mutant_indices = jnp.where(~jnp.isin(mock_data.batch_idx, mock_data.wt_indexes))[0]
    mutant_values = dk_geno_site[mutant_indices]
    
    # Calculate expected value for the first mutant (which uses the standard guess offset)
    hyper_loc = base_guesses[f"{name}_hyper_loc"]
    hyper_scale = base_guesses[f"{name}_hyper_scale"]
    hyper_shift = base_guesses[f"{name}_shift"]
    offset_val = genotype_offsets[0] # All offsets are same in guess
    
    expected_lognormal = jnp.clip(jnp.exp(hyper_loc + offset_val * hyper_scale), max=1e30)
    expected_mutant_val = hyper_shift - expected_lognormal
    
    # Check that mutant values match the calculation
    assert jnp.allclose(mutant_values, expected_mutant_val)
    
    # Because of the specific guess value, this should be close to 0.0
    assert jnp.allclose(mutant_values, 0.0)
    
    # --- 6. Check Final Expanded Shape ---
    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_dk_geno.shape == expected_shape

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes and execution.
    """
    name = "test_dk_guide"
    priors = get_priors()

    # Seed the guide execution because it samples
    with seed(rng_seed=0):
        final_dk_geno = guide(name=name,
                              data=mock_data,
                              priors=priors)

    # Expect: (1, 1, 1, 1, 1, 1, batch_size)
    expected_shape = (1, 1, 1, 1, 1, 1, mock_data.batch_size)
    assert final_dk_geno.shape == expected_shape