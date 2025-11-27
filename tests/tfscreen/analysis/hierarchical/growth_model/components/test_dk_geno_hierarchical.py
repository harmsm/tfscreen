import pytest
import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed, trace, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
# (Using the path you specified)
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno_hierarchical import (
    ModelPriors,
    define_model,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Mock Data Fixture ---

# A mock data object that provides the fields define_model needs
MockGrowthData = namedtuple("MockGrowthData", [
    "num_not_wt", 
    "num_genotype", 
    "not_wt_mask", 
    "map_genotype",
    # Add other fields with default None if other components need them
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    
    This fixture creates a scenario with 4 total genotypes:
    - Genotype 0: Wild-type (WT)
    - Genotypes 1, 2, 3: Mutants
    
    It also defines a 'map_genotype' for 8 hypothetical observations.
    """
    num_genotype = 4
    num_not_wt = 3
    
    # [WT, Mutant, Mutant, Mutant]
    # The mask is True for mutants
    not_wt_mask = jnp.array([False, True, True, True]) 
    
    # 8 observations mapping back to the 4 genotypes
    map_genotype = jnp.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=jnp.int32)
    
    return MockGrowthData(
        num_not_wt=num_not_wt,
        num_genotype=num_genotype,
        not_wt_mask=not_wt_mask,
        map_genotype=map_genotype
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
    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_not_wt,)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert jnp.allclose(guesses[f"{name}_offset"][0], -0.8240460108562919)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model using handlers.
    
    This test ensures 100% branch/line coverage by:
    1.  Using `get_guesses` to get known values.
    2.  Using `numpyro.substitute` to inject these values, making the
        function deterministic.
    3.  Running the substituted model to get the final return value.
    4.  Using `numpyro.trace` to run the model again and capture 
        intermediate values.
    5.  Checking the 'deterministic' site (per-genotype values).
    6.  Checking the final returned value (expanded values).
    """
    name = "test_dk"
    priors = get_priors()
    guesses = get_guesses(name, mock_data)
    
    # 1. Substitute sample sites with our guess values
    substituted_model = substitute(define_model, data=guesses)
    
    # 2. Run the substituted model *once* to get the final return value
    final_dk_geno = substituted_model(name=name, 
                                      data=mock_data, 
                                      priors=priors)

    # 3. Trace the execution to capture intermediate (deterministic) values
    model_trace = trace(substituted_model).get_trace(
        name=name, 
        data=mock_data, 
        priors=priors
    )
    
    # --- 1. Check the Per-Genotype Deterministic Site ---
    
    # This is the 'dk_geno_dists' variable before expansion
    assert name in model_trace
    dk_geno_per_genotype = model_trace[name]["value"]
    
    # Check shape
    assert dk_geno_per_genotype.shape == (mock_data.num_genotype,)
    
    # --- 2. Check the WT Logic (The Core "Branch") ---
    
    # The WT index is where the mask is False (or 0)
    wt_index = jnp.argmin(mock_data.not_wt_mask.astype(int))
    
    # The value for WT *must* be exactly 0.0
    assert dk_geno_per_genotype[wt_index] == 0.0
    
    # --- 3. Check the Mutant Logic ---
    
    # Get all mutant values
    mutant_values = dk_geno_per_genotype[mock_data.not_wt_mask]
    
    # Recalculate the first mutant's value by hand using the guesses
    hyper_loc = guesses[f"{name}_hyper_loc"]
    hyper_scale = guesses[f"{name}_hyper_scale"]
    hyper_shift = guesses[f"{name}_shift"]
    offset_0 = guesses[f"{name}_offset"][0] # First mutant
    
    # This is the logic from define_model:
    # dk_geno_lognormal = jnp.clip(jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale),a_max=1e30)
    # dk_geno_mutant_dists = dk_geno_hyper_shift - dk_geno_lognormal
    
    # Note: jnp.clip(..., max=1e30) is the new API
    expected_lognormal = jnp.clip(jnp.exp(hyper_loc + offset_0 * hyper_scale), max=1e30)
    expected_mutant_0 = hyper_shift - expected_lognormal
    
    # The first mutant value should match our calculation
    assert jnp.isclose(mutant_values[0], expected_mutant_0)
    
    # Because of our special guess, this value should be near zero
    assert jnp.isclose(mutant_values[0], 0.0)
    
    # --- 4. Check the Final Returned (Expanded) Tensor ---
    
    # 'final_dk_geno' is now the variable captured from the model's return
    
    # The final shape must match the map_genotype (Corrected 'final_dk_Geno' typo)
    assert final_dk_geno.shape == mock_data.map_genotype.shape 
    
    # Check that values were mapped correctly
    # map_genotype[0] is 0 (WT) -> final_dk_geno[0] should be 0.0
    assert final_dk_geno[0] == dk_geno_per_genotype[mock_data.map_genotype[0]]
    assert final_dk_geno[0] == 0.0
    
    # map_genotype[1] is 1 (Mutant 0) -> final_dk_geno[1] should be expected_mutant_0
    assert final_dk_geno[1] == dk_geno_per_genotype[mock_data.map_genotype[1]]
    assert jnp.isclose(final_dk_geno[1], expected_mutant_0)