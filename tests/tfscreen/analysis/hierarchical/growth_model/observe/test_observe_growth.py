import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, seed
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.growth import observe

# --- Mock Data Fixture ---

# A mock data object that provides the fields observe needs
MockGrowthData = namedtuple("MockGrowthData", [
    "ln_cfu",
    "ln_cfu_std",
    "num_replicate",
    "num_time",
    "num_treatment",
    "num_genotype", # This is the *total* genotype size
    "good_mask"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the growth observation.
    - Full shape: (2 reps, 1 time, 2 treatments, 10 total genotypes)
    - Batch shape: (2 reps, 1 time, 2 treatments, 4 batch genotypes)
    """
    # Define full and batch shapes
    full_shape = (2, 1, 2, 10) # (rep, time, treat, genotype_total)
    batch_shape = (2, 1, 2, 4) # (rep, time, treat, genotype_batch)
    
    # Create a mask where half the data is good (True)
    mask_array = jnp.ones(batch_shape, dtype=bool)
    mask_array = mask_array.at[0, :, :, :].set(False) # Mask first replicate
    
    return MockGrowthData(
        ln_cfu=jnp.ones(batch_shape) * 5.0,
        ln_cfu_std=jnp.ones(batch_shape) * 0.2,
        num_replicate=full_shape[0],
        num_time=full_shape[1],
        num_treatment=full_shape[2],
        num_genotype=full_shape[3], # Total size
        good_mask=mask_array
    )

def test_observe_growth_site_with_subsampling(mock_data):
    """
    Tests that the 'observe' function creates the correct observation site,
    handling both masking and subsampling.
    - Checks the site name, type, and parameters (mean, std).
    - Checks that the observation mask is correctly applied.
    - Checks that the subsample plate is correctly registered.
    """
    name = "test"

    # Create a mock prediction tensor (must match batch shape)
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 4.9

    # --- ADD RNG KEY ---
    rng_key = jax.random.PRNGKey(42)

    # --- Trace the execution (WRAPPED IN SEED) ---
    model_trace = trace(seed(observe, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    # --- 1. Check for the Observation Site ---
    obs_site_name = f"{name}_growth_obs"
    assert obs_site_name in model_trace

    site = model_trace[obs_site_name]
    
    # --- 2. Check Site Properties ---
    assert site["type"] == "sample"
    assert site["is_observed"]
    
    # --- 3. Check Subsampling ---
    # The 'genotype' plate should be marked as a subsample
    genotype_plate = model_trace[f"{name}_genotype"]
    assert genotype_plate["type"] == "plate"
    
    # The 'args' tuple in the trace stores (size, subsample_size)
    assert "args" in genotype_plate
    
    expected_size = mock_data.num_genotype
    expected_subsample_size = mock_data.ln_cfu.shape[-1]
    
    assert genotype_plate["args"] == (expected_size, expected_subsample_size)
        
    # --- 4. Check Distribution and Parameters ---
    # The distribution is wrapped in MaskedDistribution
    assert isinstance(site["fn"], dist.MaskedDistribution)
    dist_obj = site["fn"].base_dist
    assert isinstance(dist_obj, dist.Normal)
    
    # Check that the predicted mean was passed correctly
    assert jnp.all(dist_obj.loc == ln_cfu_pred)
    # Check that the standard deviation was passed correctly
    assert jnp.all(dist_obj.scale == mock_data.ln_cfu_std)
    
    # Check that the observations were passed correctly
    assert jnp.all(site["value"] == mock_data.ln_cfu)
    
    # --- 5. Check Masking and Log Prob ---
    
    # We can manually compute the log_prob and check the mask's effect.
    actual_log_prob = jnp.sum(site["fn"].log_prob(site["value"]))

    # Calculate expected log_prob
    unmasked_log_prob = dist.Normal(ln_cfu_pred, mock_data.ln_cfu_std).log_prob(mock_data.ln_cfu)
    expected_log_prob = jnp.sum(unmasked_log_prob[mock_data.good_mask])
    
    assert jnp.isclose(actual_log_prob, expected_log_prob)