import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, seed, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.growth import observe

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "ln_cfu",
    "ln_cfu_std",
    "num_replicate",
    "num_time",
    "num_treatment",
    "num_genotype", 
    "good_mask"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the growth observation.
    
    Dimensions:
    - 2 replicates
    - 1 time
    - 2 treatments
    - Batch of 4 genotypes (representing a Total of 100)
    """
    # Define shapes
    batch_shape = (2, 1, 2, 4) # (rep, time, treat, genotype_batch)
    total_genotypes = 100      # The full size of the dataset
    
    # Create a mask where one specific entry is Bad (False)
    mask_array = jnp.ones(batch_shape, dtype=bool)
    # Mark (rep=0, time=0, treat=0, gene=0) as bad
    mask_array = mask_array.at[0, 0, 0, 0].set(False)
    
    return MockGrowthData(
        ln_cfu=jnp.ones(batch_shape) * 5.0,
        ln_cfu_std=jnp.ones(batch_shape) * 0.2,
        num_replicate=batch_shape[0],
        num_time=batch_shape[1],
        num_treatment=batch_shape[2],
        num_genotype=total_genotypes, 
        good_mask=mask_array
    )

def test_observe_structure_and_distribution(mock_data):
    """
    Verifies the site names, distribution types, and shapes.
    """
    name = "test"
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0
    rng_key = jax.random.PRNGKey(0)

    # trace the model
    model_trace = trace(seed(observe, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    # 1. Check 'nu' parameter
    nu_name = f"{name}_nu"
    assert nu_name in model_trace
    # FIXED: Check instance type directly, not .base_dist
    assert isinstance(model_trace[nu_name]["fn"], dist.Gamma)

    # 2. Check Observation Site
    obs_name = f"{name}_growth_obs"
    assert obs_name in model_trace
    site = model_trace[obs_name]
    
    assert site["is_observed"]
    assert isinstance(site["fn"], dist.MaskedDistribution)
    assert isinstance(site["fn"].base_dist, dist.StudentT)

    # 3. Check shapes match input
    assert site["value"].shape == mock_data.ln_cfu.shape

def test_observe_subsampling_scaling(mock_data):
    """
    CRITICAL: Verifies that the log_prob is correctly scaled.
    
    When subsampling, the log_prob of the batch should be multiplied by:
    (Total Size / Batch Size).
    """
    name = "test"
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0 # Pred = Obs = 5.0
    
    # Fix 'nu' to a known value (e.g., 10.0) so math is deterministic
    fixed_nu = 10.0
    conditioned_model = substitute(observe, data={f"{name}_nu": fixed_nu})
    
    # Subsampling plates require an RNG key.
    rng_key = jax.random.PRNGKey(1)
    
    # Trace
    model_trace = trace(seed(conditioned_model, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )
    
    # Get the observation site
    site = model_trace[f"{name}_growth_obs"]
    
    # 1. Calculate Unscaled Log Prob manually
    base_dist = dist.StudentT(df=fixed_nu, loc=ln_cfu_pred, scale=mock_data.ln_cfu_std)
    log_probs = base_dist.log_prob(mock_data.ln_cfu)
    
    # Apply Mask
    masked_log_probs = jnp.where(mock_data.good_mask, log_probs, 0.0)
    sum_log_prob_batch = jnp.sum(masked_log_probs)
    
    # 2. Calculate Expected Scaled Log Prob
    batch_size = mock_data.ln_cfu.shape[-1]
    scale_factor = mock_data.num_genotype / batch_size # 100 / 4 = 25.0
    
    # 3. Compare with NumPyro's calculation
    # The 'scale' property of the *sample site* holds the computed scale factor
    # derived from the active plate contexts.
    site_scale = site["scale"]
    
    # Verify the observation site has the correct scaling factor applied
    assert site_scale == scale_factor

    # Verify the unscaled log probability calculated by NumPyro matches ours
    trace_log_prob = site["fn"].log_prob(site["value"])
    assert jnp.allclose(jnp.sum(trace_log_prob), sum_log_prob_batch)

def test_observe_masking_logic(mock_data):
    """
    Verifies that masked data points do not contribute to the likelihood.
    """
    name = "test"
    
    # Create a prediction that is WAY OFF for the masked point.
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0
    
    # The masked point is at [0,0,0,0]. Let's make prediction there terrible.
    # Obs=5.0, Pred=1000.0
    ln_cfu_pred = ln_cfu_pred.at[0, 0, 0, 0].set(1000.0)

    # Run model with fixed nu
    fixed_nu = 30.0
    conditioned_model = substitute(observe, data={f"{name}_nu": fixed_nu})
    
    # FIXED: Added seed handler.
    rng_key = jax.random.PRNGKey(2)
    
    model_trace = trace(seed(conditioned_model, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )
    
    site = model_trace[f"{name}_growth_obs"]
    log_probs = site["fn"].log_prob(site["value"])
    
    # Check the specific index [0,0,0,0]
    # It should be 0.0 because of the mask
    assert log_probs[0, 0, 0, 0] == 0.0
    
    # Check a valid index [0,0,0,1]
    # It should be non-zero
    assert log_probs[0, 0, 0, 1] != 0.0