import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.binding import observe, guide

# --- Mock Data Fixture ---

# Updated mock to include batching fields
MockBindingData = namedtuple("MockBindingData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "good_mask",
    "theta_std",
    "theta_obs",
    "batch_size",
    "scale_vector"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the binding observation.
    - Shape: (2 titrant_names, 3 titrant_concs, 4 genotypes)
    - A mask is applied to half the data.
    """
    shape = (2, 3, 4) # (titrant_name, titrant_conc, genotype)
    
    # Create a mask where half the data is good (True)
    mask_array = jnp.ones(shape, dtype=bool)
    mask_array = mask_array.at[0, :, :].set(False) # Mask first titrant_name
    
    batch_size = shape[2] # Full batch for basic test
    scale_vector = jnp.ones(batch_size, dtype=float)

    return MockBindingData(
        num_titrant_name=shape[0],
        num_titrant_conc=shape[1],
        num_genotype=shape[2],
        good_mask=mask_array,
        theta_std=jnp.ones(shape) * 0.1,
        theta_obs=jnp.ones(shape) * 0.5,
        batch_size=batch_size,
        scale_vector=scale_vector
    )

def test_observe_binding_site(mock_data):
    """
    Tests that the 'observe' function creates the correct observation site.
    - Checks the site name, type, and parameters (mean, std).
    - Checks that the observation mask is correctly applied.
    """
    name = "test"
    
    # Create a mock prediction tensor
    binding_pred = jnp.ones_like(mock_data.theta_obs) * 0.45
    
    # Trace the model execution
    model_trace = trace(observe).get_trace(
        name=name,
        data=mock_data,
        binding_pred=binding_pred
    )
    
    # --- 1. Check for the Observation Site ---
    obs_site_name = f"{name}_binding_obs"
    assert obs_site_name in model_trace
    
    site = model_trace[obs_site_name]
    
    # --- 2. Check Site Properties ---
    assert site["type"] == "sample"
    assert site["is_observed"]
    
    # --- 3. Check Distribution and Parameters ---
    # Check that the mask handler wrapped the distribution
    assert isinstance(site["fn"], dist.MaskedDistribution)
    
    # Get the underlying Normal distribution
    dist_obj = site["fn"].base_dist
    assert isinstance(dist_obj, dist.Normal)
    
    # Check that the predicted mean was passed correctly
    assert jnp.all(dist_obj.loc == binding_pred)
    # Check that the standard deviation was passed correctly
    assert jnp.all(dist_obj.scale == mock_data.theta_std)
    
    # Check that the observations were passed correctly
    assert jnp.all(site["value"] == mock_data.theta_obs)
    
    # --- 4. Check Masking ---
    
    # Calculate the expected log_prob manually
    # MaskedDistribution returns 0 log_prob for masked items
    actual_log_prob = jnp.sum(site["fn"].log_prob(site["value"]))

    # Unmasked calculation on only good data
    unmasked_log_prob = dist.Normal(binding_pred, mock_data.theta_std).log_prob(mock_data.theta_obs)
    expected_log_prob = jnp.sum(unmasked_log_prob[mock_data.good_mask])
    
    assert jnp.isclose(actual_log_prob, expected_log_prob)

def test_guide(mock_data):
    """
    Tests the guide function.
    The guide is currently empty/no-op, so we just check it runs without error.
    """
    name = "test_guide"
    binding_pred = jnp.ones_like(mock_data.theta_obs) * 0.45
    
    # Should run without raising
    guide(name, mock_data, binding_pred)