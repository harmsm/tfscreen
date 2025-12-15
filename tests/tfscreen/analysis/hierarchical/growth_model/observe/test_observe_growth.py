import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import trace, seed, substitute
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.growth import observe, guide

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "ln_cfu",
    "ln_cfu_std",
    "num_replicate",
    "num_time",
    "num_condition_pre",
    "num_condition_sel",
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype", 
    "batch_size",
    "scale_vector",
    "good_mask"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the growth observation.
    
    Total Shape: (1, 1, 1, 1, 1, 1, 4)
    """
    # Define sizes
    num_replicate = 1
    num_time = 1
    num_condition_pre = 1
    num_condition_sel = 1
    num_titrant_name = 1
    num_titrant_conc = 1
    
    batch_size = 4
    total_genotypes = 100
    
    # Shape: (1, 1, 1, 1, 1, 1, 4)
    shape = (num_replicate, num_time, num_condition_pre, num_condition_sel, 
             num_titrant_name, num_titrant_conc, batch_size)
    
    ln_cfu = jnp.ones(shape) * 5.0
    ln_cfu_std = jnp.ones(shape) * 0.2
    
    # Scale vector for subsampling
    scale_vector = jnp.ones(batch_size) * (total_genotypes / batch_size)
    
    # Create mask: (0,0,0,0,0,0,0) is Bad
    good_mask = jnp.ones(shape, dtype=bool)
    good_mask = good_mask.at[0,0,0,0,0,0,0].set(False)
    
    return MockGrowthData(
        ln_cfu=ln_cfu,
        ln_cfu_std=ln_cfu_std,
        num_replicate=num_replicate,
        num_time=num_time,
        num_condition_pre=num_condition_pre,
        num_condition_sel=num_condition_sel,
        num_titrant_name=num_titrant_name,
        num_titrant_conc=num_titrant_conc,
        num_genotype=total_genotypes,
        batch_size=batch_size,
        scale_vector=scale_vector,
        good_mask=good_mask
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
    """
    name = "test"
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0 # Pred = Obs
    
    # Fix 'nu'
    fixed_nu = 10.0
    conditioned_model = substitute(observe, data={f"{name}_nu": fixed_nu})
    rng_key = jax.random.PRNGKey(1)
    
    # Trace
    model_trace = trace(seed(conditioned_model, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )
    
    site = model_trace[f"{name}_growth_obs"]
    
    # 1. Calculate Expected Unscaled Log Prob (manual)
    base_dist = dist.StudentT(df=fixed_nu, loc=ln_cfu_pred, scale=mock_data.ln_cfu_std)
    log_probs = base_dist.log_prob(mock_data.ln_cfu)
    masked_log_probs = jnp.where(mock_data.good_mask, log_probs, 0.0)
    sum_log_prob_batch = jnp.sum(masked_log_probs)
    
    # 2. Verify Scale Factor
    # The site["scale"] is the vector passed to pyro.handlers.scale
    # It has shape (4,) and values 25.0
    scale_factor = 25.0 
    
    # FIX: Use jnp.all for vector comparison
    assert jnp.all(site["scale"] == scale_factor)

    # 3. Verify Unscaled Log Prob matches
    trace_log_prob = site["fn"].log_prob(site["value"])
    assert jnp.allclose(jnp.sum(trace_log_prob), sum_log_prob_batch)

def test_observe_masking_logic(mock_data):
    """
    Verifies that masked data points do not contribute to the likelihood.
    """
    name = "test"
    
    # Create a prediction that is WAY OFF for the masked point.
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0
    
    # Masked point is at (0,0,0,0,0,0,0)
    ln_cfu_pred = ln_cfu_pred.at[0,0,0,0,0,0,0].set(1000.0)

    # Run model with fixed nu
    fixed_nu = 30.0
    conditioned_model = substitute(observe, data={f"{name}_nu": fixed_nu})
    rng_key = jax.random.PRNGKey(2)
    
    model_trace = trace(seed(conditioned_model, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )
    
    site = model_trace[f"{name}_growth_obs"]
    log_probs = site["fn"].log_prob(site["value"])
    
    # Check the specific index (masked) -> 0.0
    assert log_probs[0,0,0,0,0,0,0] == 0.0
    
    # Check a valid index -> non-zero
    assert log_probs[0,0,0,0,0,0,1] != 0.0

def test_guide_structure(mock_data):
    """
    Tests that the guide creates the correct parameter site for 'nu'.
    """
    name = "test_guide"
    ln_cfu_pred = jnp.ones_like(mock_data.ln_cfu) * 5.0
    
    # Trace the guide
    rng_key = jax.random.PRNGKey(3)
    guide_trace = trace(seed(guide, rng_key)).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )
    
    # Check for 'nu' parameter and sample
    assert f"{name}_nu_loc" in guide_trace
    assert f"{name}_nu_scale" in guide_trace
    assert f"{name}_nu" in guide_trace
    
    # Check distribution
    assert isinstance(guide_trace[f"{name}_nu"]["fn"], dist.LogNormal)