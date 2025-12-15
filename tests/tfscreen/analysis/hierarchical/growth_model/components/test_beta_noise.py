import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, substitute, seed

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.components.beta_noise import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

# --- Test Cases for Helper Functions ---

def test_get_hyperparameters():
    """Tests that get_hyperparameters returns correct values and mean."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    
    # Check keys and values
    assert "beta_kappa_loc" in params
    assert "beta_kappa_scale" in params
    assert params["beta_kappa_loc"] == 25.0
    assert params["beta_kappa_scale"] == 0.05
    
    # Check logic (Mean = shape / rate)
    mean_kappa = params["beta_kappa_loc"] / params["beta_kappa_scale"]
    assert mean_kappa == 500.0

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.beta_kappa_loc == 25.0

def test_get_guesses():
    """Tests that get_guesses returns the correct key and value."""
    name = "test_noise"
    guesses = get_guesses(name, None)
    
    assert isinstance(guesses, dict)
    
    # Check that the correct key is present
    expected_key = f"{name}_beta_kappa"
    assert expected_key in guesses
    
    # Check that the guess value is the mean of the prior
    assert guesses[expected_key] == 500.0
    
    # Check that no incorrect keys are present
    assert f"{name}_beta_log_hill_n_hyper_scale" not in guesses

# --- Test Cases for define_model ---

@pytest.fixture
def model_setup():
    """Provides common setup for define_model tests."""
    name = "test_beta_noise"
    priors = get_priors()
    guesses = get_guesses(name, None)
    
    # Substitute the kappa sample with our guess value
    substituted_model = substitute(define_model, data=guesses)
    
    return {
        "name": name,
        "priors": priors,
        "guesses": guesses,
        "substituted_model": substituted_model
    }

def test_define_model_logic_and_outputs(model_setup):
    """
    Tests the core reparameterization logic of define_model.
    Checks that alpha and beta are calculated correctly.
    """
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted_model = model_setup["substituted_model"]
    
    # Use standard mean values
    fx_calc = jnp.array([0.1, 0.5, 0.9])
    
    # --- ADD RNG KEY ---
    rng_key = jax.random.PRNGKey(42)

    # --- 1. Get the final return value (WRAPPED IN SEED) ---
    seeded_run = seed(substituted_model, rng_key)
    fx_noisy_return = seeded_run(name=name,
                                 fx_calc=fx_calc,
                                 priors=priors)

    # --- 2. Trace the execution (WRAPPED IN SEED) ---
    model_trace = trace(seed(substituted_model, rng_key)).get_trace(
        name=name, 
        fx_calc=fx_calc, 
        priors=priors
    )
    
    # --- 3. Check Kappa Sample Site ---
    kappa_site = f"{name}_beta_kappa"
    assert kappa_site in model_trace
    kappa_val = model_trace[kappa_site]["value"]
    assert kappa_val == 500.0 # From get_guesses
    
    # --- 4. Check Beta Sample Site (alpha and beta params) ---
    dist_site = f"{name}_dist"
    assert dist_site in model_trace
    
    # Get the distribution object to check its parameters
    dist_obj = model_trace[dist_site]["fn"]
    assert isinstance(dist_obj, numpyro.distributions.Beta)
    
    # Check alpha = fx_calc * kappa
    expected_alpha = fx_calc * kappa_val
    assert jnp.allclose(dist_obj.concentration1, expected_alpha) # [50., 250., 450.]
    
    # Check beta = (1.0 - fx_calc) * kappa
    expected_beta = (1.0 - fx_calc) * kappa_val
    assert jnp.allclose(dist_obj.concentration0, expected_beta) # [450., 250., 50.]
    
    # --- 5. Check Deterministic Site and Return Value ---
    assert name in model_trace
    fx_noisy_sampled = model_trace[dist_site]["value"]
    fx_noisy_deterministic = model_trace[name]["value"]
    
    # All three (return, deterministic, and sample) should be the same value
    assert jnp.all(fx_noisy_return == fx_noisy_sampled)
    assert jnp.all(fx_noisy_deterministic == fx_noisy_sampled)
    assert fx_noisy_return.shape == fx_calc.shape

def test_define_model_clipping_logic(model_setup):
    """
    Tests the jnp.clip logic for extreme (0 or 1) mean values.
    """
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted_model = model_setup["substituted_model"]

    # Use extreme mean values that would result in alpha/beta < 1e-10
    # 500.0 * 1e-20 = 5e-18, which is < 1e-10
    fx_calc = jnp.array([1e-20, 0.5, 1.0 - 1e-20])

    # --- ADD RNG KEY ---
    rng_key = jax.random.PRNGKey(42)

    # --- Trace the execution (WRAPPED IN SEED) ---
    model_trace = trace(seed(substituted_model, rng_key)).get_trace(
        name=name,
        fx_calc=fx_calc,
        priors=priors
    )
    
    # Get the distribution object
    dist_obj = model_trace[f"{name}_dist"]["fn"]
    
    # Check that alpha[0] was clipped
    assert dist_obj.concentration1[0] == 1e-10
    # Check that beta[0] was not clipped
    assert jnp.isclose(dist_obj.concentration0[0], 500.0) 
    
    # Check middle value (not clipped)
    assert jnp.isclose(dist_obj.concentration1[1], 250.0)
    assert jnp.isclose(dist_obj.concentration0[1], 250.0)
    
    # Check that beta[2] was clipped
    assert jnp.isclose(dist_obj.concentration1[2], 500.0)
    assert dist_obj.concentration0[2] == 1e-10

def test_guide_logic_and_params():
    """
    Tests the guide function structure and execution.
    Verifies that parameters are created and sampling occurs correctly.
    """
    name = "test_beta_noise_guide"
    priors = get_priors()
    fx_calc = jnp.array([0.2, 0.5, 0.8])
    
    # Seed the guide execution because it samples
    with seed(rng_seed=0):
        # We need to trace the guide to inspect the created parameters
        guide_trace = trace(guide).get_trace(
            name=name,
            fx_calc=fx_calc,
            priors=priors
        )
    
    # --- 1. Check Parameter Sites ---
    # Expect kappa_loc and kappa_scale parameters
    assert f"{name}_beta_kappa_loc" in guide_trace
    assert f"{name}_beta_kappa_scale" in guide_trace
    
    # Check initial value for loc is log(prior_loc)
    expected_init_loc = jnp.log(priors.beta_kappa_loc)
    assert jnp.isclose(guide_trace[f"{name}_beta_kappa_loc"]["value"], expected_init_loc)
    
    # --- 2. Check Sample Sites ---
    # Expect kappa sample (LogNormal)
    assert f"{name}_beta_kappa" in guide_trace
    assert isinstance(guide_trace[f"{name}_beta_kappa"]["fn"], dist.LogNormal)
    
    # Expect fx_noisy sample (Beta)
    assert f"{name}_dist" in guide_trace
    dist_obj = guide_trace[f"{name}_dist"]["fn"]
    assert isinstance(dist_obj, dist.Beta)
    
    # Verify shape of sampled output
    fx_noisy = guide_trace[f"{name}_dist"]["value"]
    assert fx_noisy.shape == fx_calc.shape
    
    # Verify values are within [0, 1]
    assert jnp.all((fx_noisy >= 0.0) & (fx_noisy <= 1.0))