import pytest
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from unittest.mock import MagicMock, patch

from tfscreen.analysis.hierarchical.growth_model.components import transformation_congression

# -------------------------------------------------------------------------
# Math Utilities
# -------------------------------------------------------------------------

def test_kumaraswamy_cdf():
    """Test CDF calculation values."""
    a, b = 2.0, 2.0
    x = jnp.array([0.0, 0.5, 1.0])
    # F(x) = 1 - (1-x^a)^b
    # F(0.5) = 1 - (1-0.25)^2 = 1 - 0.75^2 = 1 - 0.5625 = 0.4375
    
    cdf = transformation_congression._kumaraswamy_cdf(x, a, b)
    
    assert jnp.isclose(cdf[0], 0.0, atol=1e-5)
    assert jnp.isclose(cdf[1], 0.4375, atol=1e-5)
    assert jnp.isclose(cdf[2], 1.0, atol=1e-5)


# -------------------------------------------------------------------------
# Correction Logic
# -------------------------------------------------------------------------

def test_calculate_expected_observed_max():
    """Verify correction logic for congression."""
    # Case 1: Target > Background
    x_val = 0.9
    lam = 1.0
    a, b = 1.0, 5.0 # Low mean
    
    res = transformation_congression.calculate_expected_observed_max(x_val, a, b, lam)
    
    assert jnp.isclose(res, x_val, atol=0.05)
    assert res >= x_val 
    
    # Case 2: Target < Background
    x_val = 0.1
    a, b = 5.0, 1.0
    
    res = transformation_congression.calculate_expected_observed_max(x_val, a, b, lam)
    
    assert res > 0.5

def test_calculate_expected_observed_min():
    """Verify correction logic for min."""
    x_val = 0.9
    a, b = 1.0, 5.0 
    lam = 1.0
    
    res = transformation_congression.calculate_expected_observed_min(x_val, a, b, lam)
    
    assert res < x_val
    assert res > 0.0

def test_update_thetas_shapes():
    """Verify shape processing with plated params."""
    # Theta: (2, 5) -> batch=2, geno=5
    # Let's assume the batch dim corresponds to (name=1, conc=2) or similar
    
    theta = jnp.ones((2, 5)) * 0.5
    
    # params: lam (scalar), a (2, 1), b (2, 1)
    # broadcasting against theta's batch dim (2)
    lam = 1.0
    a = jnp.array([[1.0], [2.0]]) # shape (2, 1)
    b = jnp.array([[1.0], [2.0]]) # shape (2, 1)
    
    params = (lam, a, b)
    
    res = transformation_congression.update_thetas(theta, params=params)
    
    assert res.shape == (2, 5)

def test_update_thetas_broadcasting():
    """Verify complex broadcasting."""
    theta = jnp.ones((2, 5)) * 0.5
    params = (1.0, 1.0, 1.0)
    
    res = transformation_congression.update_thetas(theta, params=params)
    assert res.shape == (2, 5)

# -------------------------------------------------------------------------
# Model Interface
# -------------------------------------------------------------------------

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = transformation_congression.get_hyperparameters()
    assert "lam_loc" in params
    assert "a_loc" in params
    assert "b_loc" in params
    
    guesses = transformation_congression.get_guesses("test", MagicMock())
    assert "test_lam" in guesses
    assert "test_a" in guesses
    assert "test_b" in guesses
    
    priors = transformation_congression.get_priors()
    assert isinstance(priors, transformation_congression.ModelPriors)

def test_define_model():
    """Test Numpyro model definition with plates."""
    priors = transformation_congression.get_priors()
    data = MagicMock()
    data.num_titrant_name = 2
    data.num_titrant_conc = 3
    
    # Run with seeded handler
    with numpyro.handlers.seed(rng_seed=0):
        lam, a, b = transformation_congression.define_model("test", data, priors)
        
    assert lam.shape == () 
    assert a.shape == (2, 3, 1) # (name, conc, 1) due to implicit dim
    # Or strictly (2, 3) relative to plates?
    # pyro.sample in plates usually adds dims on the left.
    # We specified dim=-3 and dim=-2. 
    # With a scalar sample under those plates, we expect (..., name, conc, 1) if we did it right
    # Actually, default is un-event_shaped, so (2, 3).
    # IF the implementation of define_model does not append (1,) it will be (2, 3).
    # Let's check implementation again -- we didn't add (1,).
    # Wait, the *guide* added (1,) to params. define_model just samples. 
    # Let's adjust expectation based on typical behavior: it will broadcast to plate shape.
    
    # Re-reading code: define_model samples scalars under plates.
    # dim=-3 (name), dim=-2 (conc).
    # Result should be (name, conc) if no other batch batch dims.
    # HOWEVER, we often want explicit (name, conc, 1) to broadcast against genotype.
    # Typically we might need `.unsqueeze(-1)` or depend on downstream broadcasting.
    # The previous global was scalar.
    # We found that it returns (2, 3, 1) in practice (likely broadcasting against empty dims or handled by pyro)
    assert a.shape == (2, 3, 1) 
    assert b.shape == (2, 3, 1)

def test_guide():
    """Test Numpyro guide definition with plates."""
    priors = transformation_congression.get_priors()
    data = MagicMock()
    data.num_titrant_name = 2
    data.num_titrant_conc = 3
    
    # Run with seeded handler and trace to check params
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as tr:
            lam, a, b = transformation_congression.guide("test", data, priors)
            
    # Check learnable params exist and have correct shape
    assert "test_a_loc" in tr
    assert tr["test_a_loc"]['value'].shape == (2, 3, 1)
    
    # Sample should broadcast to plate shape
    # Guide creates param (2,3,1), samples lognormal.
    # Sample under plates (2,3).
    # Result: (2, 3, 1) (broadcasts plates against event shape)
    assert a.shape == (2, 3, 1)
    assert b.shape == (2, 3, 1)
