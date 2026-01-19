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

def test_logit_normal_cdf():
    """Test CDF calculation values."""
    mu, sigma = 0.0, 1.0
    x = jnp.array([0.0, 0.5, 1.0])
    # logit(0.5) = 0.
    # Phi((0 - 0)/1) = Phi(0) = 0.5
    
    cdf = transformation_congression._logit_normal_cdf(x, mu, sigma)
    
    assert jnp.isclose(cdf[0], 0.0, atol=1e-5)
    assert jnp.isclose(cdf[1], 0.5, atol=1e-5)
    assert jnp.isclose(cdf[2], 1.0, atol=1e-5)

def test_empirical_cdf():
    """Test empirical CDF calculation."""
    theta = jnp.array([0.1, 0.5, 0.9])
    t_grid = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])
    
    # theta has 3 elements. y = [0.166, 0.5, 0.833] (using 0.5/n)
    # Sorted: 0.1, 0.5, 0.9
    cdf = transformation_congression._empirical_cdf(theta, t_grid)
    
    assert cdf.shape == (5,)
    assert cdf[0] == 0.16666667 # Interp 0.0 at 0.1 returns y[0]
    assert jnp.isclose(cdf[2], 0.5) # 0.5 is exactly in theta
    assert cdf[4] == 0.8333333 # Interp 1.0 at 0.9 returns y[2]


# -------------------------------------------------------------------------
# Correction Logic
# -------------------------------------------------------------------------

def test_calculate_expected_observed_max():
    """Verify correction logic for congression."""
    # Case 1: Target > Background
    x_val = 0.9
    lam = 1.0
    mu, sigma = -2.0, 1.0 # Low mean
    
    res = transformation_congression.calculate_expected_observed_max(x_val, mu, sigma, lam)
    
    assert jnp.isclose(res, x_val, atol=0.05)
    assert res >= x_val 
    
    # Case 2: Target < Background
    x_val = 0.1
    mu, sigma = 2.0, 1.0
    
    res = transformation_congression.calculate_expected_observed_max(x_val, mu, sigma, lam)
    
    assert res > 0.5

def test_calculate_expected_observed_min():
    """Verify correction logic for min."""
    x_val = 0.9
    mu, sigma = -2.0, 1.0 
    lam = 1.0
    
    res = transformation_congression.calculate_expected_observed_min(x_val, mu, sigma, lam)
    
    assert res < x_val
    assert res > 0.0

def test_update_thetas_shapes():
    """Verify shape processing with plated params."""
    theta = jnp.ones((2, 5)) * 0.5
    
    lam = 1.0
    mu = jnp.array([[0.0], [0.0]]) # shape (2, 1)
    sigma = jnp.array([[1.0], [1.0]]) # shape (2, 1)
    
    params = (lam, mu, sigma)
    
    res = transformation_congression.update_thetas(theta, params=params)
    assert res.shape == (2, 5)

    # With mask (line 129)
    mask = jnp.array([True, False, True, False, True]) # Length must match num_genotypes (5)
    res_mask = transformation_congression.update_thetas(theta, params=params, mask=mask)
    assert res_mask.shape == (2, 5)
    # Check that entries where mask is False are unchanged
    assert jnp.all(res_mask[:, 1] == theta[:, 1])
    assert jnp.all(res_mask[:, 3] == theta[:, 3])

    # Empirical mode call within shapes test
    res_emp = transformation_congression.update_thetas(theta, params=(lam,), theta_dist="empirical")
    assert res_emp.shape == (2, 5)
    
def test_update_thetas_empirical_values():
    """Verify empirical mode update_thetas with realistic values."""
    theta = jnp.array([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
    lam = 1.0
    params = (lam,)
    
    res = transformation_congression.update_thetas(theta, params=params, theta_dist="empirical")
    assert res.shape == (2, 3)
    # Result should be >= input for max-congression
    assert jnp.all(res >= theta - 1e-6)


# -------------------------------------------------------------------------
# Model Interface
# -------------------------------------------------------------------------

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = transformation_congression.get_hyperparameters()
    assert "lam_loc" in params
    assert "mu_anchoring_scale" in params
    assert "sigma_anchoring_scale" in params
    
    guesses = transformation_congression.get_guesses("test", MagicMock())
    assert "test_lam" in guesses
    assert "test_mu" in guesses
    assert "test_sigma" in guesses
    
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
        # Without anchors
        lam, mu, sigma = transformation_congression.define_model("test", data, priors)
        assert lam.shape == ()
        assert mu.shape == (2, 3, 1)
        
        # With anchors
        anchors = (jnp.zeros((2, 3, 1)), jnp.ones((2, 3, 1)))
        lam2, mu2, sigma2 = transformation_congression.define_model("test_anc", data, priors, anchors=anchors)
        assert mu2.shape == (2, 3, 1)


def test_guide():
    """Test Numpyro guide definition with plates."""
    priors = transformation_congression.get_priors()
    data = MagicMock()
    data.num_titrant_name = 2
    data.num_titrant_conc = 3
    
    # Run with seeded handler and trace to check params
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as tr:
            lam, mu, sigma = transformation_congression.guide("test", data, priors)
            
    assert "test_mu_loc" in tr
    assert tr["test_mu_loc"]['value'].shape == (2, 3, 1)
    
    assert mu.shape == (2, 3, 1)
    assert sigma.shape == (2, 3, 1)

    # Run with empirical mode
    priors_emp = transformation_congression.ModelPriors(0.0, 0.1, 0.5, 0.2, mode="empirical")
    with numpyro.handlers.seed(rng_seed=2):
        res_emp = transformation_congression.guide("test_emp", data, priors_emp)
        assert len(res_emp) == 1
        assert res_emp[0].shape == ()

    # Run with anchors to hit line 334
    with numpyro.handlers.seed(rng_seed=1):
        anchors = (jnp.zeros((2, 3, 1)), jnp.ones((2, 3, 1)))
        lam_anc, mu_anc, sigma_anc = transformation_congression.guide("test_anc", data, priors, anchors=anchors)
        assert mu_anc.shape == (2, 3, 1)

