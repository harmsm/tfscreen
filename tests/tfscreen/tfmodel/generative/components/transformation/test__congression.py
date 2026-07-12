import pytest
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.generative.components.transformation import (
    _congression as transformation_congression,
)

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
    assert jnp.isclose(cdf[4], 5/6, atol=1e-5) # Interp 1.0 at 0.9 returns y[2]


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
# population_theta: population-wide background CDF for empirical mode
# -------------------------------------------------------------------------

def test_update_thetas_population_theta_defaults_to_theta():
    """Omitting population_theta must reproduce the pre-existing behaviour of
    building the background CDF from theta itself (backward compatibility)."""
    theta = jnp.array([[0.1, 0.5, 0.9]])
    lam = 1.0
    params = (lam,)

    res_default = transformation_congression.update_thetas(
        theta, params=params, theta_dist="empirical")
    res_explicit_none = transformation_congression.update_thetas(
        theta, params=params, theta_dist="empirical", population_theta=None)

    assert jnp.allclose(res_default, res_explicit_none)


def test_update_thetas_population_theta_changes_correction():
    """A population_theta that differs from theta must change the correction:
    the empirical CDF (and hence the congression-corrected values) should
    reflect where theta falls in the *population*, not in itself."""
    theta = jnp.array([[0.5, 0.5, 0.5]])
    lam = 1.0
    params = (lam,)

    # theta is smack in the middle of a population that is itself centered
    # at 0.5 -> minimal correction expected either way, so use an asymmetric
    # population instead: one where 0.5 sits near the *bottom* of the range.
    low_population = jnp.array([[0.4, 0.45, 0.5, 0.9, 0.95]])
    high_population = jnp.array([[0.05, 0.1, 0.5, 0.55, 0.6]])

    res_low_pop = transformation_congression.update_thetas(
        theta, params=params, theta_dist="empirical", population_theta=low_population)
    res_high_pop = transformation_congression.update_thetas(
        theta, params=params, theta_dist="empirical", population_theta=high_population)

    # Same theta, different reference populations -> different corrections.
    assert not jnp.allclose(res_low_pop, res_high_pop)


def test_update_thetas_population_theta_uses_own_shape_for_lambda_broadcast():
    """population_theta may have a different (larger) trailing genotype
    dimension than theta; the correction must still broadcast to theta's
    shape without error, using population_theta's leading dims for lambda."""
    theta = jnp.array([[0.2, 0.8]])          # (1, 2) — the "batch" being corrected
    population_theta = jnp.linspace(0.0, 1.0, 50).reshape(1, 50)  # (1, 50) — full population
    lam = 1.0
    params = (lam,)

    res = transformation_congression.update_thetas(
        theta, params=params, theta_dist="empirical", population_theta=population_theta)

    assert res.shape == theta.shape
    assert jnp.all(jnp.isfinite(res))


def test_update_thetas_population_theta_ignored_for_logit_norm():
    """population_theta must be a no-op for logit_norm mode, which uses the
    smooth analytic (mu, sigma) CDF rather than raw samples."""
    theta = jnp.array([[0.1, 0.5, 0.9]])
    lam = 1.0
    mu = jnp.array([[0.0]])
    sigma = jnp.array([[1.0]])
    params = (lam, mu, sigma)

    res_without = transformation_congression.update_thetas(
        theta, params=params, theta_dist="logit_norm")
    res_with = transformation_congression.update_thetas(
        theta, params=params, theta_dist="logit_norm",
        population_theta=jnp.array([[0.99, 0.99, 0.99]]))

    assert jnp.allclose(res_without, res_with)


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


# -------------------------------------------------------------------------
# lam_mean/lam_std -- experimentally measured lambda moment-matching
# -------------------------------------------------------------------------

def test_get_hyperparameters_placeholder_default():
    """With no lam_mean/lam_std, a weakly-informative placeholder is used --
    never a specific 'real' experimental value (that would be misleading
    now that the real value must be supplied explicitly by the caller)."""
    params = transformation_congression.get_hyperparameters()
    assert params["lam_loc"] == 0.0
    assert params["lam_scale"] == 1.0


def test_get_hyperparameters_moment_matching():
    """lam_mean/lam_std (linear space) must moment-match onto the
    LogNormal's underlying-Normal parameters, i.e. the resulting
    LogNormal(lam_loc, lam_scale) has the requested arithmetic mean/std."""
    lam_mean, lam_std = 0.3572, 0.13
    params = transformation_congression.get_hyperparameters(
        lam_mean=lam_mean, lam_std=lam_std)

    lam_loc = params["lam_loc"]
    lam_scale = params["lam_scale"]

    # Analytic mean/variance of a LogNormal(loc, scale).
    implied_mean = np.exp(lam_loc + lam_scale**2 / 2.0)
    implied_var = (np.exp(lam_scale**2) - 1.0) * np.exp(2 * lam_loc + lam_scale**2)

    assert np.isclose(implied_mean, lam_mean)
    assert np.isclose(implied_var, lam_std**2)


def test_get_hyperparameters_requires_both_lam_mean_and_lam_std():
    """A lone lam_mean or lam_std (without its partner) is ambiguous and
    must raise rather than silently guessing the other."""
    with pytest.raises(ValueError, match="together"):
        transformation_congression.get_hyperparameters(lam_mean=0.36)
    with pytest.raises(ValueError, match="together"):
        transformation_congression.get_hyperparameters(lam_std=0.05)


@pytest.mark.parametrize("lam_mean,lam_std", [(0.0, 0.1), (-0.1, 0.1), (0.36, 0.0), (0.36, -0.1)])
def test_get_hyperparameters_rejects_nonpositive_lam(lam_mean, lam_std):
    """lam_mean and lam_std must both be strictly positive (LogNormal support)."""
    with pytest.raises(ValueError):
        transformation_congression.get_hyperparameters(lam_mean=lam_mean, lam_std=lam_std)


def test_get_guesses_uses_lam_mean_as_guess():
    """When lam_mean is supplied, it becomes the initial guess directly
    (rather than the old hardcoded 0.3572)."""
    guesses = transformation_congression.get_guesses("test", MagicMock(), lam_mean=0.5)
    assert guesses["test_lam"] == 0.5


def test_get_guesses_placeholder_without_lam_mean():
    guesses = transformation_congression.get_guesses("test", MagicMock())
    assert guesses["test_lam"] == 1.0


def test_get_priors_forwards_lam_mean_std():
    priors = transformation_congression.get_priors(lam_mean=0.3572, lam_std=0.13)
    assert isinstance(priors, transformation_congression.ModelPriors)
    expected = transformation_congression.get_hyperparameters(lam_mean=0.3572, lam_std=0.13)
    assert priors.lam_loc == expected["lam_loc"]
    assert priors.lam_scale == expected["lam_scale"]

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

