"""
Tests for the expected-maximum theta operator (``_congression.update_thetas``),
kept for Stage 1.5 of tfs-fit-genotypes. The registered transformation
components are tested in test_single.py and test_mixture.py.
"""
import jax.numpy as jnp

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
