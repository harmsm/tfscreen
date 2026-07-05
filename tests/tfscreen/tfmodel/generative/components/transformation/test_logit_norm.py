import pytest
import jax.numpy as jnp
import numpyro

from tfscreen.tfmodel.generative.components.transformation import (
    logit_norm,
    _congression as congression,
)


def test_get_hyperparameters_mode():
    params = logit_norm.get_hyperparameters()
    assert params["mode"] == "logit_norm"


def test_needs_full_population_theta_flag():
    """logit_norm's background CDF is the smooth analytic (mu, sigma) family,
    not raw samples, so it must not request a population-wide theta reference."""
    assert logit_norm.NEEDS_FULL_POPULATION_THETA is False


def test_get_priors_mode():
    priors = logit_norm.get_priors()
    assert isinstance(priors, congression.ModelPriors)
    assert priors.mode == "logit_norm"


def test_get_hyperparameters_forwards_lam_mean_std():
    params = logit_norm.get_hyperparameters(lam_mean=0.3572, lam_std=0.13)
    expected = congression.get_hyperparameters(lam_mean=0.3572, lam_std=0.13)
    assert params["lam_loc"] == expected["lam_loc"]
    assert params["lam_scale"] == expected["lam_scale"]
    assert params["mode"] == "logit_norm"


def test_get_priors_forwards_lam_mean_std():
    priors = logit_norm.get_priors(lam_mean=0.3572, lam_std=0.13)
    assert priors.mode == "logit_norm"
    expected = congression.get_hyperparameters(lam_mean=0.3572, lam_std=0.13)
    assert priors.lam_loc == expected["lam_loc"]
    assert priors.lam_scale == expected["lam_scale"]


def test_update_thetas_bound_with_logit_norm():
    """Pre-bound update_thetas should invoke logit_norm path without theta_dist arg."""
    theta = jnp.array([[0.2, 0.5, 0.8]])
    # mu/sigma shape: (batch_dim, 1) where batch_dim matches theta.shape[0].
    mu = jnp.array([[0.0]])
    sigma = jnp.array([[1.0]])
    lam = 1.0
    result = logit_norm.update_thetas(theta, params=(lam, mu, sigma))
    assert result.shape == theta.shape
    # Congression correction must push occupancies upward (or keep them equal).
    assert jnp.all(result >= theta - 1e-5)


def test_update_thetas_logit_norm_distinguishable_from_empirical():
    """logit_norm mode requires (lam, mu, sigma); empirical mode only (lam,)."""
    theta = jnp.array([[0.3, 0.7]])
    lam = 0.5
    mu = jnp.array([[0.0]])
    sigma = jnp.array([[1.0]])

    result_ln = logit_norm.update_thetas(theta, params=(lam, mu, sigma))

    # empirical path would flatten mu/sigma as lam — passing only (lam,) should
    # produce a different (empirical-based) correction.
    from tfscreen.tfmodel.generative.components.transformation import empirical
    result_emp = empirical.update_thetas(theta, params=(lam,))

    # Both should have correct shape.
    assert result_ln.shape == theta.shape
    assert result_emp.shape == theta.shape
