import pytest
import jax.numpy as jnp

from tfscreen.analysis.hierarchical.growth_model.components.transformation import (
    empirical,
    _congression as congression,
)


def test_get_hyperparameters_mode():
    params = empirical.get_hyperparameters()
    assert params["mode"] == "empirical"


def test_get_priors_mode():
    priors = empirical.get_priors()
    assert isinstance(priors, congression.ModelPriors)
    assert priors.mode == "empirical"


def test_update_thetas_bound_with_empirical():
    """Pre-bound update_thetas should invoke empirical path taking only (lam,)."""
    theta = jnp.array([[0.1, 0.5, 0.9]])
    lam = 1.0
    result = empirical.update_thetas(theta, params=(lam,))
    assert result.shape == theta.shape
    # Congression correction must push occupancies upward (or keep them equal).
    assert jnp.all(result >= theta - 1e-5)


def test_update_thetas_uses_empirical_not_logit_norm():
    """empirical.update_thetas must use the empirical CDF path.

    With the same lambda but a very skewed logit_norm background (high mu,
    large sigma), the empirical correction based on the actual theta distribution
    will differ from the logit_norm correction.  This confirms that the baked-in
    theta_dist="empirical" is honoured and not silently overridden.
    """
    from tfscreen.analysis.hierarchical.growth_model.components.transformation import logit_norm

    theta = jnp.array([[0.1, 0.3, 0.5, 0.7, 0.9]])
    lam = 2.0

    result_emp = empirical.update_thetas(theta, params=(lam,))

    # logit_norm requires (lam, mu, sigma); use extreme params to ensure divergence.
    # mu/sigma shape: (batch_dim, 1) matching theta.shape[0].
    mu = jnp.array([[5.0]])
    sigma = jnp.array([[0.5]])
    result_ln = logit_norm.update_thetas(theta, params=(lam, mu, sigma))

    # The two corrections should differ for at least one element.
    assert not jnp.allclose(result_emp, result_ln, atol=1e-3)
