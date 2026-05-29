import jax.numpy as jnp
import pytest
from tfscreen.growth_model.components.theta_rescale.logit import rescale


def test_logit_midpoint_is_zero():
    assert jnp.allclose(rescale(jnp.array([0.5])), jnp.array([0.0]), atol=1e-5)


def test_logit_monotone():
    theta = jnp.linspace(0.1, 0.9, 20)
    out = rescale(theta)
    assert jnp.all(jnp.diff(out) > 0)


def test_logit_high_expands_range():
    """Values above 0.5 should map above the identity line (larger absolute value)."""
    theta_high = jnp.array([0.7, 0.9])
    out = rescale(theta_high)
    assert jnp.all(out > theta_high)


def test_logit_low_compresses_toward_zero():
    """Values below 0.5 should map to negative values (below the identity line)."""
    theta_low = jnp.array([0.1, 0.3])
    out = rescale(theta_low)
    assert jnp.all(out < theta_low)


def test_logit_finite_at_boundaries():
    theta = jnp.array([0.0, 1.0])
    out = rescale(theta)
    assert jnp.all(jnp.isfinite(out))


def test_logit_shape_preserved():
    theta = jnp.ones((3, 4, 5)) * 0.5
    assert rescale(theta).shape == theta.shape
