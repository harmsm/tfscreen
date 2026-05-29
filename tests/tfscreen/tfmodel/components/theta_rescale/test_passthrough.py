import jax.numpy as jnp
from tfscreen.tfmodel.components.theta_rescale.passthrough import rescale


def test_passthrough_identity():
    theta = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert jnp.allclose(rescale(theta), theta)


def test_passthrough_shape_preserved():
    theta = jnp.ones((3, 4, 5)) * 0.5
    assert rescale(theta).shape == theta.shape
