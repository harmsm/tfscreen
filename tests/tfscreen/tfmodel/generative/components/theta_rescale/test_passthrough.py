import jax.numpy as jnp
from tfscreen.tfmodel.generative.components.theta_rescale.passthrough import rescale


def test_passthrough_identity():
    theta = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert jnp.allclose(rescale(theta), theta)


def test_passthrough_shape_preserved_3d():
    theta = jnp.ones((3, 4, 5)) * 0.5
    assert rescale(theta).shape == theta.shape


def test_passthrough_shape_preserved_1d():
    theta = jnp.linspace(0.0, 1.0, 10)
    assert rescale(theta).shape == (10,)


def test_passthrough_zero_preserved():
    theta = jnp.zeros(5)
    assert jnp.allclose(rescale(theta), jnp.zeros(5))


def test_passthrough_one_preserved():
    theta = jnp.ones(5)
    assert jnp.allclose(rescale(theta), jnp.ones(5))


def test_passthrough_does_not_clip_outside_unit_interval():
    """passthrough must not clip values — callers decide clamping."""
    theta = jnp.array([-0.5, 1.5])
    out = rescale(theta)
    assert jnp.allclose(out, theta)


def test_passthrough_2d_batch():
    theta = jnp.ones((4, 6)) * 0.3
    out = rescale(theta)
    assert out.shape == (4, 6)
    assert jnp.allclose(out, theta)
