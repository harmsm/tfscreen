"""Tests for the overflow-safe horseshoe helpers (components/_horseshoe.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from tfscreen.tfmodel.generative.components._horseshoe import (
    HalfCauchy,
    regularized_scale,
)


def _naive(lam, tau, c2):
    return np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))


@pytest.mark.parametrize("lam", [1e-6, 0.01, 0.5, 1.0, 7.0, 300.0])
@pytest.mark.parametrize("tau", [0.0, 0.01, 1.0, 3.0])
def test_regularized_scale_matches_formula(lam, tau):
    got = float(regularized_scale(jnp.float32(lam), jnp.float32(tau),
                                  jnp.float32(4.0)))
    assert got == pytest.approx(_naive(lam, tau, 4.0), rel=1e-5)


def test_regularized_scale_saturates_without_overflow():
    # lam^2 overflows float32 here; the old form gave inf / inf = NaN.
    lam = jnp.float32(1e25)
    val = regularized_scale(lam, jnp.float32(2.0), jnp.float32(9.0))
    assert float(val) == pytest.approx(1.5, rel=1e-5)    # sqrt(c2) / tau
    g = jax.grad(lambda l: regularized_scale(l, 2.0, 9.0))(lam)
    assert np.isfinite(float(g))


def test_regularized_scale_gradient_at_tau_zero():
    g = jax.grad(lambda t: regularized_scale(3.0, t, 4.0))(0.0)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("x", [1e-4, 0.3, 1.0, 12.0, 1e3])
@pytest.mark.parametrize("scale", [0.1, 1.0, 5.0])
def test_half_cauchy_matches_numpyro(x, scale):
    got = float(HalfCauchy(scale).log_prob(jnp.float32(x)))
    ref = float(dist.HalfCauchy(scale).log_prob(jnp.float32(x)))
    assert got == pytest.approx(ref, abs=1e-5)


def test_half_cauchy_finite_far_in_the_tail():
    x = jnp.float32(1e25)
    assert float(dist.HalfCauchy(1.0).log_prob(x)) == -np.inf   # numpyro's
    lp = float(HalfCauchy(1.0).log_prob(x))
    assert np.isfinite(lp)
    assert lp == pytest.approx(np.log(2 / np.pi) - 2 * np.log(1e25), rel=1e-5)
    assert np.isfinite(float(jax.grad(lambda v: HalfCauchy(1.0).log_prob(v))(x)))


def test_half_cauchy_is_a_half_cauchy():
    d = HalfCauchy(2.0)
    assert isinstance(d, dist.HalfCauchy)
    x = d.sample(jax.random.PRNGKey(0), (20_000,))
    assert float(jnp.median(x)) == pytest.approx(2.0, rel=0.05)
