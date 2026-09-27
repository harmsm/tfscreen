"""Tests for the read-count growth observer (growth_likelihood='counts')."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace
from scipy import stats

from tfscreen.tfmodel.generative.observe.growth_counts import (
    GrowthCountsObsPriors,
    count_distribution,
    get_priors,
    guide,
    log_mean,
    observe,
)

MockData = namedtuple("MockData", [
    "counts", "ln_sample_reads", "sample_ln_cfu", "num_replicate", "num_time",
    "num_condition_pre", "num_condition_sel", "num_titrant_name",
    "num_titrant_conc", "batch_size", "scale_vector", "good_mask"])

SHAPE = (1, 1, 1, 1, 1, 2, 3)


@pytest.fixture
def data():
    counts = jnp.array([0.0, 4.0, 120.0, 7.0, 0.0, 3000.0]).reshape(SHAPE)
    good = jnp.ones(SHAPE, dtype=bool).at[..., 0, 1].set(False)
    return MockData(counts=counts,
                    ln_sample_reads=jnp.full(SHAPE, jnp.log(1e6)),
                    sample_ln_cfu=jnp.full(SHAPE, jnp.log(1e8)),
                    num_replicate=1, num_time=1, num_condition_pre=1,
                    num_condition_sel=1, num_titrant_name=1,
                    num_titrant_conc=2, batch_size=3,
                    scale_vector=jnp.array([2.0, 1.0, 1.0]),
                    good_mask=good)


# ---------------------------------------------------------------------------
# the distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu,phi,inv_r", [(3.0, 5.0, 0.01), (0.02, 8.0, 0.1),
                                          (500.0, 0.5, 0.2)])
def test_moments(mu, phi, inv_r):
    d = count_distribution(jnp.log(mu), phi, inv_r)
    assert float(d.mean) == pytest.approx(mu, rel=1e-4)
    assert float(d.variance) == pytest.approx(mu * (1 + phi) + mu ** 2 * inv_r,
                                              rel=1e-4)


def test_log_prob_matches_scipy_negative_binomial():
    mu, phi, inv_r = 40.0, 3.0, 0.05
    c = 1.0 / (phi / mu + inv_r)                 # concentration
    p = c / (c + mu)                             # scipy's success probability
    k = np.arange(0, 200)
    expected = stats.nbinom.logpmf(k, c, p)
    got = count_distribution(jnp.log(mu), phi, inv_r).log_prob(jnp.asarray(k, float))
    assert np.allclose(np.asarray(got), expected, atol=1e-4)


@pytest.mark.parametrize("mu,phi,inv_r", [(3e4, 5.0, 0.01), (1e6, 5.0, 0.01),
                                          (1e6, 0.1, 1e-4), (0.01, 8.0, 0.05)])
def test_log_prob_accurate_at_high_depth(mu, phi, inv_r):
    # numpyro's own negative binomial loses several nats to float32
    # cancellation at 1e4-1e6 reads; this one must not.
    c = 1.0 / (phi / mu + inv_r)
    p = c / (c + mu)
    sd = np.sqrt(mu * (1 + phi) + mu ** 2 * inv_r)
    k = np.unique(np.clip(np.round(np.linspace(mu - 6 * sd, mu + 6 * sd, 41)),
                          0, None))
    got = count_distribution(jnp.float32(np.log(mu)), jnp.float32(phi),
                             jnp.float32(inv_r)).log_prob(jnp.asarray(k, jnp.float32))
    assert np.allclose(np.asarray(got), stats.nbinom.logpmf(k, c, p), atol=1e-3)


def test_sample_moments():
    d = count_distribution(jnp.log(50.0), 3.0, 0.02)
    x = np.asarray(d.sample(jax.random.PRNGKey(0), (200_000,)))
    assert x.mean() == pytest.approx(50.0, rel=0.01)
    assert x.var() == pytest.approx(50 * 4 + 50 ** 2 * 0.02, rel=0.03)


def test_poisson_limit():
    mu = 12.0
    k = np.arange(0, 60)
    got = count_distribution(jnp.log(mu), 1e-8, 1e-10).log_prob(jnp.asarray(k, float))
    assert np.allclose(np.asarray(got), stats.poisson.logpmf(k, mu), atol=1e-4)


def test_tiny_mean_is_finite_with_finite_gradient():
    def lp(log_mu):
        return count_distribution(log_mu, 5.0, 0.01).log_prob(jnp.array(2.0))
    assert np.isfinite(float(lp(-40.0)))
    assert np.isfinite(float(jax.grad(lp)(-40.0)))


def test_log_mean_is_frequency_times_depth(data):
    ln_cfu_pred = jnp.full(SHAPE, jnp.log(1e5))     # 1e5 of 1e8 cells
    mu = jnp.exp(log_mean(data, ln_cfu_pred))
    assert np.allclose(np.asarray(mu), 1e6 * 1e-3)


# ---------------------------------------------------------------------------
# model / guide
# ---------------------------------------------------------------------------

def _trace(fn, data, **subs):
    f = substitute(fn, data=subs) if subs else fn
    return trace(seed(f, 0)).get_trace("growth", data, jnp.full(SHAPE, 12.0),
                                       priors=get_priors())


def test_observe_sites_and_log_likelihood(data):
    tr = _trace(observe, data, growth_phi=4.0, growth_inv_r=0.02)
    assert {"growth_phi", "growth_inv_r", "growth_obs"} <= set(tr)
    site = tr["growth_obs"]
    assert site["is_observed"]
    assert np.array_equal(np.asarray(site["value"]), np.asarray(data.counts))

    # Masked cell contributes nothing; scale_vector weights by genotype.
    log_mu = log_mean(data, jnp.full(SHAPE, 12.0))
    raw = count_distribution(log_mu, 4.0, 0.02).log_prob(data.counts)
    expected = jnp.sum(jnp.where(data.good_mask, raw, 0.0) * data.scale_vector)
    lp = site["fn"].log_prob(site["value"]) * site["scale"]
    assert float(jnp.sum(lp)) == pytest.approx(float(expected), rel=1e-5)


def test_guide_registers_dispersion_only(data):
    tr = _trace(guide, data)
    assert set(k for k, v in tr.items() if v["type"] == "sample") == {
        "growth_phi", "growth_inv_r"}
    params = {k for k, v in tr.items() if v["type"] == "param"}
    assert params == {"growth_phi_loc", "growth_phi_scale",
                      "growth_inv_r_loc", "growth_inv_r_scale"}


def test_priors():
    p = get_priors()
    assert isinstance(p, GrowthCountsObsPriors)
    assert np.exp(p.phi_loc) == pytest.approx(5.0)
    assert np.exp(p.inv_r_loc) == pytest.approx(0.01)


@pytest.mark.parametrize("log_mu", [-200.0, -80.0, -40.0, 0.0, 14.0])
@pytest.mark.parametrize("log_phi", [-15.0, 1.6, 10.0])
@pytest.mark.parametrize("k", [0.0, 1.0, 30.0, 1e6])
def test_finite_value_gradient_and_hessian(log_mu, log_phi, k):
    # A genotype predicted near extinction (log mu << 0; the MAP of an
    # all-zero genotype goes there) used to underflow the concentration
    # mu / phi in float32 and turn the gradient and Hessian to NaN
    # (count-likelihood grid, 2026-09-27).
    def lp(p):
        return count_distribution(p[0], jnp.exp(p[1]), jnp.exp(p[2])).log_prob(
            jnp.float32(k))
    p = jnp.array([log_mu, log_phi, -4.6], dtype=jnp.float32)
    assert np.isfinite(float(lp(p)))
    assert np.all(np.isfinite(np.asarray(jax.grad(lp)(p))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(lp)(p))))


def test_concentration_floor_leaves_ordinary_rows_alone():
    # the floor (exp(-30)) is far below any concentration a real row has
    mu, phi, inv_r = 0.01, 8.0, 0.05
    c = 1.0 / (phi / mu + inv_r)
    k = np.arange(0, 5)
    got = count_distribution(jnp.log(mu), phi, inv_r).log_prob(jnp.asarray(k, float))
    assert np.allclose(np.asarray(got), stats.nbinom.logpmf(k, c, c / (c + mu)),
                       atol=1e-5)
