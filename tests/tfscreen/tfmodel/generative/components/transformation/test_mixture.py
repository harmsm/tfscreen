"""
Tests for the congression mixture transformation (transformation/mixture.py).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from scipy.stats import poisson

from tfscreen.tfmodel.generative.components.transformation import mixture


# ---------------------------------------------------------------------------
# Priors, guesses, define_model / guide
# ---------------------------------------------------------------------------

def test_get_hyperparameters_placeholder_default():
    params = mixture.get_hyperparameters()
    assert params == {"lam_loc": 0.0, "lam_scale": 1.0}


def test_get_hyperparameters_moment_matching():
    """(mean, std) in linear space moment-match the LogNormal."""
    lam_mean, lam_std = 0.3572, 0.13
    params = mixture.get_hyperparameters(lam_mean=lam_mean, lam_std=lam_std)
    loc, scale = params["lam_loc"], params["lam_scale"]
    assert np.isclose(np.exp(loc + scale**2 / 2.0), lam_mean)
    assert np.isclose((np.exp(scale**2) - 1.0) * np.exp(2 * loc + scale**2),
                      lam_std**2)


def test_get_hyperparameters_requires_both():
    with pytest.raises(ValueError, match="together"):
        mixture.get_hyperparameters(lam_mean=0.36)
    with pytest.raises(ValueError, match="together"):
        mixture.get_hyperparameters(lam_std=0.05)


@pytest.mark.parametrize("lam_mean,lam_std",
                         [(0.0, 0.1), (-0.1, 0.1), (0.36, 0.0), (0.36, -0.1)])
def test_get_hyperparameters_rejects_nonpositive(lam_mean, lam_std):
    with pytest.raises(ValueError):
        mixture.get_hyperparameters(lam_mean=lam_mean, lam_std=lam_std)


def test_get_guesses():
    assert mixture.get_guesses("t", MagicMock(), lam_mean=0.5) == {"t_lam": 0.5}
    assert mixture.get_guesses("t", MagicMock()) == {"t_lam": 1.0}


def test_get_priors_forwards():
    priors = mixture.get_priors(lam_mean=0.3572, lam_std=0.13)
    expected = mixture.get_hyperparameters(lam_mean=0.3572, lam_std=0.13)
    assert priors.lam_loc == expected["lam_loc"]
    assert priors.lam_scale == expected["lam_scale"]


def test_define_model_and_guide_sample_scalar_lambda():
    priors = mixture.get_priors()
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as tr:
            lam = mixture.define_model("t", MagicMock(), priors)
    assert jnp.shape(lam) == ()
    assert "t_lam" in tr and "t_lam_value" in tr

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as tr:
            lam = mixture.guide("t", MagicMock(), priors)
    assert jnp.shape(lam) == ()
    assert "t_lam_loc" in tr and "t_lam_scale" in tr


def test_needs_population_flag():
    assert mixture.NEEDS_POPULATION is True


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def _counts(sets):
    return jnp.array(np.concatenate([np.full(k, n) for n, k in
                                     enumerate(sets, start=1)]), dtype=jnp.int32)


@pytest.mark.parametrize("sets", [[12, 3, 1], [16], [0, 4], [2, 0, 5]])
@pytest.mark.parametrize("lam", [0.05, 0.357, 1.5])
def test_weights_sum_to_one(sets, lam):
    f = jnp.array([0.0, 0.3, 1.0])
    log_w = mixture._log_class_weights(jnp.array(lam), f, _counts(sets))
    np.testing.assert_allclose(np.exp(log_w).sum(axis=0), 1.0, rtol=1e-6)


@pytest.mark.parametrize("sets", [[12, 3, 1], [0, 4], [2, 0, 5]])
def test_weights_match_poisson_over_strata(sets):
    lam, f = 0.357, 0.8
    log_w = np.asarray(mixture._log_class_weights(jnp.array(lam),
                                                  jnp.array([f]),
                                                  _counts(sets)))[:, 0]
    w_cong = f * (1.0 - np.exp(-lam))
    assert np.isclose(np.exp(log_w[0]), 1.0 - w_cong)

    present = [n for n, k in enumerate(sets, start=1) if k > 0]
    pmf = poisson.pmf(present, lam)
    pmf = dict(zip(present, pmf / pmf.sum()))
    n_of_set = np.asarray(_counts(sets))
    expected = np.array([w_cong * pmf[n] / sets[n - 1] for n in n_of_set])
    np.testing.assert_allclose(np.exp(log_w[1:]), expected, rtol=1e-6)


def test_spike_only_genotype_has_no_congressed_cells_and_finite_gradient():
    f = jnp.array([0.0, 1.0])
    n = _counts([12, 3, 1])
    log_w = mixture._log_class_weights(jnp.array(0.357), f, n)
    assert np.all(np.isneginf(np.asarray(log_w[1:, 0])))
    assert np.isclose(float(log_w[0, 0]), 0.0)

    def mix(lam):
        growth = jnp.arange(log_w.shape[0], dtype=float)[:, None] * 0.1
        return jnp.sum(jax.scipy.special.logsumexp(
            mixture._log_class_weights(lam, f, n) + growth, axis=0))

    assert np.isfinite(float(jax.grad(mix)(jnp.array(0.357))))


# ---------------------------------------------------------------------------
# cell_classes against an independent loop implementation
# ---------------------------------------------------------------------------

def _toy(seed=0, num_genotype=6, num_tn=2, num_tc=3,
         batch_idx=(4, 0, 2), sets=(3, 2, 1)):
    rng = np.random.default_rng(seed)
    batch_idx = np.array(batch_idx)
    counts = np.asarray(_counts(list(sets)))
    num_sets, n_max = len(counts), len(sets)

    coresident_idx = np.full((num_genotype, num_sets, n_max), -1)
    for g in range(num_genotype):
        for k, n in enumerate(counts):
            coresident_idx[g, k, :n] = rng.integers(0, num_genotype, size=n)

    theta_pop = rng.uniform(0.05, 0.95, (1, 1, 1, 1, num_tn, num_tc,
                                         num_genotype))
    activity_pop = rng.uniform(0.5, 2.0, num_genotype)
    dk_pop = rng.normal(0.0, 0.01, num_genotype)
    bulk_fraction = rng.uniform(0.0, 1.0, num_genotype)
    bulk_fraction[batch_idx[1]] = 0.0          # a spike-only genotype

    data = SimpleNamespace(batch_idx=jnp.array(batch_idx),
                           bulk_fraction=jnp.array(bulk_fraction),
                           coresident_idx=jnp.array(coresident_idx),
                           coresident_n=jnp.array(counts))

    # The batch's own values; theta perturbed (noise acts on the focal theta
    # only), so the focal plasmid is not just its population value.
    theta = theta_pop[..., batch_idx] * 0.9
    activity = activity_pop[batch_idx].reshape(1, 1, 1, 1, 1, 1, -1)
    dk_geno = dk_pop[batch_idx].reshape(1, 1, 1, 1, 1, 1, -1)
    return data, (theta, activity, dk_geno), (theta_pop, activity_pop, dk_pop)


def _reference_classes(data, focal, population, lam):
    theta, activity, dk_geno = focal
    theta_pop, activity_pop, dk_pop = population
    batch_idx = np.asarray(data.batch_idx)
    idx = np.asarray(data.coresident_idx)
    counts = np.asarray(data.coresident_n)
    num_b, num_k = len(batch_idx), idx.shape[1]
    num_tn, num_tc = theta.shape[4], theta.shape[5]

    c_theta = np.zeros((1 + num_k, num_tn, num_tc, num_b))
    c_act = np.zeros_like(c_theta)
    c_dk = np.zeros((1 + num_k, num_b))
    c_theta[0] = theta[0, 0, 0, 0]
    c_act[0] = activity[0, 0, 0, 0, 0, 0][None, None, :]
    c_dk[0] = dk_geno.reshape(-1)

    for b, g in enumerate(batch_idx):
        for k in range(num_k):
            co = [h for h in idx[g, k] if h >= 0]
            for i in range(num_tn):
                for j in range(num_tc):
                    values = [theta[0, 0, 0, 0, i, j, b]] + \
                             [theta_pop[0, 0, 0, 0, i, j, h] for h in co]
                    acts = [activity.reshape(-1)[b]] + \
                           [activity_pop[h] for h in co]
                    best = int(np.argmax(values))
                    c_theta[1 + k, i, j, b] = values[best]
                    c_act[1 + k, i, j, b] = acts[best]
            c_dk[1 + k, b] = np.mean([dk_geno.reshape(-1)[b]] +
                                     [dk_pop[h] for h in co])

    f = np.asarray(data.bulk_fraction)[batch_idx]
    w_cong = f * (1.0 - np.exp(-lam))
    present = sorted(set(counts))
    pmf = poisson.pmf(present, lam)
    pmf = dict(zip(present, pmf / pmf.sum()))
    per_stratum = {n: int(np.sum(counts == n)) for n in present}
    w = np.zeros((1 + num_k, num_b))
    w[0] = 1.0 - w_cong
    for k, n in enumerate(counts):
        w[1 + k] = w_cong * pmf[n] / per_stratum[n]

    return c_theta, c_act, c_dk, w


def test_cell_classes_match_reference():
    data, focal, population = _toy()
    lam = 0.357
    classes = mixture.cell_classes(focal, population, jnp.array(lam), data)
    c_theta, c_act, c_dk, w = _reference_classes(data, focal, population, lam)

    num_classes = c_theta.shape[0]
    assert classes.theta.shape == (num_classes, 1, 1, 1, 1) + c_theta.shape[1:]
    np.testing.assert_allclose(np.asarray(classes.theta)[:, 0, 0, 0, 0],
                               c_theta, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(classes.activity)[:, 0, 0, 0, 0],
                               c_act, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(classes.dk_geno).reshape(num_classes, -1), c_dk,
        rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        np.exp(np.asarray(classes.log_weight)).reshape(num_classes, -1), w,
        rtol=1e-5, atol=1e-12)


def test_cell_classes_mixture_reduces_to_clean_as_lambda_vanishes():
    """As lambda -> 0 every cell is clean: the mixed growth is the clean
    class's, whatever the congressed classes' growth."""
    data, focal, population = _toy()
    classes = mixture.cell_classes(focal, population, jnp.array(1e-8), data)
    growth = jnp.arange(classes.log_weight.shape[0], dtype=float).reshape(
        -1, *([1] * (classes.log_weight.ndim - 1)))
    mixed = jax.scipy.special.logsumexp(classes.log_weight + growth, axis=0)
    np.testing.assert_allclose(np.asarray(mixed), 0.0, atol=1e-6)
