"""Tests for Stage-1.5 congression de-attenuation."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import pytest
from scipy.special import expit, logit

from tfscreen.tfmodel.generative.components.transformation._congression import (
    update_thetas,
)
from tfscreen.simulate.empirical.congression import (
    correct_theta_matrix,
    deattenuate_congression,
)
from tfscreen.simulate.empirical.fit_phenotypes import (
    GenotypeFit, PHENO_PARAMS_TRANSFORMED,
)


def _forward(theta_true, lam):
    """Apply the congression forward operator E[max(x, M)]."""
    return np.asarray(update_thetas(
        jnp.asarray(theta_true), (float(lam),), theta_dist="empirical",
        population_theta=jnp.asarray(theta_true)))


# ---------------------------------------------------------------------------
# correct_theta_matrix — the fixed-point core
# ---------------------------------------------------------------------------

def test_roundtrip_recovers_true_population():
    rng = np.random.default_rng(0)
    # 300 genotypes at 2 independent conditions.
    theta_true = expit(rng.normal(0.0, 1.5, size=(2, 300)))
    theta_obs = _forward(theta_true, lam=1.0)

    # Congression can only inflate (max), so observed >= true.
    assert np.all(theta_obs >= theta_true - 1e-9)
    # ...and it compresses the population (masks the low end up).
    assert np.std(theta_obs) < np.std(theta_true)

    theta_rec, n_iter = correct_theta_matrix(
        theta_obs, lam=1.0, tol=1e-6, max_iter=200)

    assert np.median(np.abs(theta_rec - theta_true)) < 0.03
    assert np.std(theta_rec) == pytest.approx(np.std(theta_true), abs=0.03)


def test_lambda_zero_is_noop():
    rng = np.random.default_rng(1)
    theta_obs = expit(rng.normal(0, 1, size=(1, 50)))
    theta_rec, n_iter = correct_theta_matrix(theta_obs, lam=0.0)
    assert np.allclose(theta_rec, theta_obs)
    assert n_iter == 0


def test_correction_pulls_observed_down():
    rng = np.random.default_rng(2)
    theta_true = expit(rng.normal(0, 1.5, size=(1, 300)))
    theta_obs = _forward(theta_true, lam=1.5)
    theta_rec, _ = correct_theta_matrix(theta_obs, lam=1.5, max_iter=200)
    # E[max] >= true, so the inverse pulls the observed population down.
    assert np.mean(theta_rec) < np.mean(theta_obs)


# ---------------------------------------------------------------------------
# deattenuate_congression — the fits-level wrapper
# ---------------------------------------------------------------------------

def _make_fit(genotype, theta_low, theta_high, log_K, n, dk=0.0):
    # est_t: [ln_cfu0, dk_geno, logit_low, logit_high, log_K, log_n]
    est_t = np.array([5.0, dk, logit(theta_low), logit(theta_high),
                      log_K, np.log(n)])
    cov_t = np.eye(6) * 0.01
    return GenotypeFit(
        genotype=genotype, titrant_name="iptg", n_obs=100, converged=True,
        param_names_t=["ln_cfu0[0]"] + list(PHENO_PARAMS_TRANSFORMED),
        est_t=est_t, cov_t=cov_t, pheno_slice=slice(1, 6))


def _mean_theta_high(fits, orig_keys):
    # theta_high = expit(est_t[3]) (induced-end occupancy).
    return np.mean([expit(fits[k].est_t[3]) for k in orig_keys])


def _build_fits(rng, n=25):
    fits = {}
    for i in range(n):
        fits[(f"A{i}V", "iptg")] = _make_fit(
            f"A{i}V",
            theta_low=rng.uniform(0.7, 0.98),
            theta_high=rng.uniform(0.02, 0.5),
            log_K=rng.uniform(np.log(1e-3), np.log(0.3)),   # spread -> dynamic range
            n=rng.uniform(0.8, 2.0))
    fits[("wt", "iptg")] = _make_fit("wt", 0.99, 0.01, np.log(0.017), 2.0)
    return fits


_CONCS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
_GROWTH_DF = pd.DataFrame({"titrant_name": ["iptg"] * len(_CONCS),
                          "titrant_conc": _CONCS})


def test_deattenuate_leaves_spiked_untouched_and_corrects_bulk():
    rng = np.random.default_rng(3)
    fits = _build_fits(rng)
    bulk_keys = [k for k in fits if k[0] != "wt"]

    corrected = deattenuate_congression(fits, _GROWTH_DF, lam=1.0, spiked={"wt"})

    # Spiked genotype is congression-free -> identical.
    assert np.allclose(corrected[("wt", "iptg")].est_t,
                       fits[("wt", "iptg")].est_t)
    # Bulk genotypes were corrected.
    assert any(not np.allclose(corrected[k].est_t, fits[k].est_t)
               for k in bulk_keys)
    # De-attenuation pulls the masked induced-end occupancy back down.
    assert _mean_theta_high(corrected, bulk_keys) < _mean_theta_high(fits, bulk_keys)
    # dk_geno (index 1) is untouched by the theta correction.
    for k in bulk_keys:
        assert corrected[k].est_t[1] == fits[k].est_t[1]


def test_deattenuate_lambda_none_is_noop():
    rng = np.random.default_rng(4)
    fits = _build_fits(rng)
    noop = deattenuate_congression(fits, _GROWTH_DF, lam=None, spiked={"wt"})
    for k in fits:
        assert np.allclose(noop[k].est_t, fits[k].est_t)
