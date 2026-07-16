"""Tests for the congression de-attenuation history/trajectory outputs."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy.special import expit, logit

from tfscreen.tfmodel.generative.components.transformation._congression import (
    update_thetas,
)
from tfscreen.tfmodel.genotype_fit.congression import (
    correct_theta_matrix,
    deattenuate_congression,
)
from tfscreen.tfmodel.genotype_fit.fit import GenotypeFit, PHENO_PARAMS_TRANSFORMED


def _forward(theta_true, lam):
    return np.asarray(update_thetas(
        jnp.asarray(theta_true), (float(lam),), theta_dist="empirical",
        population_theta=jnp.asarray(theta_true)))


def _make_fit(genotype, theta_low, theta_high, log_K, n, dk=0.0):
    est_t = np.array([5.0, dk, logit(theta_low), logit(theta_high),
                      log_K, np.log(n)])
    return GenotypeFit(
        genotype=genotype, titrant_name="iptg", n_obs=100, converged=True,
        param_names_t=["ln_cfu0[0]"] + list(PHENO_PARAMS_TRANSFORMED),
        est_t=est_t, cov_t=np.eye(6) * 0.01, pheno_slice=slice(1, 6))


def _build_fits(rng, n=25):
    fits = {}
    for i in range(n):
        fits[(f"A{i}V", "iptg")] = _make_fit(
            f"A{i}V", theta_low=rng.uniform(0.7, 0.98),
            theta_high=rng.uniform(0.02, 0.5),
            log_K=rng.uniform(np.log(1e-3), np.log(0.3)), n=rng.uniform(0.8, 2.0))
    fits[("wt", "iptg")] = _make_fit("wt", 0.99, 0.01, np.log(0.017), 2.0)
    return fits


_CONCS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
_GROWTH_DF = pd.DataFrame({"titrant_name": ["iptg"] * len(_CONCS),
                          "titrant_conc": _CONCS})


def test_correct_theta_matrix_history_matches_final():
    rng = np.random.default_rng(0)
    theta_true = expit(rng.normal(0.0, 1.5, size=(2, 200)))
    theta_obs = _forward(theta_true, lam=1.0)

    rec, n_iter, history = correct_theta_matrix(
        theta_obs, lam=1.0, tol=1e-6, max_iter=200, return_history=True)

    # First entry is the clamped observed matrix; last is the returned estimate.
    assert np.allclose(history[0], np.clip(theta_obs, 1e-6, 1 - 1e-6))
    assert np.allclose(history[-1], rec)
    # Two-return form is unchanged (backward compatible).
    rec2, n_iter2 = correct_theta_matrix(theta_obs, lam=1.0, tol=1e-6,
                                         max_iter=200)
    assert np.allclose(rec2, rec)
    assert n_iter2 == n_iter
    # The trajectory monotonically closes the gap to the observed population.
    err0 = np.nanmax(np.abs(_forward(history[0], 1.0) - theta_obs))
    errN = np.nanmax(np.abs(_forward(history[-1], 1.0) - theta_obs))
    assert errN <= err0


def test_correct_theta_matrix_history_lambda_zero():
    theta_obs = expit(np.random.default_rng(1).normal(0, 1, size=(1, 20)))
    rec, n_iter, history = correct_theta_matrix(theta_obs, lam=0.0,
                                                return_history=True)
    assert n_iter == 0
    assert np.allclose(rec, theta_obs)
    assert len(history) == 1


def test_deattenuate_returns_history_df():
    rng = np.random.default_rng(3)
    fits = _build_fits(rng)
    bulk_keys = [k for k in fits if k[0] != "wt"]

    corrected, history = deattenuate_congression(
        fits, _GROWTH_DF, lam=1.0, spiked={"wt"}, return_theta_history=True)

    assert list(history.columns) == [
        "genotype", "titrant_name", "titrant_conc", "iter", "theta"]
    # History covers the bulk genotypes at every concentration, never wt.
    assert set(history["genotype"]) == {k[0] for k in bulk_keys}
    assert "wt" not in set(history["genotype"])
    assert set(history["titrant_conc"]) == set(_CONCS)
    # iter starts at 0 and the correction actually iterated.
    assert history["iter"].min() == 0
    assert history["iter"].max() >= 1

    # The corrected fits match the no-history call.
    corrected_only = deattenuate_congression(
        fits, _GROWTH_DF, lam=1.0, spiked={"wt"})
    for k in fits:
        assert np.allclose(corrected[k].est_t, corrected_only[k].est_t)


def test_deattenuate_history_noop_when_lambda_none():
    fits = _build_fits(np.random.default_rng(4))
    corrected, history = deattenuate_congression(
        fits, _GROWTH_DF, lam=None, spiked={"wt"}, return_theta_history=True)
    assert history.empty
    for k in fits:
        assert np.allclose(corrected[k].est_t, fits[k].est_t)
