"""
MAP regression test for the structural prior's σ_s (thermo/prior.py).

History.  Until 2026-10-11 σ_s was a ``pyro.param`` and ΔΔG was written
``prior_mean + σ_s · offset``; MAP ran σ_s to several kT whatever the truth.
Commit c4f03d42 made σ_s a sample site and sampled per-state ΔΔG centered on
the prior means.  That fixed the U1 models, but in the U0 models (no
unfolded state) a shift equal across all states changes no constant, and
joint MAP used it to collapse one state's σ_s: on C4_U0 with true σ_s = 0.3
kT, σ_LE2 went to 0.001.

Now the prior sits on the constants, Δln K ~ MVN(P · table, P · diag(σ²) ·
Pᵀ).  We simulate per-state ΔΔG = table + Normal(0, σ) for a few hundred
mutations, observe the model's constants Δln K = P · ΔΔG with small noise,
fit by MAP with AutoDelta, and check that every σ_s comes back near the
truth (no collapse, U0 included) and that the fitted Δln K beat both a
flat-prior (PK-like) fit and the table alone.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value

from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    dlnK_guesses,
    sample_dlnK,
)

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_M = 300          # mutations
_OBS_SD = 0.05    # SD of each observed Δln K
_STEPS = 3000


def _projection(family):
    """(C, S) projection the PddG model samples Δln K through."""
    module = importlib.import_module(f"{_PKG}.{family}.PddG")
    return np.asarray(module.DLNK_PROJ), list(module.STRUCTURE_NAMES)


def _simulate(family, sigma_true, seed=0):
    proj, structs = _projection(family)
    S = len(structs)
    rng = np.random.default_rng(seed)
    table = rng.normal(0, 2, (_M, S)).astype(np.float32)
    truth = table + sigma_true * rng.normal(0, 1, (_M, S)).astype(np.float32)
    true_dlnK = truth @ proj.T
    obs = true_dlnK + _OBS_SD * rng.normal(0, 1, true_dlnK.shape)
    return (proj, structs, jnp.asarray(table), true_dlnK,
            jnp.asarray(obs, dtype=jnp.float32))


def _map(model, init):
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init))
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    return svi.run(jax.random.PRNGKey(0), _STEPS, progress_bar=False).params


def _observe(dlnK, obs):
    with numpyro.plate("mut", _M, dim=-2), \
         numpyro.plate("constant", obs.shape[1], dim=-1):
        numpyro.sample("obs", dist.Normal(dlnK, _OBS_SD), obs=obs)


def _fit_structural(proj, structs, table, obs):
    def model():
        _observe(sample_dlnK("theta", structs, _M, table, proj), obs)

    p = _map(model, dlnK_guesses("theta", table, proj))
    return (np.asarray(p["theta_dlnK_auto_loc"]),
            np.asarray(p["theta_ddG_sigma_auto_loc"]))


def _fit_flat(obs):
    """A PK-like fit: Δln K under a wide prior, no structural means."""
    def model():
        with numpyro.plate("constant", obs.shape[1], dim=-1), \
             numpyro.plate("mut_dlnK", _M, dim=-2):
            dlnK = numpyro.sample("theta_dlnK", dist.Normal(0.0, 10.0))
        _observe(dlnK, obs)

    p = _map(model, {"theta_dlnK": jnp.zeros(obs.shape)})
    return np.asarray(p["theta_dlnK_auto_loc"])


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.parametrize("family, sigma_true", [
    ("O2_C4_K3_U1_a", 1.0),
    ("O2_C4_K3_U1_a", 0.3),
    ("O2_C4_K3_U0_a", 1.0),
    ("O2_C4_K3_U0_a", 0.3),     # collapsed σ_LE2 to 0.001 in the per-state form
    ("O2_C12_K5_U0_a", 1.0),
    ("O2_C12_K5_U0_a", 0.3),
])
def test_map_recovers_sigma_and_shrinks_dlnK(family, sigma_true):
    proj, structs, table, true_dlnK, obs = _simulate(family, sigma_true)
    dlnK, sigma = _fit_structural(proj, structs, table, obs)
    flat = _fit_flat(obs)

    # Every σ_s lands near the truth: no state collapses.
    assert sigma.shape == (len(structs),)
    np.testing.assert_array_less(sigma, 1.6 * sigma_true)
    np.testing.assert_array_less(sigma_true / 1.6, sigma)

    # The fitted Δln K beat the table alone and are no worse than the flat
    # fit.  At σ = 1 kT the constants' prior SD dwarfs the 0.05 data noise,
    # so shrinkage is ~1e-4 of the error and the two fits tie; at 0.3 kT
    # the shrinkage is a real (if small) gain.
    table_dlnK = np.asarray(table) @ proj.T
    err, flat_err = _rmse(dlnK, true_dlnK), _rmse(flat, true_dlnK)
    assert err < _rmse(table_dlnK, true_dlnK)
    if sigma_true < 0.5:
        assert err < flat_err
    else:
        assert err < 1.002 * flat_err
