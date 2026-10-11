"""
MAP regression test for the per-structure ΔΔG scale σ_s (thermo/prior.py).

Until 2026-10-11 σ_s was a ``pyro.param`` and ΔΔG was written
``prior_mean + σ_s · offset``.  Under MAP that is degenerate: raising σ_s
while shrinking the offsets costs the likelihood nothing, so σ_s ran to
several kT whatever the truth and the structural prior went flat.  On this
problem the old form fit σ_s ≈ 2-6 kT for a true σ_s of 0.3 or 1 kT.

Here σ_s is a sample site and ΔΔG is centered on the prior means.  We
simulate ΔΔG = table + Normal(0, σ) for a few hundred mutations, observe
the model's projections (Δln K, and Δln K_U for a U1 model) with small
noise, fit by MAP with AutoDelta, and check that σ_s comes back near the
truth and that the fitted ΔΔG beat a flat-prior (PK-like) fit.
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
    ddG_guesses,
    sample_ddG,
)

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_M = 300          # mutations
_OBS_SD = 0.05    # SD of each observed Δln K
_STEPS = 3000


def _projection(family):
    """(S, K) matrix from ΔΔG to every constant the model's data see."""
    module = importlib.import_module(f"{_PKG}.{family}.PddG")
    rows = [np.asarray(module._PROJ)]
    if hasattr(module, "_PROJ_U"):
        rows.append(np.asarray(module._PROJ_U).reshape(1, -1))
    return jnp.asarray(np.vstack(rows).T), list(module.STRUCTURE_NAMES)


def _simulate(family, sigma_true, seed=0):
    proj, structs = _projection(family)
    S = len(structs)
    rng = np.random.default_rng(seed)
    table = jnp.asarray(rng.normal(0, 2, (_M, S)), dtype=jnp.float32)
    truth = table + sigma_true * jnp.asarray(rng.normal(0, 1, (_M, S)),
                                             dtype=jnp.float32)
    obs = truth @ proj + _OBS_SD * jnp.asarray(
        rng.normal(0, 1, (_M, proj.shape[1])), dtype=jnp.float32)
    return proj, structs, table, truth, obs


def _map(model, init):
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init))
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    return svi.run(jax.random.PRNGKey(0), _STEPS, progress_bar=False).params


def _observe(ddG, proj, obs):
    with numpyro.plate("mut", _M, dim=-2), \
         numpyro.plate("constant", proj.shape[1], dim=-1):
        numpyro.sample("obs", dist.Normal(ddG @ proj, _OBS_SD), obs=obs)


def _fit_structural(proj, structs, table, obs):
    def model():
        _observe(sample_ddG("theta", structs, _M, table), proj, obs)

    p = _map(model, ddG_guesses("theta", table))
    return (np.asarray(p["theta_ddG_auto_loc"]),
            np.asarray(p["theta_ddG_sigma_auto_loc"]))


def _fit_flat(proj, structs, obs):
    """A PK-like fit: ΔΔG under a wide prior, no structural means."""
    S = len(structs)

    def model():
        with numpyro.plate("struct", S, dim=-1), \
             numpyro.plate("mut_ddG", _M, dim=-2):
            ddG = numpyro.sample("theta_ddG", dist.Normal(0.0, 10.0))
        _observe(ddG, proj, obs)

    p = _map(model, {"theta_ddG": jnp.zeros((_M, S))})
    return np.asarray(p["theta_ddG_auto_loc"])


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.parametrize("family, sigma_true", [
    ("O2_C4_K3_U1_a", 1.0),
    ("O2_C4_K3_U1_a", 0.3),
    ("O2_C4_K3_U0_a", 1.0),
])
def test_map_recovers_sigma_and_shrinks_ddG(family, sigma_true):
    proj, structs, table, truth, obs = _simulate(family, sigma_true)
    ddG, sigma = _fit_structural(proj, structs, table, obs)
    flat = _fit_flat(proj, structs, obs)

    # σ_s per structure lands near the truth (the old pyro.param form gave
    # 2-6 kT here for either truth).
    assert sigma.shape == (len(structs),)
    np.testing.assert_array_less(sigma, 1.6 * sigma_true)
    np.testing.assert_array_less(sigma_true / 1.6, sigma)

    # The fitted ΔΔG are shrunk toward the table: closer to it than the
    # flat fit, and closer to the truth than either the flat fit or the
    # table alone.
    assert _rmse(ddG, table) < _rmse(flat, table)
    assert _rmse(ddG, truth) < _rmse(flat, truth)
    assert _rmse(ddG, truth) < _rmse(table, truth)
