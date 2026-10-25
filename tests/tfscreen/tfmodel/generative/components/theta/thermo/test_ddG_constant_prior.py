"""
U1 equivalence of the Δln K prior and the per-state ΔΔG prior (thermo/prior.py).

In the U1 models the projection P (folded rows plus the unfolding row
Δln K_U = ΔΔG_H) is square and invertible, so the prior on the constants,
Δln K ~ MVN(P · μ, P · diag(σ²) · Pᵀ), is a change of variables from the
per-state prior ΔΔG ~ Normal(μ, σ) (the form of commit c4f03d42, written out
here as a reference).  The Jacobian |det P| is constant, so the two log
densities differ by M · log|det P| everywhere and share their MAP point,
mapped through P⁻¹.
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
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    dlnK_guesses,
    posthoc_ddG,
    sample_dlnK,
)

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_M = 150
_OBS_SD = 0.1
_U1 = ["O2_C4_K3_U1_a", "O2_C12_K5_U1_a"]


def _per_state(name, structs, num_mut, means):
    """Reference: the per-state centered prior of commit c4f03d42."""
    S = len(structs)
    with numpyro.plate(f"{name}_struct_plate", S, dim=-1):
        sigma = numpyro.sample(f"{name}_ddG_sigma",
                               dist.LogNormal(DEFAULT_DDG_SIGMA_LOC,
                                              DEFAULT_DDG_SIGMA_SCALE))
        with numpyro.plate(f"{name}_mut_plate", num_mut, dim=-2):
            return numpyro.sample(f"{name}_ddG", dist.Normal(means, sigma))


def _problem(family, sigma_true=0.5, seed=0):
    module = importlib.import_module(f"{_PKG}.{family}.PddG")
    P = np.asarray(module.DLNK_PROJ)
    structs = list(module.STRUCTURE_NAMES)
    S = len(structs)
    rng = np.random.default_rng(seed)
    table = rng.normal(0, 2, (_M, S)).astype(np.float32)
    truth = table + sigma_true * rng.normal(0, 1, (_M, S)).astype(np.float32)
    obs = truth @ P.T + _OBS_SD * rng.normal(0, 1, (_M, P.shape[0]))
    return P, structs, jnp.asarray(table), jnp.asarray(obs, jnp.float32)


def _observe(dlnK, obs):
    with numpyro.plate("mut", _M, dim=-2), \
         numpyro.plate("constant", obs.shape[1], dim=-1):
        numpyro.sample("obs", dist.Normal(dlnK, _OBS_SD), obs=obs)


@pytest.mark.parametrize("family", _U1)
def test_log_densities_differ_by_the_constant_jacobian(family):
    P, structs, table, _ = _problem(family)
    S = len(structs)
    log_det = np.log(abs(np.linalg.det(P)))

    def per_state():
        _per_state("theta", structs, _M, table)

    def constants():
        sample_dlnK("theta", structs, _M, table, P)

    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(4):
        ddG = jnp.asarray(table + rng.normal(0, 1.5, (_M, S)), jnp.float32)
        sigma = jnp.asarray(np.exp(rng.normal(0, 0.5, S)), jnp.float32)
        lp_state = numpyro.infer.util.log_density(
            per_state, (), {}, {"theta_ddG": ddG, "theta_ddG_sigma": sigma})[0]
        lp_const = numpyro.infer.util.log_density(
            constants, (), {}, {"theta_dlnK": ddG @ jnp.asarray(P.T, jnp.float32),
                                "theta_ddG_sigma": sigma})[0]
        diffs.append(float(lp_state - lp_const))
    np.testing.assert_allclose(diffs, _M * log_det, rtol=1e-4, atol=0.05)


def _map(model, init, steps=4000):
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init))
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    return svi.run(jax.random.PRNGKey(0), steps, progress_bar=False).params


@pytest.mark.parametrize("family", _U1)
def test_map_point_matches_the_per_state_form(family):
    P, structs, table, obs = _problem(family)
    Pj = jnp.asarray(P, jnp.float32)

    def per_state():
        _observe(_per_state("theta", structs, _M, table) @ Pj.T, obs)

    def constants():
        _observe(sample_dlnK("theta", structs, _M, table, P), obs)

    ref = _map(per_state, {"theta_ddG": table,
                           "theta_ddG_sigma": jnp.ones(len(structs))})
    new = _map(constants, dlnK_guesses("theta", table, P))

    sigma_ref = np.asarray(ref["theta_ddG_sigma_auto_loc"])
    sigma_new = np.asarray(new["theta_ddG_sigma_auto_loc"])
    np.testing.assert_allclose(sigma_new, sigma_ref, rtol=2e-3)

    ddG_ref = np.asarray(ref["theta_ddG_auto_loc"])
    ddG_new = np.asarray(posthoc_ddG(new["theta_dlnK_auto_loc"], table,
                                     jnp.asarray(sigma_new), P))
    # Same point up to Adam's residual (typically 1e-6, at most a few 1e-3).
    np.testing.assert_allclose(ddG_new, ddG_ref, atol=5e-3)
    assert np.median(np.abs(ddG_new - ddG_ref)) < 1e-4
