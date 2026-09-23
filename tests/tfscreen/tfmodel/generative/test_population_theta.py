"""
Full-population theta must be library-ordered for every theta component.

``generative/model.py`` evaluates theta for every genotype (the congression
background) by calling the theta component's ``run_model`` on the growth
data with ``batch_idx = geno_theta_idx = arange(num_genotype)``. The
congression mixture looks co-resident genotypes up in that result by library
index, so entry g must be genotype g, whatever batch (full and reshuffled, or
a mini-batch) the latents were sampled in.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")

# The thermo PnnC/PddG variants need structure/ddG files; they share the PK
# variants' run_model.
_THETA_VARIANTS = [
    "hill_geno",
    "hill_mut",
    "categorical_geno",
    "_simple",
    "thermo.O2_C4_K3_U0_a.PK",
    "thermo.O2_C4_K3_U1_a.PK",
    "thermo.O2_C12_K5_U0_a.PK",
    "thermo.O2_C12_K5_U1_a.PK",
]


def _orchestrator(theta):
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     theta=theta)
    if theta == "_simple":
        # Per-genotype curves, so genotype order is visible.
        g = orchestrator.data.growth
        values = np.random.default_rng(0).uniform(
            0.05, 0.95,
            (g.num_titrant_name, g.num_titrant_conc, g.num_genotype))
        priors = orchestrator.priors
        orchestrator._priors = priors.replace(
            theta=priors.theta.replace(theta_values=jnp.asarray(values)))
    return orchestrator


def _shuffled_index(orchestrator, mini):
    data = orchestrator.data
    pinned = np.asarray(data.batch_idx)[:data.num_binding]
    rest = np.random.default_rng(1).permutation(
        np.asarray(data.not_binding_idx))
    idx = np.concatenate([pinned, rest])
    return idx[:data.num_binding + 3] if mini else idx


@pytest.mark.parametrize("theta", _THETA_VARIANTS)
@pytest.mark.parametrize("mini", [False, True])
@pytest.mark.parametrize("which", ["model", "guide"])
def test_population_theta_is_library_ordered(theta, mini, which):
    orchestrator = _orchestrator(theta)
    if which == "guide":
        # The guide model is jax_model partially applied to the guide control.
        control = orchestrator.jax_model_guide.keywords
    else:
        control = orchestrator.main_control_kwargs
    sample_theta, calc_theta, _ = control["theta"]

    idx = _shuffled_index(orchestrator, mini)
    growth = orchestrator.get_batch(orchestrator.data, jnp.asarray(idx)).growth

    theta_param = seed(sample_theta, rng_seed=0)("theta", growth,
                                                 orchestrator.priors.theta)
    batch_theta = np.asarray(calc_theta(theta_param, growth))

    num_genotype = growth.num_genotype
    full = jnp.arange(num_genotype)
    population_data = growth.replace(batch_idx=full, geno_theta_idx=full)
    population_theta = np.asarray(calc_theta(theta_param, population_data))

    assert population_theta.shape[-1] == num_genotype
    np.testing.assert_allclose(population_theta[..., idx], batch_theta,
                               rtol=1e-6, atol=1e-7)
