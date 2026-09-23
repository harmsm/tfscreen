"""
Library-ordered per-genotype values from the dk_geno and activity components
(``return_population=True``), for the congression mixture's co-resident
lookup. See ``generative/components/_population.py``.
"""

import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.generative.components._population import per_genotype
from tfscreen.tfmodel.generative.components.dk_geno import pinned
from tfscreen.tfmodel.generative.registry import model_registry
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")


# ---------------------------------------------------------------------------
# per_genotype helper
# ---------------------------------------------------------------------------

def _double_plus_index(x, genotype_idx):
    return 2.0 * x + genotype_idx


def test_per_genotype_library_sized():
    data = SimpleNamespace(num_genotype=4, batch_idx=jnp.array([2, 0]))
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    batch, population = per_genotype(_double_plus_index, [x], data,
                                     return_population=True)
    np.testing.assert_allclose(population, [2.0, 5.0, 8.0, 11.0])
    np.testing.assert_allclose(batch, population[jnp.array([2, 0])])


def test_per_genotype_default_is_batch_only():
    data = SimpleNamespace(num_genotype=4, batch_idx=jnp.array([2, 0]))
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    batch, population = per_genotype(_double_plus_index, [x], data)
    assert population is None
    np.testing.assert_allclose(batch, [8.0, 2.0])


def test_per_genotype_substituted_latent_has_no_population():
    """A batch-sized substitution says nothing about the rest of the library."""
    data = SimpleNamespace(num_genotype=4, batch_idx=jnp.array([2, 0]))
    x = jnp.array([3.0, 1.0])               # already sliced to the batch
    batch, population = per_genotype(_double_plus_index, [x], data,
                                     return_population=True)
    assert population is None
    np.testing.assert_allclose(batch, [8.0, 2.0])


def test_per_genotype_mixed_latents_slice_library_sized_one():
    data = SimpleNamespace(num_genotype=4, batch_idx=jnp.array([2, 0]))
    library = jnp.array([1.0, 2.0, 3.0, 4.0])
    sliced = jnp.array([10.0, 20.0])
    batch, population = per_genotype(lambda a, b, idx: a + b, [library, sliced],
                                     data, return_population=True)
    assert population is None
    np.testing.assert_allclose(batch, [13.0, 21.0])


# ---------------------------------------------------------------------------
# Every dk_geno and activity component
# ---------------------------------------------------------------------------

# Every registered dk_geno and activity variant, so a new component is
# covered automatically. dk_geno "pinned" needs a pins file to build an
# orchestrator and is tested on its own below.
_VARIANTS = [(axis, variant)
             for axis in ("dk_geno", "activity")
             for variant in sorted(model_registry[axis])
             if (axis, variant) != ("dk_geno", "pinned")]

# wt's pinned value.
_WT_VALUE = {"dk_geno": 0.0, "activity": 1.0}


@pytest.fixture(scope="module")
def orchestrators():
    return {v: ModelOrchestrator(growth_df=_GROWTH_CSV,
                                 binding_df=_BINDING_CSV,
                                 **{v[0]: v[1]})
            for v in _VARIANTS}


def _shuffled_full_batch(orchestrator, seed_value=0):
    data = orchestrator.data
    pinned_idx = np.asarray(data.batch_idx)[:data.num_binding]
    rest = np.random.default_rng(seed_value).permutation(
        np.asarray(data.not_binding_idx))
    return np.concatenate([pinned_idx, rest])


def _run(fn, name, data, priors, **kwargs):
    """Run a component under a fixed seed; return (trace, output)."""
    box = {}

    def call():
        box["out"] = fn(name, data, priors, **kwargs)

    tr = trace(seed(call, rng_seed=0)).get_trace()
    return tr, box["out"]


@pytest.mark.parametrize("axis,variant", _VARIANTS)
@pytest.mark.parametrize("which", ["define_model", "guide"])
@pytest.mark.parametrize("batch", ["full", "mini"])
def test_population_matches_batch(orchestrators, axis, variant, which, batch):
    orchestrator = orchestrators[(axis, variant)]
    idx = _shuffled_full_batch(orchestrator)
    if batch == "mini":
        idx = idx[:orchestrator.data.num_binding + 3]
    data = orchestrator.get_batch(orchestrator.data, jnp.asarray(idx)).growth
    priors = getattr(orchestrator.priors.growth, axis)
    fn = getattr(model_registry[axis][variant], which)

    _, (tensor, population) = _run(fn, axis, data, priors,
                                   return_population=True)
    _, default_tensor = _run(fn, axis, data, priors)

    num_genotype = orchestrator.data.num_genotype
    assert population.shape == (num_genotype,)
    batch_values = np.asarray(tensor).reshape(-1)
    np.testing.assert_allclose(batch_values,
                               np.asarray(population)[idx], rtol=1e-6)
    # The default call is unchanged: same batch values, no population.
    np.testing.assert_allclose(np.asarray(default_tensor), np.asarray(tensor),
                               rtol=1e-6)
    wt = np.asarray(data.wt_indexes)
    np.testing.assert_allclose(np.asarray(population)[wt], _WT_VALUE[axis])


# Components whose per-genotype latents are sampled at library size and so
# can arrive as a batch-sized substitution in the posterior forward pass.
_SUBSTITUTABLE = {
    ("dk_geno", "hierarchical_geno"): ["dk_geno_offset"],
    ("activity", "hierarchical_geno"): ["activity_offset"],
    ("activity", "horseshoe_geno"): ["activity_local_scale", "activity_offset"],
}


@pytest.mark.parametrize("axis,variant", list(_SUBSTITUTABLE))
def test_substituted_latents_give_no_population(orchestrators, axis, variant):
    orchestrator = orchestrators[(axis, variant)]
    idx = _shuffled_full_batch(orchestrator)[:orchestrator.data.num_binding + 3]
    data = orchestrator.get_batch(orchestrator.data, jnp.asarray(idx)).growth
    priors = getattr(orchestrator.priors.growth, axis)
    fn = model_registry[axis][variant].define_model

    library_trace, _ = _run(fn, axis, data, priors)
    values = {site: library_trace[site]["value"][..., idx]
              for site in _SUBSTITUTABLE[(axis, variant)]}

    def call():
        return fn(axis, data, priors, return_population=True)

    tensor, population = substitute(seed(call, rng_seed=0), data=values)()
    assert population is None
    assert np.asarray(tensor).shape[-1] == len(idx)


@pytest.mark.parametrize("which", ["define_model", "guide"])
def test_pinned_dk_geno_population(which):
    values = jnp.array([0.5, -0.01, -0.02, 0.03])
    data = SimpleNamespace(num_genotype=4, batch_idx=jnp.array([3, 0, 2]),
                           wt_indexes=jnp.array([0]))
    priors = pinned.get_priors(dk_geno_values=values)
    tensor, population = getattr(pinned, which)("dk_geno", data, priors,
                                                return_population=True)
    np.testing.assert_allclose(population, [0.0, -0.01, -0.02, 0.03])
    np.testing.assert_allclose(np.asarray(tensor).reshape(-1),
                               np.asarray(population)[[3, 0, 2]])
