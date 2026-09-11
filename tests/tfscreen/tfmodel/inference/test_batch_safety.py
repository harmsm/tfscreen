"""
Tests for tfscreen.tfmodel.inference.batch_safety.

The registry-wide tests guard against per-genotype latents sampled in plates
sized ``data.batch_size``.  Those alias every non-binding genotype onto batch
positions under any numpyro autoguide (AutoDelta/MAP included); see the
module docstring of ``batch_safety`` for the mechanism.
"""
import os

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from flax import struct

from tfscreen.tfmodel.inference.batch_safety import (
    find_batch_dependent_latents,
    find_orchestrator_batch_dependent_latents,
)
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

@struct.dataclass
class _ToyData:
    num_genotype: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)
    batch_idx: jnp.ndarray


def _toy_get_batch(data, idx):
    return _ToyData(num_genotype=data.num_genotype,
                    batch_size=int(idx.shape[0]),
                    batch_idx=idx)


def _toy_full_data(num_genotype=5):
    return _ToyData(num_genotype=num_genotype,
                    batch_size=num_genotype,
                    batch_idx=jnp.arange(num_genotype))


def _batch_sized_model(data, priors):
    numpyro.sample("global_p", dist.Normal(0.0, 1.0))
    with numpyro.plate("bad_genotype_plate", data.batch_size):
        numpyro.sample("bad_offset", dist.Normal(0.0, 1.0))


def _full_sized_model(data, priors):
    numpyro.sample("global_p", dist.Normal(0.0, 1.0))
    with numpyro.plate("good_genotype_plate", data.num_genotype):
        offset = numpyro.sample("good_offset", dist.Normal(0.0, 1.0))
    numpyro.deterministic("per_batch", offset[data.batch_idx])


def test_flags_batch_sized_latent():
    full = _toy_full_data()
    found = find_batch_dependent_latents(_batch_sized_model, None, full,
                                         _toy_get_batch,
                                         jnp.arange(5), jnp.arange(3))
    assert found == {"bad_offset": ((5,), (3,))}


def test_full_sized_latent_is_safe():
    full = _toy_full_data()
    found = find_batch_dependent_latents(_full_sized_model, None, full,
                                         _toy_get_batch,
                                         jnp.arange(5), jnp.arange(3))
    assert found == {}


def test_library_sized_latent_follows_batch_order():
    """
    Each batch position must read its own genotype's offset.  The full-batch
    training index is binding-first and reshuffled every step, so a
    permutation of the whole library is the case that matters.
    """
    from collections import namedtuple
    from numpyro.handlers import seed, substitute, trace
    from tfscreen.tfmodel.generative.components.dk_geno import (
        hierarchical_geno as dk,
    )

    Data = namedtuple("Data", ["num_genotype", "batch_size", "batch_idx",
                               "wt_indexes"])
    batch_idx = jnp.array([3, 0, 4, 1, 2])
    data = Data(num_genotype=5, batch_size=5, batch_idx=batch_idx,
                wt_indexes=jnp.array([0]))
    offsets = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    values = {"dk_geno_offset": offsets,
              "dk_geno_hyper_loc": -3.5,
              "dk_geno_hyper_scale": 0.5,
              "dk_geno_hyper_shift": 0.02}

    model = substitute(seed(dk.define_model, rng_seed=0), data=values)
    tr = trace(model).get_trace("dk_geno", data, dk.get_priors())

    expected = 0.02 - jnp.exp(-3.5 + offsets[batch_idx] * 0.5)
    expected = jnp.where(batch_idx == 0, 0.0, expected)  # wt pinned to 0
    assert jnp.allclose(tr["dk_geno"]["value"], expected)


def test_equal_length_index_sets_rejected():
    full = _toy_full_data()
    with pytest.raises(ValueError, match="different lengths"):
        find_batch_dependent_latents(_full_sized_model, None, full,
                                     _toy_get_batch,
                                     jnp.arange(3), jnp.array([4, 0, 1]))


# ---------------------------------------------------------------------------
# Registry-wide: every per-genotype component must be mini-batch safe
# ---------------------------------------------------------------------------

# One axis varied at a time from ModelOrchestrator's defaults.  dk_geno
# "pinned" is omitted because it needs a pins file; it samples no
# per-genotype latents.
_SAFE_VARIANTS = [
    ("activity", "fixed"),
    ("activity", "hierarchical_geno"),
    ("activity", "horseshoe_geno"),
    ("activity", "hierarchical_mut"),
    ("activity", "horseshoe_mut"),
    ("dk_geno", "fixed"),
    ("dk_geno", "hierarchical_geno"),
    ("ln_cfu0", "hierarchical"),
    ("ln_cfu0", "hierarchical_factored"),
    ("theta", "hill_geno"),
    ("theta", "hill_mut"),
    ("theta", "categorical_geno"),
    ("theta_growth_noise", "zero"),
    ("theta_growth_noise", "logit_normal"),
]


def _owned_by(axis, site_name):
    """True when site_name belongs to the component registered under axis."""
    if not site_name.startswith(f"{axis}_"):
        return False
    # theta's sites share a prefix with the theta noise components.
    if axis == "theta":
        return not site_name.startswith(("theta_growth_noise_",
                                         "theta_binding_noise_"))
    return True


@pytest.mark.parametrize("axis,variant", _SAFE_VARIANTS)
def test_component_latents_are_batch_safe(axis, variant):
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=6,
                                     **{axis: variant})
    found = find_orchestrator_batch_dependent_latents(orchestrator)
    found = {k: v for k, v in found.items() if _owned_by(axis, k)}
    assert found == {}, (
        f"{axis}={variant} has latents sampled at batch size "
        f"(shape with full library, shape with one fewer genotype): {found}"
    )


def test_default_model_is_batch_safe():
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=6)
    assert find_orchestrator_batch_dependent_latents(orchestrator) == {}


@pytest.mark.parametrize("theta", ["hill_geno", "categorical_geno"])
def test_binding_only_latents_are_batch_safe(theta):
    """Binding-only mode mini-batches the binding genotypes themselves."""
    orchestrator = ModelOrchestrator(None, _BINDING_CSV,
                                     binding_only=True,
                                     batch_size=3,
                                     theta=theta)
    assert find_orchestrator_batch_dependent_latents(orchestrator) == {}


def test_beta_growth_noise_is_batch_dependent():
    """
    Documented exception: beta noise's latent is the noisy theta itself,
    drawn per observation from a Beta centred on the batch's theta, so it
    cannot be sampled at library size.  The detector must keep flagging it
    so autoguides can refuse it under mini-batching.
    """
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=6,
                                     theta_growth_noise="beta")
    found = find_orchestrator_batch_dependent_latents(orchestrator)
    found = {k for k in found if _owned_by("theta_growth_noise", k)}
    assert found == {"theta_growth_noise_dist"}
