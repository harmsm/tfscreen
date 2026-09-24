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
    find_batch_order_mismatches,
    find_orchestrator_batch_dependent_latents,
    find_orchestrator_batch_order_mismatches,
    orchestrator_latent_dimension,
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
# per-genotype latents.  The thermo PnnC/PddG theta variants are omitted
# because they need structure/ddG files; they share the PK variants'
# per-mutation latents and differ only in the prior on them.
_SAFE_VARIANTS = [
    ("condition_growth", "linear"),
    ("condition_growth", "power"),
    ("condition_growth", "saturation"),
    ("growth_transition", "instant"),
    ("growth_transition", "memory"),
    ("growth_transition", "baranyi"),
    ("growth_transition", "baranyi_k"),
    ("growth_transition", "baranyi_tau"),
    ("growth_transition", "two_pop"),
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
    ("theta", "_simple"),
    ("theta", "thermo.O2_C4_K3_U0_a.PK"),
    ("theta", "thermo.O2_C4_K3_U1_a.PK"),
    ("theta", "thermo.O2_C12_K5_U0_a.PK"),
    ("theta", "thermo.O2_C12_K5_U1_a.PK"),
    ("theta_rescale", "passthrough"),
    ("theta_rescale", "logit"),
    ("transformation", "single"),
    ("transformation", "mixture"),
    ("theta_growth_noise", "zero"),
    ("theta_growth_noise", "logit_normal"),
    ("theta_binding_noise", "zero"),
    ("theta_binding_noise", "beta"),
    ("growth_noise", "zero"),
    ("growth_noise", "normal_kt"),
    ("sample_offset", "zero"),
    ("sample_offset", "normal"),
]

# Extra constructor arguments some variants require.
_VARIANT_KWARGS = {
    ("transformation", "mixture"): {"transformation_lambda": (1.0, 0.1)},
}


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
                                     **{axis: variant},
                                     **_VARIANT_KWARGS.get((axis, variant), {}))
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


@pytest.mark.parametrize("theta,activity", [
    ("hill_mut", "horseshoe_mut"),
    ("thermo.O2_C4_K3_U0_a.PK", "horseshoe_geno"),
    ("thermo.O2_C12_K5_U1_a.PK", "horseshoe_geno"),
])
def test_epistasis_latents_are_batch_safe(theta, activity):
    """Pair-indexed epistasis latents must be library-sized too."""
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=6,
                                     theta=theta,
                                     activity=activity,
                                     epistasis=True)
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


# ---------------------------------------------------------------------------
# Batch order: predictions must follow batch_idx, not library order
# ---------------------------------------------------------------------------

def _sliced_model(data, priors):
    """Library-sized latent, sliced to the batch: correct."""
    with numpyro.plate("genotype_plate", data.num_genotype):
        offset = numpyro.sample("offset", dist.Normal(0.0, 1.0))
    numpyro.deterministic("growth_pred", offset[data.batch_idx])


def _unsliced_at_full_batch_model(data, priors):
    """Slices only when mini-batching: wrong at a reordered full batch."""
    with numpyro.plate("genotype_plate", data.num_genotype):
        offset = numpyro.sample("offset", dist.Normal(0.0, 1.0))
    if data.batch_size < data.num_genotype:
        offset = offset[data.batch_idx]
    numpyro.deterministic("growth_pred", offset)


def test_order_check_passes_sliced_model():
    found = find_batch_order_mismatches(_sliced_model, None, _toy_full_data(),
                                        _toy_get_batch, jnp.arange(5),
                                        jnp.array([0, 3, 1, 4, 2]))
    assert found == {}


def test_order_check_flags_unsliced_model():
    found = find_batch_order_mismatches(_unsliced_at_full_batch_model, None,
                                        _toy_full_data(), _toy_get_batch,
                                        jnp.arange(5),
                                        jnp.array([0, 3, 1, 4, 2]))
    assert set(found) == {"growth_pred"}
    assert found["growth_pred"] > 0


def test_order_check_needs_reorderings():
    with pytest.raises(ValueError, match="reorderings"):
        find_batch_order_mismatches(_sliced_model, None, _toy_full_data(),
                                    _toy_get_batch, jnp.arange(5),
                                    jnp.array([0, 1, 2, 3, 3]))


# The full-batch training index keeps the binding genotypes first and
# reshuffles the rest every step, so run at full batch (batch_size=None).
# beta *growth* theta noise is the documented exception (tested below).
@pytest.mark.parametrize("axis,variant", _SAFE_VARIANTS)
def test_component_predictions_follow_batch_order(axis, variant):
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     **{axis: variant},
                                     **_VARIANT_KWARGS.get((axis, variant), {}))
    found = find_orchestrator_batch_order_mismatches(orchestrator)
    assert found == {}, (
        f"{axis}={variant}: predictions do not follow a reordered full "
        f"batch (largest abs difference per site): {found}"
    )


@pytest.mark.parametrize("theta,activity", [
    ("hill_mut", "hierarchical_mut"),
    ("hill_mut", "horseshoe_mut"),
    ("thermo.O2_C4_K3_U0_a.PK", "horseshoe_geno"),
])
def test_epistasis_predictions_follow_batch_order(theta, activity):
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     theta=theta,
                                     activity=activity,
                                     epistasis=True)
    assert find_orchestrator_batch_order_mismatches(orchestrator) == {}


@pytest.mark.parametrize("theta", ["hill_geno", "categorical_geno"])
def test_binding_only_predictions_follow_batch_order(theta):
    orchestrator = ModelOrchestrator(None, _BINDING_CSV,
                                     binding_only=True,
                                     theta=theta)
    assert find_orchestrator_batch_order_mismatches(orchestrator) == {}


def test_simple_theta_per_genotype_follows_batch_order():
    """
    The pre-fit's _simple theta holds per-genotype, library-ordered curves.
    It must read them through batch_idx, not by batch position.
    """
    import numpy as np
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     theta="_simple")
    g = orchestrator.data.growth
    values = np.random.default_rng(0).uniform(
        0.05, 0.95, (g.num_titrant_name, g.num_titrant_conc, g.num_genotype))
    priors = orchestrator.priors
    orchestrator._priors = priors.replace(
        theta=priors.theta.replace(theta_values=jnp.asarray(values)))
    assert find_orchestrator_batch_order_mismatches(orchestrator) == {}


def test_beta_growth_noise_is_batch_positional():
    """
    Documented exception: beta noise's latent is the noisy theta itself, one
    draw per batch position, so it does not follow a reordering. The order
    check must keep flagging it.
    """
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     theta_growth_noise="beta")
    assert "growth_pred" in find_orchestrator_batch_order_mismatches(orchestrator)


# ---------------------------------------------------------------------------
# orchestrator_latent_dimension
# ---------------------------------------------------------------------------

def test_latent_dimension_toy():
    """global_p (1) + good_offset (one per library genotype)."""
    from types import SimpleNamespace
    orch = SimpleNamespace(data=_toy_full_data(5), priors=None,
                           jax_model=_full_sized_model,
                           get_batch=_toy_get_batch)
    assert orchestrator_latent_dimension(orch) == 6


def test_latent_dimension_matches_autocontinuous():
    """Matches the latent_dim an AutoMultivariateNormal builds itself."""
    from numpyro.infer.autoguide import AutoMultivariateNormal
    from numpyro.handlers import seed

    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=6)
    num_genotype = orchestrator.data.num_genotype
    data = orchestrator.get_batch(orchestrator.data, jnp.arange(num_genotype))

    guide = AutoMultivariateNormal(orchestrator.jax_model)
    seed(guide, rng_seed=0)(data=data, priors=orchestrator.priors)

    assert orchestrator_latent_dimension(orchestrator) == guide.latent_dim
