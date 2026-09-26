"""
The congression mixture inside the full model (generative/model.py with
transformation="mixture"), on the smoke-test data.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.generative.registry import model_registry
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")


def _orchestrator(transformation, **kwargs):
    if transformation == "mixture":
        kwargs.setdefault("transformation_lambda", (0.36, 0.05))
    return ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                             transformation=transformation, **kwargs)


def _full_batch(orchestrator):
    return orchestrator.get_batch(orchestrator.data,
                                  jnp.arange(orchestrator.data.num_genotype))


def _trace(orchestrator, latents=None):
    model = seed(orchestrator.jax_model, rng_seed=0)
    if latents is not None:
        model = substitute(model, data=latents)
    return trace(model).get_trace(data=_full_batch(orchestrator),
                                  priors=orchestrator.priors)


def _latents(tr):
    return {name: site["value"] for name, site in tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)}


# ---------------------------------------------------------------------------
# Every growth component broadcasts over the class axis
# ---------------------------------------------------------------------------

_GROWTH_VARIANTS = (
    [("growth_transition", v) for v in sorted(model_registry["growth_transition"])]
    + [("condition_growth", v) for v in sorted(model_registry["condition_growth"])]
)


@pytest.mark.parametrize("axis,variant", _GROWTH_VARIANTS)
def test_mixture_runs_with_every_growth_component(axis, variant):
    mixture = _orchestrator("mixture", **{axis: variant})
    single = _orchestrator("single", **{axis: variant})

    mixture_pred = np.asarray(_trace(mixture)["growth_pred"]["value"])
    single_pred = np.asarray(_trace(single)["growth_pred"]["value"])

    assert mixture_pred.shape == single_pred.shape
    assert np.all(np.isfinite(mixture_pred))


# ---------------------------------------------------------------------------
# Reductions to the single model
# ---------------------------------------------------------------------------

def test_vanishing_lambda_reduces_to_single():
    """
    The congressed weight is about lambda, but with prior draws a congressed
    class can outgrow the clean one by tens of ln units, so lambda must be
    far below exp(-max growth gap) for the mixture to collapse visibly.
    """
    mixture = _orchestrator("mixture")
    single = _orchestrator("single")

    latents = _latents(_trace(mixture))
    latents["transformation_lam"] = jnp.array(1e-30)
    mixture_pred = np.asarray(_trace(mixture, latents)["growth_pred"]["value"])

    latents.pop("transformation_lam")
    single_pred = np.asarray(_trace(single, latents)["growth_pred"]["value"])

    np.testing.assert_allclose(mixture_pred, single_pred, rtol=1e-5, atol=1e-5)


def test_spike_only_genotype_matches_single_and_bulk_genotypes_change():
    """
    A genotype with bulk_fraction = 0 has no congressed cells, so it grows
    as in the single model at any lambda; bulk genotypes do not.
    """
    probe = _orchestrator("single")
    labels = list(probe.growth_tm.tensor_dim_labels[-1])
    spiked = [g for g in labels if g != "wt"][0]

    mixture = _orchestrator("mixture", spiked_genotypes=[spiked])
    single = _orchestrator("single", spiked_genotypes=[spiked])

    latents = _latents(_trace(mixture))
    latents["transformation_lam"] = jnp.array(1.0)
    mixture_pred = np.asarray(_trace(mixture, latents)["growth_pred"]["value"])
    latents.pop("transformation_lam")
    single_pred = np.asarray(_trace(single, latents)["growth_pred"]["value"])

    g = labels.index(spiked)
    np.testing.assert_allclose(mixture_pred[..., g], single_pred[..., g],
                               rtol=1e-5, atol=1e-5)

    bulk = [i for i in range(len(labels)) if i != g]
    assert not np.allclose(mixture_pred[..., bulk], single_pred[..., bulk])


def test_theta_growth_pred_is_the_genotype_own_theta():
    """theta_growth_pred stays uncorrected (tfs-predict-theta reads it)."""
    mixture = _orchestrator("mixture")
    single = _orchestrator("single")
    latents = _latents(_trace(mixture))
    mixture_theta = _trace(mixture, latents)["theta_growth_pred"]["value"]
    latents.pop("transformation_lam")
    single_theta = _trace(single, latents)["theta_growth_pred"]["value"]
    np.testing.assert_allclose(np.asarray(mixture_theta),
                               np.asarray(single_theta))


# ---------------------------------------------------------------------------
# Posterior forward passes over genotype chunks
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mixture_posteriors_independent_of_chunk_size():
    """
    get_posteriors runs the model over genotype chunks; the congressed cells
    still look co-residents up across the whole library, so the saved
    predictions must not depend on the chunk size.
    """
    import tempfile

    import h5py

    from tfscreen.tfmodel.inference.run_inference import RunInference

    orchestrator = _orchestrator("mixture")
    inference = RunInference(model=orchestrator, seed=1)
    svi = inference.setup_svi(adam_step_size=1e-3)
    state, _, _ = inference.run_optimization(
        svi=svi, max_num_epochs=1,
        out_prefix=os.path.join(tempfile.mkdtemp(), "fit"))

    out = {}
    for forward_batch_size in (512, 2):
        prefix = os.path.join(tempfile.mkdtemp(), "p")
        RunInference(model=orchestrator, seed=7).get_posteriors(
            svi=svi, svi_state=state, out_prefix=prefix,
            num_posterior_samples=3, forward_batch_size=forward_batch_size)
        with h5py.File(f"{prefix}_posterior.h5") as f:
            out[forward_batch_size] = f["growth_pred"][()]

    np.testing.assert_allclose(out[512], out[2], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Congression theta rule and activity defaults
# ---------------------------------------------------------------------------

def test_defaults_are_homodimer_and_fixed_activity():
    orchestrator = _orchestrator("mixture")
    assert orchestrator.settings["activity"] == "fixed"
    assert orchestrator.settings["congression_theta_rule"] == "homodimer"
    assert orchestrator.data.growth.congression_theta_rule == "homodimer"
    assert orchestrator.settings["congression_dk_rule"] == "dilution"
    assert orchestrator.settings["congression_dk_alpha"] is None
    assert orchestrator.data.growth.congression_dk_rule == "dilution"


def test_max_rule_allows_learned_activity():
    orchestrator = _orchestrator("mixture", activity="horseshoe_geno",
                                 congression_theta_rule="max")
    assert orchestrator.data.growth.congression_theta_rule == "max"


@pytest.mark.parametrize("rule", ["homodimer", "heterodimer", "max"])
def test_every_rule_runs_and_rule_survives_batching(rule):
    orchestrator = _orchestrator("mixture", congression_theta_rule=rule)
    batch = _full_batch(orchestrator)
    assert batch.growth.congression_theta_rule == rule
    pred = np.asarray(_trace(orchestrator)["growth_pred"]["value"])
    assert np.all(np.isfinite(pred))


def test_rules_give_different_predictions():
    """Same latents, different rules: the congressed classes differ."""
    base = _orchestrator("mixture", congression_theta_rule="max")
    latents = _latents(_trace(base))
    latents["transformation_lam"] = jnp.array(1.0)
    preds = {}
    for rule in ("max", "homodimer", "heterodimer"):
        o = _orchestrator("mixture", congression_theta_rule=rule)
        preds[rule] = np.asarray(_trace(o, latents)["growth_pred"]["value"])
    assert not np.allclose(preds["max"], preds["homodimer"])
    assert not np.allclose(preds["homodimer"], preds["heterodimer"])



@pytest.mark.parametrize("dk_rule,dk_alpha", [("dilution", None),
                                              ("softmin", 100.0),
                                              ("min", None)])
def test_every_dk_rule_runs_and_survives_batching(dk_rule, dk_alpha):
    orchestrator = _orchestrator("mixture", congression_dk_rule=dk_rule,
                                 congression_dk_alpha=dk_alpha)
    batch = _full_batch(orchestrator)
    assert batch.growth.congression_dk_rule == dk_rule
    assert batch.growth.congression_dk_alpha == dk_alpha
    pred = np.asarray(_trace(orchestrator)["growth_pred"]["value"])
    assert np.all(np.isfinite(pred))


def test_dk_rules_give_different_predictions():
    """Same latents, different dk rules: the congressed classes differ."""
    base = _orchestrator("mixture")
    latents = _latents(_trace(base))
    latents["transformation_lam"] = jnp.array(1.0)
    preds = {}
    for dk_rule, dk_alpha in (("dilution", None), ("min", None)):
        o = _orchestrator("mixture", congression_dk_rule=dk_rule,
                          congression_dk_alpha=dk_alpha)
        preds[dk_rule] = np.asarray(_trace(o, latents)["growth_pred"]["value"])
    assert not np.allclose(preds["dilution"], preds["min"])
