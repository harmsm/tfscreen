"""
noise/genotype_offset: one offset per binding genotype on the binding side
(joint-fit plan step 4b).
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.analysis.extraction import ExtractionContext
from tfscreen.tfmodel.generative.components.noise import genotype_offset as go
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "..", "..", "smoke-tests", "test_data")
_GROWTH = os.path.join(_SMOKE, "growth-smoke.csv")
_BINDING = os.path.join(_SMOKE, "binding-smoke.csv")
NAME = "theta_binding_noise"


def _run(fn, *args, values=None, **kwargs):
    if values:
        fn = substitute(fn, data=values)
    return trace(seed(fn, rng_seed=0)).get_trace(*args, **kwargs)


class _Data:
    def __init__(self, num_genotype, ordinal):
        self.num_genotype = num_genotype
        self.binding_ordinal = jnp.asarray(ordinal)


def test_adds_one_offset_per_binding_genotype():
    fx = jnp.full((1, 3, 4), 0.5)
    data = _Data(4, [0, 1, 2, 3])
    z = jnp.array([1.0, -1.0, 0.5, 0.0])
    tr = _run(go.define_model, NAME, fx, go.get_priors(), data,
              values={f"{NAME}_offset_scale": jnp.asarray(0.2),
                      f"{NAME}_offset_z": z})
    out = tr[f"{NAME}_offset"]["value"]
    np.testing.assert_allclose(out, 0.2 * z)
    assert tr[f"{NAME}_offset_z"]["value"].shape == (4,)
    res = substitute(seed(go.define_model, 0),
                     data={f"{NAME}_offset_scale": jnp.asarray(0.2),
                           f"{NAME}_offset_z": z})(NAME, fx, go.get_priors(), data)
    # The same offset at every concentration.
    np.testing.assert_allclose(res, 0.5 + 0.2 * np.broadcast_to(z, (1, 3, 4)))


def test_ordinal_picks_the_batch_genotypes():
    """Binding-only mini-batch: offsets are library(binding)-sized and the
    batch's positions read them through binding_ordinal."""
    fx = jnp.zeros((1, 2, 2))
    data = _Data(5, [3, 1])
    z = jnp.arange(5.0)
    res = substitute(seed(go.define_model, 0),
                     data={f"{NAME}_offset_scale": jnp.asarray(1.0),
                           f"{NAME}_offset_z": z})(NAME, fx, go.get_priors(), data)
    np.testing.assert_allclose(res[0, 0], [3.0, 1.0])


def test_fixed_scale_is_deterministic_and_guide_has_no_tau():
    priors = go.ModelPriors(offset_scale_scale=0.2, offset_scale_fixed=0.1)
    fx = jnp.zeros((1, 1, 3))
    data = _Data(3, [0, 1, 2])
    tr = _run(go.define_model, NAME, fx, priors, data)
    assert tr[f"{NAME}_offset_scale"]["type"] == "deterministic"
    assert float(tr[f"{NAME}_offset_scale"]["value"]) == pytest.approx(0.1)
    g = _run(go.guide, NAME, fx, priors, data)
    assert f"{NAME}_offset_scale" not in g
    assert f"{NAME}_offset_scale_loc" not in g
    assert g[f"{NAME}_offset_z"]["value"].shape == (3,)


def _orchestrator(**kwargs):
    return ModelOrchestrator(growth_df=_GROWTH, binding_df=_BINDING,
                             theta="hill_geno",
                             theta_binding_noise="genotype_offset", **kwargs)


def _model_trace(orch, values=None):
    fn = orch.jax_model if values is None else substitute(orch.jax_model, data=values)
    batch = orch.get_batch(orch.data, np.asarray(orch.data.batch_idx))
    return trace(seed(fn, rng_seed=0)).get_trace(data=batch, priors=orch.priors)


def test_joint_model_offsets_binding_not_growth():
    orch = _orchestrator()
    nb = orch.data.num_binding
    base = _model_trace(orch)
    assert base[f"{NAME}_offset_z"]["value"].shape == (nb,)
    fixed = {k: v["value"] for k, v in base.items()
             if v["type"] == "sample" and not v.get("is_observed", False)}
    z = jnp.linspace(-1.0, 1.0, nb)
    t0 = _model_trace(orch, {**fixed, f"{NAME}_offset_z": jnp.zeros(nb)})
    t1 = _model_trace(orch, {**fixed, f"{NAME}_offset_z": z})
    tau = float(t1[f"{NAME}_offset_scale"]["value"])
    np.testing.assert_allclose(
        np.asarray(t1["binding_pred"]["value"]),
        np.asarray(t0["binding_pred"]["value"]) + tau * np.asarray(z),
        rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(t1["theta_growth_pred"]["value"],
                               t0["theta_growth_pred"]["value"])
    np.testing.assert_allclose(t1["growth_pred"]["value"],
                               t0["growth_pred"]["value"])


def test_extract_specs_label_binding_genotypes():
    ctx = ExtractionContext(growth_tm=None, mut_labels=[], pair_labels=[],
                            growth_shares_replicates=False,
                            binding_genotypes=["wt", "M42I"])
    specs = go.get_extract_specs(ctx)
    assert [s["params_to_get"] for s in specs] == [["offset_scale"], ["offset"]]
    assert list(specs[1]["input_df"]["genotype"]) == ["wt", "M42I"]
    ctx = ExtractionContext(growth_tm=None, mut_labels=[], pair_labels=[],
                            growth_shares_replicates=False)
    assert len(go.get_extract_specs(ctx)) == 1


def test_binding_only_minibatch_reads_its_genotypes():
    orch = ModelOrchestrator(None, _BINDING, binding_only=True,
                             theta="hill_geno",
                             theta_binding_noise="genotype_offset")
    n = orch.data.binding.num_genotype
    idx = jnp.array([n - 1, 0])
    batch = orch.get_batch(orch.data, idx)
    np.testing.assert_array_equal(np.asarray(batch.binding.binding_ordinal), [n - 1, 0])
    z = jnp.arange(float(n))
    tr = trace(seed(substitute(orch.jax_model,
                               data={f"{NAME}_offset_z": z,
                                     f"{NAME}_offset_scale": jnp.asarray(1.0)}),
                    rng_seed=0)).get_trace(data=batch, priors=orch.priors)
    diff = (np.asarray(tr["binding_pred"]["value"])
            - np.asarray(tr["theta_binding_pred"]["value"]))
    np.testing.assert_allclose(diff[0, 0], [n - 1, 0.0], atol=1e-6)
