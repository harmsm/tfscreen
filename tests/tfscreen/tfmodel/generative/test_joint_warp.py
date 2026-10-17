"""
The joint fit's model pieces together (joint-fit plan step 4c): hill_geno's
absolute theta, the congression mixture's partition-function rule, linear
growth's m_warp on one condition, the growth titrant shift and the binding
genotype offsets.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.analysis.extraction import ExtractionContext
from tfscreen.tfmodel.generative.components.growth import linear
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "smoke-tests", "test_data")
LOG_A = "condition_growth_warp_log_a"
X0 = "condition_growth_warp_x0"


def _orchestrator(warp):
    orch = ModelOrchestrator(
        growth_df=os.path.join(_SMOKE, "growth-smoke.csv"),
        binding_df=os.path.join(_SMOKE, "binding-smoke.csv"),
        theta="hill_geno", transformation="mixture",
        transformation_lambda=(0.36, 0.05),
        congression_theta_rule="homodimer",
        theta_binding_noise="genotype_offset")
    p = orch.priors
    n = orch.data.growth.num_condition_rep
    flags = tuple(i == warp for i in range(n)) if warp is not None else False
    orch._priors = p.replace(
        theta=p.theta.replace(theta_growth_log_conc_shift_scale=1.0),
        growth=p.growth.replace(condition_growth=p.growth.condition_growth.replace(
            m_warp=flags)))
    return orch


def _trace(orch, values=None):
    fn = orch.jax_model if values is None else substitute(orch.jax_model, data=values)
    batch = orch.get_batch(orch.data, np.asarray(orch.data.batch_idx))
    return trace(seed(fn, rng_seed=0)).get_trace(data=batch, priors=orch.priors)


def _latents(tr):
    return {k: v["value"] for k, v in tr.items()
            if v["type"] == "sample" and not v.get("is_observed", False)}


def _warped_index(orch):
    """A selection condition (its growth depends on theta)."""
    df = orch.growth_tm.map_groups["condition_rep"].sort_values("map_condition_rep")
    labels = df["condition_rep"].astype(str).tolist()
    return next(i for i, c in enumerate(labels) if "+" in c)


def test_all_pieces_trace_together():
    orch = _orchestrator(None)
    w = _warped_index(orch)
    orch = _orchestrator(w)
    tr = _trace(orch)
    assert tr[LOG_A]["value"].shape == (1,)
    assert tr[X0]["value"].shape == (1,)
    assert "theta_growth_log_conc_shift" in tr
    assert tr["theta_binding_noise_offset_z"]["value"].shape == (orch.data.num_binding,)
    assert np.all(np.isfinite(np.asarray(tr["growth_pred"]["value"])))


def test_near_linear_warp_matches_no_warp_through_the_mixture():
    plain = _orchestrator(None)
    w = _warped_index(plain)
    warped = _orchestrator(w)
    values = _latents(_trace(plain))
    # a = e^-4: close to linear, and far enough from 0 that float32 keeps
    # the warp's differences of logistics (at e^-12 they cancel to ~4%).
    near = _trace(warped, {**values, LOG_A: jnp.array([-4.0]),
                           X0: jnp.array([0.5])})
    base = _trace(plain, values)
    np.testing.assert_allclose(np.asarray(near["growth_pred"]["value"]),
                               np.asarray(base["growth_pred"]["value"]),
                               rtol=1e-4, atol=1e-4)

    steep = _trace(warped, {**values, LOG_A: jnp.array([2.5]),
                            X0: jnp.array([0.5])})
    diff = np.abs(np.asarray(steep["growth_pred"]["value"])
                  - np.asarray(base["growth_pred"]["value"]))
    assert diff.max() > 1e-3
    # Binding never sees the warp.
    np.testing.assert_allclose(steep["binding_pred"]["value"],
                               base["binding_pred"]["value"])


def test_warp_extract_spec_names_the_warped_condition():
    orch = _orchestrator(None)
    w = _warped_index(orch)
    orch = _orchestrator(w)
    ctx = ExtractionContext(growth_tm=orch.growth_tm, mut_labels=[],
                            pair_labels=[],
                            growth_shares_replicates=orch._growth_shares_replicates,
                            condition_growth_priors=orch.priors.growth.condition_growth)
    specs = linear.get_extract_specs(ctx)
    assert len(specs) == 2
    warp = specs[1]
    assert warp["params_to_get"] == ["warp_log_a", "warp_x0"]
    df = orch.growth_tm.map_groups["condition_rep"].sort_values("map_condition_rep")
    assert len(warp["input_df"]) == 1
    assert (warp["input_df"]["map_condition_rep"].iloc[0]
            == df["map_condition_rep"].iloc[w])

    plain_ctx = ExtractionContext(growth_tm=orch.growth_tm, mut_labels=[],
                                  pair_labels=[], growth_shares_replicates=False,
                                  condition_growth_priors=linear.get_priors())
    assert len(linear.get_extract_specs(plain_ctx)) == 1


@pytest.mark.parametrize("rule", ["homodimer", "heterodimer"])
def test_partition_rules_accept_hill_geno(rule):
    orch = ModelOrchestrator(
        growth_df=os.path.join(_SMOKE, "growth-smoke.csv"),
        binding_df=os.path.join(_SMOKE, "binding-smoke.csv"),
        theta="hill_geno", transformation="mixture",
        transformation_lambda=(0.36, 0.05), congression_theta_rule=rule)
    assert np.all(np.isfinite(np.asarray(_trace(orch)["growth_pred"]["value"])))
