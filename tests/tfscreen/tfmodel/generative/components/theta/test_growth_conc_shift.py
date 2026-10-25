"""
hill_geno's growth titrant shift log s (joint-fit plan step 4a): growth
sees each curve at s times the nominal titrant, binding at the nominal one.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.tfmodel.analysis.extraction import ExtractionContext
from tfscreen.tfmodel.generative.components.theta import hill_geno
from tfscreen.tfmodel.generative.model import growth_titrant
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "..", "..", "smoke-tests", "test_data")
SITE = "theta_growth_log_conc_shift"


def _orchestrator(scale):
    orch = ModelOrchestrator(growth_df=os.path.join(_SMOKE, "growth-smoke.csv"),
                             binding_df=os.path.join(_SMOKE, "binding-smoke.csv"),
                             theta="hill_geno")
    if scale:
        p = orch.priors
        orch._priors = p.replace(theta=p.theta.replace(
            theta_growth_log_conc_shift_scale=scale))
    return orch


def _trace(orch, which="model", values=None):
    fn = orch.jax_model if which == "model" else orch.jax_model_guide
    if values:
        fn = substitute(fn, data=values)
    batch = orch.get_batch(orch.data, np.asarray(orch.data.batch_idx))
    return trace(seed(fn, rng_seed=0)).get_trace(data=batch, priors=orch.priors)


def test_off_by_default():
    orch = _orchestrator(0.0)
    assert orch.priors.theta.theta_growth_log_conc_shift_scale == 0.0
    for which in ("model", "guide"):
        assert SITE not in _trace(orch, which)


def test_on_samples_one_global_site():
    orch = _orchestrator(1.0)
    for which in ("model", "guide"):
        tr = _trace(orch, which)
        assert tr[SITE]["type"] == "sample"
        assert np.shape(tr[SITE]["value"]) == ()


def test_shift_moves_growth_theta_and_not_binding():
    orch = _orchestrator(1.0)
    base = _trace(orch)
    fixed = {k: v["value"] for k, v in base.items()
             if v["type"] == "sample" and not v.get("is_observed", False)}
    t0 = _trace(orch, values={**fixed, SITE: jnp.asarray(0.0)})
    t2 = _trace(orch, values={**fixed, SITE: jnp.asarray(np.log(2.0))})

    np.testing.assert_allclose(t2["theta_binding_pred"]["value"],
                               t0["theta_binding_pred"]["value"])
    g0 = np.asarray(t0["theta_growth_pred"]["value"])
    g2 = np.asarray(t2["theta_growth_pred"]["value"])
    assert not np.allclose(g0, g2)

    # The shifted growth theta is the curve evaluated at 2x the nominal
    # concentration.
    batch = orch.get_batch(orch.data, np.asarray(orch.data.batch_idx))
    sample_theta, calc_theta, _ = orch.main_control_kwargs["theta"]
    theta = substitute(seed(sample_theta, rng_seed=0),
                       data={**fixed, SITE: jnp.asarray(np.log(2.0))})(
        "theta", batch.growth, orch.priors.theta)
    doubled = batch.growth.replace(
        titrant_conc=batch.growth.titrant_conc * 2.0,
        log_titrant_conc=batch.growth.log_titrant_conc + np.log(2.0))
    np.testing.assert_allclose(
        g2, np.asarray(calc_theta(theta.replace(growth_log_conc_shift=None),
                                  doubled)), rtol=1e-5)


def test_growth_titrant_keeps_zero_and_passes_through_without_shift():
    orch = _orchestrator(0.0)
    g = orch.data.growth

    class Off:
        growth_log_conc_shift = None

    assert growth_titrant(Off(), g) is g

    class On:
        growth_log_conc_shift = jnp.asarray(np.log(3.0))

    out = growth_titrant(On(), g)
    conc = np.asarray(g.titrant_conc)
    np.testing.assert_allclose(np.asarray(out.titrant_conc), conc * 3.0)
    assert np.all(np.asarray(out.titrant_conc)[conc == 0] == 0)


@pytest.mark.parametrize("scale, n_specs", [(0.0, 1), (1.0, 2)])
def test_extract_spec_only_when_on(scale, n_specs):
    priors = hill_geno.get_priors().replace(
        theta_growth_log_conc_shift_scale=scale)
    ctx = ExtractionContext(growth_tm=type("TM", (), {"df": None})(),
                            mut_labels=[], pair_labels=[],
                            growth_shares_replicates=False,
                            theta_priors=priors)
    specs = hill_geno.get_extract_specs(ctx)
    assert len(specs) == n_specs
    if n_specs == 2:
        assert specs[1]["params_to_get"] == ["growth_log_conc_shift"]
