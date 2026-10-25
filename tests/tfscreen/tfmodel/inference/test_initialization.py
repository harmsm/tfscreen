"""
Tests for inference/initialization.py: site values <-> guide parameters.

The registry-wide test makes sure every component guide follows the
``{site}_loc(s)``/``{site}_scale(s)`` convention the pre-MAP hand-off relies
on, or is on an explicit list of exceptions.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from tfscreen.tfmodel.inference.initialization import (
    component_guide_init,
    component_guide_map,
    site_prior_sds,
    site_unconstrained_prior_sds,
    site_values,
    trace_model_sites,
)
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

from .test_batch_safety import (
    _BINDING_CSV,
    _GROWTH_CSV,
    _SAFE_VARIANTS,
    _variant_kwargs,
    _owned_by,
)


def _model(data=None, priors=None):
    a = numpyro.sample("a", dist.Normal(1.0, 2.0))
    numpyro.sample("b", dist.HalfNormal(3.0))
    with numpyro.plate("p", 4):
        numpyro.sample("c", dist.Normal(a, 0.5))
    numpyro.sample("d", dist.Cauchy(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(a, 1.0), obs=0.3)


def _guide(data=None, priors=None):
    numpyro.sample("a", dist.Normal(numpyro.param("a_loc", 0.0),
                                    numpyro.param("a_scale", 1.0)))
    numpyro.sample("b", dist.LogNormal(numpyro.param("b_loc", 0.0),
                                       numpyro.param("b_scale", 0.5)))
    with numpyro.plate("p", 4):
        numpyro.sample("c", dist.Normal(numpyro.param("c_locs", jnp.zeros(4)),
                                        numpyro.param("c_scales",
                                                      jnp.ones(4))))
    # does not follow the convention
    numpyro.sample("d", dist.Normal(numpyro.param("dee", 0.0), 1.0))


def test_trace_model_sites_skips_observed():
    sites = trace_model_sites(_model, None, None)
    assert set(sites) == {"a", "b", "c", "d"}


def test_component_guide_map():
    mapping, unmatched, defaults = component_guide_map(_guide, None, None)
    assert mapping["a"]["kind"] == "normal"
    assert mapping["a"]["loc"] == "a_loc" and mapping["a"]["scale"] == "a_scale"
    assert mapping["b"]["kind"] == "lognormal"
    assert mapping["c"]["loc"] == "c_locs" and mapping["c"]["shape"] == (4,)
    assert unmatched == ["d"]
    assert set(defaults) == {"a_loc", "a_scale", "b_loc", "b_scale",
                             "c_locs", "c_scales", "dee"}


def test_site_values_priorities_and_shapes():
    sites = trace_model_sites(_model, None, None)
    mapping, _, _ = component_guide_map(_guide, None, None)
    values = site_values({"b_loc": np.log(2.0),      # guide loc -> exp
                          "a": 5.0, "a_auto_loc": 6.0,  # MAP key wins
                          "c": 1.5,                   # broadcast
                          "d": np.ones(3)},           # wrong shape: skipped
                         sites, mapping)
    assert float(values["b"]) == pytest.approx(2.0)
    assert float(values["a"]) == pytest.approx(6.0)
    np.testing.assert_allclose(values["c"], 1.5)
    assert "d" not in values


def test_component_guide_init():
    mapping, _, defaults = component_guide_map(_guide, None, None)
    params, skipped = component_guide_init(
        {"a": jnp.array(2.0), "b": jnp.array(0.0), "c": jnp.full(4, 3.0)},
        mapping, defaults, init_scale=0.2)
    assert float(params["a_loc"]) == pytest.approx(2.0)
    np.testing.assert_allclose(params["c_locs"], 3.0)
    assert skipped == ["b"]          # log of 0 is not a location
    assert "b_loc" not in params
    # scales capped, smaller defaults kept
    assert float(params["a_scale"]) == pytest.approx(0.2)
    np.testing.assert_allclose(params["c_scales"], 0.2)
    assert float(params["b_scale"]) == pytest.approx(0.2)
    params, _ = component_guide_init({}, mapping, defaults, init_scale=None)
    assert params == {}


def _bounded_guide(data=None, priors=None):
    numpyro.sample("e", dist.Normal(
        numpyro.param("e_loc", 0.0),
        numpyro.param("e_scale", 1.0,
                      constraint=dist.constraints.greater_than(1e-4))))


def test_component_guide_init_stays_inside_scale_bound():
    """
    A capped scale never starts on its constraint's lower bound.

    With the cap equal to the bound (both 1e-4 for the hyperparameter
    scales), the param started at the bound, -inf in unconstrained space,
    and never moved (two-stage grid, 2026-09-29).
    """
    from numpyro.distributions.transforms import biject_to

    mapping, _, defaults = component_guide_map(_bounded_guide, None, None)
    assert mapping["e"]["scale_lower_bound"] == pytest.approx(1e-4)
    params, _ = component_guide_init({}, mapping, defaults, init_scale=1e-4)
    assert float(params["e_scale"]) == pytest.approx(2e-4)
    u = biject_to(dist.constraints.greater_than(1e-4)).inv(params["e_scale"])
    assert np.isfinite(float(u))
    # unbounded scales are unchanged
    mapping, _, defaults = component_guide_map(_guide, None, None)
    assert mapping["a"]["scale_lower_bound"] is None
    params, _ = component_guide_init({}, mapping, defaults, init_scale=0.2)
    assert float(params["a_scale"]) == pytest.approx(0.2)


def test_site_prior_sds():
    sds = site_prior_sds(trace_model_sites(_model, None, None,
                                           substitutions={"a": 0.0}))
    assert float(sds["a"]) == pytest.approx(2.0)
    np.testing.assert_allclose(sds["c"], 0.5)
    assert float(sds["b"]) == pytest.approx(3.0 * np.sqrt(1 - 2 / np.pi))
    assert "d" not in sds  # Cauchy: no finite variance


def test_site_prior_sds_skip_undefined_variance():
    """InverseGamma(2, .) with Python floats raises ZeroDivisionError from its
    variance (the horseshoe slab prior, slab_df = 4); that site is skipped,
    not the whole model."""

    def model(data=None, priors=None):
        numpyro.sample("a", dist.Normal(1.0, 2.0))
        numpyro.sample("c2", dist.InverseGamma(2.0, 2.0))

    sites = trace_model_sites(model, None, None)
    sds = site_prior_sds(sites)
    assert "c2" not in sds
    assert float(sds["a"]) == pytest.approx(2.0)
    # the unconstrained spread falls back to Monte Carlo (log units)
    assert np.isfinite(float(site_unconstrained_prior_sds(sites)["c2"]))


def test_site_unconstrained_prior_sds():
    """Prior spread in the units guide locations live in: exact SDs for real
    sites with a finite variance, a robust Monte Carlo estimate (IQR / 1.349)
    otherwise -- log units for a positive site, finite for a Cauchy."""
    sites = trace_model_sites(_model, None, None, substitutions={"a": 0.0})
    sds = site_unconstrained_prior_sds(sites)
    assert float(sds["a"]) == pytest.approx(2.0)
    np.testing.assert_allclose(sds["c"], 0.5)
    assert sds["c"].shape == (4,)
    # log|3Z|, Z ~ N(0, 1): IQR / 1.349 = 0.95 (the scale does not matter)
    assert 0.6 < float(sds["b"]) < 1.4
    # Cauchy(0, 1): IQR / 1.349 = 1.48
    assert 0.8 < float(sds["d"]) < 2.5
    # deterministic for a given seed
    again = site_unconstrained_prior_sds(sites)
    assert float(again["b"]) == float(sds["b"])


# Guide sites that do not follow the {site}_loc(s) convention, and why.
_KNOWN_UNMAPPED = {
    # horseshoe_geno names its guide params activity_lambda_offset_* and
    # activity_activity_offset_*; renaming them would break its checkpoints.
    ("activity", "horseshoe_geno"): {"activity_global_scale",
                                     "activity_local_scale",
                                     "activity_offset"},
    # noise latents guided by their priors (no location parameter)
    ("theta_growth_noise", "logit_normal"): {"theta_growth_noise_epsilon"},
    ("theta_binding_noise", "beta"): {"theta_binding_noise_dist"},
}


@pytest.mark.parametrize("axis,variant", _SAFE_VARIANTS)
def test_component_guides_follow_loc_scale_convention(axis, variant):
    orchestrator = ModelOrchestrator(batch_size=None,
                                     **_variant_kwargs(axis, variant))
    ri = RunInference(orchestrator, seed=0)
    mapping, unmatched, _ = component_guide_map(orchestrator.jax_model_guide,
                                                orchestrator.priors,
                                                ri._trace_batch())
    unmatched = {s for s in unmatched if _owned_by(axis, s)}
    assert unmatched == _KNOWN_UNMAPPED.get((axis, variant), set())
    no_scale = [s for s, e in mapping.items()
                if e["scale"] is None and _owned_by(axis, s)]
    assert no_scale == []


def test_default_model_guide_start_round_trip():
    """Guesses -> site values -> guide params reproduce the site values."""
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=None)
    ri = RunInference(orchestrator, seed=0)
    values = ri.site_values(orchestrator.init_params)
    assert "dk_geno_offset" in values  # keyed by site in the guesses
    start = ri.component_guide_start(values,
                                     guesses=orchestrator.init_params,
                                     init_scale=0.1)
    np.testing.assert_allclose(start["dk_geno_offset_locs"],
                               values["dk_geno_offset"], rtol=1e-6)
    assert float(np.max(start["dk_geno_offset_scales"])) == pytest.approx(0.1)
    # param-keyed guesses (not sites) are kept
    assert "condition_growth_k_locs" in start


def test_default_guide_start_stays_near_its_point():
    """
    SVI starts where the pre-MAP ended, not far above it.

    The starting scale is one number in every site's own units, so a large one
    is enormous on growth rates (per minute) and ln_cfu levels: at 0.1 the
    relative-fit grid's SVI started ~1000x above its pre-MAP loss and
    re-descended into other optima (2026-09-27).  Checked on the count
    likelihood, the most sensitive to it.
    """
    from numpyro.handlers import seed, substitute, trace
    from numpyro.infer.util import log_density

    from tfscreen.tfmodel.scripts.fit_model_cli import DEFAULT_GUIDE_INIT_SCALE

    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     batch_size=None,
                                     growth_likelihood="counts",
                                     sample_offset="level")
    ri = RunInference(orchestrator, seed=0)
    values = ri.site_values(orchestrator.init_params)
    data = orchestrator.get_batch(orchestrator.data,
                                  jnp.asarray(orchestrator.get_random_idx()))

    def neg_log_joint(params, key):
        draw = trace(seed(substitute(orchestrator.jax_model_guide, params),
                          key)).get_trace(data=data,
                                          priors=orchestrator.priors)
        point = {k: s["value"] for k, s in draw.items() if s["type"] == "sample"}
        lp, _ = log_density(orchestrator.jax_model, (),
                            dict(data=data, priors=orchestrator.priors), point)
        return -float(lp)

    def start(scale):
        return ri.component_guide_start(values,
                                        guesses=orchestrator.init_params,
                                        init_scale=scale)

    at_point = neg_log_joint(start(1e-12), 0)

    def excess(scale):
        params = start(scale)
        return np.median([neg_log_joint(params, k) for k in range(8)]) - at_point

    assert excess(DEFAULT_GUIDE_INIT_SCALE) < 0.01 * abs(at_point)

    # every guide param starts strictly inside its constraint, so none is
    # stuck at -inf in unconstrained space
    from numpyro.distributions.transforms import biject_to
    guide_trace = trace(seed(orchestrator.jax_model_guide, 0)).get_trace(
        data=data, priors=orchestrator.priors)
    params = start(DEFAULT_GUIDE_INIT_SCALE)
    for name, value in params.items():
        constraint = guide_trace[name]["kwargs"].get("constraint")
        if constraint is None:
            continue
        u = biject_to(constraint).inv(jnp.asarray(value))
        assert np.all(np.isfinite(np.asarray(u))), name
    # the test can fail: the old default started far from the point
    assert excess(0.1) > abs(at_point)


def test_component_guide_map_skips_wrong_shape_and_unbroadcastable_params():
    """A same-named param of the wrong shape is passed over for the one that
    matches; a param whose shape fits the site but not the distribution's
    own (unexpanded) location cannot match it."""

    def guide(data=None, priors=None):
        # e: a scalar e_loc of zeros would broadcast onto the loc; it is
        # skipped on shape and e_locs is chosen
        numpyro.param("e_loc", 0.0)
        with numpyro.plate("p", 4):
            numpyro.sample("e", dist.Normal(
                numpyro.param("e_locs", jnp.zeros(4)),
                numpyro.param("e_scales", jnp.ones(4))))
        # f: an expanded scalar Normal; f_locs has the site's shape (4,) but
        # cannot broadcast to the base distribution's scalar loc
        numpyro.param("f_locs", jnp.zeros(4))
        numpyro.sample("f", dist.Normal(numpyro.param("f_loc", 0.0),
                                        1.0).expand([4]))

    mapping, unmatched, _ = component_guide_map(guide, None, None)
    assert mapping["e"]["loc"] == "e_locs"
    assert mapping["e"]["scale"] == "e_scales"
    assert "f" not in mapping
    assert "f" in unmatched


class _NoSupport(dist.Distribution):
    """A distribution that cannot report its support."""

    def __init__(self):
        super().__init__(batch_shape=(), event_shape=())

    @property
    def support(self):
        raise NotImplementedError

    def sample(self, key, sample_shape=()):
        return jnp.zeros(sample_shape + self.batch_shape)


def test_site_unconstrained_prior_sds_skips_unusable_sites():
    """Sites without a support, or whose prior has no spread in unconstrained
    units, are left out; the others are kept."""
    from numpyro.distributions.transforms import ExpTransform

    def model(data=None, priors=None):
        numpyro.sample("a", dist.Normal(1.0, 2.0))
        numpyro.sample("nosupport", _NoSupport())
        # exp of a point mass: positive support, every draw identical, so
        # the unconstrained IQR is 0
        numpyro.sample("point", dist.TransformedDistribution(
            dist.Delta(0.0), ExpTransform()))

    sites = trace_model_sites(model, None, None)
    sds = site_unconstrained_prior_sds(sites)
    assert set(sds) == {"a"}
    assert float(sds["a"]) == pytest.approx(2.0)
