"""
Integration tests for growth_model/model.py — real component functions.

The existing test_model.py verifies orchestration (call order, argument
forwarding) using full mocks.  These tests complement it by wiring real,
simple component implementations into jax_model and asserting on the
numerical output, catching regressions in:
  - component interface compatibility (signature, return shape)
  - the core arithmetic: ln_cfu_pred = ln_cfu0 + total_growth
  - deterministic-site registration
"""

import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, seed
from collections import namedtuple
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.model import jax_model

# Real component implementations (the simplest ones — no latent parameters).
from tfscreen.analysis.hierarchical.growth_model.components.activity import fixed as activity_fixed
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno import fixed as dk_geno_fixed
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition import instant as growth_transition_instant
from tfscreen.analysis.hierarchical.growth_model.components.transformation import single as transformation_single
from tfscreen.analysis.hierarchical.growth_model.components.noise import zero as noise_zero


# ---------------------------------------------------------------------------
# Minimal namedtuple data fixtures
# ---------------------------------------------------------------------------

# Only expose the fields actually consumed by the real simple components and
# by jax_model itself (t_pre, t_sel, congression_mask).
_MockGrowthData = namedtuple(
    "_MockGrowthData",
    ["batch_size", "t_pre", "t_sel", "congression_mask"],
)

_MockBindingData = namedtuple("_MockBindingData", [])

_MockData = namedtuple("_MockData", ["growth", "binding"])

# Priors for the real components are all empty ModelPriors dataclasses.
_MockGrowthPriors = namedtuple(
    "_MockGrowthPriors",
    ["condition_growth", "ln_cfu0", "dk_geno", "activity", "transformation",
     "theta_growth_noise", "growth_transition"],
)
_MockBindingPriors = namedtuple("_MockBindingPriors", ["theta_binding_noise"])
_MockPriors = namedtuple("_MockPriors", ["theta", "growth", "binding"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
T_PRE = jnp.array(1.0)
T_SEL = jnp.array(2.0)


@pytest.fixture
def mock_data():
    growth = _MockGrowthData(
        batch_size=BATCH_SIZE,
        t_pre=T_PRE,
        t_sel=T_SEL,
        congression_mask=None,
    )
    return _MockData(growth=growth, binding=_MockBindingData())


@pytest.fixture
def real_priors():
    growth = _MockGrowthPriors(
        condition_growth=MagicMock(),  # passed to mocked condition_growth_model
        ln_cfu0=MagicMock(),           # passed to mocked ln_cfu0_model
        dk_geno=dk_geno_fixed.get_priors(),
        activity=activity_fixed.get_priors(),
        transformation=transformation_single.get_priors(),
        theta_growth_noise=noise_zero.get_priors(),
        growth_transition=growth_transition_instant.get_priors(),
    )
    binding = _MockBindingPriors(theta_binding_noise=noise_zero.get_priors())
    return _MockPriors(theta=MagicMock(), growth=growth, binding=binding)


def _build_control(mock_data, real_priors, is_guide=False):
    """
    Build the control dict for jax_model, mixing real and mock components.

    Real components: instant growth_transition, fixed activity, fixed dk_geno,
                     single transformation, zero noise.
    Mocked components: theta, ln_cfu0, condition_growth, observe_*.
    """
    # Known-value mocks so we can verify the arithmetic.
    THETA_VALUE = jnp.array(0.5)
    LN_CFU0_VALUE = jnp.array(5.0)
    G_PRE = jnp.array(2.0)
    G_SEL = jnp.array(3.0)

    theta_model = MagicMock(return_value=THETA_VALUE)
    calc_theta = MagicMock(side_effect=lambda t, d: t)
    get_moments = MagicMock(return_value=(jnp.array(0.0), jnp.array(1.0)))

    condition_growth_model = MagicMock(return_value=MagicMock())
    calculate_growth = MagicMock(return_value=(G_PRE, G_SEL))
    ln_cfu0_model = MagicMock(return_value=LN_CFU0_VALUE)

    observe_growth = MagicMock()
    observe_binding = MagicMock()

    return {
        "theta": (theta_model, calc_theta, get_moments),
        "condition_growth": condition_growth_model,
        "calculate_growth": calculate_growth,
        "ln_cfu0": ln_cfu0_model,
        "activity": activity_fixed.define_model,
        "dk_geno": dk_geno_fixed.define_model,
        "growth_transition": growth_transition_instant.define_model,
        "transformation": (transformation_single.define_model, transformation_single.update_thetas),
        "theta_growth_noise": noise_zero.define_model,
        "theta_binding_noise": noise_zero.define_model,
        "observe_growth": observe_growth,
        "observe_binding": observe_binding,
        "is_guide": is_guide,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jax_model_runs_with_real_components(mock_data, real_priors):
    """jax_model should execute without error using real simple components."""
    control = _build_control(mock_data, real_priors)
    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, real_priors, **control)


def test_jax_model_growth_pred_arithmetic(mock_data, real_priors):
    """
    ln_cfu_pred = ln_cfu0 + total_growth.

    With fixed mocks:
      ln_cfu0 = 5.0
      g_pre = 2.0,  t_pre = 1.0
      g_sel = 3.0,  t_sel = 2.0
      total_growth (instant) = g_pre*t_pre + g_sel*t_sel = 2.0 + 6.0 = 8.0
      expected ln_cfu_pred = 13.0
    """
    control = _build_control(mock_data, real_priors)
    with numpyro.handlers.seed(rng_seed=0):
        model_trace = trace(jax_model).get_trace(mock_data, real_priors, **control)

    assert "growth_pred" in model_trace
    pred = model_trace["growth_pred"]["value"]
    assert jnp.isclose(pred, 13.0, atol=1e-5), f"Expected 13.0, got {pred}"


def test_jax_model_deterministic_sites_registered(mock_data, real_priors):
    """theta_binding_pred, theta_growth_pred, binding_pred, growth_pred must appear."""
    control = _build_control(mock_data, real_priors)
    with numpyro.handlers.seed(rng_seed=0):
        model_trace = trace(jax_model).get_trace(mock_data, real_priors, **control)

    for site in ("theta_binding_pred", "theta_growth_pred", "binding_pred", "growth_pred"):
        assert site in model_trace, f"Deterministic site '{site}' missing from trace"


def test_jax_model_guide_skips_prediction(mock_data, real_priors):
    """In guide mode jax_model should pass None as prediction to both observers."""
    control = _build_control(mock_data, real_priors, is_guide=True)
    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, real_priors, **control)

    observe_growth = control["observe_growth"]
    observe_binding = control["observe_binding"]

    # Observer should be called with None as the third positional argument.
    growth_args, _ = observe_growth.call_args
    assert growth_args[2] is None

    binding_args, _ = observe_binding.call_args
    assert binding_args[2] is None


def test_jax_model_guide_still_calls_real_components(mock_data, real_priors):
    """Even in guide mode the parameter-free real components must be invoked."""
    called_components = []

    def tracking_activity(name, data, priors):
        called_components.append("activity")
        return activity_fixed.define_model(name, data, priors)

    def tracking_dk_geno(name, data, priors):
        called_components.append("dk_geno")
        return dk_geno_fixed.define_model(name, data, priors)

    control = _build_control(mock_data, real_priors, is_guide=True)
    control["activity"] = tracking_activity
    control["dk_geno"] = tracking_dk_geno

    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, real_priors, **control)

    assert "activity" in called_components
    assert "dk_geno" in called_components
