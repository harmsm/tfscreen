import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.two_pop import (
    define_model,
    guide,
    get_priors,
    get_hyperparameters,
    get_guesses,
    ModelPriors,
    _compute_growth,
)


@pytest.fixture
def mock_data():
    data = MagicMock()
    data.num_condition_rep = 2
    data.map_condition_pre = jnp.zeros((3,), dtype=int)
    return data


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------

def test_get_hyperparameters():
    params = get_hyperparameters()
    assert "ln_k_trans_loc" in params
    assert "ln_k_trans_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.ln_k_trans_loc == hp["ln_k_trans_loc"]
    assert priors.ln_k_trans_scale == hp["ln_k_trans_scale"]


def test_get_guesses(mock_data):
    guesses = get_guesses("test", mock_data)
    assert "test_ln_k_trans_locs" in guesses
    assert "test_ln_k_trans_scales" in guesses
    assert guesses["test_ln_k_trans_locs"].shape == (mock_data.num_condition_rep,)
    assert guesses["test_ln_k_trans_scales"].shape == (mock_data.num_condition_rep,)


# ---------------------------------------------------------------------------
# Analytical limit tests on _compute_growth
# ---------------------------------------------------------------------------

def test_zero_k_trans_recovers_pre_rate():
    """
    When k_trans → 0, cells never transition to the selection-phase growth mode.
    The formula simplifies to dln_cfu_sel = g_pre * t_sel, so total growth
    equals g_pre * (t_pre + t_sel).

    Derivation with k_trans → 0:
        D → g_pre - g_sel
        a = g_sel * t_sel,  b = g_pre * t_sel  (b > a since g_pre > g_sel)
        scaled_num → 0 + D * 1 = D
        dln_cfu_sel → b + ln(D) - ln(D) = g_pre * t_sel
    """
    g_pre = jnp.array([1.0, 1.5])
    g_sel = jnp.array([0.5, 0.3])
    t_pre = jnp.array([10.0, 8.0])
    t_sel = jnp.array([3.0, 5.0])
    k_trans = jnp.array([1e-8, 1e-8])   # effectively zero

    result = _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans)
    expected = g_pre * (t_pre + t_sel)
    assert jnp.allclose(result, expected, atol=1e-4)


def test_formula_matches_manual_calculation():
    """
    Verify the formula against a hand-computed value.

    Parameters: g_pre=1.0, g_sel=0.3, t_pre=0, t_sel=2, k_trans=0.2
        D = 1.0 - 0.3 - 0.2 = 0.5
        a = 0.3 * 2 = 0.6,  b = (1.0 - 0.2) * 2 = 1.6
        m = max(0.6, 1.6) = 1.6
        scaled_num = 0.2 * exp(0.6 - 1.6) + 0.5 * exp(0)
                   = 0.2 * exp(-1.0) + 0.5
                   ≈ 0.2 * 0.36788 + 0.5 = 0.07358 + 0.5 = 0.57358
        dln_cfu_sel = 1.6 + ln(0.57358) - ln(0.5)
                    = 1.6 + (-0.55523) - (-0.69315)
                    = 1.6 + 0.13792 = 1.73792
        dln_cfu_pre = 1.0 * 0 = 0
        total = 0 + 1.73792 = 1.73792
    """
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.3])
    t_pre = jnp.array([0.0])
    t_sel = jnp.array([2.0])
    k_trans = jnp.array([0.2])

    result = _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans)

    import math
    D = 0.5
    a, b = 0.6, 1.6
    m = 1.6
    scaled_num = 0.2 * math.exp(a - m) + D * math.exp(b - m)
    expected = m + math.log(scaled_num) - math.log(D)  # t_pre = 0
    assert jnp.allclose(result, jnp.array([expected]), atol=1e-5)


def test_larger_k_trans_reduces_growth_when_g_pre_gt_g_sel():
    """
    When g_pre > g_sel, a larger k_trans shifts cells faster away from the
    high-growth pre-selection mode, reducing total growth by t_sel.

    This is the monotonicity property: d(total)/d(k_trans) < 0.
    """
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.3])
    t_pre = jnp.array([0.0])
    t_sel = jnp.array([5.0])

    k_small = jnp.array([0.05])
    k_large = jnp.array([0.50])   # still < g_pre - g_sel = 0.7, so D > 0

    result_small = _compute_growth(g_pre, g_sel, t_pre, t_sel, k_small)
    result_large = _compute_growth(g_pre, g_sel, t_pre, t_sel, k_large)

    assert result_large < result_small


def test_equal_rates_fallback():
    """
    When g_pre == g_sel, D = -k_trans <= 0, so the implementation falls back to
        total = g_pre * t_pre + g_pre * t_sel

    The jnp.where guard (valid = D > 0) covers this case — no NaN should appear.
    """
    g = jnp.array([0.7, 1.2])
    t_pre = jnp.array([5.0, 3.0])
    t_sel = jnp.array([3.0, 4.0])
    k_trans = jnp.array([0.1, 0.3])

    result = _compute_growth(g, g, t_pre, t_sel, k_trans)
    expected = g * t_pre + g * t_sel   # fallback: g_pre*(t_pre + t_sel)
    assert jnp.isfinite(result).all(), "fallback must not produce NaN"
    assert jnp.allclose(result, expected, atol=1e-6)


def test_large_k_trans_fallback_is_finite():
    """
    When g_pre != g_sel but k_trans is so large that D = g_pre-g_sel-k_trans <= 0,
    the formula is outside its valid range.  The guard now falls back to
    g_pre * t_sel (k_trans -> 0 limit), so the result must be finite — no NaN —
    and all JAX gradients remain finite too.
    """
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.3])
    t_pre = jnp.array([5.0])
    t_sel = jnp.array([2.0])

    # D = 1.0 - 0.3 - 0.2 = 0.5 > 0 → valid: uses two-pop formula
    result_valid = _compute_growth(g_pre, g_sel, t_pre, t_sel, jnp.array([0.2]))
    assert jnp.isfinite(result_valid).all()

    # D = 1.0 - 0.3 - 0.9 = -0.2 <= 0 → invalid: falls back to g_pre * t_sel
    result_fallback = _compute_growth(g_pre, g_sel, t_pre, t_sel, jnp.array([0.9]))
    assert jnp.isfinite(result_fallback).all(), "guard must prevent NaN when D <= 0"
    expected_fallback = g_pre * t_pre + g_pre * t_sel
    assert jnp.allclose(result_fallback, expected_fallback, atol=1e-6)


def test_gradient_finite_across_validity_boundary():
    """
    The gradient of the output w.r.t. k_trans must be finite on both sides of
    the D=0 boundary.  This catches the jnp.where NaN-gradient problem where
    NaN * 0 propagates through the backward pass.
    """
    import jax
    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.3])
    t_pre = jnp.array([5.0])
    t_sel = jnp.array([2.0])

    def loss(k_trans):
        return _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans).sum()

    # Valid region (D > 0)
    grad_valid = jax.grad(loss)(jnp.array([0.2]))
    assert jnp.isfinite(grad_valid).all(), "gradient must be finite in valid region"

    # Invalid region (D < 0) — guard substitutes fallback, gradient must not be NaN
    grad_invalid = jax.grad(loss)(jnp.array([0.9]))
    assert jnp.isfinite(grad_invalid).all(), "gradient must be finite in fallback region"


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------

def test_define_model_structure(mock_data):
    """Sample sites exist and have the right shapes."""
    priors = get_priors()
    g_pre = jnp.array([1.0, 1.0, 1.0])
    g_sel = jnp.array([0.3, 0.3, 0.3])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([2.0, 2.0, 2.0])

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    assert "gt_ln_k_trans" in model_trace
    assert model_trace["gt_ln_k_trans"]["type"] == "sample"
    assert model_trace["gt_ln_k_trans"]["value"].shape == (mock_data.num_condition_rep,)


def test_define_model_returns_correct_shape(mock_data):
    """Return value shape matches t_sel input shape."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.3)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        result = define_model(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert result.shape == t_sel.shape


def test_define_model_matches_compute_growth(mock_data):
    """With substituted parameters, define_model matches _compute_growth directly."""
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((3,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([1.0, 1.0, 1.0])
    g_sel = jnp.array([0.3, 0.3, 0.3])
    t_pre = jnp.array([5.0, 5.0, 5.0])
    t_sel = jnp.array([1.0, 2.0, 3.0])
    ln_k = -2.0

    subs = {"gt_ln_k_trans": jnp.array([ln_k])}
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )

    k_trans = jnp.full((3,), jnp.exp(jnp.array(ln_k)))
    expected = _compute_growth(g_pre, g_sel, t_pre, t_sel, k_trans)
    assert jnp.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Guide tests
# ---------------------------------------------------------------------------

def test_guide_structure(mock_data):
    """Guide exposes all param and sample sites."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.3)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    assert "gt_ln_k_trans_locs" in guide_trace
    assert "gt_ln_k_trans_scales" in guide_trace
    assert "gt_ln_k_trans" in guide_trace


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    """Model and guide share the same set of sample sites."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.3)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    kwargs = dict(name="gt", data=mock_data, priors=priors,
                  g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel)

    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(**kwargs)
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(**kwargs)

    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples


def test_guide_returns_correct_shape(mock_data):
    """Guide return value shape matches t_sel input shape."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.3)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        result = guide(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert result.shape == t_sel.shape
