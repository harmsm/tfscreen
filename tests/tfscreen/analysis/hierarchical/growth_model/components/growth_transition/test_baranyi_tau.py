import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi_tau import (
    define_model,
    guide,
    get_priors,
    get_hyperparameters,
    get_guesses,
    ModelPriors,
)


@pytest.fixture
def mock_data():
    data = MagicMock()
    data.num_condition_rep = 2
    data.map_condition_pre = jnp.zeros((3,), dtype=int)
    return data


def test_get_hyperparameters():
    params = get_hyperparameters()
    assert "tau_0_loc" in params
    assert "tau_0_scale" in params
    assert "ln_k0_loc" in params
    assert "ln_k0_scale" in params
    assert "ln_k_loc" in params
    assert "ln_k_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.tau_0_loc == hp["tau_0_loc"]
    assert priors.ln_k0_loc == hp["ln_k0_loc"]
    assert priors.ln_k_loc == hp["ln_k_loc"]


def test_get_guesses(mock_data):
    guesses = get_guesses("test", mock_data)
    assert "test_tau_0_locs" in guesses
    assert "test_tau_0_scales" in guesses
    assert "test_ln_k0_locs" in guesses
    assert "test_ln_k0_scales" in guesses
    assert "test_ln_k_locs" in guesses
    assert "test_ln_k_scales" in guesses
    assert guesses["test_tau_0_locs"].shape == (mock_data.num_condition_rep,)
    assert guesses["test_ln_k0_locs"].shape == (mock_data.num_condition_rep,)


def test_define_model_structure(mock_data):
    """Sample sites exist and have the right shapes."""
    priors = get_priors()
    g_pre = jnp.array([1.0, 1.0, 1.0])
    g_sel = jnp.array([0.5, 0.5, 0.5])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([2.0, 2.0, 2.0])

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    assert "gt_tau_0" in model_trace
    assert "gt_ln_k0" in model_trace
    assert "gt_ln_k" in model_trace
    assert model_trace["gt_tau_0"]["type"] == "sample"
    assert model_trace["gt_ln_k0"]["type"] == "sample"
    assert model_trace["gt_ln_k"]["type"] == "sample"
    assert model_trace["gt_tau_0"]["value"].shape == (mock_data.num_condition_rep,)


def test_define_model_returns_correct_shape(mock_data):
    """Return value shape matches t_sel input shape."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.5)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        result = define_model(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert result.shape == t_sel.shape


def test_define_model_identical_rates(mock_data):
    """When g_pre == g_sel, delta_g=0 so tau=tau_0 and (g_sel-g_pre)=0: result = g_pre*(t_pre+t_sel)."""
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((3,), dtype=int)
    priors = get_priors()

    g = jnp.array([0.8, 0.8, 0.8])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([1.0, 2.0, 3.0])

    subs = {
        "gt_tau_0": jnp.array([1.0]),
        "gt_ln_k0": jnp.array([0.0]),
        "gt_ln_k": jnp.array([0.0]),
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g, g_sel=g, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g * (t_pre + t_sel)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_define_model_zero_k0_uses_base_tau(mock_data):
    """
    When k0 → 0, tau ≈ tau_0 regardless of delta_g.
    With tau_0 far before t_sel and large k, integrated_sigmoid → t_sel,
    giving the clean limit: dln_cfu ≈ g_pre*t_pre + g_sel*t_sel.
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([3.0])

    subs = {
        "gt_tau_0": jnp.array([-100.0]),  # tau_0 well before t_sel
        "gt_ln_k0": jnp.array([-20.0]),   # k0 ≈ 0 → tau stays at tau_0
        "gt_ln_k": jnp.array([3.0]),      # k = exp(3) ≈ 20, sharp
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )
    # integrated_sigmoid → t_sel  →  dln_cfu_sel → g_sel * t_sel
    expected = g_pre * t_pre + g_sel * t_sel
    assert jnp.allclose(result, expected, atol=1e-3)


def test_larger_k0_pushes_tau_later(mock_data):
    """
    Larger k0 shifts tau further right (more inertia), reducing integrated_sigmoid
    and therefore growth when g_sel > g_pre.
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([0.0])
    g_sel = jnp.array([2.0])   # delta_g = 2
    t_pre = jnp.array([0.0])
    t_sel = jnp.array([1.0])

    def run(ln_k0_val):
        subs = {
            "gt_tau_0": jnp.array([0.0]),
            "gt_ln_k0": jnp.array([ln_k0_val]),
            "gt_ln_k": jnp.array([1.0]),
        }
        return substitute(define_model, data=subs)(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    result_small_k0 = run(-5.0)   # k0 ≈ 0.007, tau ≈ tau_0 = 0 (early transition)
    result_large_k0 = run(3.0)    # k0 ≈ 20, tau = 0 + 20*2 = 40 (very late transition)

    # Late transition → less time at g_sel rate → lower growth
    assert result_large_k0 < result_small_k0


def test_define_model_very_late_tau_gives_pre_rate(mock_data):
    """
    When tau >> t_sel, integrated_sigmoid → 0:
    dln_cfu ≈ g_pre*(t_pre + t_sel).
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([3.0])

    subs = {
        "gt_tau_0": jnp.array([1000.0]),  # tau_0 way after t_sel
        "gt_ln_k0": jnp.array([-20.0]),   # k0 ≈ 0, tau stays at tau_0
        "gt_ln_k": jnp.array([3.0]),
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g_pre * (t_pre + t_sel)
    assert jnp.allclose(result, expected, atol=1e-3)


def test_guide_structure(mock_data):
    """Guide exposes all param and sample sites."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.5)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    assert "gt_tau_0_locs" in guide_trace
    assert "gt_tau_0_scales" in guide_trace
    assert "gt_ln_k0_locs" in guide_trace
    assert "gt_ln_k0_scales" in guide_trace
    assert "gt_ln_k_locs" in guide_trace
    assert "gt_ln_k_scales" in guide_trace
    assert "gt_tau_0" in guide_trace
    assert "gt_ln_k0" in guide_trace
    assert "gt_ln_k" in guide_trace


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    """Model and guide share the same set of sample sites."""
    priors = get_priors()
    g_pre = jnp.ones((3,))
    g_sel = jnp.full((3,), 0.5)
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
    g_sel = jnp.full((3,), 0.5)
    t_pre = jnp.full((3,), 10.0)
    t_sel = jnp.array([1.0, 2.0, 3.0])

    with seed(rng_seed=0):
        result = guide(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )
    assert result.shape == t_sel.shape
