import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi_k import (
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
    assert "tau_loc" in params
    assert "tau_scale" in params
    assert "ln_k0_loc" in params
    assert "ln_k0_scale" in params
    assert "ln_gamma_loc" in params
    assert "ln_gamma_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    hp = get_hyperparameters()
    assert priors.tau_loc == hp["tau_loc"]
    assert priors.ln_k0_loc == hp["ln_k0_loc"]
    assert priors.ln_gamma_loc == hp["ln_gamma_loc"]


def test_get_guesses(mock_data):
    guesses = get_guesses("test", mock_data)
    assert "test_tau_locs" in guesses
    assert "test_tau_scales" in guesses
    assert "test_ln_k0_locs" in guesses
    assert "test_ln_k0_scales" in guesses
    assert "test_ln_gamma_locs" in guesses
    assert "test_ln_gamma_scales" in guesses
    assert guesses["test_tau_locs"].shape == (mock_data.num_condition_rep,)
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

    assert "gt_tau" in model_trace
    assert "gt_ln_k0" in model_trace
    assert "gt_ln_gamma" in model_trace
    assert model_trace["gt_tau"]["type"] == "sample"
    assert model_trace["gt_ln_k0"]["type"] == "sample"
    assert model_trace["gt_ln_gamma"]["type"] == "sample"
    assert model_trace["gt_tau"]["value"].shape == (mock_data.num_condition_rep,)


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
    """When g_pre == g_sel the (g_sel - g_pre) term vanishes: result = g_pre*(t_pre + t_sel)."""
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((3,), dtype=int)
    priors = get_priors()

    g = jnp.array([0.8, 0.8, 0.8])
    t_pre = jnp.array([10.0, 10.0, 10.0])
    t_sel = jnp.array([1.0, 2.0, 3.0])

    subs = {
        "gt_tau": jnp.array([1.0]),
        "gt_ln_k0": jnp.array([0.0]),
        "gt_ln_gamma": jnp.array([0.0]),
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g, g_sel=g, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g * (t_pre + t_sel)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_define_model_very_early_tau_gives_instant_transition(mock_data):
    """
    When tau << 0 and k is large, integrated_sigmoid → t_sel, giving the clean
    physical limit: dln_cfu ≈ g_pre*t_pre + g_sel*t_sel.

    Derivation: as tau → -∞,
        logaddexp(0, k*(t_sel - tau)) → k*(t_sel - tau)
        logaddexp(0, -k*tau)          → -k*tau
        integrated_sigmoid → (k*(t_sel - tau) - (-k*tau)) / k = t_sel
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([3.0])

    # tau = -100 (well before selection starts), sharp k, negligible gamma
    subs = {
        "gt_tau": jnp.array([-100.0]),
        "gt_ln_k0": jnp.array([3.0]),     # k0 = exp(3) ≈ 20
        "gt_ln_gamma": jnp.array([-10.0]),  # gamma ≈ 0, no modulation
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g_pre * t_pre + g_sel * t_sel
    assert jnp.allclose(result, expected, atol=1e-3)


def test_define_model_very_late_tau_gives_pre_rate(mock_data):
    """
    When tau >> t_sel, integrated_sigmoid → 0, so growth stays at the pre-selection
    rate: dln_cfu ≈ g_pre*(t_pre + t_sel).
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([3.0])

    subs = {
        "gt_tau": jnp.array([1000.0]),
        "gt_ln_k0": jnp.array([3.0]),
        "gt_ln_gamma": jnp.array([-10.0]),
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g_pre * (t_pre + t_sel)
    assert jnp.allclose(result, expected, atol=1e-3)


def test_large_gamma_slows_transition(mock_data):
    """
    Large gamma reduces k for a given |Δg|, slowing the transition.
    Since g_sel > g_pre, a slower transition means less growth by t_sel.
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)
    priors = get_priors()

    g_pre = jnp.array([0.0])
    g_sel = jnp.array([10.0])   # delta_g = 10
    t_pre = jnp.array([0.0])
    t_sel = jnp.array([1.0])

    def run(ln_gamma_val):
        subs = {
            "gt_tau": jnp.array([0.5]),
            "gt_ln_k0": jnp.array([0.0]),  # k0 = 1
            "gt_ln_gamma": jnp.array([ln_gamma_val]),
        }
        return substitute(define_model, data=subs)(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
        )

    result_small_gamma = run(-10.0)  # gamma ≈ 0 → k ≈ k0 = 1 (sharper)
    result_large_gamma = run(6.9)    # gamma ≈ 1000 → k ≈ 1e-4 (very slow)

    # Slower transition (large gamma) → less exposure to g_sel rate → lower growth
    assert result_large_gamma < result_small_gamma


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

    assert "gt_tau_locs" in guide_trace
    assert "gt_tau_scales" in guide_trace
    assert "gt_ln_k0_locs" in guide_trace
    assert "gt_ln_k0_scales" in guide_trace
    assert "gt_ln_gamma_locs" in guide_trace
    assert "gt_ln_gamma_scales" in guide_trace
    assert "gt_tau" in guide_trace
    assert "gt_ln_k0" in guide_trace
    assert "gt_ln_gamma" in guide_trace


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
