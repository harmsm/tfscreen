import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi import (
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
    assert "tau_lag_loc" in params
    assert "tau_lag_scale" in params
    assert "k_sharp_loc" in params
    assert "k_sharp_scale" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.tau_lag_loc == get_hyperparameters()["tau_lag_loc"]
    assert not hasattr(priors, "pinned")


def test_get_guesses(mock_data):
    guesses = get_guesses("test", mock_data)
    assert "test_tau_lag_locs" in guesses
    assert "test_tau_lag_scales" in guesses
    assert "test_k_sharp_locs" in guesses
    assert "test_k_sharp_scales" in guesses
    assert guesses["test_tau_lag_locs"].shape == (mock_data.num_condition_rep,)


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

    assert "gt_tau_lag" in model_trace
    assert "gt_k_sharp" in model_trace
    assert model_trace["gt_tau_lag"]["type"] == "sample"
    assert model_trace["gt_k_sharp"]["type"] == "sample"
    assert model_trace["gt_tau_lag"]["value"].shape == (mock_data.num_condition_rep,)


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


def test_define_model_instant_transition_matches_sum(mock_data):
    """
    When tau_lag ≪ 0 and k_sharp is very large, the sigmoid is effectively
    a step at t=0, so dln_cfu ≈ g_pre*t_pre + g_sel*t_sel.
    """
    # Force tau_lag = -100 (well before selection starts) and k_sharp = exp(5) (very sharp)
    priors = ModelPriors(tau_lag_loc=-100.0, tau_lag_scale=0.01,
                         k_sharp_loc=5.0, k_sharp_scale=0.01)
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((1,), dtype=int)

    g_pre = jnp.array([1.0])
    g_sel = jnp.array([0.5])
    t_pre = jnp.array([10.0])
    t_sel = jnp.array([3.0])

    subs = {
        "gt_tau_lag": jnp.array([-100.0]),
        "gt_k_sharp": jnp.array([5.0]),   # log-space: k_sharp = exp(5)
    }
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel,
    )
    expected = g_pre * t_pre + g_sel * t_sel
    assert jnp.allclose(result, expected, atol=1e-3)


def test_guide_structure(mock_data):
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

    assert "gt_tau_lag_locs" in guide_trace
    assert "gt_tau_lag_scales" in guide_trace
    assert "gt_k_sharp_locs" in guide_trace
    assert "gt_k_sharp_scales" in guide_trace
    assert "gt_tau_lag" in guide_trace
    assert "gt_k_sharp" in guide_trace


def test_model_and_guide_have_compatible_sample_sites(mock_data):
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
