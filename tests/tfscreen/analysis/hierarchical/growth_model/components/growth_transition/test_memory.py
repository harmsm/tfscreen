import pytest
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.memory import (
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
    data.map_condition_pre = jnp.zeros((2,), dtype=int)
    return data


def test_get_hyperparameters():
    params = get_hyperparameters()
    assert "tau0_loc" in params
    assert "k1_loc" in params
    assert "k2_loc" in params


def test_get_priors():
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.tau0_loc == get_hyperparameters()["tau0_loc"]
    assert not hasattr(priors, "pinned")


def test_get_guesses(mock_data):
    guesses = get_guesses("test", mock_data)
    for param in ("tau0", "k1", "k2"):
        assert f"test_{param}_locs" in guesses
        assert f"test_{param}_scales" in guesses
        assert guesses[f"test_{param}_locs"].shape == (mock_data.num_condition_rep,)


def test_define_model_structure(mock_data):
    """Sample sites exist and have the right shapes."""
    priors = get_priors()
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    theta = jnp.array([0.5, 0.5])

    with seed(rng_seed=0):
        model_trace = trace(define_model).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )

    for param in ("tau0", "k1", "k2"):
        assert f"gt_{param}" in model_trace
        assert model_trace[f"gt_{param}"]["type"] == "sample"
        assert model_trace[f"gt_{param}"]["value"].shape == (mock_data.num_condition_rep,)


def test_define_model_returns_correct_shape(mock_data):
    priors = get_priors()
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    theta = jnp.array([0.5, 0.5])

    with seed(rng_seed=0):
        result = define_model(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )
    assert result.shape == t_sel.shape


def test_define_model_piecewise_logic(mock_data):
    """
    When t_sel < tau, growth = g_pre*t_pre + g_pre*t_sel.
    When t_sel >= tau, growth = g_pre*t_pre + g_pre*tau + g_sel*(t_sel-tau).
    """
    mock_data.num_condition_rep = 1
    mock_data.map_condition_pre = jnp.zeros((2,), dtype=int)

    # tau = tau0 + k1/(theta+k2) = 5 + 0 = 5
    subs = {
        "gt_tau0": jnp.array([5.0]),
        "gt_k1":   jnp.array([0.0]),
        "gt_k2":   jnp.array([1.0]),
    }

    g_pre  = jnp.array([1.0, 1.0])
    g_sel  = jnp.array([2.0, 2.0])
    t_pre  = jnp.array([10.0, 10.0])
    t_sel  = jnp.array([3.0, 8.0])   # first < tau=5, second >= tau=5
    theta  = jnp.array([0.0, 0.0])

    priors = get_priors()
    result = substitute(define_model, data=subs)(
        name="gt", data=mock_data, priors=priors,
        g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
    )

    expected_0 = 1.0 * 10.0 + 1.0 * 3.0          # t_sel < tau
    expected_1 = 1.0 * 10.0 + 1.0 * 5.0 + 2.0 * (8.0 - 5.0)  # t_sel >= tau
    assert jnp.allclose(result[0], expected_0, atol=1e-5)
    assert jnp.allclose(result[1], expected_1, atol=1e-5)


def test_guide_structure(mock_data):
    priors = get_priors()
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    theta = jnp.array([0.5, 0.5])

    with seed(rng_seed=0):
        guide_trace = trace(guide).get_trace(
            name="gt", data=mock_data, priors=priors,
            g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta,
        )

    for param in ("tau0", "k1", "k2"):
        assert f"gt_{param}_locs" in guide_trace
        assert f"gt_{param}_scales" in guide_trace
        assert f"gt_{param}" in guide_trace


def test_model_and_guide_have_compatible_sample_sites(mock_data):
    priors = get_priors()
    g_pre = jnp.array([1.0, 2.0])
    g_sel = jnp.array([0.5, 1.5])
    t_pre = jnp.array([10.0, 10.0])
    t_sel = jnp.array([20.0, 20.0])
    theta = jnp.array([0.5, 0.5])

    kwargs = dict(name="gt", data=mock_data, priors=priors,
                  g_pre=g_pre, g_sel=g_sel, t_pre=t_pre, t_sel=t_sel, theta=theta)

    with seed(rng_seed=0):
        m_tr = trace(define_model).get_trace(**kwargs)
    with seed(rng_seed=0):
        g_tr = trace(guide).get_trace(**kwargs)

    m_samples = {k for k, v in m_tr.items() if v["type"] == "sample"}
    g_samples = {k for k, v in g_tr.items() if v["type"] == "sample"}
    assert m_samples == g_samples
