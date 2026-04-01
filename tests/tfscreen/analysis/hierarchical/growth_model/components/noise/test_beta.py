import pytest
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from tfscreen.analysis.hierarchical.growth_model.components.noise.beta import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)


def test_get_hyperparameters():
    """Tests that get_hyperparameters returns correct values and mean."""
    params = get_hyperparameters()
    assert isinstance(params, dict)

    assert "beta_kappa_loc" in params
    assert "beta_kappa_scale" in params
    assert params["beta_kappa_loc"] == 25.0
    assert params["beta_kappa_scale"] == 0.05

    mean_kappa = params["beta_kappa_loc"] / params["beta_kappa_scale"]
    assert mean_kappa == 500.0

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.beta_kappa_loc == 25.0

def test_get_guesses():
    """Tests that get_guesses returns the correct key and value."""
    name = "test_noise"
    guesses = get_guesses(name, None)

    assert isinstance(guesses, dict)

    expected_key = f"{name}_beta_kappa"
    assert expected_key in guesses
    assert guesses[expected_key] == 500.0
    assert f"{name}_beta_log_hill_n_hyper_scale" not in guesses


@pytest.fixture
def model_setup():
    """Provides common setup for define_model tests."""
    name = "test_beta_noise"
    priors = get_priors()
    guesses = get_guesses(name, None)

    substituted_model = poutine.condition(define_model, data=guesses)

    return {
        "name": name,
        "priors": priors,
        "guesses": guesses,
        "substituted_model": substituted_model
    }

def test_define_model_logic_and_outputs(model_setup):
    """
    Tests the core reparameterization logic of define_model.
    Checks that alpha and beta are calculated correctly.
    """
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted_model = model_setup["substituted_model"]

    fx_calc = torch.tensor([0.1, 0.5, 0.9])

    torch.manual_seed(42)
    fx_noisy_return = substituted_model(name=name, fx_calc=fx_calc, priors=priors)

    torch.manual_seed(42)
    model_trace = poutine.trace(substituted_model).get_trace(
        name=name,
        fx_calc=fx_calc,
        priors=priors
    )

    # Check Kappa Sample Site
    kappa_site = f"{name}_beta_kappa"
    assert kappa_site in model_trace.nodes
    kappa_val = model_trace.nodes[kappa_site]["value"]
    assert float(kappa_val) == 500.0  # From get_guesses (conditioned)

    # Check Beta Sample Site
    dist_site = f"{name}_dist"
    assert dist_site in model_trace.nodes

    dist_obj = model_trace.nodes[dist_site]["fn"]
    assert isinstance(dist_obj, dist.Beta)

    # Check alpha = fx_calc * kappa
    expected_alpha = fx_calc * kappa_val
    assert torch.allclose(dist_obj.concentration1, expected_alpha)

    # Check beta = (1.0 - fx_calc) * kappa
    expected_beta = (1.0 - fx_calc) * kappa_val
    assert torch.allclose(dist_obj.concentration0, expected_beta)

    # Check Deterministic Site and Return Value
    assert name in model_trace.nodes
    fx_noisy_sampled = model_trace.nodes[dist_site]["value"]
    fx_noisy_deterministic = model_trace.nodes[name]["value"]

    assert torch.all(fx_noisy_return == fx_noisy_sampled)
    assert torch.all(fx_noisy_deterministic == fx_noisy_sampled)
    assert fx_noisy_return.shape == fx_calc.shape

def test_define_model_clipping_logic(model_setup):
    """
    Tests the torch.clamp logic for extreme (0 or 1) mean values.
    """
    name = model_setup["name"]
    priors = model_setup["priors"]
    substituted_model = model_setup["substituted_model"]

    # 500.0 * 1e-20 = 5e-18, which is < 1e-10
    fx_calc = torch.tensor([1e-20, 0.5, 1.0 - 1e-20])

    torch.manual_seed(42)
    model_trace = poutine.trace(substituted_model).get_trace(
        name=name,
        fx_calc=fx_calc,
        priors=priors
    )

    dist_obj = model_trace.nodes[f"{name}_dist"]["fn"]

    # Check that alpha[0] was clipped
    assert dist_obj.concentration1[0] == 1e-10
    # Check that beta[0] was not clipped
    assert torch.isclose(dist_obj.concentration0[0], torch.tensor(500.0))

    # Check middle value (not clipped)
    assert torch.isclose(dist_obj.concentration1[1], torch.tensor(250.0))
    assert torch.isclose(dist_obj.concentration0[1], torch.tensor(250.0))

    # Check that beta[2] was clipped
    assert torch.isclose(dist_obj.concentration1[2], torch.tensor(500.0))
    assert dist_obj.concentration0[2] == 1e-10

def test_guide_logic_and_params():
    """
    Tests the guide function structure and execution.
    Verifies that parameters are created and sampling occurs correctly.
    """
    pyro.clear_param_store()

    name = "test_beta_noise_guide"
    priors = get_priors()
    fx_calc = torch.tensor([0.2, 0.5, 0.8])

    torch.manual_seed(0)
    guide_trace = poutine.trace(guide).get_trace(
        name=name,
        fx_calc=fx_calc,
        priors=priors
    )

    # Check Parameter Sites
    assert f"{name}_beta_kappa_loc" in guide_trace.nodes
    assert f"{name}_beta_kappa_scale" in guide_trace.nodes

    # Check initial value for loc is log(prior_loc)
    expected_init_loc = torch.log(torch.tensor(priors.beta_kappa_loc))
    assert torch.isclose(guide_trace.nodes[f"{name}_beta_kappa_loc"]["value"], expected_init_loc)

    # Check Sample Sites
    assert f"{name}_beta_kappa" in guide_trace.nodes
    assert isinstance(guide_trace.nodes[f"{name}_beta_kappa"]["fn"], dist.LogNormal)

    assert f"{name}_dist" in guide_trace.nodes
    dist_obj = guide_trace.nodes[f"{name}_dist"]["fn"]
    assert isinstance(dist_obj, dist.Beta)

    # Verify shape of sampled output
    fx_noisy = guide_trace.nodes[f"{name}_dist"]["value"]
    assert fx_noisy.shape == fx_calc.shape

    # Verify values are within [0, 1]
    assert torch.all((fx_noisy >= 0.0) & (fx_noisy <= 1.0))
