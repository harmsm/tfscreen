import pytest
import torch
import pyro
import pyro.poutine as poutine

from tfscreen.analysis.hierarchical.growth_model.components.noise.zero import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

@pytest.fixture
def mock_data():
    """Provides a minimal placeholder for the 'data' argument."""
    return (1, 2)


def test_get_hyperparameters():
    """Tests that get_hyperparameters returns an empty dict."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert len(params) == 0

def test_get_guesses(mock_data):
    """Tests that get_guesses returns an empty dict."""
    name = "test_no_noise"
    guesses = get_guesses(name, mock_data)
    assert isinstance(guesses, dict)
    assert len(guesses) == 0

def test_get_priors():
    """Tests that get_priors returns a correctly instantiated empty ModelPriors."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)

def test_define_model_pass_through_logic(mock_data):
    """
    Tests that define_model correctly returns the input 'fx_calc'
    and does not add any sample or deterministic sites.
    """
    name = "test_no_noise"
    priors = get_priors()

    fx_calc_in = torch.tensor([0.1, 0.5, 0.9])

    # Check that input and output are identical
    fx_calc_out = define_model(name=name, fx_calc=fx_calc_in, priors=priors)
    assert fx_calc_out is fx_calc_in
    assert torch.all(fx_calc_out == fx_calc_in)

    # Trace the execution
    model_trace = poutine.trace(define_model).get_trace(
        name=name,
        fx_calc=fx_calc_in,
        priors=priors
    )

    # No sample or deterministic sites should have been added
    # (Pyro trace may include _INPUT/_RETURN internal nodes; filter those out)
    pyro_sites = [v for v in model_trace.nodes.values()
                  if v["type"] in ("sample", "deterministic", "param")]
    assert len(pyro_sites) == 0

def test_guide_pass_through_logic(mock_data):
    """
    Tests that guide correctly returns the input 'fx_calc'
    and does not add any sample or deterministic sites.
    """
    name = "test_no_noise_guide"
    priors = get_priors()

    fx_calc_in = torch.tensor([0.1, 0.5, 0.9])

    fx_calc_out = guide(name=name, fx_calc=fx_calc_in, priors=priors)
    assert fx_calc_out is fx_calc_in
    assert torch.all(fx_calc_out == fx_calc_in)
