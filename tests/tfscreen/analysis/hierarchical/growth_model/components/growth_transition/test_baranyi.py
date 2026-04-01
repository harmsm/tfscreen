import pytest
import torch
import pyro
import pyro.distributions as dist
from tfscreen.analysis.hierarchical.growth_model.components.growth_transition.baranyi import (
    define_model, guide, get_priors, get_hyperparameters, get_guesses
)
from unittest.mock import MagicMock, patch

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = get_hyperparameters()
    assert "tau_lag_hyper_loc_loc" in params
    assert "k_sharp_hyper_loc_loc" in params

    data = MagicMock()
    data.num_condition_rep = 5
    guesses = get_guesses("test", data)
    assert "test_tau_lag_hyper_loc" in guesses
    assert "test_tau_lag_offset" in guesses
    assert guesses["test_tau_lag_offset"].shape == (5,)

    priors = get_priors()
    assert priors is not None
    assert priors.tau_lag_hyper_loc_loc == params["tau_lag_hyper_loc_loc"]

def test_define_model():
    """Test define_model growth transition calculation."""
    data = MagicMock()
    data.num_condition_rep = 1
    data.map_condition_pre = torch.zeros((1,), dtype=torch.int64)

    priors = get_priors()

    g_pre = torch.tensor([1.0])
    g_sel = torch.tensor([0.5])
    t_pre = torch.tensor([10.0])
    t_sel = torch.tensor([2.0])
    theta = None

    # 4 hypers + 2 offsets
    sample_values = [
        1.0,              # tau_lag_hyper_loc
        0.1,              # tau_lag_hyper_scale
        1.0,              # k_sharp_hyper_loc
        0.1,              # k_sharp_hyper_scale
        torch.zeros(1),   # tau_lag_offset
        torch.zeros(1),   # k_sharp_offset
    ]

    with patch("pyro.sample", side_effect=sample_values) as mock_sample:
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = torch.arange(data.num_condition_rep)
        with patch("pyro.plate", return_value=plate_mock):
            total_growth = define_model("test", data, priors, g_pre, g_sel, t_pre, t_sel, theta)

            # tau_lag = 1.0 + 0 * 0.1 = 1.0
            # k_sharp = exp(1.0 + 0 * 0.1) = e^1 ≈ 2.71828

            k_val = torch.exp(torch.tensor(1.0))
            tau_val = torch.tensor(1.0)
            t_val = torch.tensor(2.0)

            term1 = torch.logaddexp(torch.tensor(0.0), k_val * (t_val - tau_val))
            term0 = torch.logaddexp(torch.tensor(0.0), -k_val * tau_val)
            integrated_sigmoid = (term1 - term0) / k_val

            expected_growth = 1.0 * 10.0 + 1.0 * 2.0 + (0.5 - 1.0) * integrated_sigmoid

            assert torch.allclose(total_growth, expected_growth.unsqueeze(0))
            assert mock_sample.called

def test_guide():
    """Test guide logic follows the same structure."""
    pyro.clear_param_store()

    data = MagicMock()
    data.num_condition_rep = 1
    data.map_condition_pre = torch.zeros((1,), dtype=torch.int64)
    priors = get_priors()

    g_pre = torch.tensor([1.0])
    g_sel = torch.tensor([0.5])
    t_pre = torch.tensor([10.0])
    t_sel = torch.tensor([20.0])

    # 4 hypers + 2 offsets
    sample_values = [1.0, 0.1, 1.0, 0.1, torch.zeros(1), torch.zeros(1)]

    with patch("pyro.sample", side_effect=sample_values):
        plate_mock = MagicMock()
        plate_mock.__enter__.return_value = torch.arange(data.num_condition_rep)
        with patch("pyro.plate", return_value=plate_mock):
            total_growth = guide("test", data, priors, g_pre, g_sel, t_pre, t_sel)

            k_val = torch.exp(torch.tensor(1.0))
            tau_val = torch.tensor(1.0)
            t_val = torch.tensor(20.0)

            term1 = torch.logaddexp(torch.tensor(0.0), k_val * (t_val - tau_val))
            term0 = torch.logaddexp(torch.tensor(0.0), -k_val * tau_val)
            integrated_sigmoid = (term1 - term0) / k_val

            expected_growth = 1.0 * 10.0 + 1.0 * 20.0 + (0.5 - 1.0) * integrated_sigmoid
            assert torch.allclose(total_growth, expected_growth.unsqueeze(0))
