import pytest
import torch
import pyro
import pyro.poutine as poutine
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.components.transformation import (
    _congression as transformation_congression,
)

# -------------------------------------------------------------------------
# Math Utilities
# -------------------------------------------------------------------------

def test_logit_normal_cdf():
    """Test CDF calculation values."""
    mu, sigma = 0.0, 1.0
    x = torch.tensor([0.0, 0.5, 1.0])
    # logit(0.5) = 0.
    # Phi((0 - 0)/1) = Phi(0) = 0.5

    cdf = transformation_congression._logit_normal_cdf(x, mu, sigma)

    assert torch.isclose(cdf[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(cdf[1], torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(cdf[2], torch.tensor(1.0), atol=1e-5)


def test_empirical_cdf():
    """Test empirical CDF calculation."""
    theta = torch.tensor([0.1, 0.5, 0.9])
    t_grid = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])

    # theta has 3 elements. y = [0.166, 0.5, 0.833] (using 0.5/n)
    # Sorted: 0.1, 0.5, 0.9
    cdf = transformation_congression._empirical_cdf(theta, t_grid)

    assert cdf.shape == (5,)
    assert torch.isclose(cdf[0], torch.tensor(0.16666667), atol=1e-5)
    assert torch.isclose(cdf[2], torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(cdf[4], torch.tensor(0.8333333), atol=1e-5)


# -------------------------------------------------------------------------
# Correction Logic
# -------------------------------------------------------------------------

def test_calculate_expected_observed_max():
    """Verify correction logic for congression."""
    # Case 1: Target > Background
    x_val = 0.9
    lam = 1.0
    mu, sigma = -2.0, 1.0  # Low mean

    res = transformation_congression.calculate_expected_observed_max(x_val, mu, sigma, lam)

    assert torch.isclose(res, torch.tensor(x_val), atol=0.05)
    assert res >= x_val

    # Case 2: Target < Background
    x_val = 0.1
    mu, sigma = 2.0, 1.0

    res = transformation_congression.calculate_expected_observed_max(x_val, mu, sigma, lam)

    assert res > 0.5


def test_calculate_expected_observed_min():
    """Verify correction logic for min."""
    x_val = 0.9
    mu, sigma = -2.0, 1.0
    lam = 1.0

    res = transformation_congression.calculate_expected_observed_min(x_val, mu, sigma, lam)

    assert res < x_val
    assert res > 0.0


def test_update_thetas_shapes():
    """Verify shape processing with plated params."""
    theta = torch.ones((2, 5)) * 0.5

    lam = 1.0
    mu = torch.tensor([[0.0], [0.0]])    # shape (2, 1)
    sigma = torch.tensor([[1.0], [1.0]])  # shape (2, 1)

    params = (lam, mu, sigma)

    res = transformation_congression.update_thetas(theta, params=params)
    assert res.shape == (2, 5)

    # With mask
    mask = torch.tensor([True, False, True, False, True])
    res_mask = transformation_congression.update_thetas(theta, params=params, mask=mask)
    assert res_mask.shape == (2, 5)
    # Check that entries where mask is False are unchanged
    assert torch.all(res_mask[:, 1] == theta[:, 1])
    assert torch.all(res_mask[:, 3] == theta[:, 3])

    # Empirical mode call within shapes test
    res_emp = transformation_congression.update_thetas(theta, params=(lam,), theta_dist="empirical")
    assert res_emp.shape == (2, 5)


def test_update_thetas_empirical_values():
    """Verify empirical mode update_thetas with realistic values."""
    theta = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
    lam = 1.0
    params = (lam,)

    res = transformation_congression.update_thetas(theta, params=params, theta_dist="empirical")
    assert res.shape == (2, 3)
    # Result should be >= input for max-congression
    assert torch.all(res >= theta - 1e-6)


# -------------------------------------------------------------------------
# Model Interface
# -------------------------------------------------------------------------

def test_getters():
    """Test get_hyperparameters, get_guesses, get_priors."""
    params = transformation_congression.get_hyperparameters()
    assert "lam_loc" in params
    assert "mu_anchoring_scale" in params
    assert "sigma_anchoring_scale" in params

    guesses = transformation_congression.get_guesses("test", MagicMock())
    assert "test_lam" in guesses
    assert "test_mu" in guesses
    assert "test_sigma" in guesses

    priors = transformation_congression.get_priors()
    assert isinstance(priors, transformation_congression.ModelPriors)


def test_define_model():
    """Test Pyro model definition with plates."""
    priors = transformation_congression.get_priors()
    data = MagicMock()
    data.num_titrant_name = 2
    data.num_titrant_conc = 3

    torch.manual_seed(0)
    # Without anchors
    lam, mu, sigma = transformation_congression.define_model("test_dm", data, priors)
    assert lam.shape == torch.Size([])
    assert mu.shape == (2, 3, 1)

    # With anchors
    torch.manual_seed(1)
    anchors = (torch.zeros((2, 3, 1)), torch.ones((2, 3, 1)))
    lam2, mu2, sigma2 = transformation_congression.define_model("test_dm_anc", data, priors, anchors=anchors)
    assert mu2.shape == (2, 3, 1)


def test_guide():
    """Test Pyro guide definition with plates."""
    pyro.clear_param_store()

    priors = transformation_congression.get_priors()
    data = MagicMock()
    data.num_titrant_name = 2
    data.num_titrant_conc = 3

    torch.manual_seed(0)
    tr = poutine.trace(transformation_congression.guide).get_trace("test_g", data, priors)
    lam, mu, sigma = transformation_congression.guide("test_g", data, priors)

    assert "test_g_mu_loc" in tr.nodes
    assert tr.nodes["test_g_mu_loc"]["value"].shape == (2, 3, 1)

    assert mu.shape == (2, 3, 1)
    assert sigma.shape == (2, 3, 1)

    # Run with empirical mode
    pyro.clear_param_store()
    priors_emp = transformation_congression.ModelPriors(0.0, 0.1, 0.5, 0.2, mode="empirical")
    torch.manual_seed(2)
    res_emp = transformation_congression.guide("test_emp", data, priors_emp)
    assert len(res_emp) == 1
    assert res_emp[0].shape == torch.Size([])

    # Run with anchors
    pyro.clear_param_store()
    torch.manual_seed(1)
    anchors = (torch.zeros((2, 3, 1)), torch.ones((2, 3, 1)))
    lam_anc, mu_anc, sigma_anc = transformation_congression.guide("test_anc", data, priors, anchors=anchors)
    assert mu_anc.shape == (2, 3, 1)
