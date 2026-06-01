import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tfscreen.simulate.sample_theta import sample_theta_prior, _EXCLUDED


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_sim_data():
    sd = MagicMock()
    sd.num_genotype = 4
    sd.num_titrant_conc = 3
    return sd


def _make_mock_module(G, C):
    """Return a mock theta module whose run_model returns (1, C, G)."""
    mock_module = MagicMock()
    mock_module.get_hyperparameters.return_value = {"alpha": 1.0}
    mock_module.ModelPriors.return_value = MagicMock()
    mock_module.define_model.return_value = MagicMock(name="theta_param")
    # run_model returns shape (T=1, C, G) as JAX/numpy array
    mock_module.run_model.return_value = np.ones((1, C, G)) * 0.6
    return mock_module


# ----------------------------------------------------------------------------
# Validation tests (no real JAX needed — registry is patched)
# ----------------------------------------------------------------------------

def test_raises_on_unknown_component(mock_sim_data):
    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": MagicMock()}}):
        with pytest.raises(ValueError, match="not found"):
            sample_theta_prior("does_not_exist", mock_sim_data, rng_key=0)


def test_raises_on_excluded_component(mock_sim_data):
    # "_simple" is in _EXCLUDED
    for excluded in _EXCLUDED:
        with patch("tfscreen.simulate.sample_theta.model_registry",
                   {"theta": {excluded: MagicMock()}}):
            with pytest.raises(ValueError, match="calibration-only"):
                sample_theta_prior(excluded, mock_sim_data, rng_key=0)


# ----------------------------------------------------------------------------
# Output shape tests
# ----------------------------------------------------------------------------

def test_returns_theta_gc_shape(mock_sim_data):
    G = mock_sim_data.num_genotype  # 4
    C = mock_sim_data.num_titrant_conc  # 3
    mock_module = _make_mock_module(G, C)

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        with patch("tfscreen.simulate.sample_theta.handlers"):
            theta_gc, theta_param = sample_theta_prior("hill_geno", mock_sim_data, rng_key=0)

    assert theta_gc.shape == (G, C)


def test_returns_numpy_array(mock_sim_data):
    G, C = 4, 3
    mock_module = _make_mock_module(G, C)

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        with patch("tfscreen.simulate.sample_theta.handlers"):
            theta_gc, _ = sample_theta_prior("hill_geno", mock_sim_data, rng_key=0)

    assert isinstance(theta_gc, np.ndarray)


def test_priors_overrides_applied(mock_sim_data):
    G, C = 4, 3
    mock_module = _make_mock_module(G, C)

    overrides = {"alpha": 2.5, "beta": 0.1}
    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        with patch("tfscreen.simulate.sample_theta.handlers"):
            sample_theta_prior("hill_geno", mock_sim_data, rng_key=0,
                               priors_overrides=overrides)

    # ModelPriors should have been called with the override values merged in
    call_kwargs = mock_module.ModelPriors.call_args[1]
    assert call_kwargs["alpha"] == 2.5
    assert call_kwargs["beta"] == 0.1


def test_define_model_called_with_correct_args(mock_sim_data):
    G, C = 4, 3
    mock_module = _make_mock_module(G, C)
    mock_priors = mock_module.ModelPriors.return_value

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        with patch("tfscreen.simulate.sample_theta.handlers"):
            sample_theta_prior("hill_geno", mock_sim_data, rng_key=0)

    mock_module.define_model.assert_called_once_with("theta", mock_sim_data, mock_priors)
