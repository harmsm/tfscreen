import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from tfscreen.simulate.sample_theta import sample_theta_prior, _EXCLUDED, _greedy_maximin, sample_theta_stratified


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


# ----------------------------------------------------------------------------
# Perturbation-path dispatch (simulate function present)
# ----------------------------------------------------------------------------

def _make_simulate_module(G, C):
    """Return a mock theta module that exposes a real simulate() function."""
    mock_module = MagicMock()
    mock_module.get_sim_hyperparameters.return_value = {"alpha": 1.0}
    mock_module.SimPriors.return_value = MagicMock()

    expected_theta = np.ones((G, C)) * 0.7
    expected_param = MagicMock(name="theta_param")

    def fake_simulate(name, data, sim_priors, rng_key):
        return expected_theta, expected_param

    mock_module.simulate = fake_simulate
    mock_module._expected_theta = expected_theta
    mock_module._expected_param = expected_param
    return mock_module


def test_dispatch_uses_simulate_when_present(mock_sim_data):
    G = mock_sim_data.num_genotype
    C = mock_sim_data.num_titrant_conc
    mock_module = _make_simulate_module(G, C)

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        theta_gc, theta_param = sample_theta_prior("hill_geno", mock_sim_data, rng_key=0)

    np.testing.assert_array_equal(theta_gc, mock_module._expected_theta)
    assert theta_param is mock_module._expected_param
    mock_module.define_model.assert_not_called()


def test_dispatch_prior_path_when_no_simulate(mock_sim_data):
    """Prior-predictive path is used when simulate is not a real function."""
    G, C = mock_sim_data.num_genotype, mock_sim_data.num_titrant_conc
    mock_module = _make_mock_module(G, C)
    # MagicMock.simulate is a MagicMock, not a function — should fall through to prior path

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        with patch("tfscreen.simulate.sample_theta.handlers"):
            sample_theta_prior("hill_geno", mock_sim_data, rng_key=0)

    mock_module.define_model.assert_called_once()


def test_sim_priors_overrides_applied(mock_sim_data):
    G = mock_sim_data.num_genotype
    C = mock_sim_data.num_titrant_conc
    mock_module = _make_simulate_module(G, C)

    overrides = {"alpha": 9.9}
    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}):
        sample_theta_prior("hill_geno", mock_sim_data, rng_key=0,
                           sim_priors_overrides=overrides)

    call_kwargs = mock_module.SimPriors.call_args[1]
    assert call_kwargs["alpha"] == 9.9


# ----------------------------------------------------------------------------
# _greedy_maximin
# ----------------------------------------------------------------------------

def test_greedy_maximin_returns_correct_count():
    rng = np.random.default_rng(0)
    theta_gc = rng.uniform(0, 1, (20, 5))
    selected = _greedy_maximin(theta_gc, 4)
    assert len(selected) == 4


def test_greedy_maximin_no_duplicates():
    rng = np.random.default_rng(1)
    theta_gc = rng.uniform(0, 1, (50, 8))
    selected = _greedy_maximin(theta_gc, 10)
    assert len(selected) == len(set(selected.tolist()))


def test_greedy_maximin_selects_spread_curves():
    """The selected set should span the theta space broadly."""
    # 50 curves spread across [0,1]; 2 selected should have large distance between them
    rng = np.random.default_rng(7)
    theta_gc = rng.uniform(0, 1, (50, 4))
    selected = _greedy_maximin(theta_gc, 2)
    a, b = theta_gc[selected[0]], theta_gc[selected[1]]
    dist = np.sqrt(np.sum((a - b) ** 2))
    # Two randomly chosen points from 50 uniform in [0,1]^4 would average ~0.87
    # in max-pairwise distance; our greedy selection should comfortably exceed 0.5
    assert dist > 0.5


def test_greedy_maximin_all_when_n_equals_pool():
    theta_gc = np.eye(5)
    selected = _greedy_maximin(theta_gc, 5)
    assert set(selected.tolist()) == set(range(5))


def test_greedy_maximin_all_when_n_exceeds_pool():
    theta_gc = np.eye(3)
    selected = _greedy_maximin(theta_gc, 10)
    assert list(selected) == [0, 1, 2]


# ----------------------------------------------------------------------------
# sample_theta_stratified
# ----------------------------------------------------------------------------

def _make_stratified_mock_module(pool_size, n_binding_concs, n_growth_concs):
    """
    Return a mock theta module for sample_theta_stratified tests.

    run_model is called twice (once for binding sim_data, once for growth
    sim_data).  We return distinct matrices so tests can verify which call
    produced which output.
    """
    mock_module = MagicMock()
    mock_module.get_hyperparameters.return_value = {}
    mock_module.ModelPriors.return_value = MagicMock()
    mock_module.define_model.return_value = MagicMock(name="theta_param")

    # Alternating call sides: binding first, then growth.
    rng = np.random.default_rng(42)
    binding_theta = rng.uniform(0, 1, (1, n_binding_concs, pool_size))
    growth_theta  = rng.uniform(0, 1, (1, n_growth_concs,  pool_size))
    mock_module.run_model.side_effect = [binding_theta, growth_theta]
    mock_module._binding_theta = binding_theta
    mock_module._growth_theta  = growth_theta
    return mock_module


def test_stratified_output_shapes():
    pool_size, n_select, n_binding_concs, n_growth_concs = 30, 4, 6, 3
    mock_module = _make_stratified_mock_module(pool_size, n_binding_concs, n_growth_concs)

    binding_sample_df = pd.DataFrame({"titrant_conc": np.linspace(0, 1, n_binding_concs)})
    growth_sample_df  = pd.DataFrame({"titrant_conc": np.linspace(0, 1, n_growth_concs)})

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}), \
         patch("tfscreen.simulate.sample_theta.handlers"), \
         patch("tfscreen.simulate.sample_theta.build_sim_data", return_value=MagicMock(
             num_genotype=pool_size, num_titrant_conc=n_binding_concs)):
        binding_gc, growth_gc = sample_theta_stratified(
            "hill_geno", binding_sample_df, growth_sample_df,
            rng_key=0, n_select=n_select, pool_size=pool_size,
        )

    assert binding_gc.shape == (n_select, n_binding_concs)
    assert growth_gc.shape  == (n_select, n_growth_concs)


def test_stratified_raises_when_n_select_exceeds_pool():
    binding_sample_df = pd.DataFrame({"titrant_conc": [0.0, 1.0]})
    growth_sample_df  = pd.DataFrame({"titrant_conc": [0.5]})

    with pytest.raises(ValueError, match="n_select"):
        sample_theta_stratified(
            "hill_geno", binding_sample_df, growth_sample_df,
            rng_key=0, n_select=10, pool_size=5,
        )


def test_stratified_selected_rows_come_from_pool():
    """Selected binding rows must be a subset of the pool's binding rows."""
    pool_size, n_select, n_binding_concs, n_growth_concs = 20, 3, 4, 2
    mock_module = _make_stratified_mock_module(pool_size, n_binding_concs, n_growth_concs)

    binding_sample_df = pd.DataFrame({"titrant_conc": np.linspace(0, 1, n_binding_concs)})
    growth_sample_df  = pd.DataFrame({"titrant_conc": [0.0, 1.0]})

    with patch("tfscreen.simulate.sample_theta.model_registry",
               {"theta": {"hill_geno": mock_module}}), \
         patch("tfscreen.simulate.sample_theta.handlers"), \
         patch("tfscreen.simulate.sample_theta.build_sim_data", return_value=MagicMock(
             num_genotype=pool_size, num_titrant_conc=n_binding_concs)):
        binding_gc, growth_gc = sample_theta_stratified(
            "hill_geno", binding_sample_df, growth_sample_df,
            rng_key=0, n_select=n_select, pool_size=pool_size,
        )

    # Each selected binding row must exist in the full pool binding matrix
    pool_binding = mock_module._binding_theta[0].T   # (pool_size, n_binding_concs)
    for row in binding_gc:
        match = np.any(np.all(np.isclose(pool_binding, row[None, :]), axis=1))
        assert match, "Selected binding row not found in pool"
