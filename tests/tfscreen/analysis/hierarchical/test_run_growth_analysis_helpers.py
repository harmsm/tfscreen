import pytest
from unittest.mock import MagicMock, ANY
import os
import dill

from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import (
    _run_svi,
    _run_map,
    _run_nuts,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_run_inference(mocker):
    """
    Mocks the RunInference class and its instance methods.
    """
    # Mock where RunInference is imported in run_growth_analysis
    mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis.RunInference")
    mock_ri_instance = mock_ri_class.return_value
    
    # Setup default returns for instance methods
    mock_ri_instance.setup_svi.return_value = "mock_svi_obj"
    mock_ri_instance.setup_map.return_value = "mock_map_obj"
    mock_ri_instance._iterations_per_epoch = 1
    
    # run_optimization returns (svi_state, params, converged)
    mock_ri_instance.run_optimization.return_value = ("final_state", {"p": 1}, True)

    # Patch summarize_posteriors where it is imported in run_growth_analysis
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis.summarize_posteriors")
    
    # Mock os.path.exists to return False by default
    mocker.patch("os.path.exists", return_value=False)

    return mock_ri_class, mock_ri_instance

# =============================================================================
# Tests for _run_svi
# =============================================================================

def test_run_svi_flow_converged(mock_run_inference):
    """Test standard SVI flow where convergence is reached."""
    _, ri = mock_run_inference
    
    # Execute
    state, params, converged = _run_svi(
        ri, 
        init_params=None,
        config_file="config.yaml",
        out_root="test_root",
        max_num_epochs=500,
        num_posterior_samples=100
    )
    
    # 1. Setup SVI
    ri.setup_svi.assert_called_once()
    
    # 2. Run Optimization
    ri.run_optimization.assert_called_once_with(
        "mock_svi_obj",
        init_params=None,
        out_root="test_root",
        svi_state=None,
        convergence_tolerance=ANY,
        convergence_window=ANY,
        patience=ANY,
        convergence_check_interval=ANY,
        checkpoint_interval=ANY,
        max_num_epochs=500,
        init_param_jitter=ANY
    )
    
    # 3. Get Posteriors (because converged=True)
    ri.get_posteriors.assert_called_once_with(
        svi="mock_svi_obj",
        svi_state="final_state",
        out_root="test_root",
        num_posterior_samples=100,
        sampling_batch_size=ANY,
        forward_batch_size=ANY
    )
    
    assert state == "final_state"
    assert converged is True

def test_run_svi_flow_not_converged_no_posterior(mock_run_inference):
    """Test SVI flow where it fails to converge and always_get_posterior is False."""
    _, ri = mock_run_inference
    # Force non-convergence
    ri.run_optimization.return_value = ("state", {}, False)
    
    state, params, converged = _run_svi(
        ri, 
        init_params=None,
        config_file="config.yaml",
        always_get_posterior=False
    )
    
    # Should NOT get posteriors
    ri.get_posteriors.assert_not_called()
    assert converged is False

def test_run_svi_flow_always_posterior(mock_run_inference):
    """Test SVI flow where it fails to converge but posteriors are forced."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {}, False)
    
    _run_svi(ri, init_params=None, config_file="config.yaml", always_get_posterior=True)
    
    # Should get posteriors despite no convergence
    ri.get_posteriors.assert_called_once()

def test_run_svi_not_converged_stdout(mock_run_inference, capsys):
    """Test SVI not converged message."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {}, False)
    _run_svi(ri, init_params=None, config_file="config.yaml")
    captured = capsys.readouterr()
    assert "SVI run has not yet converged." in captured.out

# =============================================================================
# Tests for _run_map
# =============================================================================

def test_run_map_flow(mock_run_inference):
    """Test MAP execution flow."""
    _, ri = mock_run_inference
    init_params = {"p": 10}
    state, params, converged = _run_map(
        ri,
        init_params=init_params,
        config_file="config.yaml",
        out_root="test_map",
        max_num_epochs=1000,
        num_posterior_samples=100
    )
    
    # 1. Setup MAP
    ri.setup_svi.assert_called_once()
    
    # 2. Run Optimization
    ri.run_optimization.assert_called_once_with(
        "mock_svi_obj",
        init_params=init_params,
        out_root="test_map",
        svi_state=None,
        convergence_tolerance=ANY,
        convergence_window=ANY,
        patience=ANY,
        convergence_check_interval=ANY,
        checkpoint_interval=ANY,
        max_num_epochs=1000,
        init_param_jitter=ANY
    )
    
    # 3. Write Params
    ri.write_params.assert_called_once_with({"p": 1}, out_root="test_map")

    # 4. Get Posteriors (REMOVED from _run_map in recent cleanup, wait, let's verify logic)
    # Actually always_get_posterior=False is default
    ri.get_posteriors.assert_not_called()

def test_run_map_not_converged(mock_run_inference, capsys):
    """Test MAP not converged message."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {"p": 1}, False)
    _run_map(ri, init_params={"p": 1}, config_file="config.yaml", always_get_posterior=False)
    captured = capsys.readouterr()
    assert "MAP run converged" not in captured.out
    assert "MAP run has not yet converged" in captured.out


# =============================================================================
# Tests for _run_nuts
# =============================================================================

@pytest.fixture
def mock_ri_for_nuts(mocker):
    """Minimal ri mock for _run_nuts tests."""
    ri = MagicMock()
    mock_mcmc = MagicMock()
    mock_mcmc.get_samples.return_value = {"param": [1.0, 2.0]}
    ri.run_nuts.return_value = mock_mcmc
    mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".run_growth_analysis.summarize_posteriors"
    )
    return ri


def test_run_nuts_calls_run_nuts(mock_ri_for_nuts):
    """_run_nuts delegates to ri.run_nuts with the correct NUTS params."""
    ri = mock_ri_for_nuts
    _run_nuts(ri, config_file="cfg.yaml",
              nuts_num_warmup=10,
              nuts_num_samples=20,
              nuts_num_chains=2,
              nuts_target_accept_prob=0.8)

    ri.run_nuts.assert_called_once_with(
        num_warmup=10,
        num_samples=20,
        num_chains=2,
        target_accept_prob=0.8,
    )


def test_run_nuts_calls_get_nuts_posteriors(mock_ri_for_nuts):
    """_run_nuts calls ri.get_nuts_posteriors with the samples and forward_batch_size."""
    ri = mock_ri_for_nuts
    expected_samples = {"param": [1.0, 2.0]}
    ri.run_nuts.return_value.get_samples.return_value = expected_samples

    _run_nuts(ri, config_file="cfg.yaml", out_root="myroot", forward_batch_size=64)

    ri.get_nuts_posteriors.assert_called_once_with(
        expected_samples,
        out_root="myroot",
        forward_batch_size=64,
    )


def test_run_nuts_calls_summarize_posteriors(mock_ri_for_nuts, mocker):
    """_run_nuts calls summarize_posteriors with the posterior file and config."""
    ri = mock_ri_for_nuts
    summarize_mock = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".run_growth_analysis.summarize_posteriors"
    )

    _run_nuts(ri, config_file="my_config.yaml", out_root="myroot")

    summarize_mock.assert_called_once_with(
        posterior_file="myroot_posterior.h5",
        config_file="my_config.yaml",
        out_root="myroot",
    )


def test_run_nuts_writes_checkpoint(tmp_path, mock_ri_for_nuts):
    """_run_nuts writes a checkpoint pkl with 'mcmc_samples' key."""
    ri = mock_ri_for_nuts
    samples = {"param": [1.0, 2.0]}
    ri.run_nuts.return_value.get_samples.return_value = samples
    out_root = str(tmp_path / "nuts_chk")

    _run_nuts(ri, config_file="cfg.yaml", out_root=out_root)

    chk_path = f"{out_root}_checkpoint.pkl"
    assert os.path.exists(chk_path)
    with open(chk_path, "rb") as f:
        chk = dill.load(f)
    assert "mcmc_samples" in chk
    assert chk["mcmc_samples"] == samples


def test_run_nuts_returns_samples(mock_ri_for_nuts):
    """_run_nuts returns the mcmc_samples dict."""
    ri = mock_ri_for_nuts
    samples = {"param": [3.0]}
    ri.run_nuts.return_value.get_samples.return_value = samples

    result = _run_nuts(ri, config_file="cfg.yaml")

    assert result is samples


def test_run_nuts_stdout(mock_ri_for_nuts, capsys):
    """_run_nuts prints a completion message."""
    _run_nuts(mock_ri_for_nuts, config_file="cfg.yaml")
    captured = capsys.readouterr()
    assert "NUTS run complete." in captured.out
