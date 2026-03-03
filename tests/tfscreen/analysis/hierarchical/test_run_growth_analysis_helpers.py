import pytest
from unittest.mock import MagicMock, ANY
import os

from tfscreen.analysis.hierarchical.run_growth_analysis import (
    _run_svi,
    _run_map
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
    mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis.RunInference")
    mock_ri_instance = mock_ri_class.return_value
    
    # Setup default returns for instance methods
    mock_ri_instance.setup_svi.return_value = "mock_svi_obj"
    mock_ri_instance.setup_map.return_value = "mock_map_obj"
    mock_ri_instance._iterations_per_epoch = 1
    
    # run_optimization returns (svi_state, params, converged)
    mock_ri_instance.run_optimization.return_value = ("final_state", {"p": 1}, True)

    # Patch summarize_posteriors where it is imported in run_growth_analysis
    mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis.summarize_posteriors")
    
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
        always_get_posterior=False
    )
    
    # Should NOT get posteriors
    ri.get_posteriors.assert_not_called()
    assert converged is False

def test_run_svi_flow_always_posterior(mock_run_inference):
    """Test SVI flow where it fails to converge but posteriors are forced."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {}, False)
    
    _run_svi(ri, init_params=None, always_get_posterior=True)
    
    # Should get posteriors despite no convergence
    ri.get_posteriors.assert_called_once()

def test_run_svi_not_converged_stdout(mock_run_inference, capsys):
    """Test SVI not converged message."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {}, False)
    _run_svi(ri, init_params=None)
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
    _run_map(ri, init_params={"p": 1}, always_get_posterior=False)
    captured = capsys.readouterr()
    assert "MAP run converged" not in captured.out
    assert "MAP run has not yet converged" in captured.out
