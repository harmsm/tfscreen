import pytest
from unittest.mock import MagicMock, patch, ANY
import os

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.analyze_theta import (
    analyze_theta,
    _run_svi,
    _run_map,
    main
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_run_inference(mocker):
    """
    Mocks the RunInference class and its instance methods.
    """
    mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.RunInference")
    mock_ri_instance = mock_ri_class.return_value
    
    # Setup default returns for instance methods
    mock_ri_instance.setup_svi.return_value = "mock_svi_obj"
    mock_ri_instance.setup_map.return_value = "mock_map_obj"
    
    # run_optimization returns (svi_state, params, converged)
    mock_ri_instance.run_optimization.return_value = ("final_state", {"p": 1}, True)
    
    return mock_ri_class, mock_ri_instance

@pytest.fixture
def mock_growth_model(mocker):
    """
    Mocks the GrowthModel class.
    """
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.GrowthModel")
    mock_gm_instance = mock_gm_class.return_value
    mock_gm_instance.init_params = {"a": 1.0} # Needed for MAP
    return mock_gm_class, mock_gm_instance

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
        num_steps=500,
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
        checkpoint_interval=ANY,
        num_steps=500
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

# =============================================================================
# Tests for _run_map
# =============================================================================

def test_run_map_flow(mock_run_inference):
    """Test MAP execution flow."""
    _, ri = mock_run_inference
    init_params = {"p": 10}
    
    with patch("os.path.isfile", return_value=True), \
         patch("os.remove") as mock_remove:
        
        state, params, converged = _run_map(
            ri,
            init_params=init_params,
            out_root="test_map",
            map_num_steps=1000
        )
        
        # Should delete old losses file
        mock_remove.assert_called_once_with("test_map_losses.csv")
    
    # 1. Setup MAP
    ri.setup_map.assert_called_once()
    
    # 2. Run Optimization
    ri.run_optimization.assert_called_once_with(
        "mock_map_obj",
        init_params=init_params,
        out_root="test_map",
        svi_state=None,
        convergence_tolerance=ANY,
        convergence_window=ANY,
        checkpoint_interval=ANY,
        num_steps=1000
    )
    
    # 3. Write Params
    ri.write_params.assert_called_once_with({"p": 1}, out_root="test_map")

# =============================================================================
# Tests for analyze_theta (Orchestrator)
# =============================================================================

def test_analyze_theta_svi_mode(mock_growth_model, mock_run_inference):
    """Test analyze_theta executing SVI path."""
    gm_class, gm_inst = mock_growth_model
    ri_class, ri_inst = mock_run_inference
    
    # Mock the internal helper to verify it's called
    with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi") as mock_run_svi:
        
        analyze_theta(
            growth_df="growth.csv",
            binding_df="binding.csv",
            seed=42,
            analysis_method="svi",
            batch_size=512,
            spiked=["A10G"]
        )
        
        # 1. Initialize GrowthModel
        gm_class.assert_called_once()
        assert gm_class.call_args[1]["batch_size"] == 512
        assert gm_class.call_args[1]["spiked_genotypes"] == ["A10G"]
        
        # 2. Initialize RunInference
        ri_class.assert_called_once_with(gm_inst, 42)
        
        # 3. Call _run_svi
        mock_run_svi.assert_called_once()
        
        # Verify passed arguments
        # ri is positional arg 0
        args, kwargs = mock_run_svi.call_args
        assert args[0] == ri_inst
        assert kwargs["init_params"] is None

def test_analyze_theta_map_mode(mock_growth_model, mock_run_inference):
    """Test analyze_theta executing MAP path."""
    _, gm_inst = mock_growth_model
    _, ri_inst = mock_run_inference
    
    with patch("tfscreen.analysis.hierarchical.analyze_theta._run_map") as mock_run_map:
        
        analyze_theta(
            growth_df="g", binding_df="b", seed=1,
            analysis_method="map"
        )
        
        mock_run_map.assert_called_once()
        
        # Verify passed arguments
        # _run_map(ri, init_params, ...) both positional
        args, kwargs = mock_run_map.call_args
        assert args[0] == ri_inst
        assert args[1] == gm_inst.init_params

def test_analyze_theta_posterior_mode(mock_growth_model, mock_run_inference):
    """Test analyze_theta executing 'posterior' path (SVI with 0 steps)."""
    _, ri_inst = mock_run_inference
    
    with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi") as mock_run_svi:
        
        analyze_theta(
            growth_df="g", binding_df="b", seed=1,
            analysis_method="posterior"
        )
        
        mock_run_svi.assert_called_once()
        
        # Check positional args
        args, kwargs = mock_run_svi.call_args
        assert args[0] == ri_inst
        
        # Check keyword overrides
        assert kwargs["num_steps"] == 0
        assert kwargs["always_get_posterior"] is True

def test_analyze_theta_invalid_method(mock_growth_model, mock_run_inference):
    """Test ValueError on invalid analysis method."""
    with pytest.raises(ValueError, match="not recognized"):
        analyze_theta("g", "b", 1, analysis_method="magic_wand")

# =============================================================================
# Tests for main
# =============================================================================

def test_main():
    """Test that main wraps analyze_theta correctly."""
    with patch("tfscreen.analysis.hierarchical.analyze_theta.generalized_main") as mock_gen_main:
        main()
        mock_gen_main.assert_called_once_with(
            analyze_theta,
            manual_arg_types={"seed": int, "checkpoint_file": str, "spiked": list},
            manual_arg_nargs={"spiked": "+"}
        )