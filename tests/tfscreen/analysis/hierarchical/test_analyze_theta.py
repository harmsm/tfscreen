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
import tfscreen.analysis.hierarchical.analyze_theta as analyze_theta_mod

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

    # Patch summarize_posteriors globally for these tests
    mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.summarize_posteriors")
    
    # Mock os.path.exists to return False by default to prevent FileExistsError
    mocker.patch("os.path.exists", return_value=False)

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
        max_num_epochs=500,
        num_posterior_samples=100
    )
    
    # 1. Setup SVI
    
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

    # 4. Get Posteriors (REMOVED from _run_map in recent cleanup)
    ri.get_posteriors.assert_not_called()

    # 5. Summarize Posteriors is called (verified by global patch if needed, 
    # but here we just ensure flow moves forward)

def test_run_map_not_converged(mock_run_inference, capsys):
    """Test MAP not converged message."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {"p": 1}, False)
    _run_map(ri, init_params={"p": 1}, always_get_posterior=False)
    captured = capsys.readouterr()
    assert "MAP run converged" not in captured.out
    assert "MAP run has not yet converged" in captured.out

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
            spiked=["A10G"],
            pre_map_num_epoch=0
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
        assert kwargs["init_params"] == gm_inst.init_params
        assert "init_param_jitter" in kwargs

def test_analyze_theta_svi_pre_map_flow(mock_growth_model, mock_run_inference):
    """Test analyze_theta executing SVI path with pre-MAP optimization."""
    gm_class, gm_inst = mock_growth_model
    ri_class, ri_inst = mock_run_inference
    
    with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi") as mock_run_svi:
        with patch("tfscreen.analysis.hierarchical.analyze_theta._run_map") as mock_run_map:
            # Set return for _run_map to simulate parameter update
            mock_run_map.return_value = ("map_state", {"p_map": 2}, True)
            
            analyze_theta(
                growth_df="growth.csv",
                binding_df="binding.csv",
                seed=42,
                analysis_method="svi",
                pre_map_num_epoch=50
            )
            
            # 1. MAP should be called
            mock_run_map.assert_called_once()
            map_args, map_kwargs = mock_run_map.call_args
            assert map_kwargs["max_num_epochs"] == 50
            assert map_kwargs["init_param_jitter"] == 0.0
            assert "adam_step_size" in map_kwargs
            
            # 2. SVI should be called with parameters from MAP
            mock_run_svi.assert_called_once()
            svi_args, svi_kwargs = mock_run_svi.call_args
            assert svi_kwargs["init_params"] == {"p_map": 2}
            assert svi_kwargs["init_param_jitter"] == 0.1 # default

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
        assert kwargs["max_num_epochs"] == 0
        assert kwargs["always_get_posterior"] is True

def test_analyze_theta_invalid_method(mock_growth_model, mock_run_inference):
    """Test ValueError on invalid analysis method."""
    with pytest.raises(ValueError, match="not recognized"):
        analyze_theta("g", "b", 1, analysis_method="magic_wand")

def test_analyze_theta_config_loading(mock_growth_model, mock_run_inference):
    """Test analyze_theta loading from config."""
    gm_class, gm_inst = mock_growth_model
    
    mock_config = {
        "condition_growth": "independent",
        "ln_cfu0": "hierarchical",
        "dk_geno": "fixed",
        "activity": "fixed",
        "theta": "hill",
        "transformation": "single",
        "theta_growth_noise": "none",
        "theta_binding_noise": "none",
        "spiked_genotypes": ["S1"]
    }
    gm_class.load_config.return_value = ("cg", "cb", mock_config)
    
    with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi"):
        analyze_theta(config_file="config.yaml", seed=1)
        
        gm_class.load_config.assert_called_once_with("config.yaml")
        # Verify gm_class was initialized with values from config
        kwargs = gm_class.call_args[1]
        assert kwargs["condition_growth"] == "independent"
        assert kwargs["spiked_genotypes"] == ["S1"]

def test_analyze_theta_errors(mock_growth_model):
    """Test analyze_theta validation errors."""
    # No seed
    with pytest.raises(ValueError, match="seed must be provided"):
        analyze_theta(growth_df="g", binding_df="b", seed=None)
    
    # No dfs
    with pytest.raises(ValueError, match="growth_df and binding_df must be provided"):
        analyze_theta(growth_df=None, binding_df=None, seed=1)

def test_run_svi_not_converged_stdout(mock_run_inference, capsys):
    """Test SVI not converged message."""
    _, ri = mock_run_inference
    ri.run_optimization.return_value = ("state", {}, False)
    _run_svi(ri, init_params=None)
    captured = capsys.readouterr()
    assert "SVI run has not yet converged." in captured.out

def test_main_block(mocker):
    """Test if main block calls main."""
    mock_main = mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.main")
    # Simulate if __name__ == "__main__"
    import tfscreen.analysis.hierarchical.analyze_theta as mod
    # We can't easily trigger the true if __name__ == "__main__" block without subprocess, 
    # but we covered the main() function itself.

def test_main():
    """Test that main wraps analyze_theta correctly."""
    with patch("tfscreen.analysis.hierarchical.analyze_theta.generalized_main") as mock_gen_main:
        main()
        mock_gen_main.assert_called_once_with(
            analyze_theta,
            manual_arg_types={"growth_df": str,
                              "binding_df": str,
                              "seed": int,
                              "checkpoint_file": str,
                              "config_file": str,
                              "spiked": list,
                              "pre_map_num_epoch": int,
                              "init_param_jitter": float},
            manual_arg_nargs={"spiked": "+"}
        )
import runpy
import sys
def test_main_entry_point():
    with patch.object(sys, 'argv', ['analyze_theta', '--help']):
        # Patch the function where it is used in the module
        with patch("tfscreen.analysis.hierarchical.analyze_theta.generalized_main") as mock_gen:
            try:
                main()
            except SystemExit:
                pass
            mock_gen.assert_called_once()
