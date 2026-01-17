import pytest
import os
from unittest.mock import patch, MagicMock
from tfscreen.analysis.hierarchical.analyze_theta import analyze_theta

@pytest.fixture
def mock_growth_model(mocker):
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.GrowthModel")
    mock_gm_instance = mock_gm_class.return_value
    mock_gm_instance.init_params = {"a": 1.0}
    return mock_gm_class, mock_gm_instance

@pytest.fixture
def mock_run_inference(mocker):
    mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.analyze_theta.RunInference")
    mock_ri_instance = mock_ri_class.return_value
    
    # Setup default returns for instance methods
    mock_ri_instance.setup_svi.return_value = "mock_svi_obj"
    mock_ri_instance.run_optimization.return_value = ("final_state", {"p": 1}, True)
    mock_ri_instance._iterations_per_epoch = 1
    
    return mock_ri_class, mock_ri_instance

def test_analyze_theta_checkpoint_exists_no_resume(mock_growth_model, mock_run_inference):
    """
    Test that analyze_theta raises FileExistsError if checkpoint exists and
    checkpoint_file is None.
    """
    # Mock os.path.exists to return True for the default checkpoint
    with patch("os.path.exists", return_value=True):
        with pytest.raises(FileExistsError, match="already exists"):
            analyze_theta(
                growth_df="g.csv",
                binding_df="b.csv",
                seed=1,
                out_root="test_root",
                checkpoint_file=None
            )

def test_analyze_theta_premap_checkpoint_exists_no_resume(mock_growth_model, mock_run_inference):
    """
    Test that analyze_theta raises FileExistsError if premap checkpoint exists.
    """
    # Mock os.path.exists: return True for premap_checkpoint.pkl
    def side_effect(path):
        if "premap_checkpoint.pkl" in path:
            return True
        return False

    with patch("os.path.exists", side_effect=side_effect):
        with pytest.raises(FileExistsError, match="already exists"):
            analyze_theta(
                growth_df="g.csv",
                binding_df="b.csv",
                seed=1,
                out_root="test_root",
                checkpoint_file=None,
                pre_map_num_epoch=1000
            )

def test_analyze_theta_checkpoint_exists_with_resume(mock_growth_model, mock_run_inference):
    """
    Test that analyze_theta proceeds if checkpoint exists but user provided checkpoint_file.
    """
    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi") as mock_run_svi:
            analyze_theta(
                growth_df="g.csv",
                binding_df="b.csv",
                seed=1,
                out_root="test_root",
                checkpoint_file="existing.pkl"
            )
            # Should not raise exception, should call _run_svi
            mock_run_svi.assert_called_once()

def test_analyze_theta_no_checkpoint_no_resume(mock_growth_model, mock_run_inference):
    """
    Test that analyze_theta proceeds if no checkpoint exists and checkpoint_file is None.
    """
    # Mock os.path.exists to return False
    with patch("os.path.exists", return_value=False):
        with patch("tfscreen.analysis.hierarchical.analyze_theta._run_svi") as mock_run_svi:
            analyze_theta(
                growth_df="g.csv",
                binding_df="b.csv",
                seed=1,
                out_root="test_root",
                checkpoint_file=None
            )
            # Should not raise exception, should call _run_svi
            mock_run_svi.assert_called_once()
