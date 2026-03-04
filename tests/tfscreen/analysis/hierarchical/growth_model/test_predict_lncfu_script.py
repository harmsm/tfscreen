import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.scripts.predict_lncfu import predict_lncfu_cli

def test_predict_lncfu_cli(tmp_path):
    """Test the predict_lncfu_cli function logic."""
    
    config_file = str(tmp_path / "config.yaml")
    posterior_file = str(tmp_path / "post.h5")
    output_file = str(tmp_path / "out.csv")
    
    # Create a dummy dataframe for the mock prediction to return
    dummy_df = pd.DataFrame({"genotype": ["wt"], "median": [10.0]})
    
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_lncfu.read_configuration") as mock_read_config, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_lncfu.predict_lncfu") as mock_predict:
        
        # Setup mocks
        mock_read_config.return_value = (MagicMock(), {})
        mock_predict.return_value = dummy_df
        
        # Call the CLI function
        predict_lncfu_cli(config_file=config_file,
                          posterior_file=posterior_file,
                          output_file=output_file,
                          num_samples=10,
                          genotypes=["wt", "mut"],
                          t_sel=[0.0, 1.0])
        
        # Verify read_configuration was called
        mock_read_config.assert_called_once_with(config_file)
        
        # Verify predict_lncfu was called with correct arguments
        args, kwargs = mock_predict.call_args
        assert kwargs["param_posteriors"] == posterior_file
        assert kwargs["num_samples"] == 10
        assert kwargs["genotypes"] == ["wt", "mut"]
        assert kwargs["t_sel"] == [0.0, 1.0]
        
        # Verify output was saved
        saved_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(saved_df, dummy_df)

def test_main_entry_point():
    """Test the main entry point (argument parsing)."""
    from tfscreen.analysis.hierarchical.growth_model.scripts.predict_lncfu import main
    
    # Mocking generalized_main to avoid full execution
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_lncfu.generalized_main") as mock_gen_main:
        main()
        
        # Verify generalized_main was called with predict_lncfu_cli
        args, kwargs = mock_gen_main.call_args
        assert args[0].__name__ == "predict_lncfu_cli"
        assert "manual_arg_types" in kwargs
        assert "manual_arg_nargs" in kwargs
        assert kwargs["manual_arg_nargs"]["t_sel"] == "+"
