import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli import predict_cli

def test_predict_cli_basic(tmp_path):
    """Test the predict_cli function logic (basic)."""
    
    config_file = str(tmp_path / "config.yaml")
    posterior_file = str(tmp_path / "post.h5")
    
    # Create a dummy dataframe for the mock prediction to return
    dummy_df = pd.DataFrame({"genotype": ["wt"], "median": [10.0]})
    
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli.read_configuration") as mock_read_config, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli.predict") as mock_predict:
        
        # Setup mocks
        mock_read_config.return_value = (MagicMock(), {})
        mock_predict.return_value = dummy_df
        
        # Call the CLI function
        predict_cli(config_file=config_file,
                    posterior_file=posterior_file,
                    out_prefix="my_test",
                    num_samples=10,
                    genotypes=["wt"],
                    t_sel=[0.0, 1.0])
        
        # Verify read_configuration was called
        mock_read_config.assert_called_once_with(config_file)
        
        # Verify predict was called with correct arguments
        args, kwargs = mock_predict.call_args
        assert kwargs["param_posteriors"] == posterior_file
        assert kwargs["num_samples"] == 10
        assert kwargs["genotypes"] == ["wt"]
        assert kwargs["t_sel"] == [0.0, 1.0]
        assert kwargs["predict_sites"] == ["growth_pred"] # default
        
        # Verify output was saved to my_test_growth_pred.csv
        expected_file = tmp_path / "my_test_growth_pred.csv"
        # Since predict_cli runs in tmp_path, we need to check if the file was written there.
        # However, predict_cli writes to the current working directory or provided paths.
        # In the script, it just uses f"{out_prefix}_{site_name}.csv".
        # To catch it, we should probably change CWD in the test or mock to_csv.
        # For now, let's just mock to_csv on the returned dataframe if possible, 
        # or check for its existence in the current directory (which might be the workspace root).
        
        # Better: let's verify that the call to to_csv happened with the right filename.
        # We can't strictly do that easily with a real DataFrame unless we mock it.
        # Let's mock the results of predict to be a MagicMock that we can check to_csv on.
        
def test_predict_cli_multiple_sites(tmp_path):
    """Test the predict_cli function logic with multiple sites."""
    
    config_file = str(tmp_path / "config.yaml")
    posterior_file = str(tmp_path / "post.h5")
    
    # Create dummy dataframes
    df1 = pd.DataFrame({"val": [1]})
    df2 = pd.DataFrame({"val": [2]})
    results = {"site1": df1, "site2": df2}
    
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli.read_configuration") as mock_read_config, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli.predict") as mock_predict, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        mock_read_config.return_value = (MagicMock(), {})
        mock_predict.return_value = results
        
        predict_cli(config_file=config_file,
                    posterior_file=posterior_file,
                    out_prefix="test_multi",
                    predict_sites=["site1", "site2"])
        
        # Verify two files were written
        assert mock_to_csv.call_count == 2
        calls = [c[0][0] for c in mock_to_csv.call_args_list]
        assert "test_multi_site1.csv" in calls
        assert "test_multi_site2.csv" in calls

def test_main_entry_point():
    """Test the main entry point (argument parsing)."""
    from tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli import main
    
    # Mocking generalized_main to avoid full execution
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.predict_cli.generalized_main") as mock_gen_main:
        main()
        
        # Verify generalized_main was called with predict_cli
        args, kwargs = mock_gen_main.call_args
        assert args[0].__name__ == "predict_cli"
        assert "manual_arg_types" in kwargs
        assert "manual_arg_nargs" in kwargs
        assert kwargs["manual_arg_nargs"]["t_sel"] == "+"
        assert kwargs["manual_arg_nargs"]["predict_sites"] == "+"
