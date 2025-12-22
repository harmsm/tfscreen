import pytest
import os
import yaml
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from tfscreen.analysis.hierarchical.summarize_posteriors import summarize_posteriors, main

@pytest.fixture
def mock_config():
    return {
        "settings": {
            "condition_growth": "hierarchical",
            "ln_cfu0": "hierarchical",
            "dk_geno": "hierarchical",
            "activity": "hierarchical",
            "theta": "hill",
            "transformation": "none",
            "theta_growth_noise": "none",
            "theta_binding_noise": "none",
            "spiked_genotypes": None
        },
        "growth_df": "growth.csv",
        "binding_df": "binding.csv"
    }

def test_summarize_posteriors_full(tmpdir, mock_config):
    config_file = os.path.join(tmpdir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)
    
    posterior_file = os.path.join(tmpdir, "post.npz")
    np.savez(posterior_file, a=np.array([1]))
    
    with patch("tfscreen.analysis.hierarchical.summarize_posteriors.GrowthModel") as MockGM:
        mock_gm = MockGM.return_value
        mock_gm.extract_parameters.return_value = {"param1": pd.DataFrame({"x": [1]})}
        mock_gm.extract_growth_predictions.return_value = pd.DataFrame({"y": [2]})
        mock_gm.extract_theta_curves.return_value = pd.DataFrame({"z": [3]})
        
        out_root = os.path.join(tmpdir, "tfs")
        summarize_posteriors(posterior_file, config_file, out_root=out_root)
        
        assert os.path.exists(f"{out_root}_sum_param1.csv")
        assert os.path.exists(f"{out_root}_sum_growth_pred.csv")
        assert os.path.exists(f"{out_root}_sum_theta_curves.csv")

def test_summarize_posteriors_errors():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        summarize_posteriors("p.npz", "nonexistent.yaml")
    
    # Create a dummy config with all required keys
    full_config = {
        "settings": {k: "dummy" for k in [
            "condition_growth", "ln_cfu0", "dk_geno", "activity", "theta", 
            "transformation", "theta_growth_noise", "theta_binding_noise", "spiked_genotypes"
        ]},
        "growth_df": "g", "binding_df": "b"
    }
    with patch("builtins.open", mock_open(read_data=yaml.dump(full_config))):
        with patch("os.path.exists", side_effect=lambda x: x == "config.yaml"):
            with patch("tfscreen.analysis.hierarchical.summarize_posteriors.GrowthModel"):
                with pytest.raises(FileNotFoundError, match="Posterior file not found"):
                    summarize_posteriors("missing.npz", "config.yaml")

def test_main():
    with patch("tfscreen.analysis.hierarchical.summarize_posteriors.generalized_main") as mock_gen:
        main()
        mock_gen.assert_called_once_with(summarize_posteriors)

def test_summarize_posteriors_no_hill(tmpdir, mock_config):
    mock_config["settings"]["theta"] = "categorical"
    config_file = os.path.join(tmpdir, "config_cat.yaml")
    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)
    
    posterior_file = os.path.join(tmpdir, "post.npz")
    np.savez(posterior_file, a=np.array([1]))
    
    with patch("tfscreen.analysis.hierarchical.summarize_posteriors.GrowthModel") as MockGM:
        mock_gm = MockGM.return_value
        mock_gm.extract_parameters.return_value = {}
        mock_gm.extract_growth_predictions.return_value = pd.DataFrame()
        
        out_root = os.path.join(tmpdir, "tfs_cat")
        summarize_posteriors(posterior_file, config_file, out_root=out_root)
        
        # Should NOT have theta_curves
        assert not os.path.exists(f"{out_root}_sum_theta_curves.csv")

import runpy
import sys
def test_entry_point():
    # Mock sys.argv to avoid parser error/exit
    with patch.object(sys, 'argv', ['summarize_posteriors', '--help']):
        # Patch the root function to avoid re-import issues with runpy
        with patch("tfscreen.util.cli.generalized_main.generalized_main") as mock_gen:
            try:
                runpy.run_module("tfscreen.analysis.hierarchical.summarize_posteriors", run_name="__main__")
            except SystemExit:
                pass
            mock_gen.assert_called_once()
