import pytest
from unittest.mock import MagicMock, patch, ANY
import os
import pandas as pd
import numpy as np
import yaml

from tfscreen.analysis.hierarchical.configure_growth_analysis import configure_growth_analysis
from tfscreen.analysis.hierarchical.run_growth_analysis import run_growth_analysis

@pytest.fixture
def mock_growth_model(mocker):
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.configure_growth_analysis.GrowthModel")
    mock_gm_instance = mock_gm_class.return_value
    mock_gm_instance.settings = {"batch_size": 512, "theta": "hill"}
    
    # Create simple mock priors
    mock_priors = MagicMock()
    mock_priors.val = 1.0
    mock_gm_instance.priors = mock_priors
    
    # Real tensor manager maps for configure test
    mock_tm = MagicMock()
    mock_tm.map_groups = {
        "condition": pd.DataFrame({"replicate": [1], "condition": ["A"], "map_condition": [0]})
    }
    mock_gm_instance.growth_tm = mock_tm
    
    mock_gm_instance.init_params = {
        "condition_growth_offset": np.array([0.5]),
        "scalar_param": 2.0
    }
    return mock_gm_class, mock_gm_instance

def test_configure_growth_analysis(mock_growth_model, tmpdir):
    _, _ = mock_growth_model
    
    out_root = os.path.join(tmpdir, "test")
    
    # Pass dummy strings which are required now
    configure_growth_analysis(
        growth_df="g.csv",
        binding_df="b.csv",
        out_root=out_root
    )
    
    assert os.path.exists(f"{out_root}_config.yaml")
    assert os.path.exists(f"{out_root}_guesses.csv")
    
    with open(f"{out_root}_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    assert config["data"]["growth"] == "g.csv"
    assert config["components"]["batch_size"] == 512
    assert config["init_params"]["scalar_param"] == 2.0
    
    df = pd.read_csv(f"{out_root}_guesses.csv")
    assert "condition_growth_offset" in df["parameter"].values
    assert 0.5 in df["value"].values
    assert "replicate" in df.columns

@pytest.fixture
def run_mocks(mocker):
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis.GrowthModel")
    mock_gm_instance = mock_gm_class.return_value
    mock_gm_instance.priors = MagicMock()
    mock_gm_instance.init_params = {"arr_param": np.array([0.0])}
    
    mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis.RunInference")
    mock_ri_instance = mock_ri_class.return_value
    
    mock_svi = mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis._run_svi")
    mock_map = mocker.patch("tfscreen.analysis.hierarchical.run_growth_analysis._run_map")
    
    # Configure map mock to return 3 values
    mock_map.return_value = (None, {"scalar": 2.0, "arr_param": np.array([1.5])}, True)
    
    return mock_gm_class, mock_ri_class, mock_svi, mock_map

def test_run_growth_analysis_svi(run_mocks, tmpdir):
    gm_class, ri_class, mock_svi, mock_map = run_mocks
    
    config_file = os.path.join(tmpdir, "config.yaml")
    config = {
        "data": {"growth": "g.csv", "binding": "b.csv"},
        "components": {"batch_size": 128},
        "priors": {"a": 1.0},
        "init_params": {"scalar": 2.0}
    }
    with open(config_file, "w") as f:
        yaml.dump(config, f)
        
    guesses_file = os.path.join(tmpdir, "guesses.csv")
    pd.DataFrame({
        "parameter": ["arr_param"],
        "flat_index": [0],
        "value": [1.5]
    }).to_csv(guesses_file, index=False)
    
    run_growth_analysis(
        config_file=config_file,
        guesses_file=guesses_file,
        seed=42,
        analysis_method="svi"
    )
    
    gm_class.assert_called_once()
    assert gm_class.call_args[1]["batch_size"] == 128
    
    ri_class.assert_called_once()
    
    mock_svi.assert_called_once()
    kwargs = mock_svi.call_args[1]
    
    assert kwargs["init_params"]["scalar"] == 2.0
    np.testing.assert_array_equal(kwargs["init_params"]["arr_param"], np.array([1.5]))

def test_run_growth_analysis_errors(tmpdir):
    with pytest.raises(ValueError, match="seed must be provided"):
        run_growth_analysis(config_file="missing.yaml", seed=None)
        
    with pytest.raises(FileNotFoundError):
        run_growth_analysis(config_file="missing.yaml", seed=42)
