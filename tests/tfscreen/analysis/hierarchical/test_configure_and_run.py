import pytest
import os
import pandas as pd
import numpy as np
import yaml
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

from tfscreen.analysis.hierarchical.growth_model.scripts.configure_growth_analysis import configure_growth_analysis
from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import run_growth_analysis
from tfscreen.analysis.hierarchical.growth_model.configuration_io import (
    read_configuration,
    _update_dataclass
)

@pytest.fixture
def mock_gm(mocker):
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.configure_growth_analysis.GrowthModel")
    mock_gm_inst = mock_gm_class.return_value
    
    # Mock settings
    mock_gm_inst.settings = {"batch_size": 128, "theta": "hill"}
    
    # Mock priors
    mock_priors = MagicMock()
    mock_priors.val = 1.0
    mock_gm_inst.priors = mock_priors
    
    # Mock init_params
    mock_gm_inst.init_params = {
        "scalar_p": 0.5,
        "arr_p": np.array([1.0, 2.0]),
        "cond_p": np.array([0.1]),
        "theta_p": np.array([0.2]),
        "geno_p": np.array([0.3]),
        "ln_cfu0_p": np.array([0.4])
    }
    
    # Mock tensor manager maps
    mock_tm = MagicMock()
    mock_tm.map_groups = {
        "condition": pd.DataFrame({"replicate": ["R1"], "condition": ["C1"], "map_condition_rep": [0]}),
        "genotype": pd.DataFrame({"genotype": ["G1"], "map_genotype": [0]}),
        "theta": pd.DataFrame({"titrant_name": ["T1"], "titrant_conc": [0.1], "genotype": ["G1"], "map_theta": [0]}),
        "ln_cfu0": pd.DataFrame({"replicate": ["R1"], "condition_pre": ["CP1"], "genotype": ["G1"], "map_ln_cfu0": [0]})
    }
    mock_gm_inst.growth_tm = mock_tm
    
    return mock_gm_class, mock_gm_inst

def test_configure_growth_analysis_coverage(mock_gm, tmpdir):
    _, mock_gm_inst = mock_gm
    out_root = os.path.join(tmpdir, "test")
    
    # Mock priors with dict and dataclass members for coverage
    mock_sub = MagicMock()
    mock_sub.__dataclass_fields__ = {}
    mock_sub.sub_val = 1.0
    
    mock_priors = MagicMock()
    mock_priors.dc = mock_sub
    mock_priors.d = {"k": 2.0, "arr": np.zeros((2,))}
    mock_priors.scalar = 3.0
    mock_gm_inst.priors = mock_priors
    
    # Add some specific parameter names to trigger branch coverage
    mock_gm_inst.init_params = {
        "scalar": 1.0,
        "condition_growth_offset": np.array([0.5]),
        "theta_logit_low": np.array([0.6]),
        "dk_geno_loc": np.array([0.7]),
        "activity_loc": np.array([0.8]),
        "ln_cfu0_loc": np.array([0.9]),
        "unknown_arr": np.zeros((2,)), # 1D
        "unknown_2d": np.zeros((2, 2)), # 2D
        "unknown_3d": np.zeros((2, 2, 2)) # 3D
    }
    
    configure_growth_analysis(growth_df="g.csv", binding_df="b.csv", out_root=out_root)
    
    assert os.path.exists(f"{out_root}_config.yaml")
    assert os.path.exists(f"{out_root}_priors.csv")
    assert os.path.exists(f"{out_root}_guesses.csv")
    
    # Check guesses content for mapping coverage
    guesses_df = pd.read_csv(f"{out_root}_guesses.csv")
    assert "replicate" in guesses_df.columns # condition mapping
    assert "titrant_name" in guesses_df.columns # theta mapping
    assert "genotype" in guesses_df.columns # genotype mapping
    assert "dim_0" in guesses_df.columns # fallback mapping

def test_read_configuration_logic(tmpdir, mocker):
    config_path = os.path.join(tmpdir, "config.yaml")
    priors_path = os.path.join(tmpdir, "priors.csv")
    guesses_path = os.path.join(tmpdir, "guesses.csv")
    
    config = {
        "data": {"growth": "g.csv", "binding": "b.csv"},
        "components": {"batch_size": 256},
        "priors_file": "priors.csv",
        "guesses_file": "guesses.csv"
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    pd.DataFrame({"parameter": ["p1", "model_type"], "value": [1.0, "empirical"]}).to_csv(priors_path, index=False)
    # Test different guess formats
    pd.DataFrame({
        "parameter": ["param1", "param2", "param3"],
        "value": [0.5, 0.6, 0.7],
        "flat_index": [0, 0, 0]
    }).to_csv(guesses_path, index=False)
    
    mock_gm_class = mocker.patch("tfscreen.analysis.hierarchical.growth_model.configuration_io.GrowthModel")
    mock_gm_inst = mock_gm_class.return_value
    mock_gm_inst.init_params = {"param1": 0.0, "param2": jnp.zeros((1,)), "param3": 0.0}
    
    # Mock update_dataclass
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.configuration_io._update_dataclass"
, return_value=MagicMock())
    
    gm, init_params = read_configuration(config_path)
    
    assert gm == mock_gm_inst
    assert init_params["param1"] == 0.5
    assert isinstance(init_params["param2"], jnp.ndarray)

def test_read_configuration_errors(tmpdir):
    config_path = os.path.join(tmpdir, "bad_config.yaml")
    
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        read_configuration("nonexistent.yaml")
        
    # All subsequent tests involve parsing the yaml and potentially initializing GM
    with patch("tfscreen.analysis.hierarchical.growth_model.configuration_io.GrowthModel"):
        # 2. priors_file missing in yaml
        with open(config_path, "w") as f:
            yaml.dump({"data":{"growth":"g", "binding":"b"}, "components":{}}, f)
        with pytest.raises(ValueError, match="priors_file not specified"):
            read_configuration(config_path)
            
        # 3. priors_file not found on disk
        with open(config_path, "w") as f:
            yaml.dump({"data":{"growth":"g", "binding":"b"}, "components":{}, "priors_file": "missing.csv"}, f)
        with pytest.raises(FileNotFoundError, match="Priors file not found"):
            read_configuration(config_path)

        # 4. guesses_file missing in yaml
        priors_path = os.path.join(tmpdir, "priors.csv")
        pd.DataFrame({"parameter":["p"], "value":[1]}).to_csv(priors_path, index=False)
        with open(config_path, "w") as f:
             yaml.dump({"data":{"growth":"g", "binding":"b"}, "components":{}, "priors_file": "priors.csv"}, f)
        with pytest.raises(ValueError, match="guesses_file not specified"):
            read_configuration(config_path)

        # 5. Missing parameters in guesses
        guesses_path = os.path.join(tmpdir, "guesses.csv")
        pd.DataFrame({"parameter":["param1"], "value":[1]}).to_csv(guesses_path, index=False)
        with open(config_path, "w") as f:
             yaml.dump({"data":{"growth":"g", "binding":"b"}, "components":{}, "priors_file": "priors.csv", "guesses_file": "guesses.csv"}, f)
        
        with patch("tfscreen.analysis.hierarchical.growth_model.configuration_io.GrowthModel") as mock_gm_class:
            mock_gm_inst = mock_gm_class.return_value
            mock_gm_inst.init_params = {"param1": 0, "param2": 0} # param2 missing in CSV
            with pytest.raises(ValueError, match="Missing initial guesses for parameters"):
                read_configuration(config_path)

        # 6. Guesses file not found on disk
        with open(config_path, "w") as f:
             yaml.dump({"data":{"growth":"g", "binding":"b"}, "components":{}, "priors_file": "priors.csv", "guesses_file": "missing_guesses.csv"}, f)
        with pytest.raises(FileNotFoundError, match="Guesses file not found"):
            read_configuration(config_path)

def test_configure_growth_analysis_errors(tmpdir):
    # Missing dfs
    with pytest.raises(ValueError, match="growth_df and binding_df must be provided"):
        configure_growth_analysis(growth_df=None, binding_df=None)

def test_update_dataclass_recursion():
    from dataclasses import dataclass
    @dataclass
    class Sub:
        x: float = 0.0
    @dataclass
    class Main:
        sub: Sub = None
        y: float = 0.0
        
    m = Main(sub=Sub())
    # Test recursion and update
    m2 = _update_dataclass(m, "", {"sub.x": 10.0, "y": 20.0})
    assert m2.sub.x == 10.0
    assert m2.y == 20.0
    
    # Test no update
    m3 = _update_dataclass(m, "", {"z": 30.0})
    assert m3 == m

def test_mains(mocker):
    # Test main functions via mocker to cover boilerplate
    mocker.patch("sys.argv", ["tfs-configure", "--growth_df", "g.csv", "--binding_df", "b.csv"])
    # Mock the internal call so we don't actually run it
    mock_run = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.configure_growth_analysis.configure_growth_analysis", autospec=True)
    from tfscreen.analysis.hierarchical.growth_model.scripts.configure_growth_analysis import main
    main()
    assert mock_run.called

    mocker.patch("sys.argv", ["tfs-run", "c.yaml", "--seed", "42"])
    mock_run_analysis = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis.run_growth_analysis", autospec=True)
    from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import main as main_run
    main_run()
    assert mock_run_analysis.called

def test_run_growth_analysis_svi_full(tmpdir, mocker):
    with patch("os.path.exists", return_value=False):
        config_path = os.path.join(tmpdir, "config.yaml")
        # Minimal valid setup for read_configuration
        mock_gm_run = MagicMock()
        mock_gm_run.settings = {"batch_size": 256}
        mock_gm_run._ln_cfu_df = "g.csv"
        mock_gm_run._binding_df = "b.csv"
        mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis.read_configuration", return_value=(mock_gm_run, {"p":1.0}))
        
        # Mock write_configuration to avoid yaml problems with mocks
        mock_ri_class = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis.RunInference")
        mock_svi = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis._run_svi")
        mock_map = mocker.patch("tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis._run_map")
        mock_map.return_value = (None, {"p":1.1}, True)

        # Test SVI with pre-map. 
        run_growth_analysis(config_path, seed=42, analysis_method="svi", pre_map_num_epoch=100)
        mock_map.assert_called_once()
        mock_svi.assert_called_once()
        
        # Test MAP
        mock_map.reset_mock()
        run_growth_analysis(config_path, seed=42, analysis_method="map")
        mock_map.assert_called_once()
        
        # Test posterior
        run_growth_analysis(config_path, seed=42, analysis_method="posterior")
        # posterior calls _run_svi with max_num_epochs=0
        assert mock_svi.call_count == 2
        
        # Test invalid method
        with pytest.raises(ValueError, match="not recognized"):
            run_growth_analysis(config_path, seed=42, analysis_method="invalid")

def test_run_growth_analysis_no_seed():
    with pytest.raises(ValueError, match="seed must be provided"):
        run_growth_analysis("c.yaml")
