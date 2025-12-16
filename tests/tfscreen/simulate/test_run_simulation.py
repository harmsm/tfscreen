
import pytest
import os
import pandas as pd
from unittest.mock import MagicMock
from tfscreen.simulate.run_simulation import run_simulation, _setup_file_output
import tfscreen.util

@pytest.fixture
def mock_config():
    return {"some": "config"}

@pytest.fixture
def mock_result_dfs():
    lib_df = pd.DataFrame({"lib": [1]})
    pheno_df = pd.DataFrame({"pheno": [1]})
    ddG_df = pd.DataFrame({"ddG": [1]})
    sample_df = pd.DataFrame({"sample": [1]})
    counts_df = pd.DataFrame({"counts": [1]})
    return lib_df, pheno_df, ddG_df, sample_df, counts_df

# -----------------------------------------------------------------------------
# Tests for _setup_file_output
# -----------------------------------------------------------------------------

def test_setup_file_output_none():
    assert _setup_file_output(None, "prefix") is None

def test_setup_file_output_bad_prefix(tmp_path):
    with pytest.raises(ValueError, match="output_prefix must be a string"):
        _setup_file_output(tmp_path, 123)

def test_setup_file_output_create_dir(tmp_path):
    out_dir = tmp_path / "new_dir"
    res = _setup_file_output(out_dir, "test_")
    assert out_dir.exists()
    assert res["library"] == str(out_dir / "test_library.csv")
    assert res["counts"] == str(out_dir / "test_counts.csv")

def test_setup_file_output_dir_exists_is_file(tmp_path):
    f = tmp_path / "file"
    f.touch()
    with pytest.raises(FileExistsError, match="exists and is not a directory"):
        _setup_file_output(f, "test_")

def test_setup_file_output_files_exist(tmp_path):
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    (out_dir / "test_library.csv").touch()
    
    with pytest.raises(ValueError, match="output files already exist"):
        _setup_file_output(out_dir, "test_")

def test_setup_file_output_success_existing_dir(tmp_path):
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    res = _setup_file_output(out_dir, "test_")
    assert res is not None

# -----------------------------------------------------------------------------
# Tests for run_simulation
# -----------------------------------------------------------------------------

def test_run_simulation_config_error(mocker):
    mocker.patch("tfscreen.util.read_yaml", return_value=None)
    with pytest.raises(RuntimeError, match="Aborting simulation due to configuration error"):
        run_simulation("bad_config.yaml", None)

def test_run_simulation_success(mocker, mock_config, mock_result_dfs):
    lib_df, pheno_df, ddG_df, sample_df, counts_df = mock_result_dfs
    
    # Mock dependencies
    mocker.patch("tfscreen.util.read_yaml", return_value=mock_config)
    mock_lib_pred = mocker.patch("tfscreen.simulate.run_simulation.library_prediction", return_value=(lib_df, pheno_df, ddG_df))
    mock_sel_exp = mocker.patch("tfscreen.simulate.run_simulation.selection_experiment", return_value=(sample_df, counts_df))
    
    # Mock file output setup
    mock_file_dict = {"library": "lib.csv", "phenotype": "pheno.csv", "genotype_ddG": "ddG.csv", "sample": "sample.csv", "counts": "counts.csv"}
    mocker.patch("tfscreen.simulate.run_simulation._setup_file_output", return_value=mock_file_dict)
    
    # Mock to_csv
    mocker.patch.object(pd.DataFrame, "to_csv")

    results = run_simulation("config.yaml", "out_dir", "prefix_")

    # Verify calls
    mock_lib_pred.assert_called_once_with(mock_config)
    mock_sel_exp.assert_called_once_with(mock_config, lib_df, pheno_df)
    
    assert results["library"].equals(lib_df)
    assert results["counts"].equals(counts_df)
    
    # Verify file writing
    assert pd.DataFrame.to_csv.call_count == 5 
    # Logic for checking arguments could be added but call_count gives good confidence here given the simple loop logic

def test_run_simulation_no_output(mocker, mock_config, mock_result_dfs):
    lib_df, pheno_df, ddG_df, sample_df, counts_df = mock_result_dfs
    
    mocker.patch("tfscreen.util.read_yaml", return_value=mock_config)
    mocker.patch("tfscreen.simulate.run_simulation.library_prediction", return_value=(lib_df, pheno_df, ddG_df))
    mocker.patch("tfscreen.simulate.run_simulation.selection_experiment", return_value=(sample_df, counts_df))
    
    # Explicitly check _setup_file_output is called with None
    mock_setup = mocker.patch("tfscreen.simulate.run_simulation._setup_file_output", return_value=None)
    
    mocker.patch.object(pd.DataFrame, "to_csv")

    results = run_simulation("config.yaml", None)
    
    mock_setup.assert_called_once_with(None, "tfscreen_")
    assert pd.DataFrame.to_csv.call_count == 0
