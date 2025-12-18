
import pytest
import pandas as pd
from unittest.mock import MagicMock
from tfscreen.simulate.library_prediction import library_prediction

@pytest.fixture
def mock_config():
    return {
        "condition_blocks": [{"some": "block"}],
        "observable_calculator": "mock_calc",
        "observable_calc_kwargs": {"k": "v"},
        "ddG_spreadsheet": "ddG.csv",
        "calibration_file": "cal.csv",
        "mut_growth_rate_shape": 1.0,
        "mut_growth_rate_scale": 0.5
    }

def test_library_prediction_success(mocker, mock_config):
    # Mock read_yaml
    mocker.patch("tfscreen.util.read_yaml", return_value=mock_config)

    # Mock LibraryManager
    mock_lm_cls = mocker.patch("tfscreen.simulate.library_prediction.library_manager.LibraryManager")
    mock_lm_instance = mock_lm_cls.return_value
    mock_library_df = pd.DataFrame({"genotype": ["WT", "M1"]})
    mock_lm_instance.build_library_df.return_value = mock_library_df

    # Mock build_sample_dataframes
    mock_sample_df = pd.DataFrame({"condition": ["A", "B"]})
    mock_build_sample = mocker.patch("tfscreen.simulate.library_prediction.build_sample_dataframes", return_value=mock_sample_df)

    # Mock thermo_to_growth
    mock_phenotype_df = pd.DataFrame({"phenotype": [1.0, 0.5]})
    mock_genotype_ddG_df = pd.DataFrame({"ddG": [0.0, 1.0]})
    mock_thermo = mocker.patch("tfscreen.simulate.library_prediction.thermo_to_growth", return_value=(mock_phenotype_df, mock_genotype_ddG_df))

    # Call function
    lib_df, pheno_df, ddG_df = library_prediction(cf="config.yaml")

    # Assertions
    tfscreen.util.read_yaml.assert_called_once_with("config.yaml", override_keys=None)
    mock_lm_cls.assert_called_once_with(mock_config)
    mock_lm_instance.build_library_df.assert_called_once()
    
    mock_build_sample.assert_called_once_with(mock_config["condition_blocks"], replicate=1)
    
    # Check thermo_to_growth call
    mock_thermo.assert_called_once()
    _, kwargs = mock_thermo.call_args
    assert kwargs["genotypes"].equals(mock_library_df["genotype"])
    assert kwargs["sample_df"].equals(mock_sample_df)
    assert kwargs["observable_calculator"] == mock_config["observable_calculator"]
    assert kwargs["observable_calc_kwargs"] == mock_config["observable_calc_kwargs"]
    assert kwargs["ddG_df"] == mock_config["ddG_spreadsheet"]
    assert kwargs["calibration_data"] == mock_config["calibration_file"]
    assert kwargs["mut_growth_rate_shape"] == mock_config["mut_growth_rate_shape"]
    assert kwargs["mut_growth_rate_scale"] == mock_config["mut_growth_rate_scale"]

    # Check returns
    assert lib_df.equals(mock_library_df)
    assert pheno_df.equals(mock_phenotype_df)
    assert ddG_df.equals(mock_genotype_ddG_df)

def test_library_prediction_config_error(mocker):
    # Mock read_yaml returning None
    mocker.patch("tfscreen.util.read_yaml", return_value=None)
    
    with pytest.raises(RuntimeError, match="Aborting simulation due to configuration error"):
        library_prediction("bad_config.yaml")

import tfscreen.util # Need this imported to patch it correctly via the module where it is used
