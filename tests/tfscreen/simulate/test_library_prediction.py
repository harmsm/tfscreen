import pytest
import pandas as pd
import tfscreen.util
from unittest.mock import MagicMock
from tfscreen.simulate.library_prediction import library_prediction


@pytest.fixture
def mock_config():
    return {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "theta_rng_seed": 7,
        "struct_ensemble_path": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
    }


def test_library_prediction_success(mocker, mock_config):
    mocker.patch("tfscreen.util.read_yaml", return_value=mock_config)

    mock_lm_cls = mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    )
    mock_library_df = pd.DataFrame({"genotype": ["wt", "M1"]})
    mock_lm_cls.return_value.build_library_df.return_value = mock_library_df

    mock_sample_df = pd.DataFrame({"titrant_conc": [0.0, 1.0]})
    mock_build_sample = mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=mock_sample_df,
    )

    mock_sim_data = MagicMock()
    mock_build_sim_data = mocker.patch(
        "tfscreen.simulate.library_prediction.build_sim_data",
        return_value=mock_sim_data,
    )

    mock_phenotype_df = pd.DataFrame({"theta": [0.5, 0.3]})
    mock_genotype_theta_df = pd.DataFrame({"genotype": ["wt", "M1"]})
    mock_thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(mock_phenotype_df, mock_genotype_theta_df),
    )

    mock_jax_key = mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey",
                                return_value="mock_key")

    lib_df, pheno_df, theta_df = library_prediction(cf="config.yaml")

    tfscreen.util.read_yaml.assert_called_once_with("config.yaml", override_keys=None)
    mock_lm_cls.assert_called_once_with(mock_config)
    mock_build_sample.assert_called_once_with(mock_config["condition_blocks"], replicate=1)
    mock_build_sim_data.assert_called_once_with(
        library_df=mock_library_df,
        sample_df=mock_sample_df,
        struct_ensemble_path=None,
    )
    mock_jax_key.assert_called_once_with(7)

    mock_thermo.assert_called_once()
    _, kwargs = mock_thermo.call_args
    assert kwargs["theta_component"] == "mock_theta"
    assert kwargs["sim_data"] is mock_sim_data
    assert kwargs["growth_params"] == mock_config["growth"]
    assert kwargs["dk_geno_hyper_loc"] == mock_config["dk_geno_hyper_loc"]
    assert kwargs["dk_geno_hyper_scale"] == mock_config["dk_geno_hyper_scale"]
    assert kwargs["dk_geno_hyper_shift"] == mock_config["dk_geno_hyper_shift"]
    assert kwargs["activity_component"] == "fixed"       # default when key absent
    assert kwargs["activity_priors_overrides"] is None

    assert lib_df.equals(mock_library_df)
    assert pheno_df.equals(mock_phenotype_df)
    assert theta_df.equals(mock_genotype_theta_df)


def test_library_prediction_config_error():
    with pytest.raises(FileNotFoundError):
        library_prediction("nonexistent_config.yaml")
