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
        "seed": 7,
        "thermo_data": None,
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
    mock_parameters_df = pd.DataFrame({"genotype": ["wt", "M1"],
                                        "dk_geno": [0.0, -0.01],
                                        "activity": [1.0, 1.0]})
    mock_thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(mock_phenotype_df, mock_genotype_theta_df, mock_parameters_df),
    )

    mock_jax_key = mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey",
                                return_value="mock_key")

    lib_df, pheno_df, theta_df, params_df = library_prediction(cf="config.yaml")

    tfscreen.util.read_yaml.assert_called_once_with("config.yaml", override_keys=None)
    mock_lm_cls.assert_called_once_with(mock_config)
    mock_build_sample.assert_called_once_with(mock_config["condition_blocks"], replicate=1)
    mock_build_sim_data.assert_called_once_with(
        library_df=mock_library_df,
        sample_df=mock_sample_df,
        thermo_data=None,
    )
    mock_jax_key.assert_called_once_with(7)   # seed value

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
    assert kwargs["rng"] is not None                     # seeded RNG passed through

    assert lib_df.equals(mock_library_df)
    assert pheno_df.equals(mock_phenotype_df)
    assert theta_df.equals(mock_genotype_theta_df)
    assert params_df.equals(mock_parameters_df)


def test_library_prediction_config_error():
    with pytest.raises(FileNotFoundError):
        library_prediction("nonexistent_config.yaml")


def test_library_prediction_dk_geno_zero_passes_flag(mocker, mock_config):
    """dk_geno_zero=True in config must be forwarded to thermo_to_growth."""
    config = {**mock_config, "dk_geno_zero": True}
    mocker.patch("tfscreen.util.read_yaml", return_value=config)
    mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    ).return_value.build_library_df.return_value = pd.DataFrame({"genotype": ["wt", "M1"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0]}),
    )
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data", return_value=MagicMock())
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")

    mock_thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(
            pd.DataFrame({"theta": [0.5]}),
            pd.DataFrame({"genotype": ["wt"]}),
            pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]}),
        ),
    )

    library_prediction(cf="config.yaml")

    _, kwargs = mock_thermo.call_args
    assert kwargs["dk_geno_zero"] is True


def test_library_prediction_dk_geno_zero_makes_hyper_params_optional(mocker):
    """When dk_geno_zero=True the three dk_geno_hyper_* keys may be absent."""
    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "seed": 0,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_zero": True,
        # dk_geno_hyper_* intentionally absent
    }
    mocker.patch("tfscreen.util.read_yaml", return_value=config)
    mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    ).return_value.build_library_df.return_value = pd.DataFrame({"genotype": ["wt"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0]}),
    )
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data", return_value=MagicMock())
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")
    mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(
            pd.DataFrame({"theta": [0.5]}),
            pd.DataFrame({"genotype": ["wt"]}),
            pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]}),
        ),
    )

    # Must not raise KeyError despite missing dk_geno_hyper_* keys
    library_prediction(cf="config.yaml")


def test_library_prediction_missing_hyper_params_raises_without_dk_geno_zero(mocker):
    """Without dk_geno_zero, missing dk_geno_hyper_* must raise KeyError."""
    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "seed": 0,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        # dk_geno_hyper_* absent and dk_geno_zero not set
    }
    mocker.patch("tfscreen.util.read_yaml", return_value=config)
    mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    ).return_value.build_library_df.return_value = pd.DataFrame({"genotype": ["wt"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0]}),
    )
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data", return_value=MagicMock())
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")

    with pytest.raises(KeyError):
        library_prediction(cf="config.yaml")


# ---------------------------------------------------------------------------
# Unified seed: seed drives both JAX key and numpy RNG
# ---------------------------------------------------------------------------

def _make_mock_thermo_return():
    return (
        pd.DataFrame({"theta": [0.5]}),
        pd.DataFrame({"genotype": ["wt"]}),
        pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]}),
    )


def _patch_library_deps(mocker, config):
    mocker.patch("tfscreen.util.read_yaml", return_value=config)
    mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    ).return_value.build_library_df.return_value = pd.DataFrame({"genotype": ["wt"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0]}),
    )
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data", return_value=MagicMock())


def test_jax_key_derived_from_seed(mocker):
    """JAX PRNGKey is called with the value of seed."""
    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "seed": 42,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
    }
    _patch_library_deps(mocker, config)
    mock_jax_key = mocker.patch(
        "tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k"
    )
    mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=_make_mock_thermo_return(),
    )

    library_prediction(cf="config.yaml")

    mock_jax_key.assert_called_once_with(42)


def test_jax_key_defaults_to_zero_when_no_seed(mocker):
    """When seed is absent the JAX PRNGKey falls back to 0."""
    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
    }
    _patch_library_deps(mocker, config)
    mock_jax_key = mocker.patch(
        "tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k"
    )
    mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=_make_mock_thermo_return(),
    )

    library_prediction(cf="config.yaml")

    mock_jax_key.assert_called_once_with(0)


def test_numpy_rng_passed_to_thermo_to_growth(mocker):
    """A seeded numpy RNG is always forwarded to thermo_to_growth."""
    import numpy as np
    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "seed": 99,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
    }
    _patch_library_deps(mocker, config)
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")
    mock_thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=_make_mock_thermo_return(),
    )

    library_prediction(cf="config.yaml")

    _, kwargs = mock_thermo.call_args
    assert isinstance(kwargs["rng"], np.random.Generator)


def test_numpy_rng_seeded_with_seed(mocker):
    """The numpy RNG passed to thermo_to_growth is seeded from seed,
    so two calls with the same seed produce the same first draw."""
    import numpy as np
    captured_rngs = []

    def capture_rng(*args, **kwargs):
        captured_rngs.append(kwargs["rng"])
        return _make_mock_thermo_return()

    config = {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "mock_theta",
        "seed": 5,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
    }

    for _ in range(2):
        _patch_library_deps(mocker, config)
        mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")
        mocker.patch(
            "tfscreen.simulate.library_prediction.thermo_to_growth",
            side_effect=capture_rng,
        )
        library_prediction(cf="config.yaml")

    draw_a = captured_rngs[0].integers(0, 2**32)
    draw_b = captured_rngs[1].integers(0, 2**32)
    assert draw_a == draw_b, "Same seed must produce identical RNG state"
