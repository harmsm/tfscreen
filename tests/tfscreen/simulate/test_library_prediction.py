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

    lib_df, pheno_df, theta_df, params_df, binding_df = library_prediction(cf="config.yaml")

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
    assert binding_df is None


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


# ---------------------------------------------------------------------------
# genotype_params_file integration
# ---------------------------------------------------------------------------

def _make_params_csv(tmp_path, content):
    p = tmp_path / "params.csv"
    p.write_text(content)
    return str(p)


def _base_config_with_binding(params_path, genotypes=None):
    binding = {
        "titrant_name": "iptg",
        "titrant_conc": [0.0, 0.001, 0.01],
        "noise": 0.0,
        "genotype_params_file": params_path,
    }
    if genotypes is not None:
        binding["genotypes"] = genotypes
    return {
        "condition_blocks": [{"some": "block"}],
        "theta_component": "hill_geno",
        "seed": 7,
        "thermo_data": None,
        "theta_priors": None,
        "growth": {"cond_A": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
        "binding_data": binding,
    }


def _patch_for_params_file(mocker, config):
    """Patch all heavy deps; returns (mock_thermo, mock_sim_data)."""
    mocker.patch("tfscreen.util.read_yaml", return_value=config)

    mock_library_df = pd.DataFrame({"genotype": ["wt", "A47V"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager"
    ).return_value.build_library_df.return_value = mock_library_df

    import numpy as np
    import jax.numpy as jnp
    mock_sim_data = MagicMock()
    mock_sim_data.log_titrant_conc = jnp.array(
        [np.log(1e-20), np.log(0.001), np.log(0.01)]
    )
    mock_sim_data.num_mutation = 1
    mock_sim_data.num_pair = 0
    mock_sim_data.mut_nnz_mut_idx = np.array([0], dtype=np.int32)
    mock_sim_data.mut_nnz_geno_idx = np.array([1], dtype=np.int32)
    mock_sim_data.pair_nnz_pair_idx = None
    mock_sim_data.pair_nnz_geno_idx = None
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0, 0.001, 0.01]}),
    )
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sim_data",
        return_value=mock_sim_data,
    )
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey", return_value="k")

    mock_thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(
            pd.DataFrame({"theta": [0.5, 0.3]}),
            pd.DataFrame({"genotype": ["wt", "A47V"]}),
            pd.DataFrame({
                "genotype": ["wt", "A47V"],
                "dk_geno": [0.0, -0.01],
                "activity": [1.0, 1.0],
            }),
        ),
    )
    return mock_thermo, mock_sim_data


def test_params_file_produces_binding_theta_df(mocker, tmp_path):
    """When genotype_params_file is set, binding_theta_df is populated."""
    csv = _make_params_csv(
        tmp_path,
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "wt,0.99,0.01,-4.1,2.0\n"
        "A47V,0.97,0.03,-3.8,1.8\n",
    )
    config = _base_config_with_binding(csv)
    _patch_for_params_file(mocker, config)

    # hill_geno module must be importable for sim_priors
    from tfscreen.tfmodel.generative.registry import model_registry
    assert "hill_geno" in model_registry["theta"]

    _, _, _, _, binding_df = library_prediction(cf="config.yaml")

    assert binding_df is not None
    assert set(binding_df["genotype"].unique()) == {"wt", "A47V"}
    assert set(binding_df.columns) >= {"genotype", "titrant_name", "titrant_conc", "theta_true"}


def test_params_file_theta_gc_override_passed_to_thermo(mocker, tmp_path):
    """theta_gc_override passed to thermo_to_growth contains measured genotypes."""
    csv = _make_params_csv(
        tmp_path,
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "wt,0.99,0.01,-4.1,2.0\n"
        "A47V,0.97,0.03,-3.8,1.8\n",
    )
    config = _base_config_with_binding(csv)
    mock_thermo, _ = _patch_for_params_file(mocker, config)

    library_prediction(cf="config.yaml")

    _, kwargs = mock_thermo.call_args
    override = kwargs.get("theta_gc_override", {})
    assert "wt"   in override
    assert "A47V" in override


def test_unsupported_theta_component_raises(mocker, tmp_path):
    """genotype_params_file with a non-Hill component must raise ValueError."""
    csv = _make_params_csv(
        tmp_path,
        "genotype,theta_low,theta_high,log_hill_K,hill_n\nwt,0.99,0.01,-4.1,2.0\n",
    )
    config = _base_config_with_binding(csv)
    config["theta_component"] = "thermo.O2_C12_K5_U0_a.PK"

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

    with pytest.raises(ValueError, match="hill"):
        library_prediction(cf="config.yaml")


def test_params_file_and_genotypes_coexist(mocker, tmp_path):
    """genotype_params_file and genotypes can coexist; both appear in binding_df."""
    csv = _make_params_csv(
        tmp_path,
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "A47V,0.97,0.03,-3.8,1.8\n",
    )
    config = _base_config_with_binding(csv, genotypes=["wt"])
    mock_thermo, mock_sim_data = _patch_for_params_file(mocker, config)

    # Patch sample_theta_prior (called for simulated WT in the 'genotypes' path)
    import numpy as np
    mock_wt_gc = np.array([[0.99, 0.50, 0.01]])   # shape (1, 3): 1 genotype × 3 concs
    mock_theta_param = MagicMock()
    mocker.patch(
        "tfscreen.simulate.library_prediction.sample_theta_prior",
        return_value=(mock_wt_gc, mock_theta_param),
    )

    _, _, _, _, binding_df = library_prediction(cf="config.yaml")

    assert binding_df is not None
    genos = set(binding_df["genotype"].unique())
    # WT from simulated path + A47V from params file
    assert "A47V" in genos
    assert "wt" in genos
