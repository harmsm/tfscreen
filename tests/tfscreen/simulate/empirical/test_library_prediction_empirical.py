"""Integration test for the phenotype_source=empirical branch of library_prediction."""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

import pytest

import tfscreen.util
from tfscreen.simulate.library_prediction import (
    library_prediction, _resolve_phenotype_model_path,
)
from tfscreen.simulate.empirical.fit_phenotypes import (
    PHENO_PARAMS_TRANSFORMED, PHENO_PARAMS_NATURAL,
)
from tfscreen.simulate.empirical.population import PopulationModel


def _make_model():
    mu = np.array([0.005, 1.2, -1.2, np.log(0.02), np.log(1.3)])
    cov = np.diag([1e-4, 0.2, 0.2, 0.1, 0.05])
    return PopulationModel(
        param_names_t=list(PHENO_PARAMS_TRANSFORMED),
        param_names_natural=list(PHENO_PARAMS_NATURAL),
        mu=mu, cov=cov, n_used=100,
        wt_ref={"theta_low": 0.99, "theta_high": 0.01,
                "log_hill_K": np.log(0.017), "hill_n": 2.0})


def test_resolve_phenotype_model_path(tmp_path):
    # CLI saves a single <out_prefix>_phenotype_model.json.
    saved = _make_model().save(str(tmp_path / "emp_phenotype_model"))
    assert saved.endswith(".json")
    model_json = str(tmp_path / "emp_phenotype_model.json")

    # Bare out_prefix resolves via the _phenotype_model.json convenience...
    assert _resolve_phenotype_model_path(str(tmp_path / "emp")) == model_json
    # ...the .json path works directly...
    assert _resolve_phenotype_model_path(model_json) == model_json
    # ...and without the extension.
    assert (_resolve_phenotype_model_path(str(tmp_path / "emp_phenotype_model"))
            == model_json)


def test_resolve_phenotype_model_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No phenotype model file"):
        _resolve_phenotype_model_path(str(tmp_path / "nope"))


def test_library_prediction_empirical_injects_overrides(mocker, tmp_path):
    model = _make_model()
    model.save(str(tmp_path / "pop_model"))

    cf = {
        "condition_blocks": [{"b": 1}],
        "theta_component": "hill_geno",
        "seed": 3,
        "thermo_data": None,
        "growth": {"cond": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
        "phenotype_source": "empirical",
        "empirical": {"phenotype_model": str(tmp_path / "pop_model")},
    }
    mocker.patch("tfscreen.util.read_yaml", return_value=cf)

    lm = mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager")
    lm.return_value.build_library_df.return_value = pd.DataFrame(
        {"genotype": ["wt", "A1B", "C2D"]})

    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0, 1.0]}))

    sim_data = MagicMock()
    sim_data.log_titrant_conc = np.log(np.array([1e-20, 1.0]))
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data",
                 return_value=sim_data)

    thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(pd.DataFrame({"theta": [0.5]}),
                      pd.DataFrame({"genotype": ["wt"]}),
                      pd.DataFrame({"genotype": ["wt", "A1B", "C2D"],
                                    "dk_geno": [0.0, 0.0, 0.0],
                                    "activity": [1.0, 1.0, 1.0]})))
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey",
                 return_value="k")

    library_prediction(cf="config.yaml")

    _, kwargs = thermo.call_args

    # Empirical overrides were built for every genotype and threaded through.
    assert set(kwargs["theta_gc_override"]) == {"wt", "A1B", "C2D"}
    assert set(kwargs["dk_geno_override"]) == {"wt", "A1B", "C2D"}
    assert all(len(v) == 2 for v in kwargs["theta_gc_override"].values())

    # wt is pinned to dk_geno == 0.
    assert kwargs["dk_geno_override"]["wt"] == 0.0

    # Activity forced to fixed unit (A absorbed into fitted theta).
    assert kwargs["activity_component"] == "fixed"
    assert kwargs["activity_wt"] == 1.0
    assert kwargs["activity_mut_scale"] == 0.0

    # theta_component forced to hill_geno; theta prior overrides dropped.
    assert kwargs["theta_component"] == "hill_geno"
    assert kwargs["theta_priors_overrides"] is None
    assert kwargs["theta_sim_priors_overrides"] is None


def test_simulate_known_keys_include_empirical():
    from tfscreen.simulate.selection_experiment import SIMULATE_KNOWN_KEYS
    assert "phenotype_source" in SIMULATE_KNOWN_KEYS
    assert "empirical" in SIMULATE_KNOWN_KEYS


def test_empirical_forces_hill_geno_and_warns(mocker, tmp_path):
    _make_model().save(str(tmp_path / "pop_model"))
    cf = {
        "condition_blocks": [{"b": 1}],
        "theta_component": "hill_mut",           # will be forced to hill_geno
        "theta_sim_priors": {"epi_tau_scale": 0.1},   # hill_mut-only; dropped
        "seed": 3,
        "thermo_data": None,
        "growth": {"cond": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
        "phenotype_source": "empirical",
        "empirical": {"phenotype_model": str(tmp_path / "pop_model")},
    }
    mocker.patch("tfscreen.util.read_yaml", return_value=cf)
    lm = mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager")
    lm.return_value.build_library_df.return_value = pd.DataFrame(
        {"genotype": ["wt", "A1B"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0, 1.0]}))
    sim_data = MagicMock()
    sim_data.log_titrant_conc = np.log(np.array([1e-20, 1.0]))
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data",
                 return_value=sim_data)
    thermo = mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(pd.DataFrame({"theta": [0.5]}),
                      pd.DataFrame({"genotype": ["wt"]}),
                      pd.DataFrame({"genotype": ["wt", "A1B"],
                                    "dk_geno": [0.0, 0.0],
                                    "activity": [1.0, 1.0]})))
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey",
                 return_value="k")

    with pytest.warns(UserWarning, match="forces theta_component=hill_geno"):
        library_prediction(cf="config.yaml")

    _, kwargs = thermo.call_args
    assert kwargs["theta_component"] == "hill_geno"
    assert kwargs["theta_sim_priors_overrides"] is None


def test_empirical_requires_model_path(mocker):
    cf = {
        "condition_blocks": [{"b": 1}],
        "theta_component": "hill_geno",
        "seed": 3,
        "thermo_data": None,
        "growth": {"cond": {"m": 1.0, "b": 0.0}},
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
        "phenotype_source": "empirical",
        # no 'empirical' block
    }
    mocker.patch("tfscreen.util.read_yaml", return_value=cf)
    lm = mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager")
    lm.return_value.build_library_df.return_value = pd.DataFrame(
        {"genotype": ["wt", "A1B"]})
    mocker.patch(
        "tfscreen.simulate.library_prediction.build_sample_dataframes",
        return_value=pd.DataFrame({"titrant_conc": [0.0, 1.0]}))
    sim_data = MagicMock()
    sim_data.log_titrant_conc = np.log(np.array([1e-20, 1.0]))
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data",
                 return_value=sim_data)
    mocker.patch("tfscreen.simulate.library_prediction.jax.random.PRNGKey",
                 return_value="k")

    with pytest.raises(ValueError, match="phenotype_model"):
        library_prediction(cf="config.yaml")
