"""
The ``ddG_truth_sd`` simulate setting: SD of the simulated per-structure ΔΔG.

PddG/PnnC draw ``theta_ddG = prior_mean + sigma_s * offset`` with
``offset ~ Normal(0, 1)``; in the simulator's draw that sum is the ground
truth.  ``ddG_truth_sd`` sets ``sigma_s`` for the draw only (default: the
init value, 1 kT), in the unit of ``thermo_options['ddG_units']``.
"""
import os
import warnings
from unittest.mock import MagicMock

import jax
import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro import handlers

from tfscreen.simulate.sample_theta import ddG_truth_sigma_s, sample_theta_prior
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.thermo_to_growth import thermo_to_growth
from tfscreen.tfmodel.generative.registry import model_registry

C12 = "thermo.O2_C12_K5_U0_a.PddG"
C4 = "thermo.O2_C4_K3_U0_a.PddG"
C12_STRUCTS = ["H", "HO", "L", "LO", "HE2", "LE2"]
C4_STRUCTS = ["H", "HD", "L", "LE2"]
RT = 1.987204259e-3 * 298.15          # kcal/mol per kT at 298.15 K

SAMPLE_DF = pd.DataFrame({
    "titrant_name": "iptg",
    "titrant_conc": [0.0, 0.01, 1.0],
    "condition_pre": "pre",
    "condition_sel": "sel",
})
GROWTH = {"pre": {"m": 0.0, "b": 0.01}, "sel": {"m": 0.01, "b": 0.01}}
_AA = "ACDEFGHIKLMNPQRSTVWY"


def _library(num_sites):
    """wt plus every substitution at ``num_sites`` alanine positions."""
    muts = [f"A{site}{aa}" for site in range(1, num_sites + 1)
            for aa in _AA if aa != "A"]
    return ["wt"] + muts, muts


def _table(path, muts, structs, scale=1.0, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 1, (len(muts), len(structs))) * scale,
                      columns=structs)
    df.insert(0, "mut", muts)
    df.to_csv(path, index=False)
    return str(path), df.set_index("mut")


def _draw(component, genotypes, thermo_data, thermo_options=None,
          ddG_truth_sd=None, seed=0):
    lib = pd.DataFrame({"genotype": genotypes})
    sim_data = build_sim_data(lib, SAMPLE_DF, thermo_data=thermo_data,
                              theta_component=component,
                              thermo_options=thermo_options)
    override = ddG_truth_sigma_s(ddG_truth_sd, component, thermo_options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return thermo_to_growth(lib["genotype"], sim_data, SAMPLE_DF,
                                component, jax.random.PRNGKey(seed), GROWTH,
                                rng=np.random.default_rng(seed),
                                return_mutation_truth=True,
                                theta_model_params_override=override)


def _truth_minus_table(truth, table, structs, factor=1.0):
    mut = truth["mutation"].set_index("mutation")
    return pd.DataFrame({s: mut[f"ddG_{s}"] - table.loc[mut.index, s] * factor
                         for s in structs})


# ── semantics ────────────────────────────────────────────────────────────────

def test_zero_sd_gives_the_table_exactly_in_kT(tmp_path):
    genotypes, muts = _library(2)
    path, table = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    truth = _draw(C12, genotypes, path, {"ddG_units": "kcal/mol"},
                  ddG_truth_sd=0)[3]
    diff = _truth_minus_table(truth, table, C12_STRUCTS, factor=1 / RT)
    np.testing.assert_allclose(diff.to_numpy(), 0.0, atol=1e-5)
    assert np.isfinite(truth["mutation"].drop(columns="mutation")
                       .to_numpy()).all()


def test_positive_sd_sets_the_spread_independently_per_structure(tmp_path):
    genotypes, muts = _library(30)                  # 570 mutations
    path, table = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    truth = _draw(C12, genotypes, path, ddG_truth_sd=0.5)[3]
    diff = _truth_minus_table(truth, table, C12_STRUCTS)
    n = len(diff)
    # SE of an SD estimate is about sd / sqrt(2n): 0.015 here.
    np.testing.assert_allclose(diff.std(ddof=1), 0.5, atol=0.06)
    np.testing.assert_allclose(diff.mean(), 0.0, atol=4 * 0.5 / np.sqrt(n))
    corr = np.corrcoef(diff.to_numpy().T)
    off_diag = corr[~np.eye(len(C12_STRUCTS), dtype=bool)]
    assert np.abs(off_diag).max() < 4 / np.sqrt(n)


def test_per_structure_mapping(tmp_path):
    genotypes, muts = _library(30)
    path, table = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    sds = {"LE2": 2.0, "H": 0.0, "HO": 0.25, "L": 1.0, "LO": 0.5, "HE2": 0.0}
    truth = _draw(C12, genotypes, path, ddG_truth_sd=sds)[3]
    diff = _truth_minus_table(truth, table, C12_STRUCTS)
    for s, sd in sds.items():
        if sd == 0:
            np.testing.assert_allclose(diff[s], 0.0, atol=1e-5)
        else:
            assert diff[s].std(ddof=1) == pytest.approx(sd, rel=0.12)


def test_sd_in_kcal_is_converted_like_the_table(tmp_path):
    """A kcal/mol table with a kcal/mol SD gives the same truth (in kT) as
    the kT table with the SD in kT."""
    genotypes, muts = _library(2)
    path_kT, table = _table(tmp_path / "kT.csv", muts, C4_STRUCTS)
    kcal = table * RT
    kcal.reset_index().to_csv(tmp_path / "kcal.csv", index=False)
    ref = _draw(C4, genotypes, path_kT, ddG_truth_sd=0.5 / RT)[3]["mutation"]
    got = _draw(C4, genotypes, str(tmp_path / "kcal.csv"),
                {"ddG_units": "kcal/mol"}, ddG_truth_sd=0.5)[3]["mutation"]
    pd.testing.assert_frame_equal(ref, got, atol=1e-5, rtol=0)
    sig = ddG_truth_sigma_s(0.5, C4, {"ddG_units": "kcal/mol"})
    np.testing.assert_allclose(sig["theta_ddG_sigma_s"], 0.5 / RT, rtol=1e-6)
    sig = ddG_truth_sigma_s(0.5, C4, {"ddG_units": "kJ/mol"})
    np.testing.assert_allclose(sig["theta_ddG_sigma_s"],
                               0.5 / (8.314462618e-3 * 298.15), rtol=1e-6)


def test_prior_mean_stays_the_table(tmp_path):
    """The recorded prior mean (what a fit's ddG_prior_mean_* reports) is
    the unperturbed table whatever the truth SD."""
    genotypes, muts = _library(2)
    path, table = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    opts = {"ddG_units": "kcal/mol"}
    sim_data = build_sim_data(pd.DataFrame({"genotype": genotypes}), SAMPLE_DF,
                              thermo_data=path, theta_component=C12,
                              thermo_options=opts)
    _, _, det = sample_theta_prior(
        C12, sim_data, jax.random.PRNGKey(0), return_deterministic=True,
        model_params_overrides=ddG_truth_sigma_s(0.7, C12, opts))
    np.testing.assert_allclose(det["theta_ddG_prior_mean"],
                               table.loc[muts, C12_STRUCTS].to_numpy() / RT,
                               rtol=1e-6)


# ── default and the fit side are unchanged ───────────────────────────────────

# Seeded C4_U0 PddG truth from test_thermo_truth (commit 9f252025): the
# default and an explicit 1 kT must both reproduce it.
_C4_PDDG_TRUTH_BEFORE = pd.DataFrame({
    "ddG_H":   [-0.261347, -0.399871, 0.819358],
    "ddG_HD":  [-2.974988, -0.291039, 2.898206],
    "ddG_L":   [0.500638, -1.941736, -1.234181],
    "ddG_LE2": [-1.335636, -0.765674, -0.370668],
})


@pytest.mark.parametrize("sd", [None, 1.0, {s: 1 for s in C4_STRUCTS}])
def test_default_and_one_kT_reproduce_the_seeded_truth(sd, tmp_path):
    genotypes = ["wt", "H29A", "H29C", "L62A", "H29A/L62A", "H29C/L62A"]
    muts = ["H29A", "H29C", "L62A"]
    path, _ = _table(tmp_path / "ddG.csv", muts, C4_STRUCTS)
    mut = _draw(C4, genotypes, path, ddG_truth_sd=sd)[3]["mutation"]
    for col in _C4_PDDG_TRUTH_BEFORE:
        np.testing.assert_allclose(mut[col], _C4_PDDG_TRUTH_BEFORE[col],
                                   atol=2e-6)


def test_override_leaves_the_fit_side_init_alone(tmp_path):
    """sigma_s starts at 1 in the model whenever nothing overrides it (the
    fit), also after a simulator draw that did override it."""
    genotypes, muts = _library(1)
    path, _ = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    sim_data = build_sim_data(pd.DataFrame({"genotype": genotypes}), SAMPLE_DF,
                              thermo_data=path, theta_component=C12)
    sample_theta_prior(C12, sim_data, jax.random.PRNGKey(0),
                       model_params_overrides=ddG_truth_sigma_s(0.0, C12))
    module = model_registry["theta"][C12]
    priors = module.ModelPriors(**module.get_hyperparameters())
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        module.define_model("theta", sim_data, priors)
    site = tr["theta_ddG_sigma_s"]
    assert site["type"] == "param"
    np.testing.assert_array_equal(site["value"], np.ones(len(C12_STRUCTS)))


# ── validation ───────────────────────────────────────────────────────────────

def test_none_is_no_override():
    assert ddG_truth_sigma_s(None, C12) is None
    assert ddG_truth_sigma_s(None, "hill_geno") is None


@pytest.mark.parametrize("value, match", [
    (-0.1, ">= 0"),
    (float("nan"), ">= 0"),
    (True, "number"),
    ("0.5", "number"),
    ({"H": 1.0}, "missing"),
    ({**{s: 1.0 for s in C12_STRUCTS}, "HD": 1.0}, "unknown"),
    ({**{s: 1.0 for s in C12_STRUCTS}, "H": -1}, r"\[H\]"),
])
def test_bad_values_raise(value, match):
    with pytest.raises(ValueError, match=match):
        ddG_truth_sigma_s(value, C12)


@pytest.mark.parametrize("component", ["hill_geno", "thermo.O2_C12_K5_U0_a.PK"])
def test_non_structural_component_raises(component):
    with pytest.raises(ValueError, match="PddG"):
        ddG_truth_sigma_s(0.5, component)


def test_pnnc_accepts_it():
    sig = ddG_truth_sigma_s(0.5, "thermo.O2_C4_K3_U0_a.PnnC")
    np.testing.assert_allclose(sig["theta_ddG_sigma_s"], [0.5] * 4)


def test_unknown_override_name_raises(tmp_path):
    genotypes, muts = _library(1)
    path, _ = _table(tmp_path / "ddG.csv", muts, C12_STRUCTS)
    sim_data = build_sim_data(pd.DataFrame({"genotype": genotypes}), SAMPLE_DF,
                              thermo_data=path, theta_component=C12)
    with pytest.raises(ValueError, match="not param sites"):
        sample_theta_prior(C12, sim_data, jax.random.PRNGKey(0),
                           model_params_overrides={"theta_nope": 1.0})


# ── config plumbing: tfs-simulate and tfs-setup-sim-grid ─────────────────────

def test_known_simulate_key():
    from tfscreen.simulate.selection_experiment import SIMULATE_KNOWN_KEYS
    assert "ddG_truth_sd" in SIMULATE_KNOWN_KEYS


def _mock_library_prediction(mocker, cf):
    mocker.patch("tfscreen.util.read_yaml", return_value=cf)
    lm = mocker.patch(
        "tfscreen.simulate.library_prediction.library_manager.LibraryManager")
    lm.return_value.build_library_df.return_value = pd.DataFrame(
        {"genotype": ["wt"]})
    mocker.patch("tfscreen.simulate.library_prediction.build_sample_dataframes",
                 return_value=pd.DataFrame({"titrant_conc": [0.0, 1.0]}))
    mocker.patch("tfscreen.simulate.library_prediction.build_sim_data",
                 return_value=MagicMock())
    return mocker.patch(
        "tfscreen.simulate.library_prediction.thermo_to_growth",
        return_value=(pd.DataFrame(), pd.DataFrame(),
                      pd.DataFrame({"genotype": ["wt"]})))


_CF = {"condition_blocks": [{}], "theta_component": C12, "growth": {},
       "dk_geno_zero": True, "thermo_data": "x.csv"}


@pytest.mark.parametrize("cf_extra, expected", [
    ({}, None),
    ({"ddG_truth_sd": 0.5}, [0.5] * 6),
    ({"ddG_truth_sd": 0.5, "thermo_options": {"ddG_units": "kcal/mol"}},
     [0.5 / RT] * 6),
])
def test_library_prediction_passes_the_override(mocker, cf_extra, expected):
    from tfscreen.simulate.library_prediction import library_prediction
    thermo = _mock_library_prediction(mocker, {**_CF, **cf_extra})
    library_prediction(cf="config.yaml")
    got = thermo.call_args.kwargs["theta_model_params_override"]
    if expected is None:
        assert got is None
    else:
        np.testing.assert_allclose(got["theta_ddG_sigma_s"], expected,
                                   rtol=1e-6)


@pytest.mark.parametrize("cf_extra", [
    {"theta_component": "hill_geno"},
    {"phenotype_source": "empirical"},
])
def test_library_prediction_rejects_it(mocker, cf_extra):
    from tfscreen.simulate.library_prediction import library_prediction
    thermo = _mock_library_prediction(
        mocker, {**_CF, **cf_extra, "ddG_truth_sd": 0.5})
    with pytest.raises(ValueError, match="ddG_truth_sd"):
        library_prediction(cf="config.yaml")
    thermo.assert_not_called()


def test_setup_sim_grid_varies_it(tmp_path):
    from tfscreen.simulate.scripts.setup_sim_grid_cli import setup_sim_grid
    base = tmp_path / "simulate_config.yaml"
    base.write_text(yaml.dump({"theta_component": C12, "ddG_truth_sd": 1.0,
                               "thermo_options": {"ddG_units": "kcal/mol"}}))
    mapping = {s: 0.25 for s in C12_STRUCTS}
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.dump({
        "base_config": "simulate_config.yaml",
        "run_name": "sd{{ ddG_truth_sd }}",
        "simulate": [{"name": "dial", "variants": [
            {"ddG_truth_sd": 0}, {"ddG_truth_sd": 0.5}]}],
    }))
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(grid), out_dir=out)
    got = []
    for run in runs:
        with open(os.path.join(out, run["run"], "tfs_sim_config.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        got.append(cfg["ddG_truth_sd"])
        assert cfg["thermo_options"] == {"ddG_units": "kcal/mol"}
    assert got == [0, 0.5]

    # A per-structure mapping is one variant value (the whole key).
    grid.write_text(yaml.dump({
        "base_config": "simulate_config.yaml",
        "simulate": [{"name": "dial", "variants": [{"ddG_truth_sd": mapping}]}],
    }))
    out2 = str(tmp_path / "grid_out2")
    runs = setup_sim_grid(str(grid), out_dir=out2)
    with open(os.path.join(out2, runs[0]["run"], "tfs_sim_config.yaml")) as fh:
        assert yaml.safe_load(fh)["ddG_truth_sd"] == mapping
