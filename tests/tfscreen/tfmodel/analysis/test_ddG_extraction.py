"""
Per-structure ΔΔG extraction for the structure-based thermodynamic models.

PddG and PnnC sample the per-mutation constants (``theta_dlnK``) under a
structural prior and record a post-hoc per-structure ΔΔG (``theta_ddG``,
M × S, in the model's ``STRUCTURE_NAMES`` order; exact in U1, the
conditional mean under the per-state prior in U0) and its structural prior
mean (``theta_ddG_prior_mean``: the CSV values or the MLP output).  The
simulator writes the truth as ``ddG_{structure}`` columns of
``*_sim_mutation_parameters.csv``.  ``tfs-extract-params`` must write one
table per structure under those names and ``tfs-summarize-fit`` must score
them (U1) or flag them (U0), with the structure labels right whatever order
the input file uses.
"""
import glob
import os
import types
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.tfmodel.analysis.extraction import _extract_param_est
from tfscreen.tfmodel.generative.registry import model_registry

_MODELS = [f"thermo.{fam}.{kind}"
           for fam in ("O2_C4_K3_U0_a", "O2_C4_K3_U1_a",
                       "O2_C12_K5_U0_a", "O2_C12_K5_U1_a")
           for kind in ("PddG", "PnnC")]

_MUTS = ["H29A", "H29C", "L62A"]
_GENOTYPES = ["wt"] + _MUTS + ["H29A/L62A"]


def _ctx(pair_labels=()):
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[_GENOTYPES, ["iptg"]],
        df=pd.DataFrame({"genotype": _GENOTYPES,
                         "genotype_idx": range(len(_GENOTYPES)),
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    return types.SimpleNamespace(growth_tm=growth_tm, mut_labels=list(_MUTS),
                                 pair_labels=list(pair_labels))


def _ddG_specs(component):
    """The per-mutation ΔΔG specs (``ddG_*``, ``ddG_prior_mean_*``)."""
    specs = model_registry["theta"][component].get_extract_specs(_ctx())
    return [s for s in specs
            if "out_names" in s and s["get_columns"] == ["mutation"]]


def _sigma_spec(component):
    specs = model_registry["theta"][component].get_extract_specs(_ctx())
    found = [s for s in specs if s.get("out_names") == ["ddG_sigma"]]
    assert len(found) == 1
    return found[0]


# ── extract specs ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("component", _MODELS)
def test_specs_name_one_table_per_structure(component):
    structs = model_registry["theta"][component].STRUCTURE_NAMES
    specs = _ddG_specs(component)
    names = [n for s in specs for n in s["out_names"]]
    assert names == [f"{site}_{s}" for s in structs
                     for site in ("ddG", "ddG_prior_mean")]
    for spec in specs:
        assert spec["in_run_prefix"] == "theta_"
        assert spec["get_columns"] == ["mutation"]
        assert list(spec["input_df"]["mutation"]) == _MUTS


@pytest.mark.parametrize("component", _MODELS)
def test_specs_read_each_structure_column(component):
    """Each table reads its structure's column of the (M, S) site."""
    structs = model_registry["theta"][component].STRUCTURE_NAMES
    M, S = len(_MUTS), len(structs)
    # Sample k, mutation m, structure s holds 100 k + 10 m + s.
    vals = (100 * np.arange(2)[:, None, None] + 10 * np.arange(M)[None, :, None]
            + np.arange(S)[None, None, :]).astype(float)
    post = {"theta_ddG": vals, "theta_ddG_prior_mean": -vals}
    tables = {}
    for spec in _ddG_specs(component):
        tables.update(_extract_param_est(param_posteriors=post,
                                         q_to_get={"q0": 0.0, "q1": 1.0},
                                         **spec))
    for s, struct in enumerate(structs):
        df = tables[f"ddG_{struct}"]
        assert list(df.columns) == ["mutation", "q0", "q1"]
        np.testing.assert_allclose(df["q0"], 10 * np.arange(M) + s)
        np.testing.assert_allclose(df["q1"], 100 + 10 * np.arange(M) + s)
        np.testing.assert_allclose(tables[f"ddG_prior_mean_{struct}"]["q1"],
                                   -(10 * np.arange(M) + s))


@pytest.mark.parametrize("component", _MODELS)
def test_sigma_spec_gives_one_row_per_structure(component):
    """σ_s (theta_ddG_sigma, shape S) becomes one ddG_sigma table keyed by
    structure, in STRUCTURE_NAMES order."""
    structs = model_registry["theta"][component].STRUCTURE_NAMES
    spec = _sigma_spec(component)
    assert spec["params_to_get"] == ["ddG_sigma"]
    assert spec["in_run_prefix"] == "theta_"
    assert spec["get_columns"] == ["structure"]
    post = {"theta_ddG_sigma": np.array([np.arange(len(structs)) + 1.0,
                                         np.arange(len(structs)) + 3.0])}
    df = _extract_param_est(param_posteriors=post,
                            q_to_get={"q0": 0.0, "q1": 1.0},
                            **spec)["ddG_sigma"]
    assert list(df["structure"]) == list(structs)
    np.testing.assert_allclose(df["q0"], np.arange(len(structs)) + 1.0)
    np.testing.assert_allclose(df["q1"], np.arange(len(structs)) + 3.0)


def test_out_names_must_match_params():
    with pytest.raises(ValueError, match="out_names"):
        _extract_param_est(input_df=pd.DataFrame({"m": [0]}),
                           params_to_get=["a", "b"], out_names=["x"],
                           map_column="m", get_columns=[], in_run_prefix="",
                           param_posteriors={"a": np.zeros((1, 1))},
                           q_to_get={"q0.5": 0.5})


# ── tfs-summarize-fit scoring ────────────────────────────────────────────────

def _q_table(values, spread):
    df = pd.DataFrame({"mutation": _MUTS})
    for q, z in (("q0.025", -2.0), ("q0.5", 0.0), ("q0.975", 2.0)):
        df[q] = np.asarray(values) + z * spread
    return df


def _summarize_run(tmp_path, with_ddG, config_yaml=None):
    from tfscreen.tfmodel.scripts.summarize_fit_cli import _summarize_params
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"genotype": _GENOTYPES, "ln_K_op": 0.0}).to_csv(
        run / "tfs_sim_parameters.csv", index=False)
    truth = pd.DataFrame({"mutation": _MUTS, "d_ln_K_op": [1.0, 2.0, 3.0],
                          "ddG_H": [0.5, -1.0, 2.0], "ddG_HD": [3.0, 0.0, 1.0]})
    truth.to_csv(run / "tfs_sim_mutation_parameters.csv", index=False)
    _q_table([1.1, 2.1, 2.9], 0.2).to_csv(run / "tfs_params_d_ln_K_op.csv",
                                          index=False)
    if with_ddG:
        for s in ("H", "HD"):
            _q_table(truth[f"ddG_{s}"] + 0.1, 0.2).to_csv(
                run / f"tfs_params_ddG_{s}.csv", index=False)
            _q_table(truth[f"ddG_{s}"] - 0.3, 0.0).to_csv(
                run / f"tfs_params_ddG_prior_mean_{s}.csv", index=False)
    out = tmp_path / "summary"
    out.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _summarize_params(str(run), str(out / "tfs_summarize"), config_yaml)
    return truth, out, [str(w.message) for w in caught]


def test_summarize_scores_ddG_tables_against_truth(tmp_path):
    truth, out, msgs = _summarize_run(tmp_path, with_ddG=True)
    assert not [m for m in msgs if "_params_ddG" in m]
    for s in ("H", "HD"):
        got = pd.read_csv(out / f"tfs_summarize_params_ddG_{s}.csv")
        np.testing.assert_allclose(got["ref"], truth[f"ddG_{s}"])
        prior = pd.read_csv(out / f"tfs_summarize_params_ddG_prior_mean_{s}.csv")
        np.testing.assert_allclose(prior["ref"], truth[f"ddG_{s}"])
        # Posterior tables get a calibration; the prior mean (a point) not.
        assert (out / f"tfs_summarize_params_ddG_{s}_pit.csv").exists()
        assert not glob.glob(str(out / f"*ddG_prior_mean_{s}_*"))


def test_summarize_scores_ddG_tables_for_a_u1_fit(tmp_path):
    cfg = {"components": {"theta": "thermo.O2_C4_K3_U1_a.PddG"}}
    truth, out, _ = _summarize_run(tmp_path, with_ddG=True, config_yaml=cfg)
    got = pd.read_csv(out / "tfs_summarize_params_ddG_H.csv")
    np.testing.assert_allclose(got["ref"], truth["ddG_H"])
    assert (out / "tfs_summarize_params_ddG_H_pit.csv").exists()


@pytest.mark.parametrize("theta", ["thermo.O2_C4_K3_U0_a.PddG",
                                   "thermo.O2_C12_K5_U0_a.PnnC"])
def test_summarize_flags_u0_per_state_ddG(tmp_path, capsys, theta):
    """U0: per-state ΔΔG's common mode comes from the prior, so the table is
    written with the truth as ref_unidentified and is not scored; the
    prior-mean tables and d_ln_K are scored as usual."""
    cfg = {"components": {"theta": theta}}
    truth, out, msgs = _summarize_run(tmp_path, with_ddG=True, config_yaml=cfg)
    assert "no unfolded state (U0)" in capsys.readouterr().out
    for s in ("H", "HD"):
        got = pd.read_csv(out / f"tfs_summarize_params_ddG_{s}.csv")
        assert "ref" not in got.columns
        np.testing.assert_allclose(got["ref_unidentified"], truth[f"ddG_{s}"])
        assert not got["per_state_identified"].any()
        assert not glob.glob(str(out / f"tfs_summarize_params_ddG_{s}_*"))
        assert not (out / f"tfs_summarize_params_ddG_{s}.pdf").exists()
        prior = pd.read_csv(out / f"tfs_summarize_params_ddG_prior_mean_{s}.csv")
        np.testing.assert_allclose(prior["ref"], truth[f"ddG_{s}"])
    got = pd.read_csv(out / "tfs_summarize_params_d_ln_K_op.csv")
    np.testing.assert_allclose(got["ref"], truth["d_ln_K_op"])

    # tfs-summarize-calibration reads only tables with a ref column.
    from tfscreen.analysis.calibration_grid import read_quantities
    names = [q for q, _ in read_quantities(str(out))]
    assert "ddG_H" not in names and "d_ln_K_op" in names


def test_ddG_per_state_identified_by_family():
    from tfscreen.tfmodel.scripts.summarize_fit_cli import (
        _ddG_per_state_identified,
    )
    for fam, expected in (("O2_C4_K3_U0_a", False), ("O2_C4_K3_U1_a", True),
                          ("O2_C12_K5_U0_a", False), ("O2_C12_K5_U1_a", True)):
        for kind in ("PddG", "PnnC"):
            cfg = {"components": {"theta": f"thermo.{fam}.{kind}"}}
            assert _ddG_per_state_identified(cfg) is expected
    assert _ddG_per_state_identified(None) is True
    assert _ddG_per_state_identified(
        {"components": {"theta": "thermo.O2_C4_K3_U0_a.PK"}}) is True


def test_summarize_scores_ddG_sigma_against_truth(tmp_path):
    from tfscreen.tfmodel.scripts.summarize_fit_cli import _summarize_ddG_sigma
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"structure": ["H", "HD"], "ddG_sigma": [0.5, 1.5]}).to_csv(
        run / "tfs_sim_ddG_sigma.csv", index=False)
    pd.DataFrame({"structure": ["HD", "H"], "q0.025": [1.0, 0.3],
                  "q0.5": [1.4, 0.6], "q0.975": [1.9, 0.9]}).to_csv(
        run / "tfs_params_ddG_sigma.csv", index=False)
    out = tmp_path / "summary"
    out.mkdir()
    _summarize_ddG_sigma(str(run), str(out / "tfs_summarize"))
    got = pd.read_csv(out / "tfs_summarize_params_ddG_sigma.csv")
    np.testing.assert_allclose(got["ref"], [1.5, 0.5])


def test_summarize_ddG_sigma_silent_for_a_PK_fit(tmp_path):
    from tfscreen.tfmodel.scripts.summarize_fit_cli import _summarize_ddG_sigma
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"structure": ["H"], "ddG_sigma": [1.0]}).to_csv(
        run / "tfs_sim_ddG_sigma.csv", index=False)
    out = tmp_path / "summary"
    out.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _summarize_ddG_sigma(str(run), str(out / "tfs_summarize"))
    assert not [w for w in caught if "ddG_sigma" in str(w.message)]
    assert not list(out.iterdir())


def test_summarize_notes_missing_ddG_for_a_PK_fit(tmp_path, capsys):
    """A PK fit of a PddG simulation writes no ddG: one note, no warnings,
    and its d_ln_K tables are still scored."""
    truth, out, msgs = _summarize_run(tmp_path, with_ddG=False)
    assert not [m for m in msgs if "_params_ddG" in m]
    assert "wrote no *_params_ddG_*.csv" in capsys.readouterr().out
    got = pd.read_csv(out / "tfs_summarize_params_d_ln_K_op.csv")
    np.testing.assert_allclose(got["ref"], truth["d_ln_K_op"])


# ── end to end: simulate → MAP → point posterior → extract → summarize ──────

_SIM_CONFIG = dict(
    reading_frame=0,
    first_amplicon_residue=27,
    wt_seq=("gccagccacgtttctgcgaaaacgcgggaaaaagtggaagcggcgatggcggagctgaattacatt"
            "cccaaccgcgtggcacaacaactggcgggcaaacagtcgttgctgattggcgttgccacctccag"
            "tctggccctgcacgcgccgtcgcaaattgtcgcggcgattaaatctcgcgccgatcaactgggtg"
            "ccagcgtggtggtgtcgatg"),
    degen_sites=("......nnt" + "." * 96 + "nnt" + "." * 108),
    tiles=("......111" + "." * 96 + "222" + "." * 108),
    expected_5p="cgcgtggtgaaccag",
    expected_3p="gtagaacgaagcggc",
    tile_combos=["single-1", "single-2", "double-1-2"],
    condition_blocks=[dict(library="kanR", titrant_name="iptg",
                           titrant_conc=[0, 0.001, 0.1, 1.0],
                           condition_pre="kanR-kan", t_pre=30,
                           condition_sel="kanR+kan", t_sel=[160, 200])],
    growth={"kanR+kan": {"m": -0.009933, "b": 0.0},
            "kanR-kan": {"m": 0.000808, "b": 0.00467}},
    dk_geno_hyper_loc=-3.5, dk_geno_hyper_scale=1.0, dk_geno_hyper_shift=0.02,
    activity_wt=1.0, activity_mut_scale=0.0,
    transform_sizes={"single-1": 2000, "single-2": 2000, "double-1-2": 2000},
    library_mixture={"single-1": 100, "single-2": 100, "double-1-2": 10},
    lib_assembly_skew_sigma=0.5,
    transformation_poisson_lambda=0,
    congression_theta_rule="homodimer",
    congression_dk_rule="dilution",
    cfu0=65000000.0,
    tube_noise_sigma=0.002,
    total_num_reads=300000,
    prob_index_hop=0.0,
    seed=1,
    presplit_data={"noise": 0.0},
)

# Residues the library above mutates (H29, L62).
_RESNUMS = [29, 62]


def _reordered_ddG_csv(path, structs):
    """A ΔΔG CSV with the structures reversed, an extra column and the
    'Mutation' header, so labels can only be right if read by name."""
    rng = np.random.default_rng(0)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    muts = [f"{wt}{pos}{aa}" for wt, pos in (("H", 29), ("L", 62))
            for aa in aas if aa != wt]
    df = pd.DataFrame(rng.normal(0, 1.5, (len(muts), len(structs))),
                      columns=list(structs))
    df.insert(0, "Mutation", muts)
    df["unused"] = 99.0
    df = df[["Mutation", "unused"] + list(reversed(structs))]
    df.to_csv(path, index=False)
    return str(path)


def _reordered_ensemble(path, structs):
    """A structure ensemble whose structure_names are in reversed order."""
    with h5py.File(path, "w") as hf:
        hf.create_dataset("structure_names",
                          data=np.array(list(reversed(structs)),
                                        dtype=h5py.string_dtype()))
        for i, name in enumerate(structs):
            rng = np.random.default_rng(i)
            grp = hf.create_group(name)
            dist_mat = np.abs(rng.normal(5, 2, (2, 2))).astype(np.float32)
            np.fill_diagonal(dist_mat, 0.0)
            grp.create_dataset("logP", data=rng.normal(size=(2, 20)).astype(np.float32))
            grp.create_dataset("residue_nums", data=np.asarray(_RESNUMS, dtype=np.int32))
            grp.create_dataset("dist_matrix", data=dist_mat)
            grp.create_dataset("n_chains_bearing_mut", data=np.int32(2))
    return str(path)


@pytest.mark.slow
@pytest.mark.parametrize("component", ["thermo.O2_C4_K3_U0_a.PddG",
                                       "thermo.O2_C12_K5_U0_a.PnnC"])
def test_ddG_tables_end_to_end(tmp_path, component):
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config
    from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
    from tfscreen.tfmodel.scripts.extract_params_cli import extract_params
    from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
    from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
    from tfscreen.tfmodel.scripts.summarize_fit_cli import summarize_fit

    module = model_registry["theta"][component]
    structs = module.STRUCTURE_NAMES
    if component.endswith("PddG"):
        thermo_data = _reordered_ddG_csv(tmp_path / "ddG.csv", structs)
    else:
        thermo_data = _reordered_ensemble(tmp_path / "ens.h5", structs)

    cfg = dict(_SIM_CONFIG, theta_component=component, thermo_data=thermo_data)
    with open(tmp_path / "sim.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    run = tmp_path / "run"
    run.mkdir()
    cwd = os.getcwd()
    os.chdir(run)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_simulation_from_config(str(tmp_path / "sim.yaml"),
                                       out_prefix="tfs_sim", num_replicates=1,
                                       write_raw=False, write_growth=True)
            configure_model(growth_df="tfs_sim_growth.csv",
                            presplit_df="tfs_sim_presplit.csv",
                            theta_model=component,
                            library_config="tfs_sim_input-config.yaml",
                            growth_shares_replicates=True,
                            thermo_data=thermo_data)
            fit_model("tfs_configure_config.yaml", seed=1,
                      analysis_method="map", max_num_epochs=20,
                      convergence_window_steps=10)
            sample_posterior("tfs_configure_config.yaml",
                             "tfs_fit_model_checkpoint.pkl",
                             laplace="point", num_posterior_samples=5,
                             skip_growth_observations=True)
            extract_params("tfs_configure_config.yaml",
                           "tfs_posterior.h5")
            summarize_fit(".")
    finally:
        os.chdir(cwd)

    truth = pd.read_csv(run / "tfs_sim_mutation_parameters.csv").set_index("mutation")
    with h5py.File(run / "tfs_posterior.h5", "r") as hf:
        ddG = np.median(hf["theta_ddG"][:], axis=0)            # (M, S)
        prior = np.median(hf["theta_ddG_prior_mean"][:], axis=0)

    # The truth's projection to Δln K holds in the model's order, so its ddG
    # columns are labeled in STRUCTURE_NAMES order.  (The post-hoc U0 ΔΔG
    # reproduces the fitted constants exactly too.)
    proj = np.asarray(module._PROJ)
    truth_d = truth[[f"ddG_{s}" for s in structs]].values @ proj.T
    for k, col in enumerate(_d_ln_K_order(module)[:proj.shape[0]]):
        np.testing.assert_allclose(truth_d[:, k], truth[col], atol=1e-4)

    # Same for the fit: the extracted ddG tables, stacked by structure name,
    # project onto the extracted d_ln_K tables.
    fit_ddG = np.column_stack([
        pd.read_csv(run / f"tfs_params_ddG_{s}.csv")["q0.5"] for s in structs])
    fit_d = fit_ddG @ proj.T
    for k, col in enumerate(_d_ln_K_order(module)[:proj.shape[0]]):
        np.testing.assert_allclose(
            fit_d[:, k], pd.read_csv(run / f"tfs_params_{col}.csv")["q0.5"],
            atol=1e-4)

    if component.endswith("PddG"):
        csv = pd.read_csv(thermo_data).set_index("Mutation")

    for s, struct in enumerate(structs):
        table = pd.read_csv(run / f"tfs_params_ddG_{struct}.csv")
        # The fit's mutations (those seen in the growth data), in its order.
        assert len(table) == ddG.shape[0]
        assert set(table["mutation"]) <= set(truth.index)
        np.testing.assert_allclose(table["q0.5"], ddG[:, s], atol=1e-5)
        prior_table = pd.read_csv(run / f"tfs_params_ddG_prior_mean_{struct}.csv")
        np.testing.assert_allclose(prior_table["q0.5"], prior[:, s], atol=1e-5)
        if component.endswith("PddG"):
            # The prior mean is the CSV column of the same name.
            np.testing.assert_allclose(
                prior_table["q0.5"],
                csv.loc[prior_table["mutation"], struct].values, atol=1e-5)

        # Both end-to-end models are U0: the per-state table is flagged,
        # with the truth as ref_unidentified.
        scored = pd.read_csv(run / "summary" / f"tfs_summarize_params_ddG_{struct}.csv")
        np.testing.assert_allclose(scored["ref_unidentified"],
                                   truth.loc[scored["mutation"], f"ddG_{struct}"])
        assert (run / "summary" / f"tfs_summarize_params_ddG_prior_mean_{struct}.csv").exists()

    # σ_s: the simulator's truth (1 kT, the default) per structure, the
    # fit's extracted table, and the summary joining them on structure.
    sim_sigma = pd.read_csv(run / "tfs_sim_ddG_sigma.csv")
    assert list(sim_sigma["structure"]) == list(structs)
    np.testing.assert_allclose(sim_sigma["ddG_sigma"], 1.0)
    fit_sigma = pd.read_csv(run / "tfs_params_ddG_sigma.csv")
    assert list(fit_sigma["structure"]) == list(structs)
    assert (fit_sigma["q0.5"] > 0).all()
    scored = pd.read_csv(run / "summary" / "tfs_summarize_params_ddG_sigma.csv")
    np.testing.assert_allclose(scored["ref"], 1.0)


def _d_ln_K_order(module):
    """The model's d_ln_K sites in the order of its ``_PROJ`` rows."""
    import inspect
    import re
    return re.findall(r'deterministic\(f"\{name\}_(d_ln_K_\w+)"',
                      inspect.getsource(module.define_model))
