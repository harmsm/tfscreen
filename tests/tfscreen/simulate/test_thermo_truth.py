"""
Ground truth written for the thermodynamic theta models.

The thermodynamic components (``thermo.O2_C4_K3_*`` and ``thermo.O2_C12_K5_*``)
carry titrant-independent per-genotype constants as (G,) arrays and their
per-mutation, per-pair and per-structure values only as deterministic sites.
These tests draw a tiny library from each family and check that the
simulator's truth covers every parameter ``tfs-extract-params`` writes for the
model (names from the component's ``get_extract_specs``).
"""
import types
import warnings
from unittest.mock import patch

import jax
import numpy as np
import pandas as pd
import pytest
from flax.struct import dataclass

from tfscreen.genetics.build_mut_geno_matrix import build_mut_geno_matrix
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.thermo_to_growth import (
    _mutation_truth_tables,
    _theta_param_to_df,
    thermo_to_growth,
)
from tfscreen.tfmodel.generative.registry import model_registry

GENOTYPES = ["wt", "H29A", "H29C", "L62A", "H29A/L62A", "H29C/L62A"]

SAMPLE_DF = pd.DataFrame({
    "titrant_name": "iptg",
    "titrant_conc": [0.0, 0.01, 1.0],
    "condition_pre": "pre",
    "condition_sel": "sel",
})

GROWTH = {"pre": {"m": 0.0, "b": 0.01}, "sel": {"m": 0.01, "b": 0.01}}

# Columns of the ddG prior CSV (struct names of the lac-dimer and MWC models).
_C4_STRUCTS = ["H", "HD", "L", "LE2"]
_C12_STRUCTS = ["H", "HO", "L", "LO", "HE2", "LE2"]


def _ddG_csv(path, component="O2_C4"):
    structs = _C12_STRUCTS if "O2_C12" in component else _C4_STRUCTS
    rng = np.random.default_rng(3)
    muts = ["H29A", "H29C", "L62A"]
    df = pd.DataFrame(rng.normal(0, 1, (len(muts), len(structs))),
                      columns=structs)
    df.insert(0, "mut", muts)
    df.to_csv(path, index=False)
    return str(path)


def _draw(component, thermo_data=None, return_mutation_truth=True, seed=0,
          **sim_kwargs):
    lib = pd.DataFrame({"genotype": GENOTYPES})
    sim_data = build_sim_data(lib, SAMPLE_DF, thermo_data=thermo_data,
                              **sim_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return thermo_to_growth(lib["genotype"], sim_data, SAMPLE_DF, component,
                                jax.random.PRNGKey(seed), GROWTH,
                                rng=np.random.default_rng(seed),
                                return_mutation_truth=return_mutation_truth)


def _extract_names(component):
    """Parameter names tfs-extract-params writes for ``component``."""
    mut_labels, pair_labels, *_ = build_mut_geno_matrix(GENOTYPES)
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[GENOTYPES, ["iptg"]],
        df=pd.DataFrame({"genotype": GENOTYPES,
                         "genotype_idx": range(len(GENOTYPES)),
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    ctx = types.SimpleNamespace(growth_tm=growth_tm, mut_labels=mut_labels,
                                pair_labels=pair_labels)
    specs = model_registry["theta"][component].get_extract_specs(ctx)
    names = [p for spec in specs
             for p in spec.get("out_names", spec["params_to_get"])]
    # A prior-mean table (ddG_prior_mean_H) is scored against the ddG truth
    # (ddG_H); it has no truth column of its own.
    return [n.replace("ddG_prior_mean_", "ddG_") for n in names]


# PK covers each family; PddG cases cover the per-structure ddG truth and,
# for U1, the unfolding shift projected from ddG_H.
CASES = [
    ("thermo.O2_C4_K3_U0_a.PK", ["ln_K_op", "ln_K_HL"]),
    ("thermo.O2_C4_K3_U1_a.PK", ["ln_K_op", "ln_K_HL", "ln_K_U"]),
    ("thermo.O2_C12_K5_U0_a.PK", ["ln_K_h_l", "ln_K_h_o", "ln_K_l_o"]),
    ("thermo.O2_C12_K5_U1_a.PK", ["ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u"]),
    ("thermo.O2_C4_K3_U0_a.PddG", ["ln_K_op", "ln_K_HL"]),
    ("thermo.O2_C4_K3_U1_a.PddG", ["ln_K_op", "ln_K_HL", "ln_K_U"]),
    ("thermo.O2_C12_K5_U1_a.PddG", ["ln_K_h_l", "ln_K_h_o", "ln_K_l_o", "ln_K_u"]),
]


@pytest.mark.parametrize("component, scalar_ks", CASES)
def test_truth_covers_every_extracted_parameter(component, scalar_ks, tmp_path):
    td = (_ddG_csv(tmp_path / "ddG.csv", component)
          if component.endswith("PddG") else None)
    _, _, params, truth = _draw(component, thermo_data=td)
    mutation_df, pair_df = truth["mutation"], truth["pair"]

    # Titrant-independent (G,) constants are per-genotype columns now.
    for col in scalar_ks:
        assert col in params.columns
        assert params[col].notna().all()

    assert mutation_df is not None and pair_df is not None
    assert list(mutation_df["mutation"]) == ["H29A", "H29C", "L62A"]
    assert set(pair_df["pair"]) == {"H29A/L62A", "H29C/L62A"}

    truth_names = set(params.columns) | set(mutation_df.columns) | set(pair_df.columns)
    if truth["ddG_sigma"] is not None:      # PddG/PnnC: the σ_s truth table
        truth_names.add("ddG_sigma")
    missing = [n for n in _extract_names(component) if n not in truth_names]
    assert missing == []


@pytest.mark.parametrize("component, scalar_ks", CASES)
def test_mutation_truth_matches_genotype_truth(component, scalar_ks, tmp_path):
    """d_X of a mutation is its single mutant's X minus wt's X; a double's X
    is wt + both deltas + the pair's epistasis."""
    td = (_ddG_csv(tmp_path / "ddG.csv", component)
          if component.endswith("PddG") else None)
    _, _, params, truth = _draw(component, thermo_data=td)
    p = params.set_index("genotype")
    mut = truth["mutation"].set_index("mutation")
    pair = truth["pair"].set_index("pair")
    for col in scalar_ks:
        for m in ["H29A", "H29C", "L62A"]:
            assert p.loc[m, col] - p.loc["wt", col] == pytest.approx(
                mut.loc[m, f"d_{col}"], abs=1e-4)
        if f"epi_{col}" in pair.columns:
            expected = (p.loc["wt", col] + mut.loc["H29A", f"d_{col}"]
                        + mut.loc["L62A", f"d_{col}"]
                        + pair.loc["H29A/L62A", f"epi_{col}"])
            assert p.loc["H29A/L62A", col] == pytest.approx(expected, abs=1e-4)


def test_pddg_writes_per_structure_ddG(tmp_path):
    td = _ddG_csv(tmp_path / "ddG.csv")
    _, _, _, truth = _draw("thermo.O2_C4_K3_U0_a.PddG", thermo_data=td)
    mut = truth["mutation"]
    for s in _C4_STRUCTS:
        assert f"ddG_{s}" in mut.columns
    # The model's projection: d_ln_K_op = ddG_H - ddG_HD.
    np.testing.assert_allclose(mut["d_ln_K_op"], mut["ddG_H"] - mut["ddG_HD"],
                               atol=1e-5)


@pytest.mark.parametrize("component, structs, u", [
    ("thermo.O2_C4_K3_U1_a.PddG", _C4_STRUCTS, "ln_K_U"),
    ("thermo.O2_C12_K5_U1_a.PddG", _C12_STRUCTS, "ln_K_u"),
])
def test_u1_pddg_unfolding_truth_is_ddG_H(component, structs, u, tmp_path):
    """K_U = [U]/[H] and mutations leave U alone, so d_ln_K_U = ddG_H, and the
    per-genotype ln_K_U truth varies across genotypes."""
    _, _, params, truth = _draw(component,
                                thermo_data=_ddG_csv(tmp_path / "ddG.csv",
                                                     component))
    mut = truth["mutation"]
    for s in structs:
        assert f"ddG_{s}" in mut.columns
    np.testing.assert_allclose(mut[f"d_{u}"], mut["ddG_H"], atol=1e-5)
    assert f"epi_{u}" in truth["pair"].columns
    assert params[u].nunique() > 1


def test_mutation_truth_does_not_change_the_draw():
    pheno_a, theta_a, params_a = _draw("thermo.O2_C4_K3_U0_a.PK",
                                       return_mutation_truth=False)
    pheno_b, theta_b, params_b, _ = _draw("thermo.O2_C4_K3_U0_a.PK")
    pd.testing.assert_frame_equal(theta_a, theta_b)
    pd.testing.assert_frame_equal(params_a, params_b)
    pd.testing.assert_frame_equal(pheno_a, pheno_b)


def test_hill_components_have_no_mutation_truth():
    """Components sampled through simulate() have no trace to read."""
    _, _, params, truth = _draw("hill_geno")
    assert truth == {"mutation": None, "pair": None, "ddG_sigma": None}
    assert "log_hill_K" in params.columns


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class _OneDParam:
    ln_K_op: np.ndarray    # (G,)
    ln_K_E: np.ndarray     # (1, G)
    tf_total: float        # scalar


def test_theta_param_to_df_reads_one_dimensional_fields():
    param = _OneDParam(ln_K_op=np.array([1.0, 2.0, 3.0]),
                       ln_K_E=np.array([[4.0, 5.0, 6.0]]),
                       tf_total=1e-6)
    df = _theta_param_to_df(param, ["b", "a"], [1, 0], num_genotype_sim=3)
    assert list(df.columns) == ["genotype", "ln_K_op", "ln_K_E"]
    assert list(df["ln_K_op"]) == [2.0, 1.0]
    assert list(df["ln_K_E"]) == [5.0, 4.0]


def test_theta_param_to_df_without_num_genotype_skips_one_dimensional():
    param = _OneDParam(ln_K_op=np.array([1.0, 2.0, 3.0]),
                       ln_K_E=np.array([[4.0, 5.0, 6.0]]),
                       tf_total=1e-6)
    df = _theta_param_to_df(param, ["b", "a"], [1, 0])
    assert list(df.columns) == ["genotype", "ln_K_E"]


def test_mutation_truth_tables_suffixes_multiple_titrants():
    genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
    sim_data = types.SimpleNamespace(num_mutation=2, num_pair=1,
                                     struct_names=None)
    det = {"theta_d_ln_K_E": np.array([[1.0, 2.0], [3.0, 4.0]]),
           "theta_d_ln_K_op": np.array([5.0, 6.0]),
           "theta_epi_ln_K_op": np.array([7.0]),
           "theta_ln_K_op": np.zeros(4)}      # per-genotype: not mutation truth
    mut, pair = _mutation_truth_tables(det, genotypes, sim_data)
    assert list(mut.columns) == ["mutation", "d_ln_K_E_T0", "d_ln_K_E_T1",
                                 "d_ln_K_op"]
    assert list(pair.columns) == ["pair", "epi_ln_K_op"]
    assert pair["pair"].tolist() == ["A1B/C2D"]


def test_mutation_truth_tables_empty_without_mutations():
    sim_data = types.SimpleNamespace(num_mutation=0, num_pair=0,
                                     struct_names=None)
    assert _mutation_truth_tables({"theta_x": 1}, ["wt"], sim_data) == (None, None)


# ---------------------------------------------------------------------------
# tfs-simulate writes the tables
# ---------------------------------------------------------------------------

def test_simulate_cli_writes_mutation_and_pair_truth(tmp_path):
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config

    lib_df = pd.DataFrame({"genotype": ["wt", "A1B"]})
    params_df = pd.DataFrame({"genotype": ["wt", "A1B"], "dk_geno": [0.0, 0.0],
                              "activity": [1.0, 1.0], "ln_K_op": [1.0, 2.0]})
    mut_df = pd.DataFrame({"mutation": ["A1B"], "d_ln_K_op": [1.0]})
    sample_df = pd.DataFrame({"replicate": [1], "library": ["x"]})
    counts_df = pd.DataFrame({"sample": [0], "genotype": ["wt"], "counts": [1]})

    with patch("tfscreen.util.read_yaml", return_value={"seed": 0, "growth": {}}), \
         patch("tfscreen.simulate.scripts.simulate_cli.library_prediction",
               return_value=(lib_df, lib_df, lib_df, params_df, None,
                             {"mutation": mut_df, "pair": None}, None)), \
         patch("tfscreen.simulate.scripts.simulate_cli.selection_experiment",
               return_value=(sample_df, counts_df)):
        run_simulation_from_config("config.yaml",
                                   out_prefix=str(tmp_path / "tfs_sim"),
                                   num_replicates=1, write_raw=False,
                                   write_growth=False)

    written = pd.read_csv(tmp_path / "tfs_sim_mutation_parameters.csv")
    pd.testing.assert_frame_equal(written, mut_df)
    assert not (tmp_path / "tfs_sim_pair_parameters.csv").exists()


# ── ΔΔG units, column handling and the default being unchanged ──────────────

def test_default_ddG_units_leave_the_draw_unchanged(tmp_path):
    comp = "thermo.O2_C4_K3_U0_a.PddG"
    td = _ddG_csv(tmp_path / "ddG.csv", comp)
    base = _draw(comp, thermo_data=td)
    for kwargs in ({"theta_component": comp},
                   {"theta_component": comp,
                    "thermo_options": {"ddG_units": "kT"}}):
        other = _draw(comp, thermo_data=td, **kwargs)
        for a, b in zip(base[:3], other[:3]):
            pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(base[3]["mutation"], other[3]["mutation"])
        pd.testing.assert_frame_equal(base[3]["pair"], other[3]["pair"])


def test_kcal_ddG_matches_kT_draw(tmp_path):
    """The same energies given in kcal/mol, with ddG_units: kcal/mol, give
    the kT draw."""
    comp = "thermo.O2_C4_K3_U0_a.PddG"
    td = _ddG_csv(tmp_path / "ddG.csv", comp)
    RT = 1.987204259e-3 * 298.15
    df = pd.read_csv(td)
    df[_C4_STRUCTS] = df[_C4_STRUCTS] * RT
    td_kcal = str(tmp_path / "ddG_kcal.csv")
    df.to_csv(td_kcal, index=False)
    ref = _draw(comp, thermo_data=td, theta_component=comp)[3]["mutation"]
    got = _draw(comp, thermo_data=td_kcal, theta_component=comp,
                thermo_options={"ddG_units": "kcal/mol"})[3]["mutation"]
    pd.testing.assert_frame_equal(ref, got, atol=1e-5, rtol=0)
    # Read as kT, the kcal file would shrink every ΔΔG by RT.
    raw = _draw(comp, thermo_data=td_kcal, theta_component=comp)[3]["mutation"]
    assert not np.allclose(raw["ddG_H"], ref["ddG_H"], atol=1e-3)


def test_reordered_csv_with_extras_labels_ddG_truth(tmp_path):
    """Extra columns, a 'Mutation' column and a pandas index are accepted, and
    the ddG truth columns follow the model's structure order, not the file's."""
    comp = "thermo.O2_C12_K5_U0_a.PddG"
    td = _ddG_csv(tmp_path / "ddG.csv", comp)
    df = pd.read_csv(td).rename(columns={"mut": "Mutation"})
    df["unfolded"] = 0.0
    df = df[["Mutation", "unfolded"] + list(reversed(_C12_STRUCTS))]
    td2 = str(tmp_path / "ddG_messy.csv")
    df.to_csv(td2)                               # writes 'Unnamed: 0'
    ref = _draw(comp, thermo_data=td, theta_component=comp)[3]["mutation"]
    got = _draw(comp, thermo_data=td2, theta_component=comp)[3]["mutation"]
    pd.testing.assert_frame_equal(ref, got)
    # Without the component the file order is kept; the truth labels still
    # follow the model's order.
    rev = str(tmp_path / "rev.csv")
    pd.read_csv(td).to_csv(rev, index=False,
                           columns=["mut"] + list(reversed(_C12_STRUCTS)))
    got2 = _draw(comp, thermo_data=rev)[3]["mutation"]
    pd.testing.assert_frame_equal(ref, got2)


# ---------------------------------------------------------------------------
# small branches of the truth helpers
# ---------------------------------------------------------------------------

@dataclass
class _WrongLengthParam:
    ln_K_op: np.ndarray    # (G,)
    other: np.ndarray      # (2,): not per-genotype when G = 3


def test_theta_param_to_df_skips_fields_not_sized_by_genotype():
    param = _WrongLengthParam(ln_K_op=np.array([1.0, 2.0, 3.0]),
                              other=np.array([7.0, 8.0]))
    df = _theta_param_to_df(param, ["a", "b", "c"], [0, 1, 2],
                            num_genotype_sim=3)
    assert list(df.columns) == ["genotype", "ln_K_op"]


def test_theta_at_concs_hill_override_and_nan_for_curve_less_override():
    from tfscreen.simulate.thermo_to_growth import _theta_at_concs

    concs = np.array([0.0, 0.1, 1.0])
    fake_module = types.SimpleNamespace(
        run_model=lambda param, data: (np.full((len(concs), 3), 0.5),))
    hill = {"theta_low": 0.1, "theta_high": 0.9, "log_hill_K": 0.0,
            "hill_n": 2.0}
    with patch.dict(model_registry["theta"], {"fake": fake_module}), \
         patch("tfscreen.simulate.sim_data_class.with_titrant_conc",
               side_effect=lambda sd, c: sd):
        out = _theta_at_concs("fake", None, None, [1.0, 0.0, 0.1],
                              ["wt", "A1B", "C2D"], [0, 1, 2],
                              theta_gc_override={"A1B": np.ones(3),
                                                 "C2D": np.ones(3)},
                              theta_params_override={"A1B": hill})
    wide = out.pivot(index="genotype", columns="titrant_conc", values="theta")
    np.testing.assert_allclose(wide.loc["wt"], [0.5, 0.5, 0.5])
    log_c = np.log([1e-20, 0.1, 1.0])
    expected = 0.1 + 0.8 / (1.0 + np.exp(-2.0 * log_c))
    np.testing.assert_allclose(wide.loc["A1B"], expected)
    assert wide.loc["C2D"].isna().all()


def test_mutation_truth_tables_label_mismatch_raises():
    sim_data = types.SimpleNamespace(num_mutation=3, num_pair=0,
                                     struct_names=None)
    with pytest.raises(ValueError, match=r"labels \(2/0\) do not match "
                                         r"sim_data \(3/0\)"):
        _mutation_truth_tables({"theta_d_x": np.zeros(3)},
                               ["wt", "A1B", "C2D"], sim_data)


def test_mutation_truth_tables_ddG_without_structure_names_numbers_them():
    sim_data = types.SimpleNamespace(num_mutation=2, num_pair=0,
                                     struct_names=("H",))   # wrong length
    det = {"theta_ddG": np.array([[1.0, 2.0], [3.0, 4.0]])}
    mut, pair = _mutation_truth_tables(det, ["wt", "A1B", "C2D"], sim_data)
    assert pair is None
    assert list(mut.columns) == ["mutation", "ddG_0", "ddG_1"]
    assert mut["ddG_1"].tolist() == [2.0, 4.0]


def test_ddG_sigma_truth_table_numbers_structures_without_names():
    from tfscreen.simulate.thermo_to_growth import _ddG_sigma_truth_table

    sites = {"theta_ddG_sigma": np.array([0.5, 1.5])}
    df = _ddG_sigma_truth_table(sites, None)
    assert df["structure"].tolist() == ["0", "1"]
    assert df["ddG_sigma"].tolist() == [0.5, 1.5]
    df = _ddG_sigma_truth_table(sites, ["H", "L", "X"])     # wrong length
    assert df["structure"].tolist() == ["0", "1"]
    assert _ddG_sigma_truth_table({}, ["H"]) is None
