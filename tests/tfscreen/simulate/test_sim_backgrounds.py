"""
tfs-simulate with genetic backgrounds, operators and TF levels
(planning/dimer-thermo-model.md step 3e), end to end: the raw experiment
processes like real data and passes tfs-configure-model's background check.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.genetics.library_design import library_composition_table
from tfscreen.process_raw.scripts.process_counts_cli import process_counts
from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config
from tfscreen.tfmodel.scripts.configure_model_cli import (
    check_backgrounds_in_data,
)

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "examples", "simulate-and-analyze")
OUTSIDE = "M1A"          # the far background's mutation, outside the amplicon

CONTEXTS = {"units": "kT",
            "operators": {"O1": {"reference": True},
                          "O2": {"shift": 2.0, "shift_sd": 0.5}},
            "tf_levels": {"low": {"conc_uM": 3.0, "ln_sd": 0.3},
                          "high": {"conc_uM": 30.0, "ln_sd": 0.3}}}


def _design():
    rows = []
    libs = (("kanR", "reference"), ("kanR_far", "far"))
    for library, background in libs:
        for op in ("O1", "O2"):
            for level in ("low", "high"):
                for conc in (0.0, 1.0):
                    for t in (100, 140):
                        rows.append({
                            "sample": f"{library}-{op}-{level}-{conc}-{t}",
                            "library": library, "replicate": 1,
                            "condition_pre": "kanR-kan", "t_pre": 30,
                            "condition_sel": "kanR+kan", "t_sel": t,
                            "titrant_name": "iptg", "titrant_conc": conc,
                            "operator": op, "tf_level": level,
                            "background": background})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    cf.pop("condition_blocks")
    for k in ("binding_data", "presplit_data", "theta_sim_priors"):
        cf.pop(k, None)
    d = tmp_path_factory.mktemp("bg")
    _design().to_csv(d / "design.csv", index=False)
    with open(d / "contexts.yaml", "w") as fh:
        yaml.safe_dump(CONTEXTS, fh)
    cf.update(design=str(d / "design.csv"), contexts=str(d / "contexts.yaml"),
              theta_component="thermo.dimer.PK", total_num_reads=2_000_000,
              backgrounds={"far": {"outside_mutations": [OUTSIDE]}})
    path = d / "config.yaml"
    path.write_text(yaml.safe_dump(cf))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        run_simulation_from_config(str(path), out_prefix=str(d / "sim"),
                                   seed=2)
    finally:
        os.chdir(cwd)
    return d, cf


def test_library_has_both_backgrounds(run):
    d, _ = run
    lib = pd.read_csv(d / "sim_library.csv")
    assert set(lib["background"]) == {"reference", "far"}
    far = lib.loc[lib["background"] == "far", "genotype"].astype(str)
    assert far.str.contains(OUTSIDE).all()
    assert "spiked" not in set(lib.loc[lib["background"] == "far",
                                       "library_origin"])


def test_each_tube_reads_its_own_background(run):
    d, _ = run
    tubes = pd.read_csv(d / "sim_tubes.csv")
    assert {"operator", "tf_level", "background"} <= set(tubes.columns)
    assert len(tubes) == len(_design())
    for _, tube in tubes.iterrows():
        c = pd.read_csv(d / "sim_counts" / f"counts_{tube['sample']}.csv")
        named = c.loc[c["genotype"] != "__unknown__", "genotype"]
        far = named.str.contains(OUTSIDE)
        assert far.all() if tube["background"] == "far" else not far.any()
        assert c["counts"].sum() > 0
        assert (c.set_index("genotype")["counts"]
                .get(OUTSIDE if tube["background"] == "far" else "wt", 0) > 0)


def test_raw_experiment_passes_the_configure_checks(run):
    d, cf = run
    growth = pd.read_csv(process_counts(
        str(d / "sim_tubes.csv"), str(d / "sim_counts"),
        out_prefix=str(d / "processed"), verbose=False))
    assert {"operator", "tf_level", "background"} <= set(growth.columns)
    table = library_composition_table(str(d / "config.yaml"))
    check_backgrounds_in_data(table, growth)


def test_context_truth_is_written(run):
    d, _ = run
    shift = pd.read_csv(d / "sim_operator_shift.csv").set_index("operator")
    assert shift.loc["O1", "operator_shift"] == 0.0
    assert shift.loc["O2", "operator_shift"] != 0.0
    ln_tf = pd.read_csv(d / "sim_ln_tf_total.csv").set_index("tf_level")
    assert np.exp(ln_tf.loc["high", "ln_tf_total"]) > \
        np.exp(ln_tf.loc["low", "ln_tf_total"])
    theta = pd.read_csv(d / "sim_genotype_theta.csv")
    assert {"operator", "tf_level"} <= set(theta.columns)
    assert theta.groupby(["operator", "tf_level"]).ngroups == 4


def test_subsample_keeps_the_same_members_in_each_background(tmp_path):
    from tfscreen.genetics import backgrounds as B
    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    cf.pop("condition_blocks")
    for k in ("binding_data", "presplit_data"):
        cf.pop(k, None)
    design = _design()
    design = design[(design["operator"] == "O1")
                    & (design["tf_level"] == "low")]
    design.to_csv(tmp_path / "design.csv", index=False)
    cf.update(design=str(tmp_path / "design.csv"), total_num_reads=500_000,
              backgrounds={"far": {"outside_mutations": [OUTSIDE]}},
              library_subsample={"double-1-2": 20},
              library_subsample_seed=5)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cf))
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        run_simulation_from_config(str(path), out_prefix=str(tmp_path / "s"),
                                   seed=1, write_raw=False)
    finally:
        os.chdir(cwd)
    lib = pd.read_csv(tmp_path / "s_library.csv")
    far = lib[lib["background"] == "far"]
    ref = lib[lib["background"] == "reference"]
    far_members = set(B.design_names(far["genotype"], frozenset({OUTSIDE})))
    ref_dbl = set(ref.loc[ref["library_origin"] == "double-1-2", "genotype"])
    far_dbl = set(B.design_names(
        far.loc[far["library_origin"] == "double-1-2", "genotype"],
        frozenset({OUTSIDE})))
    assert far_dbl == ref_dbl and len(far_dbl - {"wt"}) <= 20 + 9
    assert far_members <= set(ref["genotype"]) | {"wt"}
    growth = pd.read_csv(tmp_path / "s_growth.csv")
    check_backgrounds_in_data(library_composition_table(str(path)), growth)
