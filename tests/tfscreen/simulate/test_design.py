"""Simulation designs: a tube table in place of condition_blocks."""
import os

import pandas as pd
import pytest
import yaml

from tfscreen.simulate.build_sample_dataframes import (
    CONDITION_COLUMNS,
    build_sample_dataframes,
    design_conditions,
    read_design,
)

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "examples", "simulate-and-analyze")


def _design():
    rows = []
    for rep, shift in ((1, 0), (2, 7)):
        for conc in (0.0, 1.0):
            for t in (100, 120):
                if rep == 2 and conc == 1.0 and t == 120:
                    continue          # a missing tube
                rows.append({"sample": f"k-r{rep}-{conc}-{t + shift}",
                             "library": "kanR", "replicate": rep,
                             "condition_pre": "kanR-kan", "t_pre": 30,
                             "condition_sel": "kanR+kan", "t_sel": t + shift,
                             "titrant_name": "iptg", "titrant_conc": conc,
                             "od600": 0.3})
    return pd.DataFrame(rows)


def test_read_design_keeps_tube_table_columns(tmp_path):
    path = tmp_path / "d.csv"
    _design().to_csv(path, index=False)
    d = read_design(str(path))
    assert list(d.columns) == ["sample", "library", "replicate",
                               "condition_pre", "t_pre", "condition_sel",
                               "t_sel", "titrant_name", "titrant_conc"]
    assert len(d) == 7


@pytest.mark.parametrize("change,match", [
    (lambda d: d.drop(columns="t_sel"), "missing column"),
    (lambda d: d.assign(sample="same"), "repeats sample"),
    (lambda d: pd.concat([d, d.iloc[[0]].assign(sample="x")]), "same replicate"),
])
def test_read_design_refuses(change, match):
    with pytest.raises(ValueError, match=match):
        read_design(change(_design()))


def test_design_conditions_is_the_union_over_replicates():
    conds = design_conditions(read_design(_design()))
    # replicate 1: 4 tubes; replicate 2: 3 tubes at other times -> 7 conditions
    assert len(conds) == 7
    assert set(conds["replicate"]) == {1}
    blocks = build_sample_dataframes([{
        "library": "kanR", "titrant_name": "iptg", "titrant_conc": [0.0],
        "condition_pre": "kanR-kan", "t_pre": 30, "condition_sel": "kanR+kan",
        "t_sel": [100]}])
    assert list(conds.columns) == list(blocks.columns)


def test_simulate_follows_a_design(tmp_path, monkeypatch):
    """Each replicate gets exactly its design's tubes, under the design's
    names, and the raw tube table is the design."""
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config

    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    cf.pop("condition_blocks")
    cf.pop("binding_data", None)
    cf.pop("presplit_data", None)
    design = _design()
    design.to_csv(tmp_path / "design.csv", index=False)
    cf["design"] = str(tmp_path / "design.csv")
    with open(tmp_path / "cfg.yaml", "w") as fh:
        yaml.safe_dump(cf, fh)

    monkeypatch.chdir(EXAMPLE)
    run_simulation_from_config(str(tmp_path / "cfg.yaml"),
                               out_prefix=str(tmp_path / "sim"), seed=3)

    tubes = pd.read_csv(tmp_path / "sim_tubes.csv")
    cols = ["sample", "replicate"] + CONDITION_COLUMNS
    got = tubes[cols].sort_values("sample").reset_index(drop=True)
    want = design[cols].sort_values("sample").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    files = sorted(os.listdir(tmp_path / "sim_counts"))
    assert files == sorted(f"counts_{s}.csv" for s in design["sample"])
    growth = pd.read_csv(tmp_path / "sim_growth.csv")
    assert set(zip(growth["replicate"], growth["t_sel"])) == \
        set(zip(design["replicate"], design["t_sel"]))


def test_design_and_blocks_refused(tmp_path):
    from tfscreen.simulate import library_prediction

    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    _design().to_csv(tmp_path / "design.csv", index=False)
    cf["design"] = str(tmp_path / "design.csv")
    with pytest.raises(ValueError, match="not both"):
        library_prediction(cf)


def test_cfu0_per_library_sets_each_librarys_totals(tmp_path, monkeypatch):
    """cfu0 can be a {library: cells} map; tube totals scale with it."""
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config

    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    for k in ("binding_data", "presplit_data"):
        cf.pop(k, None)
    monkeypatch.chdir(EXAMPLE)

    base = float(str(cf["cfu0"]).replace("_", ""))
    totals = {}
    for name, cfu0 in (("scalar", base),
                       ("map", {"kanR": base * 4.0, "pheS": base})):
        c = dict(cf, cfu0=cfu0)
        with open(tmp_path / f"{name}.yaml", "w") as fh:
            yaml.safe_dump(c, fh)
        run_simulation_from_config(str(tmp_path / f"{name}.yaml"),
                                   out_prefix=str(tmp_path / name), seed=5,
                                   num_replicates=1, write_raw=False)
        g = pd.read_csv(tmp_path / f"{name}_growth.csv")
        totals[name] = (g.drop_duplicates(["library", "condition_sel", "t_sel",
                                           "titrant_conc"])
                        .groupby("library")["sample_cfu"].sum())
    ratio = totals["map"] / totals["scalar"]
    assert ratio["kanR"] == pytest.approx(4.0, rel=1e-6)
    assert ratio["pheS"] == pytest.approx(1.0, rel=1e-6)


def test_cfu0_map_must_name_every_library(tmp_path, monkeypatch):
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config

    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    for k in ("binding_data", "presplit_data"):
        cf.pop(k, None)
    cf["cfu0"] = {"kanR": 1e7}
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cf, fh)
    monkeypatch.chdir(EXAMPLE)
    with pytest.raises(ValueError, match="per library"):
        run_simulation_from_config(str(tmp_path / "c.yaml"),
                                   out_prefix=str(tmp_path / "x"),
                                   num_replicates=1, write_raw=False)
