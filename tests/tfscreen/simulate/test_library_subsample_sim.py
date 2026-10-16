"""tfs-simulate with a subsampled library (library_genotypes,
library_subsample; genetics/library_subsample.py)."""
import glob
import os

import pandas as pd
import pytest
import yaml

from tfscreen.genetics import LibraryManager, library_composition_table
from tfscreen.genetics.library_subsample import build_library, subsample_design
from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config
from tfscreen.tfmodel.scripts.configure_model_cli import (
    check_genotypes_in_library,
)

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "examples", "simulate-and-analyze")


def _design():
    rows = []
    for rep in (1, 2):
        for conc in (0.0, 1.0):
            for t in (100, 140):
                rows.append({"sample": f"k{rep}-{conc}-{t}", "library": "kanR",
                             "replicate": rep, "condition_pre": "kanR-kan",
                             "t_pre": 30, "condition_sel": "kanR+kan",
                             "t_sel": t, "titrant_name": "iptg",
                             "titrant_conc": conc})
    return pd.DataFrame(rows)


@pytest.fixture
def cfg_dir(tmp_path):
    """A small simulate config (design, no binding/presplit) in its own
    directory, with the design file next to it."""
    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    cf.pop("condition_blocks")
    for k in ("binding_data", "presplit_data"):
        cf.pop(k, None)
    d = tmp_path / "cfg"
    d.mkdir()
    _design().to_csv(d / "design.csv", index=False)
    cf["design"] = str(d / "design.csv")
    cf["library_subsample"] = {"double-1-2": 40}
    cf["library_subsample_seed"] = 7
    return d, cf


def _run(d, cf, name, **kwargs):
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cf))
    out = d.parent / name
    kwargs.setdefault("seed", 3)
    run_simulation_from_config(str(path), out_prefix=str(out / "sim"),
                               **kwargs)
    return path, out


def test_subsampled_library_reaches_every_output(cfg_dir, monkeypatch, capsys):
    d, cf = cfg_dir
    monkeypatch.chdir(EXAMPLE)
    path, out = _run(d, cf, "a")
    log = capsys.readouterr().out
    assert "Library subsample (library_subsample_depth: full_library" in log

    kept, summary = build_library(str(path))
    kept_genos = set(kept["genotype"].astype(str))
    full = LibraryManager(cf).build_library_df()
    assert len(kept_genos) < full["genotype"].nunique()

    lib = pd.read_csv(out / "sim_library.csv")
    assert set(lib["genotype"]) == kept_genos
    dbl = set(lib.loc[lib["library_origin"] == "double-1-2", "genotype"])
    assert len(dbl - {"wt"}) <= 40 + 9          # 40 drawn, plus wt/spikes

    growth = pd.read_csv(out / "sim_growth.csv")
    assert set(growth["genotype"]) <= kept_genos | {"__unknown__"}
    # Every raw counts file lists exactly the kept library
    files = sorted(glob.glob(str(out / "sim_counts" / "counts_*.csv")))
    assert len(files) == 8
    total = 0
    for f in files:
        c = pd.read_csv(f)
        assert set(c["genotype"]) == kept_genos | {"__unknown__"}
        total += c["counts"].sum()
    # full_library depth: reads scaled by the kept share of the pool
    design = subsample_design(cf, summary)
    assert total == pytest.approx(design["total_num_reads"], abs=8)
    assert design["total_num_reads"] < cf["total_num_reads"]

    # tfs-configure-model's library snapshot from the same config is the
    # simulated library, and it accepts the simulated growth table
    comp = library_composition_table(str(path))
    assert set(comp["genotype"]) == kept_genos
    check_genotypes_in_library(set(comp["genotype"]), growth, "growth_df")


def test_subsample_is_reproducible_from_seed(cfg_dir, monkeypatch):
    d, cf = cfg_dir
    monkeypatch.chdir(EXAMPLE)
    _, a = _run(d, cf, "a")
    _, b = _run(d, cf, "b")
    pd.testing.assert_frame_equal(pd.read_csv(a / "sim_library.csv"),
                                  pd.read_csv(b / "sim_library.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(a / "sim_growth.csv"),
                                  pd.read_csv(b / "sim_growth.csv"))
    cf["library_subsample_seed"] = 8
    _, c = _run(d, cf, "c")
    assert (set(pd.read_csv(a / "sim_library.csv")["genotype"])
            != set(pd.read_csv(c / "sim_library.csv")["genotype"]))


def test_genotype_list_relative_to_config(cfg_dir, monkeypatch):
    """library_genotypes is read relative to the config file, wherever the
    simulation runs; wt and spikes are added to the list."""
    d, cf = cfg_dir
    full = LibraryManager(cf).build_library_df()
    full["genotype"] = full["genotype"].astype(str)
    listed = sorted(set(full.loc[full["library_origin"] != "double-1-2",
                                 "genotype"]))
    dbl = sorted(g for g in set(full.loc[full["library_origin"]
                                         == "double-1-2", "genotype"])
                 if "/" in g)[:12]
    (d / "keep.txt").write_text("\n".join(listed + dbl) + "\n")
    cf.pop("library_subsample")
    cf["library_genotypes"] = "keep.txt"
    monkeypatch.chdir(EXAMPLE)         # not the config's directory
    _, out = _run(d, cf, "a", write_raw=False)
    lib = pd.read_csv(out / "sim_library.csv")
    assert set(lib["genotype"]) <= set(listed + dbl) | {"wt"}
    assert set(dbl) <= set(lib["genotype"])


def test_genotype_outside_library_refused(cfg_dir, monkeypatch):
    d, cf = cfg_dir
    (d / "keep.txt").write_text("Q999W\n")
    cf["library_genotypes"] = "keep.txt"
    monkeypatch.chdir(EXAMPLE)
    with pytest.raises(ValueError, match="not in the described library"):
        _run(d, cf, "a")


def test_as_written_keeps_configured_reads(cfg_dir, monkeypatch):
    d, cf = cfg_dir
    cf["library_subsample_depth"] = "as_written"
    monkeypatch.chdir(EXAMPLE)
    _, out = _run(d, cf, "a", num_replicates=2)
    total = sum(pd.read_csv(f)["counts"].sum()
                for f in glob.glob(str(out / "sim_counts" / "counts_*.csv")))
    assert total == pytest.approx(cf["total_num_reads"], abs=8)
