"""Tests for tfscreen.simulate.raw_output (the simulator's raw formats)."""
import os

import numpy as np
import pandas as pd
import pytest

from tfscreen.process_raw import counts_to_lncfu
from tfscreen.process_raw.od600 import od600_to_cfu_per_mL
from tfscreen.process_raw.scripts.process_counts_cli import process_counts
from tfscreen.simulate.raw_output import sample_names, write_raw_experiment

CALIBRATION = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "examples", "od600", "od600_calibration.yaml")
GENOTYPES = ["wt", "A1V", "C2D", "E3F"]


def _experiment(n_tubes=12, od600=None, seed=0):
    """Simulator-shaped frames: integer sample ids, design columns, totals,
    counts only for genotypes seen (no __unknown__)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tubes):
        rows.append({"sample": i, "titrant_name": "iptg",
                     "titrant_conc": [0.0, 1.0][i % 2],
                     "condition_pre": "kanR-kan", "t_pre": 30,
                     "condition_sel": "kanR+kan", "t_sel": 60 * (i // 2),
                     "replicate": 1, "library": "kanR", "activity": 1.0,
                     "sample_cfu": 1e8 * (1 + i), "sample_cfu_std": 0.0})
    tubes = pd.DataFrame(rows).set_index(pd.Index(range(n_tubes)))
    if od600 is not None:
        tubes["od600"] = od600
        tubes["od600_detectable"] = np.asarray(od600) > 0.0929
        cfu, sd, _ = od600_to_cfu_per_mL(np.asarray(od600), CALIBRATION)
        tubes["sample_cfu"] = cfu * 5.0
        tubes["sample_cfu_std"] = sd * 5.0
    counts = []
    for i in range(n_tubes):
        # E3F never seen: the raw files must still list it, with 0
        for g in GENOTYPES[:3]:
            counts.append({"sample": i, "genotype": g,
                           "counts": int(rng.integers(20, 2000)),
                           "dk_geno": 0.0, "theta": 0.5})
    return tubes, pd.DataFrame(counts)


def _as_fastq_counts(counts, samples):
    """The counts as tfs-process-fastq would report them: every library
    genotype in every tube, zeros included, plus __unknown__."""
    full = pd.MultiIndex.from_product([samples, GENOTYPES + ["__unknown__"]],
                                      names=["sample", "genotype"])
    c = counts.set_index(["sample", "genotype"])["counts"].reindex(full,
                                                                   fill_value=0)
    return c.reset_index()


def test_sample_names_are_not_substrings_of_each_other():
    names = sample_names(1500)
    assert len(set(names)) == 1500
    assert all(len(n) == len(names[0]) for n in names)
    assert not any(a != b and a in b for a in names[:20] for b in names)


def test_raw_files_are_in_fastq_format(tmp_path):
    tubes, counts = _experiment()
    out = write_raw_experiment(tubes, counts, GENOTYPES, str(tmp_path / "sim"))
    files = sorted(os.listdir(out["counts_dir"]))
    assert files[0] == "counts_tube0001.csv" and len(files) == 12
    c = pd.read_csv(os.path.join(out["counts_dir"], files[0]))
    assert list(c.columns) == ["genotype", "counts"]
    assert list(c["genotype"]) == GENOTYPES + ["__unknown__"]
    assert c.set_index("genotype").loc[["E3F", "__unknown__"], "counts"].tolist() == [0, 0]
    t = pd.read_csv(out["tubes"])
    # design columns and totals only; no simulator truth
    assert "activity" not in t.columns and "dk_geno" not in t.columns
    assert {"sample_cfu", "sample_cfu_std"} <= set(t.columns)
    assert out["command"].startswith("tfs-process-counts ")


def test_raw_round_trip_reproduces_direct_growth_table(tmp_path):
    """tfs-process-counts on the raw files gives the table tfs-simulate
    builds in memory from the same counts in fastq form. (From the
    simulator's sparse counts the in-memory table differs only by the
    pseudocounts of genotypes never seen, which a real counts file lists.)"""
    tubes, counts = _experiment()
    direct = counts_to_lncfu(tubes, _as_fastq_counts(counts, tubes.index))
    out = write_raw_experiment(tubes, counts, GENOTYPES, str(tmp_path / "sim"))
    processed = pd.read_csv(process_counts(out["tubes"], out["counts_dir"],
                                           out_prefix=str(tmp_path / "p"),
                                           verbose=False))
    key = ["t_sel", "titrant_conc", "genotype"]
    m = direct.astype({"genotype": str}).merge(processed, on=key,
                                               suffixes=("_d", "_p"))
    assert len(m) == len(direct) == len(processed)
    for col in ("counts", "sample_reads", "ln_cfu", "ln_cfu_var"):
        np.testing.assert_allclose(m[f"{col}_d"], m[f"{col}_p"], rtol=1e-12)


def test_raw_od600_route(tmp_path):
    od = [0.05] + [0.15 + 0.03 * i for i in range(11)]   # first unreadable
    tubes, counts = _experiment(od600=od)
    out = write_raw_experiment(tubes, counts, GENOTYPES, str(tmp_path / "sim"),
                               od600_config={"calibration": CALIBRATION,
                                             "tube_volume_mL": 5.0})
    assert out["dropped"] == ["tube0001"]
    t = pd.read_csv(out["tubes"])
    assert "sample_cfu" not in t.columns and "tube0001" not in set(t["sample"])
    assert not os.path.exists(os.path.join(out["counts_dir"], "counts_tube0001.csv"))
    assert "--od600_file" in out["command"]
    assert os.path.exists(out["calibration"])

    processed = pd.read_csv(process_counts(
        out["tubes"], out["counts_dir"], out_prefix=str(tmp_path / "p"),
        od600_file=out["od600"], od600_calibration_file=out["calibration"],
        tube_volume_mL=5.0, verbose=False))
    kept = tubes.iloc[1:]
    direct = counts_to_lncfu(kept, _as_fastq_counts(counts, kept.index))
    m = direct.astype({"genotype": str}).merge(
        processed, on=["t_sel", "titrant_conc", "genotype"], suffixes=("_d", "_p"))
    assert len(m) == len(direct)
    np.testing.assert_allclose(m["sample_cfu_d"], m["sample_cfu_p"], rtol=1e-12)
    np.testing.assert_allclose(m["ln_cfu_d"], m["ln_cfu_p"], rtol=1e-12)


def test_counts_outside_library_refused(tmp_path):
    tubes, counts = _experiment(n_tubes=2)
    with pytest.raises(ValueError, match="outside the library"):
        write_raw_experiment(tubes, counts, ["wt"], str(tmp_path / "sim"))


def test_unassigned_reads_round_trip(tmp_path):
    """Simulated __unknown__ reads (unassigned_read_fraction) land in the raw
    files' __unknown__ row and count toward the tube's reads, in both the
    in-memory table and tfs-process-counts."""
    tubes, counts = _experiment()
    unknown = pd.DataFrame({"sample": tubes.index, "genotype": "__unknown__",
                            "counts": 5000})
    counts = pd.concat([counts, unknown], ignore_index=True)
    direct = counts_to_lncfu(tubes, _as_fastq_counts(counts, tubes.index))
    out = write_raw_experiment(tubes, counts, GENOTYPES, str(tmp_path / "sim"))
    raw = pd.read_csv(f"{out['counts_dir']}/counts_tube0001.csv")
    assert raw.set_index("genotype").loc["__unknown__", "counts"] == 5000
    processed = pd.read_csv(process_counts(out["tubes"], out["counts_dir"],
                                           out_prefix=str(tmp_path / "p"),
                                           verbose=False))
    assert "__unknown__" not in set(processed["genotype"])
    assigned = counts[counts.genotype != "__unknown__"].groupby("sample")["counts"].sum()
    reads = processed.groupby("t_sel")["sample_reads"].first()
    assert (reads > assigned.max()).all()
    key = ["t_sel", "titrant_conc", "genotype"]
    m = direct.astype({"genotype": str}).merge(processed, on=key,
                                               suffixes=("_d", "_p"))
    for col in ("sample_reads", "ln_cfu"):
        np.testing.assert_allclose(m[f"{col}_d"], m[f"{col}_p"], rtol=1e-12)
