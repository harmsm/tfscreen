import os

import numpy as np
import pandas as pd
import pytest

from tfscreen.process_raw import od600 as O
from tfscreen.process_raw.scripts.process_counts_cli import (
    add_tube_totals,
    process_counts,
)

CALIBRATION = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "examples", "od600", "od600_calibration.yaml")


def _tubes(libraries=("kanR",), presplit_cols=True):
    rows = []
    for lib in libraries:
        for t in (0, 60):
            rows.append({"sample": f"{lib}-t{t}", "library": lib,
                         "replicate": 1, "condition_pre": f"{lib}-pre",
                         "condition_sel": f"{lib}+sel", "t_pre": 30,
                         "t_sel": t, "titrant_name": "iptg",
                         "titrant_conc": 0.0})
    df = pd.DataFrame(rows)
    if not presplit_cols:
        df = df.drop(columns=["replicate", "condition_pre"])
    return df


def _write_counts(d, tubes):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for s, lib in zip(tubes["sample"], tubes["library"]):
        genotypes = ["wt", "A1V", "C2D", "__unknown__"]
        if lib == "pheS":
            genotypes = ["wt", "E3F", "__unknown__"]
        pd.DataFrame({"genotype": genotypes,
                      "counts": rng.integers(50, 500, len(genotypes))}
                     ).to_csv(os.path.join(d, f"counts_{s}_.csv"), index=False)
    return d


# ---------------------------------------------------------------------------
# tube totals
# ---------------------------------------------------------------------------

def test_add_tube_totals_from_od600_column():
    tubes = _tubes()
    tubes["od600"] = [0.2, 0.4]
    out = add_tube_totals(tubes, od600_calibration_file=CALIBRATION, tube_volume_mL=5.0)
    cfu, sd, _ = O.od600_to_cfu_per_mL(np.array([0.2, 0.4]), CALIBRATION)
    assert np.allclose(out["sample_cfu"], cfu * 5.0)
    assert np.allclose(out["sample_cfu_std"], sd * 5.0)
    # the shared curve error and the per-reading error are kept apart
    assert np.allclose(out["sample_cfu_std"] ** 2,
                       out["sample_cfu_curve_std"] ** 2
                       + out["sample_cfu_reading_std"] ** 2)
    assert out["od600_in_calibrated_range"].all()


def test_add_tube_totals_from_od600_file(tmp_path):
    tubes = _tubes()
    od = pd.DataFrame({"sample": tubes["sample"], "od600": [0.2, 0.4]})
    od.to_csv(tmp_path / "od.csv", index=False)
    out = add_tube_totals(tubes, od600_file=str(tmp_path / "od.csv"),
                          od600_calibration_file=CALIBRATION, tube_volume_mL=5.0)
    assert list(out["od600"]) == [0.2, 0.4]


def test_add_tube_totals_passthrough_supplied_totals():
    tubes = _tubes()
    tubes["sample_cfu"] = [1e8, 2e8]
    tubes["sample_cfu_std"] = [1e7, 2e7]
    pd.testing.assert_frame_equal(add_tube_totals(tubes), tubes)


@pytest.mark.parametrize("kwargs,match", [
    ({"tube_volume_mL": 5.0}, "calibration"),
    ({"od600_calibration_file": CALIBRATION}, "tube_volume_mL"),
])
def test_add_tube_totals_od600_needs_calibration_and_volume(kwargs, match):
    tubes = _tubes()
    tubes["od600"] = [0.2, 0.4]
    with pytest.raises(ValueError, match=match):
        add_tube_totals(tubes, **kwargs)


def test_add_tube_totals_refuses_totals_and_od600():
    tubes = _tubes()
    tubes["od600"] = [0.2, 0.4]
    tubes["sample_cfu"] = [1e8, 2e8]
    with pytest.raises(ValueError, match="Use one or the other"):
        add_tube_totals(tubes, od600_calibration_file=CALIBRATION, tube_volume_mL=5.0)


def test_add_tube_totals_calibration_without_od600():
    with pytest.raises(ValueError, match="no OD600"):
        add_tube_totals(_tubes(), od600_calibration_file=CALIBRATION)


def test_add_tube_totals_missing_reading(tmp_path):
    tubes = _tubes()
    pd.DataFrame({"sample": tubes["sample"][:1], "od600": [0.2]}).to_csv(
        tmp_path / "od.csv", index=False)
    with pytest.raises(ValueError, match="No OD600 reading"):
        add_tube_totals(tubes, od600_file=str(tmp_path / "od.csv"),
                        od600_calibration_file=CALIBRATION, tube_volume_mL=5.0)


def test_tube_totals_refuses_undetectable_reading():
    od = pd.DataFrame({"sample": ["a", "b"], "od600": [0.01, 0.4]})
    with pytest.raises(ValueError, match="detection threshold"):
        O.tube_totals_from_od600(od, CALIBRATION, 5.0)


def test_tube_totals_refuses_duplicate_reading():
    od = pd.DataFrame({"sample": ["a", "a"], "od600": [0.3, 0.4]})
    with pytest.raises(ValueError, match="more than one reading"):
        O.tube_totals_from_od600(od, CALIBRATION, 5.0)


# ---------------------------------------------------------------------------
# process_counts end to end
# ---------------------------------------------------------------------------

def test_process_counts_growth_from_od600(tmp_path):
    tubes = _tubes()
    tubes["od600"] = [0.2, 0.4]
    tubes.to_csv(tmp_path / "tubes.csv", index=False)
    counts = _write_counts(str(tmp_path / "counts"), tubes)

    out = process_counts(str(tmp_path / "tubes.csv"), counts,
                         out_prefix=str(tmp_path / "growth"),
                         od600_calibration_file=CALIBRATION, tube_volume_mL=5.0,
                         verbose=False)
    df = pd.read_csv(out)
    assert out == str(tmp_path / "growth.csv")
    for col in ("genotype", "counts", "sample_reads", "frequency", "ln_cfu",
                "ln_cfu_var", "sample_cfu", "sample_cfu_curve_std",
                "condition_sel", "t_sel"):
        assert col in df.columns
    assert "obs_file" not in df.columns
    assert "__unknown__" not in set(df["genotype"])
    # ln_cfu = ln(frequency) + ln(tube total)
    assert np.allclose(df["ln_cfu"],
                       np.log(df["frequency"]) + np.log(df["sample_cfu"]))


def test_process_counts_several_libraries_match_separate_runs(tmp_path):
    tubes = _tubes(libraries=("kanR", "pheS"))
    tubes["sample_cfu"] = [1e8, 3e8, 2e8, 5e8]
    tubes["sample_cfu_std"] = tubes["sample_cfu"] * 0.1
    counts = _write_counts(str(tmp_path / "counts"), tubes)

    tubes.to_csv(tmp_path / "all.csv", index=False)
    joint = pd.read_csv(process_counts(str(tmp_path / "all.csv"), counts,
                                       out_prefix=str(tmp_path / "joint"),
                                       verbose=False))

    parts = []
    for lib, sub in tubes.groupby("library"):
        sub.to_csv(tmp_path / f"{lib}.csv", index=False)
        parts.append(pd.read_csv(process_counts(
            str(tmp_path / f"{lib}.csv"), counts,
            out_prefix=str(tmp_path / f"sep_{lib}"), verbose=False)))
    separate = pd.concat(parts)

    key = ["library", "sample", "genotype"]
    joint = joint.sort_values(key, ignore_index=True)
    separate = separate.sort_values(key, ignore_index=True)[joint.columns]
    pd.testing.assert_frame_equal(joint, separate)
    # genotypes stay within their own library
    assert "E3F" not in set(joint.loc[joint["library"] == "kanR", "genotype"])


def test_process_counts_presplit(tmp_path):
    tubes = _tubes()
    tubes["sample_cfu"] = [1e8, 3e8]
    tubes["sample_cfu_std"] = [1e7, 3e7]
    tubes.to_csv(tmp_path / "tubes.csv", index=False)
    counts = _write_counts(str(tmp_path / "counts"), tubes)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        out = process_counts("tubes.csv", counts, presplit=True, verbose=False)
    finally:
        os.chdir(cwd)
    assert out == "tfs_presplit.csv"
    df = pd.read_csv(tmp_path / out)
    # tfs-configure-model's presplit reader needs library
    assert list(df.columns) == ["library", "replicate", "condition_pre",
                                "genotype", "ln_cfu", "ln_cfu_std"]
    # canonical genotype order within each replicate x condition_pre: wt first
    assert df["genotype"].iloc[0] == "wt"


def test_process_counts_presplit_needs_design_columns(tmp_path):
    tubes = _tubes(presplit_cols=False)
    tubes.to_csv(tmp_path / "tubes.csv", index=False)
    with pytest.raises(ValueError, match="replicate"):
        process_counts(str(tmp_path / "tubes.csv"), str(tmp_path), presplit=True)


def test_process_counts_missing_totals(tmp_path):
    tubes = _tubes()
    tubes.to_csv(tmp_path / "tubes.csv", index=False)
    counts = _write_counts(str(tmp_path / "counts"), tubes)
    with pytest.raises(ValueError, match="sample_cfu"):
        process_counts(str(tmp_path / "tubes.csv"), counts,
                       out_prefix=str(tmp_path / "x"), verbose=False)


def test_process_counts_default_library(tmp_path):
    tubes = _tubes().drop(columns="library")
    tubes["sample_cfu"] = [1e8, 3e8]
    tubes["sample_cfu_std"] = [1e7, 3e7]
    tubes.to_csv(tmp_path / "tubes.csv", index=False)
    tubes["library"] = "kanR"
    counts = _write_counts(str(tmp_path / "counts"), tubes)
    df = pd.read_csv(process_counts(str(tmp_path / "tubes.csv"), counts,
                                    out_prefix=str(tmp_path / "g"),
                                    verbose=False))
    assert set(df["library"]) == {"default"}
