"""Tubes read below the OD600 detection threshold are dropped the same way
by tfs-simulate's direct growth table and by the raw route through
tfs-process-counts."""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.process_raw.scripts.process_counts_cli import process_counts
from tfscreen.simulate.raw_output import undetectable_tubes
from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
EXAMPLE = os.path.join(ROOT, "examples", "simulate-and-analyze")
CALIBRATION = os.path.abspath(os.path.join(ROOT, "examples", "od600",
                                           "od600_calibration.yaml"))
TUBE = ["replicate", "titrant_conc", "t_sel"]


def test_undetectable_tubes_mask():
    df = pd.DataFrame({"od600_detectable": [True, False, True]})
    np.testing.assert_array_equal(undetectable_tubes(df, {"x": 1}),
                                  [False, True, False])
    assert not undetectable_tubes(df, None).any()
    assert not undetectable_tubes(pd.DataFrame({"a": [1]}), {"x": 1}).any()


def test_direct_and_raw_routes_keep_the_same_tubes(tmp_path, monkeypatch,
                                                   capsys):
    with open(os.path.join(EXAMPLE, "simulate_config.yaml")) as fh:
        cf = yaml.safe_load(fh)
    cf.pop("condition_blocks")
    for k in ("binding_data", "presplit_data"):
        cf.pop(k, None)
    rows = []
    for rep in (1, 2):
        for conc in (0.0, 1.0):
            for t in (60, 120, 240):
                rows.append({"sample": f"k{rep}-{conc}-{t}", "library": "kanR",
                             "replicate": rep, "condition_pre": "kanR-kan",
                             "t_pre": 30, "condition_sel": "kanR+kan",
                             "t_sel": t, "titrant_name": "iptg",
                             "titrant_conc": conc})
    pd.DataFrame(rows).to_csv(tmp_path / "design.csv", index=False)
    cf["design"] = str(tmp_path / "design.csv")
    # cfu0 low enough that an early, repressed kanR+kan tube reads below the
    # calibration's detection threshold (0.093)
    cf["cfu0"] = 5e8
    cf["od600"] = {"calibration": CALIBRATION, "tube_volume_mL": 5.0,
                   "sample_cfu_from_od600": True}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cf))
    monkeypatch.chdir(EXAMPLE)
    out = str(tmp_path / "sim")
    run_simulation_from_config(str(tmp_path / "c.yaml"), out_prefix=out,
                               seed=1)
    log = capsys.readouterr().out

    od = pd.read_csv(f"{out}_od600.csv")
    low = od.loc[~od["od600_detectable"].astype(bool), TUBE]
    assert len(low) >= 1, "the fixture should have an undetectable tube"
    assert f"Dropped {len(low)} of {len(od)} tube(s)" in log

    direct = pd.read_csv(f"{out}_growth.csv")
    processed = pd.read_csv(process_counts(
        f"{out}_tubes.csv", f"{out}_counts", out_prefix=str(tmp_path / "p"),
        od600_file=f"{out}_tube_od600.csv",
        od600_calibration_file=f"{out}_od600_calibration.yaml",
        tube_volume_mL=5.0, verbose=False))

    def tubes(df):
        return set(map(tuple, df[TUBE].drop_duplicates().to_numpy()))

    assert tubes(direct) == tubes(processed)
    assert not tubes(direct) & set(map(tuple, low.to_numpy()))
    assert len(tubes(direct)) == len(od) - len(low)
    key = TUBE + ["genotype"]
    m = direct.merge(processed, on=key, suffixes=("_d", "_p"))
    assert len(m) == len(direct) == len(processed)
    np.testing.assert_array_equal(m["counts_d"], m["counts_p"])
    np.testing.assert_allclose(m["sample_cfu_d"], m["sample_cfu_p"],
                               rtol=1e-9)
