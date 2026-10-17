import os

import pandas as pd
import pytest

from tfscreen.process_raw.binding import BINDING_COLUMNS
from tfscreen.process_raw.scripts.process_binding_cli import process_binding


def _wells():
    rows = []
    for rep, offset in ((1, 0.0), (2, 0.1)):
        for c, t in ((0.0, 0.9), (1e-6, 0.5)):
            rows.append(dict(genotype="wt", biorep=rep, date=f"d{rep}",
                             protein_dimer_conc_molar=2e-6,
                             iptg_conc_molar=c, theta=t + offset))
    rows.append(dict(genotype="M42I", biorep=1, date="d3",
                     protein_dimer_conc_molar=2e-6, iptg_conc_molar=0.0,
                     theta=0.3))
    return pd.DataFrame(rows)


def test_process_binding_writes_tables(tmpdir):
    src = os.path.join(tmpdir, "wells.csv")
    _wells().to_csv(src, index=False)
    out = os.path.join(tmpdir, "b")
    process_binding(src, out_prefix=out)
    df = pd.read_csv(f"{out}.csv")
    assert list(df.columns) == BINDING_COLUMNS
    assert set(df["genotype"]) == {"wt", "M42I"}
    summary = pd.read_csv(f"{out}_summary.csv")
    assert "n_replicates" in summary.columns


def test_process_binding_empty_replicate_cols(tmpdir):
    src = os.path.join(tmpdir, "wells.csv")
    _wells().drop(columns=["biorep", "date"]).to_csv(src, index=False)
    out = os.path.join(tmpdir, "b")
    process_binding(src, replicate_cols="", theta_std=0.1, out_prefix=out)
    assert os.path.exists(f"{out}.csv")


def test_process_binding_refuses_overwrite(tmpdir):
    src = os.path.join(tmpdir, "b.csv")
    _wells().to_csv(src, index=False)
    with pytest.raises(ValueError, match="overwrite"):
        process_binding(src, out_prefix=os.path.join(tmpdir, "b"))
