"""Tests for tfs-calibrate-od600."""
import os

import pytest

from tfscreen.process_raw import od600 as O
from tfscreen.process_raw.scripts.calibrate_od600_cli import (
    calibrate_od600,
    main,
)

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "examples", "od600")


def test_writes_all_outputs(tmp_path):
    out = str(tmp_path / "sub" / "cal")
    calibrate_od600(os.path.join(EXAMPLE, "replicates.csv"),
                    os.path.join(EXAMPLE, "plate_counts.csv"),
                    out_prefix=out)
    for suffix in (".yaml", "_replicates.csv", "_plate_counts.csv", ".pdf"):
        assert os.path.exists(out + suffix)
    cal = O.read_calibration(out + ".yaml")
    assert cal["source"]["plate_count_file"] == "plate_counts.csv"


def test_main_parses_flags(tmp_path):
    out = str(tmp_path / "cal")
    main_args = [os.path.join(EXAMPLE, "replicates.csv"),
                 os.path.join(EXAMPLE, "plate_counts.csv"),
                 "--out_prefix", out, "--degree", "1",
                 "--detection_threshold", "0.1"]
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["tfs-calibrate-od600"] + main_args):
        main()
    cal = O.read_calibration(out + ".yaml")
    assert cal["degree"] == 1 and cal["detection_threshold"] == 0.1


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_cli_module_runs_as_script(monkeypatch):
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["tfs-calibrate-od600", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("tfscreen.process_raw.scripts.calibrate_od600_cli",
                         run_name="__main__")
    assert exc.value.code == 0
