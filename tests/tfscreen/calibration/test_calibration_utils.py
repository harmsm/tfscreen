
import pytest
import numpy as np
import pandas as pd
import os
import json
from tfscreen.calibration.io import read_calibration, write_calibration

def test_read_calibration_from_dict():
    d = {"some": "data"}
    assert read_calibration(d) == d

def test_read_write_calibration_roundtrip(tmp_path):
    f = tmp_path / "cal.json"
    
    cal_data = {
        "k_bg": {
            "m": {"t1": 1.0},
            "b": {"t1": 2.0}
        },
        "dk_cond": {
            "m": {"c1": 0.5},
            "b": {"c1": 0.1}
        },
        "theta_param": {
            "t1": [1.0, 2.0, 3.0, 4.0]
        }
    }
    
    write_calibration(cal_data, str(f))
    
    # Read back
    read_data = read_calibration(str(f))
    
    # Check data integrity
    assert read_data["k_bg"]["m"]["t1"] == 1.0
    assert read_data["theta_param"]["t1"] == [1.0, 2.0, 3.0, 4.0]
    
    # Check dataframes were created
    assert "k_bg_df" in read_data
    assert "dk_cond_df" in read_data
    
    assert read_data["k_bg_df"].loc["t1", "m"] == 1.0
    assert read_data["dk_cond_df"].loc["c1", "b"] == 0.1

def test_read_calibration_error_mismatch():
    # Test error when m exists but b does not
    cal_data = {
        "k_bg": {
            "m": {"t1": 1.0},
            "b": {} # Missing t1
        }
    }
    
    # We can't use write_calibration because it doesn't validate this? 
    # Actually write_calibration just dumps dict. 
    # But read_calibration checks consistency.
    
    import json
    f_path = "bad_cal.json"
    with open(f_path, "w") as f:
        json.dump(cal_data, f)
        
    try:
        with pytest.raises(ValueError, match="titrant t1 only seen in k_bg"):
            read_calibration(f_path)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)

def test_read_calibration_error_mismatch_dk():
    cal_data = {
        "k_bg": {"m": {}, "b": {}},
        "dk_cond": {
            "m": {"c1": 1.0},
            "b": {} 
        }
    }
    f_path = "bad_cal_dk.json"
    with open(f_path, "w") as f:
        json.dump(cal_data, f)
        
    try:
        with pytest.raises(ValueError, match="condition c1 only seen in dk_cond"):
            # Looking at source code:
            # err = f"condition {k} only seen in k_bg slope not intercept\n"
            # It says "condition {k}" but then "only seen in k_bg". typo in source?
            # Yes, line 61: "only seen in k_bg slope not intercept"
            read_calibration(f_path)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)
