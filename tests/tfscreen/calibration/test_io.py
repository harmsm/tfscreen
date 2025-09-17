import pytest
import json
import numpy as np
import pandas as pd

from tfscreen.calibration.io import (
    read_calibration,
    write_calibration
)

import pytest
import json
import numpy as np
import pandas as pd
import os # write_calibration now uses os.fsync

# Assuming functions are in this namespace for testing
from tfscreen.calibration.io import (
    read_calibration,
    write_calibration
)

def test_read_write_calibration_roundtrip(tmp_path):
    """
    Tests that writing and reading a calibration dict correctly processes the data,
    including the addition of a background entry.
    """
    # 1. ARRANGE
    json_file = tmp_path / "test_calibration.json"
    original_dict = {
        "m": {("wt", "iptg"): 0.5, ("mut1", "iptg"): 0.6},
        "b": {("wt", "iptg"): 0.1, ("mut1", "iptg"): 0.2},
        "theta_param": {"iptg": np.array([1.0, -0.9, 10.0, 2.0])},
        "bg_model_param": {"iptg": np.array([0.025, 0.001])}
    }

    # 2. ACT
    write_calibration(original_dict, json_file)
    roundtrip_dict = read_calibration(json_file)

    # 3. ASSERT
    # Check the contents of the intermediate JSON file
    with open(json_file, 'r') as f:
        on_disk_data = json.load(f)
    assert on_disk_data["m"]["wt|||iptg"] == 0.5
    assert isinstance(on_disk_data["theta_param"]["iptg"], list)

    # Check that 'linear_df' was created and has the correct shape and content
    assert "linear_df" in roundtrip_dict
    linear_df = roundtrip_dict.pop("linear_df")
    
    # Original dict had 2 keys for one titrant; the reader adds one background key.
    assert len(linear_df) == 3
    # Check that the new background row exists and has the correct values of 0.0
    assert ("background", "iptg") in linear_df.index
    assert linear_df.loc[("background", "iptg"), "m"] == 0.0
    assert linear_df.loc[("background", "iptg"), "b"] == 0.0
    # Check that original data is still present and correct
    assert linear_df.loc[("wt", "iptg"), "m"] == 0.5

    # The reader also adds keys to the 'm' and 'b' dicts. We must remove them
    # before comparing the rest of the dictionary to the original.
    assert ("background", "iptg") in roundtrip_dict["m"]
    roundtrip_dict["m"].pop(("background", "iptg"))
    roundtrip_dict["b"].pop(("background", "iptg"))
    
    # Now, the rest of the dictionary should match the original
    assert roundtrip_dict.keys() == original_dict.keys()
    assert roundtrip_dict["m"] == original_dict["m"]
    
    # For keys with numpy arrays, compare them element-wise
    for k in original_dict["theta_param"]:
        np.testing.assert_array_equal(
            roundtrip_dict["theta_param"][k], original_dict["theta_param"][k]
        )
    for k in original_dict["bg_model_param"]:
        np.testing.assert_array_equal(
            roundtrip_dict["bg_model_param"][k], original_dict["bg_model_param"][k]
        )

def test_read_calibration_multiple_titrants(tmp_path):
    """
    Tests that the reader correctly adds background rows for every unique titrant.
    """
    # 1. ARRANGE
    json_file = tmp_path / "multi_titrant.json"
    input_dict = {
        "m": {("wt", "iptg"): 0.5, ("wt", "atc"): 0.8},
        "b": {("wt", "iptg"): 0.1, ("wt", "atc"): 0.3},
        "theta_param": {},
        "bg_model_param": {}
    }
    write_calibration(input_dict, json_file)

    # 2. ACT
    result_dict = read_calibration(json_file)

    # 3. ASSERT
    linear_df = result_dict["linear_df"]
    # Should have 2 original rows + 2 background rows
    assert len(linear_df) == 4
    
    # Check that background rows were added for BOTH titrants
    assert ("background", "iptg") in linear_df.index
    assert linear_df.loc[("background", "iptg"), "m"] == 0.0
    
    assert ("background", "atc") in linear_df.index
    assert linear_df.loc[("background", "atc"), "m"] == 0.0

def test_read_calibration_pass_through():
    """
    Tests that passing a dict to read_calibration returns it unchanged.
    This test is unaffected by the changes and remains valid.
    """
    my_dict = {"test": 1}
    result = read_calibration(my_dict)
    assert result is my_dict # Should return the same object, not a copy