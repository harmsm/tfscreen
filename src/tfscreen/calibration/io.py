import numpy as np
import pandas as pd

import json
import copy
import os

def read_calibration(json_file):
    """
    Read and parse a calibration dictionary from a JSON file.

    This function loads a JSON file containing calibration data. It performs
    necessary transformations, such as converting string-delimited keys into
    tuples and lists into NumPy arrays. It also adds a 'linear_df' DataFrame
    to the dictionary for convenient downstream lookups of slope and intercept
    parameters.

    Parameters
    ----------
    json_file : str or dict
        The file path to the JSON file to read. If a dictionary is passed,
        it is returned directly without modification.

    Returns
    -------
    dict
        The fully parsed calibration dictionary, ready for use in Python.
    """

    # If this is already a dictionary, use it directly but ensure copy
    if issubclass(type(json_file),dict):
        calibration_dict = copy.deepcopy(json_file)
    else:
        with open(json_file,'r') as f:
            calibration_dict = json.load(f)

    # Ensure DataFrames are present
    if "k_bg_df" not in calibration_dict:
        k_bg_data = calibration_dict.get("k_bg", {})
        if k_bg_data:
            calibration_dict["k_bg_df"] = pd.DataFrame(k_bg_data)

    if "dk_cond_df" not in calibration_dict:
        dk_cond_data = calibration_dict.get("dk_cond", {})
        if dk_cond_data:
            calibration_dict["dk_cond_df"] = pd.DataFrame(dk_cond_data)

    return calibration_dict


def write_calibration(calibration_dict,
                      json_file):
    """
    Write a calibration dictionary to a JSON file.

    This function prepares a Python-native calibration dictionary for JSON
    serialization. It also removes the derived 'dk_cond_df' and 'k_bg_df' before
    writing to ensure the stored format is minimal and consistent.

    Parameters
    ----------
    calibration_dict : dict
        The calibration dictionary to write.
    json_file : str
        The file path for the output JSON file.
    """

    calibration_out = copy.deepcopy(calibration_dict)

    # Remove the linear_df attribute for consistency. Parameter values are 
    # stored **only** in dictionaries in the json. 
    if "dk_cond_df" in calibration_out:
        calibration_out.pop("dk_cond_df")

    if "k_bg_df" in calibration_out:
        calibration_out.pop("k_bg_df")
    
    for k in calibration_dict["theta_param"]:
        calibration_out["theta_param"][k] = list(calibration_out["theta_param"][k])

    with open(json_file,'w') as f:
        json.dump(calibration_out,f,indent=2,sort_keys=True)

        # Force it to write so we can immediately read back in
        f.flush()
        os.fsync(f.fileno())


