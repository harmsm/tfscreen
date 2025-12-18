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

    # If this is already a dictionary, return it
    if issubclass(type(json_file),dict):
        return json_file

    with open(json_file,'r') as f:
        calibration_input = json.load(f)

    calibration_dict = copy.deepcopy(calibration_input)

    k_bg_keys = []
    ms = []
    bs = []
    for k in calibration_dict["k_bg"]["m"]:
        if k not in calibration_dict["k_bg"]["b"]:
            err = f"titrant {k} only seen in k_bg slope not intercept\n"
            raise ValueError(err)
        k_bg_keys.append(k)
        bs.append(calibration_dict["k_bg"]["b"][k])
        ms.append(calibration_dict["k_bg"]["m"][k])

    k_bg_df = pd.DataFrame({
        "m":ms,
        "b":bs,
    },index=k_bg_keys)
    calibration_dict["k_bg_df"] = k_bg_df

    dk_cond_keys = []
    ms = []
    bs = []
    for k in calibration_dict["dk_cond"]["m"]:
        if k not in calibration_dict["dk_cond"]["b"]:
            err = f"condition {k} only seen in dk_cond slope not intercept\n"
            raise ValueError(err)
        dk_cond_keys.append(k)
        bs.append(calibration_dict["dk_cond"]["b"][k])
        ms.append(calibration_dict["dk_cond"]["m"][k])

    dk_cond_df = pd.DataFrame({
        "m":ms,
        "b":bs,
    },index=dk_cond_keys)
    calibration_dict["dk_cond_df"] = dk_cond_df

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


