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

    # split string keys delimited by ||| into tuples
    for level in ["m","b"]:
        for k in calibration_input[level]:
            new_k = tuple(k.split("|||"))
            calibration_dict[level][new_k] = calibration_input[level][k]
            calibration_dict[level].pop(k)

    # Convert numpy arrays to lists
    for level in ["theta_param","bg_model_param"]:
        for k in calibration_input[level]:
            calibration_dict[level][k] = np.array(calibration_input[level][k])

    # Build linear_df

    # Get the keys that will form the index.
    keys = list(calibration_dict["m"].keys())

    # Add ("background",titrant) keys (zero slope and intercept)
    unique_titrant = set([k[1] for k in keys])
    for t in unique_titrant:
        new_key = ("background",t)
        keys.append(new_key)
        calibration_dict["m"][new_key] = 0.0
        calibration_dict["b"][new_key] = 0.0
    
    # Explicitly create the MultiIndex.
    index = pd.MultiIndex.from_tuples(keys, names=['condition', 'titrant_name'])
    
    # Build the DataFrame using the created index.
    # Look up 'b' values using the keys from 'm' to ensure correct alignment.
    linear_df = pd.DataFrame({
        "m": calibration_dict["m"].values(),
        "b": [calibration_dict["b"][k] for k in keys]
    }, index=index)

    calibration_dict["linear_df"] = linear_df

    return calibration_dict


def write_calibration(calibration_dict,
                      json_file):
    """
    Write a calibration dictionary to a JSON file.

    This function prepares a Python-native calibration dictionary for JSON
    serialization. It performs necessary transformations, including converting
    tuple keys (used for 'm' and 'b' parameters) into strings and converting
    NumPy arrays into lists. It also removes the derived 'linear_df' before
    writing to ensure the stored format is minimal and consistent.

    Parameters
    ----------
    calibration_dict : dict
        The calibration dictionary to write.
    json_file : str
        The file path for the output JSON file.
    """

    calibration_out = copy.deepcopy(calibration_dict)

    # Convert tuple keys to string keys delimited by |||
    for level in ["m","b"]:
        for k in calibration_dict[level]:
            new_k = f"{k[0]}|||{k[1]}"
            calibration_out[level][new_k] = calibration_dict[level][k]
            calibration_out[level].pop(k)

    # Convert numpy arrays to lists
    for level in ["theta_param","bg_model_param"]:
        for k in calibration_dict[level]:
            calibration_out[level][k] = list(calibration_dict[level][k])

    # Remove the linear_df attribute for consistency. Parameter values are 
    # stored **only** in dictionaries in the json. 
    if "linear_df" in calibration_out:
        calibration_out.pop("linear_df")

    with open(json_file,'w') as f:
        json.dump(calibration_out,f,indent=2,sort_keys=True)

        # Force it to write so we can immediately read back in
        f.flush()
        os.fsync(f.fileno())


