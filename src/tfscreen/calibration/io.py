import numpy as np

import json
import copy

def read_calibration(json_file):
    """
    Read a calibration dictionary out of a json file.

    Parameters
    ----------
    json_file : str
        path to json file to read

    Returns
    -------
    calibration : dict
        calibration dictionary
    """

    # If this is already a dictionary, return it
    if issubclass(type(json_file),dict):
        return json_file

    with open(json_file,'r') as f:
        calibration_dict = json.load(f)

    calibration_dict["cov_matrix"] = np.array(calibration_dict["cov_matrix"])
    calibration_dict["param_values"] = np.array(calibration_dict["param_values"])
    
    # Build param_dict, a convenient dictionary for looking up paramter values
    param_dict = {}
    for i, p in enumerate(calibration_dict["param_names"]):
        param_dict[p] = calibration_dict["param_values"][i]
    calibration_dict["param_dict"] = param_dict

    return calibration_dict


def write_calibration(calibration_dict,
                      json_file):
    """
    Write a calibration dictionary to a json file.

    Parameters
    ----------
    calibration : dict
        calibration dictionary
    json_file : str
        path to json file to write
    """

    calibration_dict = copy.deepcopy(calibration_dict)
    calibration_dict["param_values"] = [float(f)
                                        for f in calibration_dict["param_values"]]
    
    cov = calibration_dict["cov_matrix"]
    
    cov_list = []
    for i in range(cov.shape[0]):
        cov_list.append([])
        for j in range(cov.shape[1]):
            cov_list[-1].append(float(cov[i,j]))

    calibration_dict["cov_matrix"] = cov_list

    # Remove the param_dict attribute for consistency. Parameter values are 
    # stored **only** in param_values. 
    if "param_dict" in calibration_dict:
        calibration_dict.pop("param_dict")

    with open(json_file,'w') as f:
        json.dump(calibration_dict,f,indent=2,sort_keys=True)


