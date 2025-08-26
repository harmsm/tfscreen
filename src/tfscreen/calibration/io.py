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

    with open(json_file,'r') as f:
        calibration = json.load(f)

    calibration["cov_matrix"] = np.array(calibration["cov_matrix"])
    calibration["param_values"] = np.array(calibration["param_values"])

    return calibration


def write_calibration(calibration,
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

    calibration = copy.deepcopy(calibration)
    calibration["param_values"] = [float(f)
                                   for f in calibration["param_values"]]
    
    cov = calibration["cov_matrix"]
    
    cov_list = []
    for i in range(cov.shape[0]):
        cov_list.append([])
        for j in range(cov.shape[1]):
            cov_list[-1].append(float(cov[i,j]))

    calibration["cov_matrix"] = cov_list

    with open(json_file,'w') as f:
        json.dump(calibration,f,indent=2,sort_keys=True)


