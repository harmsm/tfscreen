import yaml
import re

def _normalize_types(node):
    """
    Recursively processes data to:
    1. Convert strings in scientific notation to numbers (float or int).
    2. Convert floats that are whole numbers (e.g., 12.0) to integers.
    """
    
    # Regex to find strings that are valid scientific notation.
    sci_notation_pattern = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)$')

    # Recurse through lists and dictionaries first.
    if isinstance(node, dict):
        return {k: _normalize_types(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_types(elem) for elem in node]

    # --- Type Conversion Logic ---

    # 1. Handle strings: Check if they are scientific notation.
    if isinstance(node, str):
        if sci_notation_pattern.match(node):
            node = float(node)  # Convert the string to a float.
        else:
            return node # It's a regular string, so we're done with it.

    # 2. Handle floats: Check if the value is a whole number.
    # This check runs on original floats and those just converted from strings.
    if isinstance(node, float):
        if node.is_integer():
            return int(node)
        return node

    # Return any other data types (like existing ints, bools, etc.) as is.
    return node

def _rekey_as_tuples(some_dict):
    """
    Rekey strings of numbers as tuples.
    """

    new_dict = {}
    for k in some_dict:
        new_k = k
        if not issubclass(type(k),tuple):
            new_k = tuple(list(str(k)))
        new_dict[new_k] = some_dict[k]

    return new_dict

def load_simulation_config(filepath: str) -> dict:
    """
    Loads a YAML configuration file from the specified path.

    Parameters
    ----------
    filepath : str
        The path to the YAML configuration file.

    Returns
    -------
    config : dict
        A dictionary containing the configuration parameters.
    """

    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{filepath}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    
    # clean up floats and ints
    config = _normalize_types(config)

    # Rekey things like "11" and "12" as ('1','1') and ('1','2')
    to_rekey = ["transform_sizes","library_mixture"]
    for k in to_rekey:
        if k in config:
            config[k] = _rekey_as_tuples(config[k])

    return config