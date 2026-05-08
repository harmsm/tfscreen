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


def read_yaml(cf: str | dict,
              override_keys: dict | None=None) -> dict:
    """
    Loads a YAML configuration file from the specified path.

    Parameters
    ----------
    cf : str or dict
        If string, this is the path to the YAML configuration file. If a dict,
        pass through (assume its already read)

    Returns
    -------
    config : dict
        A dictionary containing the configuration parameters.
    """

    if issubclass(type(cf),dict):
        return cf

    try:
        with open(cf, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{cf}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    
    # clean up floats and ints
    config = _normalize_types(config)

    # Replace keys from the configuration with keyword arguments passed in.
    if override_keys is not None:
        for k in override_keys:
            if k not in config:
                err = f"override_keys has a key '{k}' that was not in configuration."
                raise ValueError(err)
            config[k] = override_keys[k]

    # Validate growth vs condition_blocks when both are present.
    if "growth" in config and "condition_blocks" in config:
        defined = set(config["growth"].keys())
        used = set()
        for block in config["condition_blocks"]:
            used.add(block["condition_pre"])
            used.add(block["condition_sel"])
        missing_from_growth = used - defined
        extra_in_growth = defined - used
        problems = []
        if missing_from_growth:
            problems.append(
                "conditions in condition_blocks but absent from growth: "
                + str(sorted(missing_from_growth))
            )
        if extra_in_growth:
            problems.append(
                "keys in growth not used by any condition_block: "
                + str(sorted(extra_in_growth))
            )
        if problems:
            raise ValueError(
                "growth / condition_blocks mismatch in config:\n  "
                + "\n  ".join(problems)
            )

    return config