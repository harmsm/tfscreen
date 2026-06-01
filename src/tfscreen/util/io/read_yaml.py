import yaml
import re


def _normalize_types(node):
    """
    Recursively convert quoted scientific-notation strings to floats.

    PyYAML parses unquoted numbers natively (``1.5e-7`` → float, ``100_000``
    → int).  This function handles only the edge case where a numeric value
    was quoted in the YAML source (e.g. ``"2.5e7"``), which PyYAML would
    otherwise leave as a string.

    Note: integers and floats are *not* interconverted.  Write count-like
    values as integers (``25_000_000``, ``100_000``) and rate/fraction values
    as floats (``0.01``, ``1.5e-7``).  PyYAML preserves that distinction.
    """
    sci_notation_pattern = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)$')

    if isinstance(node, dict):
        return {k: _normalize_types(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_types(elem) for elem in node]

    if isinstance(node, str) and sci_notation_pattern.match(node):
        return float(node)

    return node


def read_yaml(cf: str | dict,
              override_keys: dict | None = None) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    cf : str or dict
        Path to the YAML file, or a dict (returned as-is).
    override_keys : dict, optional
        Key/value pairs to overwrite in the loaded config.  Every key must
        already exist in the config; unknown keys raise ``ValueError``.

    Returns
    -------
    dict
        Parsed configuration with type normalization applied.

    Raises
    ------
    FileNotFoundError
        If *cf* is a path that does not exist.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    ValueError
        If *override_keys* contains a key absent from the config, or if
        the ``growth`` and ``condition_blocks`` sections are inconsistent.
    """
    if isinstance(cf, dict):
        return cf

    with open(cf) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Could not parse YAML file '{cf}': {exc}") from exc

    config = _normalize_types(config)

    if override_keys is not None:
        for k in override_keys:
            if k not in config:
                raise ValueError(
                    f"override_keys has key '{k}' that is not in the config."
                )
            config[k] = override_keys[k]

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
