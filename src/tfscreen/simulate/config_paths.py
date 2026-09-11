"""
File-valued keys in a simulate config.

A simulate config names input files in a few places.  Per the YAML
conventions, relative paths are relative to the config file's own directory,
not the working directory -- ``resolve_config_paths`` applies that rule.
``tfs-setup-sim-grid`` uses ``iter_path_entries`` to copy every such file into
each run directory, so runs are self-contained.
"""

import copy
import os

# Top-level keys holding a file path.
TOP_LEVEL_PATH_KEYS = ("thermo_data", "calibration_file")

# Nested keys that may hold a file path.  A choose_by value is a path only
# when it is not one of the builtin selection keywords.
NESTED_PATH_KEYS = (
    ("binding_data", "spiked_binding", "choose_by"),
    ("binding_data", "library_binding", "choose_by"),
    ("empirical", "phenotype_model"),
)
CHOOSE_BY_KEYWORDS = frozenset({"stratified", "random"})


def iter_path_entries(cfg):
    """
    Yield ``(container, key, key_path)`` for every file-valued entry in ``cfg``.

    ``container[key]`` is a non-empty string naming a file (choose_by
    keywords are skipped).  ``key_path`` is the dotted name, for messages.
    Absent blocks are simply skipped.
    """
    for key in TOP_LEVEL_PATH_KEYS:
        value = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(value, str) and value:
            yield cfg, key, key

    for key_path in NESTED_PATH_KEYS:
        node = cfg
        for key in key_path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict):
            continue
        value = node.get(key_path[-1])
        if isinstance(value, str) and value and value not in CHOOSE_BY_KEYWORDS:
            yield node, key_path[-1], ".".join(key_path)


def resolve_config_paths(cfg, base_dir):
    """
    Return a deep copy of ``cfg`` with relative file paths made absolute.

    Relative values are joined to ``base_dir`` (the config file's directory);
    absolute values are left alone.  The input is never mutated.
    """
    out = copy.deepcopy(cfg)
    for container, key, _ in iter_path_entries(out):
        value = container[key]
        if not os.path.isabs(value):
            container[key] = os.path.normpath(os.path.join(base_dir, value))
    return out


def existing_input_file(key_path, value):
    """
    Return the file a path entry refers to, or None if it does not exist.

    ``empirical.phenotype_model`` may be given as the model JSON, the path
    without ``.json``, or the bare ``tfs-build-empirical`` out_prefix (see
    ``library_prediction._resolve_phenotype_model_path``).
    """
    candidates = [value]
    if key_path == "empirical.phenotype_model":
        candidates += [f"{value}.json", f"{value}_phenotype_model.json"]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None
