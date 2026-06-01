"""
Shared utilities for tfs-setup-grid and tfs-setup-sim-grid.

Provides run-name generation, Jinja2 environment construction, and config-file
path rewriting.  Both grid CLIs import from here; keep this module free of any
growth-model- or simulate-specific logic.
"""

import os
import re

import jinja2
import yaml


def sanitize(s):
    """Make a string safe for use as a directory-name component."""
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]", "_", str(s))).strip("_")


def make_jinja_env(strict=True):
    """Return a Jinja2 Environment with the ``basename`` filter registered."""
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined if strict else jinja2.Undefined
    )
    env.filters["basename"] = os.path.basename
    return env


def make_run_name(name_template, all_vars, index):
    """Return a filesystem-safe directory name for this run."""
    prefix = f"run_{index:04d}"
    if name_template:
        try:
            raw = make_jinja_env(strict=False).from_string(name_template).render(**all_vars)
        except jinja2.TemplateError as e:
            raise ValueError(f"run_name template error: {e}") from e
        suffix = sanitize(raw)
    else:
        suffix = "_".join(
            sanitize(str(v)) for v in all_vars.values() if str(v).strip()
        )
    return f"{prefix}_{suffix}" if suffix else prefix


def relativize_node(node, subdir):
    """Recursively rewrite absolute path strings in a config node to be relative to subdir."""
    if isinstance(node, dict):
        return {k: relativize_node(v, subdir) for k, v in node.items()}
    if isinstance(node, list):
        return [relativize_node(v, subdir) for v in node]
    if isinstance(node, str) and node and os.path.isabs(node) and os.path.exists(node):
        return os.path.relpath(node, subdir)
    return node


def relativize_config_paths(yaml_path, subdir):
    """Rewrite absolute paths anywhere in a written config YAML to be relative to subdir."""
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    updated = relativize_node(cfg, subdir)
    if updated != cfg:
        with open(yaml_path, "w") as fh:
            yaml.dump(updated, fh, default_flow_style=False, sort_keys=False)


def relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir):
    """Return tmpl_vars with any path-like string values rewritten relative to subdir.

    A value is treated as a path when it resolves (relative to grid_yaml_dir, or
    as-is if absolute) to a file or directory that exists on disk.
    """
    out = {}
    for key, val in tmpl_vars.items():
        if isinstance(val, str) and val:
            abs_path = val if os.path.isabs(val) else os.path.normpath(
                os.path.join(grid_yaml_dir, val)
            )
            if os.path.exists(abs_path):
                out[key] = os.path.relpath(abs_path, subdir)
                continue
        out[key] = val
    return out
