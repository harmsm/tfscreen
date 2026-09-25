"""
Shared utilities for tfs-setup-grid and tfs-setup-sim-grid.

Provides run-name generation, Jinja2 environment construction and input-file
staging.  Both grid CLIs import from here; keep
this module free of any growth-model- or simulate-specific logic.

Input staging
-------------
A grid output directory must be movable as a unit (on and off a cluster, onto
another partition), so a run may not refer to anything outside it.
``InputStager`` copies each input file into ``<out_prefix>/inputs/`` once and
returns the path a run subdirectory uses to reach it (``../inputs/<name>``);
``check_no_outside_paths`` fails on a config value that names a file but is not
one of the keys the caller stages.
"""

import filecmp
import os
import re
import shutil

import jinja2

# Directory under out_prefix holding the input files shared by all runs.
INPUTS_DIRNAME = "inputs"


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


# ---------------------------------------------------------------------------
# Input staging (self-contained, movable grid directories)
# ---------------------------------------------------------------------------

class InputStager:
    """Copy input files into ``<out_prefix>/inputs/`` once each.

    ``stage(path, what)`` returns the path a run subdirectory uses to reach the
    copy (``../inputs/<name>``). The same source file always maps to the same
    copy. A different file whose name is taken (by an earlier source in this
    setup, or by a different file already in the directory) gets a numeric
    suffix, so an existing run is never left pointing at changed contents.

    With ``dry_run=True`` it only checks that each path is an existing file and
    copies nothing, so a whole grid can be validated before anything is written.

    Parameters
    ----------
    out_prefix : str
        Root directory of the grid.
    tool : str
        Name of the calling CLI, used in error messages.
    dry_run : bool, optional
        Validate only; copy nothing. Default False.
    """

    def __init__(self, out_prefix, tool, dry_run=False):
        self.inputs_dir = os.path.join(os.path.abspath(out_prefix), INPUTS_DIRNAME)
        self.tool = tool
        self.dry_run = dry_run
        self._by_source = {}
        self._claimed = set()

    def stage(self, path, what):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} names a file that does not exist: {path}")
        if not os.path.isfile(path):
            raise ValueError(
                f"{what} names a directory ({path}). {self.tool} copies input "
                f"files into the grid so it can be moved, and does not copy "
                f"directories. Point it at a file."
            )
        source = os.path.realpath(path)
        if self.dry_run:
            return os.path.join("..", INPUTS_DIRNAME, os.path.basename(source))
        if source in self._by_source:
            return self._by_source[source]

        os.makedirs(self.inputs_dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(source))
        n = 1
        while True:
            name = f"{stem}{ext}" if n == 1 else f"{stem}_{n}{ext}"
            dest = os.path.join(self.inputs_dir, name)
            if name not in self._claimed:
                if not os.path.exists(dest):
                    shutil.copy2(source, dest)
                    break
                if os.path.isfile(dest) and filecmp.cmp(source, dest, shallow=False):
                    break
            n += 1

        self._claimed.add(name)
        rel = os.path.join("..", INPUTS_DIRNAME, name)
        self._by_source[source] = rel
        return rel


def check_no_outside_paths(cfg, known_key_paths, source_dirs, hint):
    """Fail if a config value outside ``known_key_paths`` names an existing file.

    Such a value would be written as-is and so depend on a location outside the
    grid. An absolute value is checked as-is (file or directory); a relative
    value is checked as a file against the directory it came from.

    Parameters
    ----------
    cfg : dict
        Config (possibly nested) to walk.
    known_key_paths : iterable of tuple
        Key paths (tuples of keys into ``cfg``) the caller stages itself.
    source_dirs : str or dict
        Directory relative values resolve against: one directory for the whole
        config, or a dict mapping each top-level key to its directory.
    hint : str
        Appended to the error message: what to do if the value really is a
        file the run reads.
    """
    known = {tuple(k) for k in known_key_paths}

    def base_dir(key_path):
        if isinstance(source_dirs, dict):
            return source_dirs[key_path[0]]
        return source_dirs

    def walk(node, key_path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, key_path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, key_path + (i,))
        elif isinstance(node, str) and node and key_path not in known:
            if os.path.isabs(node):
                target = node if os.path.exists(node) else None
            else:
                cand = os.path.join(base_dir(key_path), node)
                target = cand if os.path.isfile(cand) else None
            if target is not None:
                dotted = ".".join(str(k) for k in key_path)
                raise ValueError(
                    f"Config key '{dotted}' = '{node}' names a file or "
                    f"directory ({os.path.normpath(target)}), but '{dotted}' is "
                    f"not one of the file-path keys copied into the grid. The "
                    f"grid must be movable, so a run cannot refer to a location "
                    f"outside it. {hint}"
                )

    walk(cfg, ())


def stage_template_vars(tmpl_vars, grid_yaml_dir, stager):
    """Return tmpl_vars with each value naming an existing file replaced by its copy.

    A value is a file when it resolves (relative to grid_yaml_dir, or as-is if
    absolute) to an existing file. A value naming an existing directory is an
    error (``.`` excepted: it means the run directory).
    """
    out = {}
    for key, val in tmpl_vars.items():
        if isinstance(val, str) and val and val != ".":
            path = val if os.path.isabs(val) else os.path.join(grid_yaml_dir, val)
            if os.path.isfile(path):
                val = stager.stage(path, f"Template variable '{key}'")
            elif os.path.isdir(path):
                stager.stage(path, f"Template variable '{key}'")  # raises
        out[key] = val
    return out


def render_run_template(jinja_template, tmpl_vars, run_name):
    """Render the run template, naming the run on an undefined variable."""
    try:
        return jinja_template.render(**tmpl_vars)
    except jinja2.UndefinedError as exc:
        raise ValueError(
            f"Undefined template variable for run '{run_name}': {exc}"
        ) from exc
