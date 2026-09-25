"""
tfs-setup-sim-grid — set up a directory grid for simulation runs.

Creates one subdirectory per combination of simulate-parameter variants,
writes a modified ``tfs_sim_config.yaml`` in each (base config + overrides),
and optionally renders a Jinja2 template with the per-run template variables.

GRID YAML FORMAT
----------------

    base_config: ../simulate_config.yaml   # base config to override

    run_name: "{{ theta_component }}__noise{{ growth_rate_noise }}"
    output_file: run.sh   # Jinja2 template; looked up next to this YAML

    simulate:              # key-value overrides applied to the base config
      - name: thermodynamic_model
        variants:
          - theta_component: thermo.O2_C12_K5_U0_a.PK
          - theta_component: hill_geno
      - name: noise
        variants:
          - growth_rate_noise: 0.01
          - growth_rate_noise: 0.05

    template:              # variables injected into the Jinja2 template only
      - name: num_replicates
        variants:
          - num_replicates: 3

NOTES
-----
- The Cartesian product is taken across **all** blocks (simulate + template).
- ``simulate`` variables override top-level keys in the base config.  Nested
  keys are not supported — override the entire top-level key if needed.
- ``simulate`` variables are NOT injected into the template; ``template``
  variables are NOT written to the config.  To share a variable, list it in both.
- ``run_name`` may reference variables from either section.
- Use the ``basename`` Jinja2 filter to strip path info from filenames:
      run_name: "{{ thermo_data | basename }}__noise{{ growth_rate_noise }}"

INPUT FILES
-----------
The grid directory is self-contained and can be moved as a unit (on and off a
cluster, onto another partition). Every input file a run needs is copied into
``<out_prefix>/inputs/`` once, and each run's config and rendered template
refer to it as ``../inputs/<name>``. Nothing in a run points outside the grid.

- Config file paths are the keys listed in ``_SIM_PATH_KEYS``, nested ones
  included (``thermo_data``, ``empirical.phenotype_model``,
  ``binding_data.*.choose_by``, ...). A relative path resolves against the
  base config's directory for a base-config value and against the grid YAML's
  directory for a ``simulate`` override. A ``choose_by`` keyword
  (``stratified``/``random``) is not a path and is left alone.
- A ``template`` variable that names an existing file (relative to the grid
  YAML, or absolute) is copied the same way.
- Two different files with the same name are kept apart by a numeric suffix
  (``hill_params_2.csv``); the same file used by many runs is copied once.
- Setup fails rather than write a run that depends on a location outside the
  grid: a missing input file, an input that is a directory, or a config value
  outside ``_SIM_PATH_KEYS`` that names an existing file (add its key to
  ``_SIM_PATH_KEYS`` if the simulator reads it).
"""

import copy
import filecmp
import itertools
import json
import os
import shutil

import jinja2
import yaml

from tfscreen.util.cli import generalized_main
from tfscreen.util.grid_utils import (
    make_jinja_env as _make_jinja_env,
    make_run_name as _make_run_name,
)

# Keys in the simulate config that hold file paths, as key paths into the
# (possibly nested) config dict. Add any new file-valued key here; setup fails on
# a config value outside this list that names an existing file.
_SIM_PATH_KEYS = (
    ("thermo_data",),
    ("calibration_file",),
    ("empirical", "phenotype_model"),
    ("binding_data", "spiked_binding", "choose_by"),
    ("binding_data", "library_binding", "choose_by"),
)

# Values of a path key that are keywords, not paths (see
# simulate/library_prediction.py::_is_file_choice).
_PATH_KEYWORDS = frozenset({"stratified", "random"})

# Fixed filename for the per-run config written into each subdirectory.
_SIM_CONFIG_FILENAME = "tfs_sim_config.yaml"

# Directory under out_prefix holding the input files shared by all runs.
_INPUTS_DIRNAME = "inputs"


# ---------------------------------------------------------------------------
# Block expansion
# ---------------------------------------------------------------------------

def _expand_block(block):
    """Return a list of variant dicts for one block entry (``variants`` form only)."""
    if "variants" in block:
        return list(block["variants"])
    raise ValueError(
        f"Block '{block.get('name', '?')}' must have a 'variants' list. "
        f"The 'auto' form is not supported for simulate grids."
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_paths(vars_dict, base_dir):
    """Return a copy of vars_dict with _SIM_PATH_KEYS resolved to absolute paths.

    Nested key paths are followed only through dicts; missing or empty values,
    absolute paths, non-strings and ``_PATH_KEYWORDS`` are left unchanged. The
    input is not modified.
    """
    out = copy.deepcopy(vars_dict)
    for key_path in _SIM_PATH_KEYS:
        node = out
        for key in key_path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            continue
        val = node.get(key_path[-1])
        if (not isinstance(val, str) or not val or os.path.isabs(val)
                or val in _PATH_KEYWORDS):
            continue
        node[key_path[-1]] = os.path.normpath(os.path.join(base_dir, val))
    return out


class _InputStager:
    """Copy input files into ``<out_prefix>/inputs/`` once each.

    ``stage(path)`` returns the path a run subdirectory uses to reach the copy
    (``../inputs/<name>``). The same source file always maps to the same copy.
    A different file whose name is taken (by an earlier source in this setup,
    or by a different file already in the directory) gets a numeric suffix, so
    an existing run is never left pointing at changed contents.

    With ``dry_run=True`` it only checks that each path is an existing file and
    copies nothing, so a whole grid can be validated before anything is written.
    """

    def __init__(self, out_prefix, dry_run=False):
        self.inputs_dir = os.path.join(os.path.abspath(out_prefix), _INPUTS_DIRNAME)
        self.dry_run = dry_run
        self._by_source = {}
        self._claimed = set()

    def stage(self, path, what):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} names a file that does not exist: {path}")
        if not os.path.isfile(path):
            raise ValueError(
                f"{what} names a directory ({path}). tfs-setup-sim-grid copies "
                f"input files into the grid so it can be moved, and does not "
                f"copy directories. Point it at a file."
            )
        source = os.path.realpath(path)
        if self.dry_run:
            return os.path.join("..", _INPUTS_DIRNAME, os.path.basename(source))
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
        rel = os.path.join("..", _INPUTS_DIRNAME, name)
        self._by_source[source] = rel
        return rel


def _get_node(cfg, key_path):
    """Return the dict holding ``key_path[-1]``, or None if the path is absent."""
    node = cfg
    for key in key_path[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def _phenotype_model_file(path):
    """Return the file a ``phenotype_model`` value loads (mirrors the simulator)."""
    for cand in (path, f"{path}.json", f"{path}_phenotype_model.json"):
        if os.path.isfile(cand):
            return cand
    return path


def _stage_config_files(run_cfg, stager):
    """Replace each ``_SIM_PATH_KEYS`` value in run_cfg (in place) with its copy.

    Values must already be absolute (see ``_resolve_paths``).
    """
    for key_path in _SIM_PATH_KEYS:
        node = _get_node(run_cfg, key_path)
        if node is None:
            continue
        val = node.get(key_path[-1])
        if not isinstance(val, str) or not val or val in _PATH_KEYWORDS:
            continue
        if key_path == ("empirical", "phenotype_model"):
            val = _phenotype_model_file(val)
        node[key_path[-1]] = stager.stage(
            val, f"Config key '{'.'.join(key_path)}'"
        )


def _check_no_outside_paths(run_cfg, source_dirs):
    """Fail if a config value outside ``_SIM_PATH_KEYS`` names an existing file.

    Such a value would be written as-is and so depend on a location outside the
    grid. ``source_dirs`` maps each top-level key to the directory its relative
    values came from.
    """
    known = set(_SIM_PATH_KEYS)

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
                cand = os.path.join(source_dirs[key_path[0]], node)
                target = cand if os.path.isfile(cand) else None
            if target is not None:
                dotted = ".".join(str(k) for k in key_path)
                raise ValueError(
                    f"Config key '{dotted}' = '{node}' names a file or "
                    f"directory ({os.path.normpath(target)}), but '{dotted}' is "
                    f"not one of the file-path keys tfs-setup-sim-grid copies "
                    f"into the grid. The grid must be movable, so a run cannot "
                    f"refer to a location outside it. If the simulator reads "
                    f"this file, add the key to _SIM_PATH_KEYS in "
                    f"{__name__}; otherwise change the value."
                )

    walk(run_cfg, ())


def _stage_template_vars(tmpl_vars, grid_yaml_dir, stager):
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


def _render(jinja_template, tmpl_vars, run_name):
    """Render the run template, naming the run on an undefined variable."""
    try:
        return jinja_template.render(**tmpl_vars)
    except jinja2.UndefinedError as exc:
        raise ValueError(
            f"Undefined template variable for run '{run_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_sim_grid(grid_yaml, out_prefix="sim_grid"):
    """
    Set up a directory grid for simulation runs.

    Reads a grid YAML file, loads a base simulate config, expands all simulate
    and template blocks into their Cartesian product, and for each combination:

    1. Creates a subdirectory under *out_prefix*.
    2. Merges the simulate-block overrides into the base config and writes
       ``tfs_sim_config.yaml`` inside that subdirectory.
    3. Renders the Jinja2 template (``output_file``) with the template
       variables and writes the result to the subdirectory.
    4. Writes ``combo.json`` recording the variable assignments for that run.

    A ``grid_summary.json`` is written to *out_prefix* listing all created runs.
    Input files named by the configs or template variables are copied into
    ``<out_prefix>/inputs/`` and referenced as ``../inputs/<name>``, so the
    *out_prefix* directory can be moved as a unit. Every run is validated
    before anything is written.

    Parameters
    ----------
    grid_yaml : str
        Path to the grid YAML file.
    out_prefix : str, optional
        Root directory for the grid.  Created if it does not exist.
        Default ``"sim_grid"``.

    Returns
    -------
    list of dict
        One dict per created run, each with keys ``run``, ``simulate``,
        and ``template``.
    """
    grid_yaml = os.path.abspath(grid_yaml)
    if not os.path.exists(grid_yaml):
        raise FileNotFoundError(f"Grid YAML not found: {grid_yaml}")

    grid_yaml_dir = os.path.dirname(grid_yaml)

    with open(grid_yaml) as fh:
        grid = yaml.safe_load(fh)

    base_config_path = grid.get("base_config")
    if not base_config_path:
        raise ValueError(
            "Grid YAML must have a 'base_config' key pointing to a simulate config."
        )

    if not os.path.isabs(base_config_path):
        base_config_path = os.path.normpath(os.path.join(grid_yaml_dir, base_config_path))

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_cfg_dir = os.path.dirname(base_config_path)

    with open(base_config_path) as fh:
        base_cfg = yaml.safe_load(fh)

    run_name_template = grid.get("run_name")
    output_file = grid.get("output_file")

    sim_block_specs = grid.get("simulate") or []
    tmpl_block_specs = grid.get("template") or []

    if not sim_block_specs and not tmpl_block_specs:
        raise ValueError(
            "Grid YAML must have at least one of 'simulate' or 'template' blocks."
        )

    sim_variant_lists = [_expand_block(b) for b in sim_block_specs]
    tmpl_variant_lists = [_expand_block(b) for b in tmpl_block_specs]

    n_sim = len(sim_variant_lists)
    all_combos = []
    for combo_tuple in itertools.product(*(sim_variant_lists + tmpl_variant_lists)):
        sim_vars = {}
        tmpl_vars = {}
        for i, variant in enumerate(combo_tuple):
            if i < n_sim:
                sim_vars.update(variant)
            else:
                tmpl_vars.update(variant)
        all_combos.append((sim_vars, tmpl_vars))

    jinja_template = None
    if output_file:
        template_path = os.path.join(grid_yaml_dir, output_file)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        template_text = open(template_path).read()
        try:
            jinja_template = _make_jinja_env(strict=True).from_string(template_text)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error in {output_file}: {e}") from e

    # Build every run's config and template variables, and validate them
    # (including rendering the template) before writing anything, so a bad input
    # cannot leave a half-built grid.
    runs_to_write = []
    checker = _InputStager(out_prefix, dry_run=True)
    for i, (sim_vars, tmpl_vars) in enumerate(all_combos, start=1):
        all_vars = {**sim_vars, **tmpl_vars}
        run_name = _make_run_name(run_name_template, all_vars, i)

        # Base config with known path keys resolved to absolute (relative to
        # base_cfg_dir), then simulate overrides (relative to grid_yaml_dir).
        run_cfg = _resolve_paths(base_cfg, base_cfg_dir)
        run_cfg.update(_resolve_paths(sim_vars, grid_yaml_dir))
        source_dirs = {k: grid_yaml_dir if k in sim_vars else base_cfg_dir
                       for k in run_cfg}
        _check_no_outside_paths(run_cfg, source_dirs)
        _stage_config_files(copy.deepcopy(run_cfg), checker)
        if jinja_template is not None:
            _render(jinja_template, _stage_template_vars(
                tmpl_vars, grid_yaml_dir, checker), run_name)

        runs_to_write.append((run_name, run_cfg, sim_vars, tmpl_vars))

    os.makedirs(out_prefix, exist_ok=True)
    stager = _InputStager(out_prefix)

    all_runs = []

    for run_name, run_cfg, sim_vars, tmpl_vars in runs_to_write:
        subdir = os.path.abspath(os.path.join(out_prefix, run_name))
        os.makedirs(subdir, exist_ok=True)

        # Copy input files into <out_prefix>/inputs/; the config refers to them
        # as ../inputs/<name>.
        _stage_config_files(run_cfg, stager)
        cfg_path = os.path.join(subdir, _SIM_CONFIG_FILENAME)
        with open(cfg_path, "w") as fh:
            yaml.dump(run_cfg, fh, default_flow_style=False, sort_keys=False)

        # Render Jinja2 template with template-section variables only.
        if jinja_template is not None:
            rendered = _render(jinja_template, _stage_template_vars(
                tmpl_vars, grid_yaml_dir, stager), run_name)
            out_filename = os.path.basename(output_file)
            with open(os.path.join(subdir, out_filename), "w") as fh:
                fh.write(rendered)

        combo_record = {"simulate": sim_vars, "template": tmpl_vars}
        with open(os.path.join(subdir, "combo.json"), "w") as fh:
            json.dump(combo_record, fh, indent=2)
            fh.write("\n")

        all_runs.append({"run": run_name, "simulate": sim_vars, "template": tmpl_vars})
        print(f"  {subdir}", flush=True)

    summary = {
        "grid_yaml": grid_yaml,
        "out_prefix": os.path.abspath(out_prefix),
        "runs": all_runs,
    }
    summary_path = os.path.join(out_prefix, "grid_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    n_ok = len(all_runs)
    print(
        f"\n{n_ok} run{'s' if n_ok != 1 else ''} created under {out_prefix}/",
        flush=True,
    )

    return all_runs


def main():
    generalized_main(
        setup_sim_grid,
        manual_arg_types={"grid_yaml": str, "out_prefix": str},
    )


if __name__ == "__main__":
    main()
