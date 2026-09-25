"""
tfs-setup-grid — set up a directory grid for model runs.

Creates one subdirectory per combination of model-component and template
variables, calls ``tfs-configure-model`` in each, and renders a Jinja2
template with the per-run template variables.

GRID YAML FORMAT
----------------

    run_name: "{{ condition_growth }}__{{ growth_transition }}__{{ seed }}"
    output_file: run.srun   # Jinja2 template filename; looked up next to this YAML

    configure_model:         # variables forwarded to tfs-configure-model
      - name: data
        variants:
          - binding_df: ../data/binding.csv
            growth_df:  ../data/growth.csv
      - name: condition_growth
        auto: condition_growth   # enumerate all registered components for this axis
      - name: growth_transition
        auto: growth_transition
      - name: theta_epistasis   # co-varying: variants always stay paired
        variants:
          - theta_model: hill_mut
            epistasis: true
          - theta_model: hill_geno
            epistasis: false

    template:                # variables injected into the Jinja2 template only
      - name: seed
        variants:
          - seed: 0
          - seed: 42

NOTES
-----
- The Cartesian product is taken across **all** blocks (configure_model + template).
- configure_model variables are NOT injected into the template; template variables
  are NOT forwarded to tfs-configure-model.  To share a variable, list it in both.
- ``run_name`` and the run index prefix may reference variables from either section.
- Use the ``basename`` Jinja2 filter to strip path info from filenames in run_name:
      run_name: "{{ binding_df | basename }}__{{ condition_growth }}"
- Incompatible component combinations are skipped automatically and logged in
  ``grid_summary.json``.

INPUT FILES
-----------
The grid directory is self-contained and can be moved as a unit (on and off a
cluster, onto another partition). Every input file a run needs is copied into
``<out_prefix>/inputs/`` once, and each run's config and rendered template
refer to it as ``../inputs/<name>``. Nothing in a run points outside the grid.

- The configure_model file arguments are the keys in ``_PATH_KEYS``
  (``binding_df``, ``growth_df``, ``presplit_df``, ``base_growth_df``,
  ``thermo_data``, ``library_config``). A relative value resolves against the
  grid YAML's directory. tfs-configure-model reads the original file; the
  written ``tfs_configure_config.yaml`` then names the copy wherever it
  recorded that file (``data.*``, ``components.thermo_data``,
  ``components.presplit_df``/``base_growth_df``, ``library.source``).
- The priors, guesses and library-composition CSVs are per-run outputs written
  next to the config (``priors_file``/``guesses_file``/``library_file``, read
  relative to the config) and are left alone.
- Data paths in the config are read relative to the working directory, so
  each run is launched from its own directory (as the templates do).
- A ``template`` variable that names an existing file (relative to the grid
  YAML, or absolute) is copied the same way.
- Two different files with the same name are kept apart by a numeric suffix
  (``growth_2.csv``); the same file used by many runs is copied once.
- Setup fails rather than write a run that depends on a location outside the
  grid. Before anything is written: a missing input file, an input that is a
  directory, a configure_model value outside ``_PATH_KEYS`` that names an
  existing file, or a template that does not render. After tfs-configure-model
  runs: any other path in the written config that names an existing file
  outside the grid (add its argument to ``_PATH_KEYS``).
"""

import itertools
import json
import os
import shutil

import jinja2
import yaml

from tfscreen.util.cli import generalized_main
from tfscreen.util.grid_utils import (
    InputStager as _InputStager,
    check_no_outside_paths as _check_no_outside_paths,
    make_jinja_env as _make_jinja_env,
    make_run_name as _make_run_name,
    render_run_template as _render_run_template,
    stage_template_vars as _stage_template_vars,
)
from tfscreen.tfmodel.generative.registry import model_registry
from tfscreen.tfmodel.scripts.configure_model_cli import (
    configure_model,
)

# configure_model parameter names that take a "_model" suffix but are stored in
# the YAML / registry without it (e.g. "condition_growth" → "condition_growth_model").
_COMPONENT_AXES = frozenset({
    "condition_growth",
    "growth_transition",
    "ln_cfu0",
    "dk_geno",
    "activity",
    "theta",
    "transformation",
    "theta_rescale",
    "theta_growth_noise",
    "theta_binding_noise",
    "growth_noise",
})

# configure_model arguments that are input file paths. Each is copied into
# <out_prefix>/inputs/ and the written config names the copy. Add any new
# file-valued configure_model argument here; setup fails on a configure_model
# value outside this set that names an existing file.
_PATH_KEYS = frozenset({
    "binding_df", "growth_df", "presplit_df", "base_growth_df", "thermo_data",
    "library_config",
})

# Top-level keys of the written config naming files configure_model writes
# into the run directory (read relative to the config): per-run outputs, not
# inputs to copy.
_RUN_OUTPUT_KEYS = (("priors_file",), ("guesses_file",), ("library_file",))

# Fixed output prefix used inside every per-combination run.
_CONFIGURE_OUT_PREFIX = "tfs_configure"

_TOOL = "tfs-setup-grid"


# ---------------------------------------------------------------------------
# Block expansion
# ---------------------------------------------------------------------------

def _expand_block(block):
    """Return a list of variant dicts for one block entry.

    Supports two forms:
      - ``auto: <axis>`` — enumerate all registered components for that axis;
        the variable name is the block's ``name`` (defaults to the axis name).
      - ``variants: [...]`` — explicit list of variant dicts.
    """
    if "auto" in block:
        axis = block["auto"]
        if axis not in model_registry:
            raise ValueError(
                f"Block '{block.get('name', '?')}': unknown registry axis '{axis}'. "
                f"Available axes: {sorted(model_registry.keys())}"
            )
        var_name = block.get("name", axis)
        return [{var_name: v} for v in sorted(model_registry[axis].keys())]

    if "variants" in block:
        return list(block["variants"])

    raise ValueError(
        f"Block '{block.get('name', '?')}' must have either 'auto' or 'variants'."
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_cm_paths(cm_vars, base_dir):
    """Return a copy of cm_vars with _PATH_KEYS resolved to absolute paths."""
    out = dict(cm_vars)
    for key in _PATH_KEYS:
        if key in out and out[key] and not os.path.isabs(out[key]):
            out[key] = os.path.normpath(os.path.join(base_dir, out[key]))
    return out


def _input_paths(resolved_cm):
    """Return ``{key: path}`` for the file arguments present in resolved_cm."""
    return {k: resolved_cm[k] for k in sorted(_PATH_KEYS)
            if isinstance(resolved_cm.get(k), str) and resolved_cm[k]}


def _check_cm_vars(cm_vars, grid_yaml_dir, stager):
    """Validate one combination's configure_model variables (writes nothing).

    Fails if a value outside ``_PATH_KEYS`` names an existing file, or if a
    ``_PATH_KEYS`` value is missing or a directory.
    """
    _check_no_outside_paths(
        cm_vars, [(k,) for k in _PATH_KEYS], grid_yaml_dir,
        hint=(f"If tfs-configure-model reads this file, add the argument to "
              f"_PATH_KEYS in {__name__}; otherwise change the value."),
    )
    for key, path in _input_paths(_resolve_cm_paths(cm_vars, grid_yaml_dir)).items():
        stager.stage(path, f"configure_model variable '{key}'")


def _stage_written_config(cfg, path_map, subdir):
    """Point a written config at the staged input copies (returns a new dict).

    Every string value equal to an input path in ``path_map`` (original
    absolute path → ``../inputs/<name>``) is replaced by its copy, wherever
    tfs-configure-model recorded it. Any other value that names an existing
    file outside the run directory is an error: the run would not be movable.
    """
    staged_keys = []

    def rewrite(node, key_path):
        if isinstance(node, dict):
            return {k: rewrite(v, key_path + (k,)) for k, v in node.items()}
        if isinstance(node, list):
            return [rewrite(v, key_path + (i,)) for i, v in enumerate(node)]
        if isinstance(node, str) and node in path_map:
            staged_keys.append(key_path)
            return path_map[node]
        return node

    out = rewrite(cfg, ())
    _check_no_outside_paths(
        out, staged_keys + list(_RUN_OUTPUT_KEYS), subdir,
        hint=(f"tfs-configure-model recorded a path {_TOOL} does not know how "
              f"to copy. If it is an input file, add the configure_model "
              f"argument that supplies it to _PATH_KEYS in {__name__}."),
    )
    return out


# ---------------------------------------------------------------------------
# configure_model kwargs preparation
# ---------------------------------------------------------------------------

def _cm_kwargs(cm_vars):
    """Translate combo variable names to configure_model parameter names.

    Registry-axis keys (e.g. ``condition_growth``) gain a ``_model`` suffix
    to match configure_model's parameter names (e.g. ``condition_growth_model``).
    All other keys are forwarded unchanged.
    """
    kwargs = {}
    for k, v in cm_vars.items():
        if k in _COMPONENT_AXES:
            kwargs[f"{k}_model"] = v
        else:
            kwargs[k] = v
    return kwargs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_grid(grid_yaml, out_prefix="grid"):
    """
    Set up a directory grid for model runs.

    Reads a grid YAML file, expands all configure_model and template blocks
    into their Cartesian product, and for each combination:

    1. Creates a subdirectory under *out_prefix*.
    2. Calls tfs-configure-model (via its Python API) to generate
       ``tfs_configure_config.yaml``, ``tfs_configure_priors.csv``, and
       ``tfs_configure_guesses.csv`` inside that subdirectory.  Incompatible
       combinations are skipped and logged.
    3. Renders the Jinja2 template (``output_file``) with the template
       variables and writes the result to the subdirectory.
    4. Writes ``combo.json`` recording the variable assignments for that run.

    A ``grid_summary.json`` is written to *out_prefix* listing all created
    runs and any skipped combinations. Input files named by the
    configure_model or template variables are copied into
    ``<out_prefix>/inputs/`` and referenced as ``../inputs/<name>``, so the
    *out_prefix* directory can be moved as a unit. Every combination's inputs
    and template are validated before anything is written.

    Parameters
    ----------
    grid_yaml : str
        Path to the grid YAML file.
    out_prefix : str, optional
        Root directory for the grid.  Created if it does not exist.
        Default ``"grid"``.

    Returns
    -------
    list of dict
        One dict per successfully created run, each with keys ``run``,
        ``configure_model``, and ``template``.
    """
    grid_yaml = os.path.abspath(grid_yaml)
    if not os.path.exists(grid_yaml):
        raise FileNotFoundError(f"Grid YAML not found: {grid_yaml}")

    grid_yaml_dir = os.path.dirname(grid_yaml)

    with open(grid_yaml) as fh:
        grid = yaml.safe_load(fh)

    run_name_template = grid.get("run_name")
    output_file = grid.get("output_file")

    cm_block_specs = grid.get("configure_model") or []
    tmpl_block_specs = grid.get("template") or []

    if not cm_block_specs and not tmpl_block_specs:
        raise ValueError(
            "Grid YAML must have at least one of 'configure_model' or 'template' blocks."
        )

    # Expand each block into its list of variant dicts.
    cm_variant_lists = [_expand_block(b) for b in cm_block_specs]
    tmpl_variant_lists = [_expand_block(b) for b in tmpl_block_specs]

    # Build Cartesian product; track which tuple positions are cm vs template.
    n_cm = len(cm_variant_lists)
    all_combos = []
    for combo_tuple in itertools.product(*(cm_variant_lists + tmpl_variant_lists)):
        cm_vars = {}
        tmpl_vars = {}
        for i, variant in enumerate(combo_tuple):
            if i < n_cm:
                cm_vars.update(variant)
            else:
                tmpl_vars.update(variant)
        all_combos.append((cm_vars, tmpl_vars))

    # Load and compile Jinja2 template if an output_file is specified.
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

    # Validate every combination's input files and render its template before
    # writing anything, so a bad input cannot leave a half-built grid.
    # (Component incompatibilities only surface when configure_model runs; those
    # combinations are skipped below, as before.)
    runs_to_write = []
    checker = _InputStager(out_prefix, _TOOL, dry_run=True)
    for i, (cm_vars, tmpl_vars) in enumerate(all_combos, start=1):
        all_vars = {**cm_vars, **tmpl_vars}
        run_name = _make_run_name(run_name_template, all_vars, i)
        _check_cm_vars(cm_vars, grid_yaml_dir, checker)
        if jinja_template is not None:
            _render_run_template(jinja_template, _stage_template_vars(
                tmpl_vars, grid_yaml_dir, checker), run_name)
        runs_to_write.append((run_name, cm_vars, tmpl_vars))

    os.makedirs(out_prefix, exist_ok=True)
    stager = _InputStager(out_prefix, _TOOL)

    all_runs = []
    skipped = []

    for run_name, cm_vars, tmpl_vars in runs_to_write:
        all_vars = {**cm_vars, **tmpl_vars}
        subdir = os.path.abspath(os.path.join(out_prefix, run_name))

        # configure_model reads the original input files (relative paths
        # resolved against the grid YAML's directory).
        resolved_cm = _resolve_cm_paths(cm_vars, grid_yaml_dir)
        cm_kw = _cm_kwargs(resolved_cm)
        cm_out_prefix = os.path.join(subdir, _CONFIGURE_OUT_PREFIX)

        os.makedirs(subdir, exist_ok=True)

        try:
            configure_model(out_prefix=cm_out_prefix, **cm_kw)
        except Exception as exc:
            reason = str(exc)
            skipped.append({"run": run_name, "reason": reason, "combo": all_vars})
            shutil.rmtree(subdir, ignore_errors=True)
            print(f"  SKIP {run_name}: {reason}", flush=True)
            continue

        # Copy the inputs into <out_prefix>/inputs/ and point the written
        # config at the copies (../inputs/<name>).
        path_map = {
            path: stager.stage(path, f"configure_model variable '{key}'")
            for key, path in _input_paths(resolved_cm).items()
        }
        cfg_path = f"{cm_out_prefix}_config.yaml"
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        try:
            cfg = _stage_written_config(cfg, path_map, subdir)
        except ValueError:
            shutil.rmtree(subdir, ignore_errors=True)
            raise
        with open(cfg_path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        # Render Jinja2 template with template-section variables only.
        if jinja_template is not None:
            rendered = _render_run_template(jinja_template, _stage_template_vars(
                tmpl_vars, grid_yaml_dir, stager), run_name)
            out_filename = os.path.basename(output_file)
            with open(os.path.join(subdir, out_filename), "w") as fh:
                fh.write(rendered)

        # Write combo.json recording the original (pre-resolution) variable values.
        combo_record = {"configure_model": cm_vars, "template": tmpl_vars}
        with open(os.path.join(subdir, "combo.json"), "w") as fh:
            json.dump(combo_record, fh, indent=2)
            fh.write("\n")

        all_runs.append({"run": run_name, "configure_model": cm_vars, "template": tmpl_vars})
        print(f"  {subdir}", flush=True)

    # Write grid_summary.json.
    summary = {
        "grid_yaml": grid_yaml,
        "out_prefix": os.path.abspath(out_prefix),
        "runs": all_runs,
        "skipped": skipped,
    }
    summary_path = os.path.join(out_prefix, "grid_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    n_ok = len(all_runs)
    n_skip = len(skipped)
    print(
        f"\n{n_ok} run{'s' if n_ok != 1 else ''} created under {out_prefix}/",
        flush=True,
    )
    if n_skip:
        print(
            f"{n_skip} combination{'s' if n_skip != 1 else ''} skipped "
            f"(see grid_summary.json)",
            flush=True,
        )

    return all_runs


def main():
    generalized_main(
        setup_grid,
        manual_arg_types={"grid_yaml": str, "out_prefix": str},
    )


if __name__ == "__main__":
    main()
