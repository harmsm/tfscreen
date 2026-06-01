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
          - theta_model: hill
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
- Relative paths in configure_model blocks (binding_df, growth_df,
  thermo_data) are resolved relative to the grid YAML's directory, then
  re-expressed relative to each subdirectory in the written config file.
"""

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
    relativize_config_paths as _relativize_config_paths,
    relativize_template_vars as _relativize_template_vars,
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
})

# configure_model arguments that are file paths and need abs→rel rewriting.
_PATH_KEYS = frozenset({"binding_df", "growth_df", "thermo_data"})

# Fixed output prefix used inside every per-combination run.
_CONFIGURE_OUT_PREFIX = "tfs_configure"


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
    runs and any skipped combinations.

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

    os.makedirs(out_prefix, exist_ok=True)

    all_runs = []
    skipped = []

    for i, (cm_vars, tmpl_vars) in enumerate(all_combos, start=1):
        all_vars = {**cm_vars, **tmpl_vars}
        run_name = _make_run_name(run_name_template, all_vars, i)
        subdir = os.path.abspath(os.path.join(out_prefix, run_name))

        # Resolve relative paths in cm_vars to absolute before calling configure_model.
        resolved_cm = _resolve_cm_paths(cm_vars, grid_yaml_dir)
        cm_kw = _cm_kwargs(resolved_cm)
        cm_out_prefix = os.path.join(subdir, _CONFIGURE_OUT_PREFIX)

        os.makedirs(subdir, exist_ok=True)

        try:
            configure_model(out_prefix=cm_out_prefix, **cm_kw)
            _relativize_config_paths(f"{cm_out_prefix}_config.yaml", subdir)
        except Exception as exc:
            reason = str(exc)
            skipped.append({"run": run_name, "reason": reason, "combo": all_vars})
            shutil.rmtree(subdir, ignore_errors=True)
            print(f"  SKIP {run_name}: {reason}", flush=True)
            continue

        # Render Jinja2 template with template-section variables only.
        if jinja_template is not None:
            rendered_tmpl_vars = _relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir)
            try:
                rendered = jinja_template.render(**rendered_tmpl_vars)
            except jinja2.UndefinedError as exc:
                raise ValueError(
                    f"Undefined template variable for run '{run_name}': {exc}"
                ) from exc
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
