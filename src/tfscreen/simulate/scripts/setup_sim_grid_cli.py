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
- Input files named in the config (``thermo_data``, ``calibration_file``,
  a ``binding_data.*_binding.choose_by`` params file, ``empirical.phenotype_model``;
  see ``tfscreen.simulate.config_paths``) are **copied into every run
  directory**, and the written config names the local copy.  Runs are thus
  self-contained: they work from any working directory and after being moved
  or synced to another machine.  Relative paths are resolved first -- against
  the grid YAML's directory for ``simulate`` overrides, against the base
  config's directory for base-config values.  A missing file fails at setup.
"""

import itertools
import json
import os
import shutil

import jinja2
import yaml

from tfscreen.simulate.config_paths import (
    existing_input_file,
    iter_path_entries,
    resolve_config_paths,
)
from tfscreen.util.cli import generalized_main
from tfscreen.util.grid_utils import (
    make_jinja_env as _make_jinja_env,
    make_run_name as _make_run_name,
    relativize_config_paths as _relativize_config_paths,
    relativize_template_vars as _relativize_template_vars,
)


# Fixed filename for the per-run config written into each subdirectory.
_SIM_CONFIG_FILENAME = "tfs_sim_config.yaml"


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
    """
    Return a deep copy of vars_dict with its file paths resolved to absolute.

    The file-valued keys are defined in ``tfscreen.simulate.config_paths``
    (e.g. ``thermo_data`` and ``binding_data.spiked_binding.choose_by`` when
    it names a params file rather than a builtin keyword).  The copy is deep
    so resolving nested values never mutates the shared base config.
    """
    return resolve_config_paths(vars_dict, base_dir)


def _localize_inputs(run_cfg, subdir):
    """
    Copy every input file named in ``run_cfg`` into ``subdir``.

    Rewrites each file-valued entry to the copied file's basename, so the run
    directory is self-contained: it works from any working directory and
    after being moved or synced elsewhere.  ``run_cfg`` paths must already be
    absolute (``_resolve_paths``).  Fails fast when a file does not exist or
    two different files share a basename.
    """
    copied = {}
    for container, key, key_path in iter_path_entries(run_cfg):
        source = existing_input_file(key_path, container[key])
        if source is None:
            raise FileNotFoundError(
                f"Input file for '{key_path}' not found: {container[key]}"
            )
        source = os.path.abspath(source)
        name = os.path.basename(source)
        if name in copied and copied[name] != source:
            raise ValueError(
                f"Two different input files share the name '{name}' "
                f"({copied[name]} and {source}); rename one."
            )
        if name not in copied:
            shutil.copy2(source, os.path.join(subdir, name))
            copied[name] = source
        container[key] = name


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

    os.makedirs(out_prefix, exist_ok=True)

    all_runs = []

    for i, (sim_vars, tmpl_vars) in enumerate(all_combos, start=1):
        all_vars = {**sim_vars, **tmpl_vars}
        run_name = _make_run_name(run_name_template, all_vars, i)
        subdir = os.path.abspath(os.path.join(out_prefix, run_name))
        os.makedirs(subdir, exist_ok=True)

        # Build per-run config: base config with known path keys resolved to
        # absolute (relative to base_cfg_dir), then apply simulate overrides
        # (path keys resolved relative to grid_yaml_dir).
        run_cfg = _resolve_paths(base_cfg, base_cfg_dir)
        run_cfg.update(_resolve_paths(sim_vars, grid_yaml_dir))

        # Copy input files (binding params, thermo data, ...) into the run
        # directory and point the config at the local copies.
        _localize_inputs(run_cfg, subdir)

        # Write config; _relativize_config_paths then rewrites any remaining
        # absolute paths that exist on disk to be relative to subdir.
        cfg_path = os.path.join(subdir, _SIM_CONFIG_FILENAME)
        with open(cfg_path, "w") as fh:
            yaml.dump(run_cfg, fh, default_flow_style=False, sort_keys=False)
        _relativize_config_paths(cfg_path, subdir)

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
