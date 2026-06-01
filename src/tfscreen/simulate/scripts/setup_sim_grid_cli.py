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
- Relative paths in ``simulate`` blocks (e.g. ``thermo_data``) are resolved
  relative to the grid YAML's directory, then re-expressed relative to each
  subdirectory in the written config.
- Relative paths already in the base config are resolved relative to the base
  config's location, then re-expressed relative to each subdirectory.
"""

import itertools
import json
import os

import jinja2
import yaml

from tfscreen.util.cli import generalized_main
from tfscreen.util.grid_utils import (
    make_jinja_env as _make_jinja_env,
    make_run_name as _make_run_name,
    relativize_config_paths as _relativize_config_paths,
    relativize_template_vars as _relativize_template_vars,
)

# Top-level keys in the simulate config that hold file paths.
_SIM_PATH_KEYS = frozenset({"thermo_data", "calibration_file"})

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
    """Return a copy of vars_dict with _SIM_PATH_KEYS resolved to absolute paths."""
    out = dict(vars_dict)
    for key in _SIM_PATH_KEYS:
        if key in out and out[key] and not os.path.isabs(out[key]):
            out[key] = os.path.normpath(os.path.join(base_dir, out[key]))
    return out


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

        # Write config; _relativize_config_paths then rewrites any absolute
        # paths that exist on disk to be relative to subdir.
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
