#!/usr/bin/env python3
"""
create_grid.py — Generate a grid of HPC run directories from a Jinja2 template
and a YAML combo spec.

Usage:
    python create_grid.py template.srun grid.yaml output_dir/

TEMPLATE FORMAT
    The template file uses Jinja2 syntax. Variable placeholders look like:
        {{ BINDING_CSV_FILE }}

    Note: Jinja2's {{ }} delimiters do not conflict with bash ${ } syntax, so
    both can coexist in the same .srun file.

    Boolean flags that may be present or absent should be expressed as variables
    whose value is either the flag string or an empty string, e.g.:
        {{ EPISTASIS }}      # resolves to "--epistasis" or ""
    Leave these unquoted so an empty value produces no argument.

GRID YAML FORMAT
    run_name: "{{ MODEL_TO_RUN }}_{{ EPISTASIS }}"   # optional Jinja2 name template
    output_file: run.srun                             # optional; default: template filename

    blocks:
      - name: theta_model          # one value chosen per run
        variants:
          - MODEL_TO_RUN: "hill_mut"
          - MODEL_TO_RUN: "hill"

      - name: epistasis            # another independent block
        variants:
          - EPISTASIS: "--epistasis"
          - EPISTASIS: ""

      - name: coupled_pair         # multiple keys in one variant move together
        variants:
          - MODEL_TO_RUN: "hill_mut"
            EPISTASIS: "--epistasis"
          - MODEL_TO_RUN: "hill"
            EPISTASIS: ""

    The Cartesian product is taken across blocks; values within a single variant
    dict always move together (they are never split).

OUTPUT
    <output_dir>/
      run_0001_hill_mut_epistasis/
        run.srun      — rendered template with variables substituted
        combo.json    — machine-readable dict of this run's variable assignments
      run_0002_hill_mut/
        ...
      grid_summary.json  — list of all runs with their combos
"""

import argparse
import itertools
import json
import os
import re  # used by _sanitize
import sys
from pathlib import Path

import jinja2
import yaml


# ---------------------------------------------------------------------------
# Grid loading and combo generation
# ---------------------------------------------------------------------------

def load_grid(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_combos(grid):
    """Return list of variable dicts (one per run) as the Cartesian product across blocks."""
    blocks = grid.get("blocks", [])
    if not blocks:
        return [{}]
    variant_lists = [block["variants"] for block in blocks]
    combos = []
    for combo_tuple in itertools.product(*variant_lists):
        merged = {}
        for variant in combo_tuple:
            merged.update(variant)
        combos.append(merged)
    return combos


# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------

def _sanitize(s):
    """Make a string safe for use as a directory name component."""
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]", "_", str(s))).strip("_")


def make_run_name(name_template, combo, index):
    """Return a filesystem-safe directory name for this run."""
    prefix = f"run_{index:04d}"
    if name_template:
        try:
            raw = jinja2.Environment().from_string(name_template).render(**combo)
        except jinja2.TemplateError as e:
            sys.exit(f"ERROR: run_name template failed: {e}")
        suffix = _sanitize(raw)
    else:
        suffix = "_".join(_sanitize(v) for v in combo.values() if str(v).strip())

    return f"{prefix}_{suffix}" if suffix else prefix


# ---------------------------------------------------------------------------
# Relative path adjustment
# ---------------------------------------------------------------------------

def adjust_combo_paths(combo, template_dir, run_dir):
    """
    Rewrite relative paths in combo values whose key ends with '_FILE'.

    The path is taken as the last whitespace-delimited token of the value, so
    both bare paths and flag+path strings are handled:
        BINDING_CSV_FILE: "data/binding.csv"
        DATA_INPUT_FILE:  "--struct_ensemble_path data/struct_ensemble.h5"

    Absolute paths and empty values are left unchanged.
    """
    adjusted = {}
    for key, value in combo.items():
        if key.endswith("_FILE") and value:
            parts = str(value).split()
            path = parts[-1]
            if not os.path.isabs(path):
                abs_path = os.path.normpath(os.path.join(template_dir, path))
                parts[-1] = os.path.relpath(abs_path, run_dir)
                value = " ".join(parts)
        adjusted[key] = value
    return adjusted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate a grid of HPC run directories from a Jinja2 template and YAML combo spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("template", help="Jinja2 template file (e.g. run.srun)")
    ap.add_argument("grid_yaml", help="YAML file defining the combo grid")
    ap.add_argument("output_dir", help="Root directory for generated run subdirectories")
    args = ap.parse_args()

    template_path = Path(args.template).resolve()
    grid_path = Path(args.grid_yaml).resolve()
    output_root = Path(args.output_dir).resolve()

    if not template_path.exists():
        sys.exit(f"ERROR: template file not found: {template_path}")
    if not grid_path.exists():
        sys.exit(f"ERROR: grid YAML not found: {grid_path}")

    template_dir = str(template_path.parent)
    template_text = template_path.read_text()

    grid = load_grid(grid_path)
    run_name_template = grid.get("run_name", None)
    output_filename = grid.get("output_file", template_path.name)

    combos = build_combos(grid)
    if not combos:
        sys.exit("ERROR: grid produced no combinations — check your blocks/variants")

    jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    try:
        jinja_template = jinja_env.from_string(template_text)
    except jinja2.TemplateSyntaxError as e:
        sys.exit(f"ERROR: template syntax error: {e}")

    output_root.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for i, combo in enumerate(combos, start=1):
        run_name = make_run_name(run_name_template, combo, i)
        run_dir = output_root / run_name
        run_dir.mkdir(exist_ok=True)

        combo = adjust_combo_paths(combo, template_dir, str(run_dir))

        try:
            rendered = jinja_template.render(**combo)
        except jinja2.UndefinedError as e:
            sys.exit(f"ERROR: undefined variable for combo {combo}: {e}")

        (run_dir / output_filename).write_text(rendered)
        (run_dir / "combo.json").write_text(json.dumps(combo, indent=2) + "\n")

        all_runs.append({"run": run_name, "combo": combo})
        print(f"  {run_dir}")

    summary = {
        "template": str(template_path),
        "grid": str(grid_path),
        "runs": all_runs,
    }
    (output_root / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{len(combos)} run{'s' if len(combos) != 1 else ''} → {output_root}")


if __name__ == "__main__":
    main()
