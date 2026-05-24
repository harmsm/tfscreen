"""
tfs-summarize-grid — summarize the results of a model grid created by tfs-setup-grid.

Scans every immediate subdirectory of a grid directory for ``combo.json``.
For each completed run it reads the component settings and any output files
present, then writes a flat summary CSV with one row per run.

A run is considered complete when ``tfs_configure_config.yaml`` exists in the
subdirectory (i.e. tfs-configure-model finished successfully during setup).
"""

import json
import os

import pandas as pd

from tfscreen.util.cli import generalized_main


def summarize_grid(grid_dir, out_prefix=None):
    """
    Summarize a model grid created by tfs-setup-grid.

    Scans every immediate subdirectory of *grid_dir* for ``combo.json`` and
    collects the per-run variable assignments into a summary CSV.

    Parameters
    ----------
    grid_dir : str
        Path to the grid root directory created by ``tfs-setup-grid``.
    out_prefix : str or None, optional
        Prefix for the output CSV; ``{out_prefix}.csv`` is written.
        Defaults to ``{grid_dir}/grid_summary``.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per run found.  An empty DataFrame is
        returned when no runs are found.
    """
    grid_dir = os.path.abspath(grid_dir)
    if not os.path.isdir(grid_dir):
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")

    if out_prefix is None:
        out_prefix = os.path.join(grid_dir, "grid_summary")

    entries = sorted(
        d for d in os.listdir(grid_dir)
        if os.path.isdir(os.path.join(grid_dir, d))
    )

    rows = []
    for name in entries:
        subdir = os.path.join(grid_dir, name)
        combo_path = os.path.join(subdir, "combo.json")
        if not os.path.exists(combo_path):
            continue

        with open(combo_path) as fh:
            combo = json.load(fh)

        row = {"run": name}

        # Flatten configure_model and template variable dicts into the row.
        for section in ("configure_model", "template"):
            for k, v in (combo.get(section) or {}).items():
                row[k] = v

        # Note whether tfs-configure-model output is present.
        row["configure_complete"] = os.path.exists(
            os.path.join(subdir, "tfs_configure_config.yaml")
        )

        rows.append(row)

    if not rows:
        print(f"No runs found in {grid_dir}/", flush=True)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    n_complete = df["configure_complete"].sum() if "configure_complete" in df.columns else 0
    print(
        f"Summary written to {csv_path} "
        f"({len(df)} run(s), {n_complete} complete)",
        flush=True,
    )

    return df


def main():
    generalized_main(
        summarize_grid,
        manual_arg_types={"grid_dir": str, "out_prefix": str},
    )


if __name__ == "__main__":
    main()
