"""
CLI for grading per-genotype theta stability across N estimate runs.
"""

import pandas as pd

from tfscreen.analysis.compare_theta import (
    compare_theta as _compare_theta,
    aggregate_theta,
    stability_crosstabs,
)
from tfscreen.util.cli import generalized_main, read_lines


def _format_crosstabs(crosstabs):
    """Render the crosstab dict as a human-readable text block."""
    blocks = []
    for name, table in crosstabs.items():
        blocks.append(f"# {name}")
        blocks.append(table.to_string())
        blocks.append("")
    return "\n".join(blocks)


def compare_theta(estimates_file,
                  out_prefix="tfs_compare_theta",
                  reference=None,
                  min_coverage=0.5,
                  overdispersion_threshold=2.0,
                  write_aggregate=False):
    """
    Grade per-genotype theta stability across N independent estimate runs.

    Reads N theta-estimate CSVs (each with the standard quantile columns:
    genotype, titrant_conc [, titrant_name], q0.001 ... q0.5 ... q0.999) and
    scores every genotype on two axes: reproducibility (run-to-run spread of the
    q0.5 point estimate, in native theta units -- the graded tier) and
    self-consistency (whether that spread is explained by each run's reported
    uncertainty -- a flag). Writes a per-genotype table to {out_prefix}.csv and
    the interpretation crosstabs to {out_prefix}_crosstabs.txt.

    Parameters
    ----------
    estimates_file : str
        Path to a text file listing the estimate CSV paths, one per line
        ('#' comments allowed). In the default (mean) mode at least two are
        required.
    out_prefix : str, optional
        Prefix for the output files. Default 'tfs_compare_theta'.
    reference : str or None, optional
        Path to a reference-run CSV. If given, switches to reference mode: each
        estimate run is scored by its deviation from this run (e.g. k-fold
        dropouts vs. a full-data fit). If omitted, runs are compared to their
        symmetric cross-run mean.
    min_coverage : float, optional
        Minimum fraction of estimate runs a genotype must appear in to be
        graded; below this it is tiered 'low_coverage'. Default 0.5.
    overdispersion_threshold : float, optional
        Overdispersion above this sets the 'overdispersed' flag and the
        'overconfident' crosstab column. Default 2.0.
    write_aggregate : bool, optional
        If set, also write {out_prefix}_aggregate.csv: a long-form aggregate
        theta-vs-condition table combining the N estimate runs as an equal-weight
        mixture of their per-run posteriors (quantiles reconstructed and mixed).
        The aggregate error folds in both per-run posterior width and run-to-run
        spread and does not shrink with N. Only the N estimate runs are mixed --
        a reference run (if any) is never included. Default False.
    """
    paths = read_lines(estimates_file)
    if not paths:
        raise ValueError(f"No estimate paths found in '{estimates_file}'.")

    print(f"Reading {len(paths)} estimate file(s)...", flush=True)
    estimate_dfs = [pd.read_csv(p) for p in paths]

    reference_df = None
    if reference is not None:
        print(f"Reading reference {reference}...", flush=True)
        reference_df = pd.read_csv(reference)

    result = _compare_theta(
        estimate_dfs,
        reference_df=reference_df,
        min_coverage=min_coverage,
        overdispersion_threshold=overdispersion_threshold,
    )

    out_file = f"{out_prefix}.csv"
    result.to_csv(out_file, index=False)
    print(f"Wrote {len(result)} genotype rows to {out_file}", flush=True)

    # Tier breakdown to stdout.
    counts = result["tier"].value_counts()
    print("Tier breakdown:", flush=True)
    for tier in ["A", "B", "C", "D", "low_coverage"]:
        if tier in counts.index:
            print(f"  {tier:>12}: {counts[tier]}", flush=True)

    crosstabs = stability_crosstabs(
        result, overdispersion_threshold=overdispersion_threshold
    )
    crosstab_file = f"{out_prefix}_crosstabs.txt"
    with open(crosstab_file, "w") as fh:
        fh.write(_format_crosstabs(crosstabs))
    print(f"Wrote interpretation crosstabs to {crosstab_file}", flush=True)

    if write_aggregate:
        print("Building aggregate theta-vs-condition table...", flush=True)
        aggregate = aggregate_theta(estimate_dfs)
        aggregate_file = f"{out_prefix}_aggregate.csv"
        aggregate.to_csv(aggregate_file, index=False)
        print(f"Wrote {len(aggregate)} aggregate rows to {aggregate_file}",
              flush=True)


def main():
    generalized_main(compare_theta,
                     manual_arg_types={"reference": str})


if __name__ == "__main__":
    main()
