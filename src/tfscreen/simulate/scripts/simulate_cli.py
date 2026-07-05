import os
import yaml
import numpy as np
import pandas as pd

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.simulate.binding_data import generate_binding_df
from tfscreen.simulate.presplit_data import generate_presplit_df
from tfscreen.simulate.base_growth_data import generate_base_growth_df
from tfscreen.process_raw import counts_to_lncfu
from tfscreen.util.cli.generalized_main import generalized_main


def run_simulation_from_config(
    config_file,
    output_dir,
    output_prefix="tfs_sim_",
    num_replicates=2,
    seed=None,
):
    """
    Simulate a TF selection experiment from a YAML configuration file.

    Runs library_prediction once to establish ground-truth phenotypes, then
    simulates num_replicates independent experimental replicates using
    selection_experiment. Writes library, parameters, genotype_theta (long-form:
    genotype/titrant_name/titrant_conc/theta), and analysis-ready growth CSV
    files. If the config contains a 'binding_data' block, also writes a
    simulated binding curve CSV (see
    tfscreen.simulate.binding_data.generate_binding_df). If it contains a
    'base_growth_data' block, also writes a simulated direct growth-rate
    calibration CSV (see
    tfscreen.simulate.base_growth_data.generate_base_growth_df). If it
    contains a 'presplit_data' block, also writes a simulated presplit CSV
    (see tfscreen.simulate.presplit_data.generate_presplit_df).

    Parameters
    ----------
    config_file : str
        Path to the YAML run configuration file.
    output_dir : str
        Directory to write output CSV files into (created if absent).
    output_prefix : str
        Prefix for all output filenames. Default 'tfs_sim_'.
    num_replicates : int
        Number of independent experimental replicates to simulate. Default 2.
    seed : int, optional
        Random seed. Overrides seed in the config file when provided.
    """
    cf = tfscreen.util.read_yaml(config_file)
    if seed is not None:
        cf["seed"] = seed

    os.makedirs(output_dir, exist_ok=True)

    def out_path(name):
        return os.path.join(output_dir, f"{output_prefix}{name}.csv")

    config_out = os.path.join(output_dir, f"{output_prefix}input-config.yaml")

    output_names = ["library", "parameters", "genotype_theta", "growth"]
    if "binding_data" in cf:
        output_names.append("binding")
    if "presplit_data" in cf:
        output_names.append("presplit")
    if "base_growth_data" in cf:
        output_names.append("base_growth")

    existing = [out_path(n) for n in output_names if os.path.exists(out_path(n))]
    if os.path.exists(config_out):
        existing.append(config_out)
    if existing:
        paths = ", ".join(existing)
        raise FileExistsError(
            f"Output files already exist: {paths}\n"
            f"Delete them or choose a different output_dir / output_prefix "
            f"before re-running."
        )

    base_seed = cf.get("seed", None)
    rng = np.random.default_rng(base_seed)

    # -------------------------------------------------------------------------
    # Ground-truth library and phenotypes (deterministic given the config)

    library_df, phenotype_df, genotype_theta_df, parameters_df, binding_theta_df = library_prediction(cf)

    # -------------------------------------------------------------------------
    # Simulate independent replicates

    all_sample_parts = []
    all_counts_parts = []
    sample_id_offset = 0

    for rep in range(1, num_replicates + 1):
        print(f"\n--- Replicate {rep} of {num_replicates} ---", flush=True)

        # Give each replicate a distinct (but reproducible) random seed so
        # that replicates differ even when a base seed is set.
        rep_cf = dict(cf)
        rep_cf["seed"] = (
            base_seed * num_replicates + rep if base_seed is not None else None
        )

        rep_phenotype_df = phenotype_df.copy()
        rep_phenotype_df["replicate"] = rep

        sample_df_rep, counts_df_rep = selection_experiment(
            rep_cf, library_df, rep_phenotype_df
        )

        # Shift sample IDs so they are globally unique across replicates
        max_id = int(sample_df_rep.index.max()) + 1
        sample_df_rep = sample_df_rep.copy()
        counts_df_rep = counts_df_rep.copy()
        sample_df_rep.index = sample_df_rep.index + sample_id_offset
        sample_df_rep["sample"] = sample_df_rep.index
        counts_df_rep["sample"] = counts_df_rep["sample"] + sample_id_offset

        all_sample_parts.append(sample_df_rep)
        all_counts_parts.append(counts_df_rep)
        sample_id_offset += max_id

    combined_sample_df = pd.concat(all_sample_parts)
    combined_counts_df = pd.concat(all_counts_parts, ignore_index=True)

    growth_df = counts_to_lncfu(combined_sample_df, combined_counts_df)

    # -------------------------------------------------------------------------
    # Write outputs

    library_df.to_csv(out_path("library"), index=False)
    parameters_df.to_csv(out_path("parameters"), index=False)
    genotype_theta_df.to_csv(out_path("genotype_theta"), index=False)
    growth_df.to_csv(out_path("growth"), index=False)
    print(f"\nWrote: {', '.join(out_path(n) for n in ['library', 'parameters', 'genotype_theta', 'growth'])}")

    if "binding_data" in cf:
        binding_df = generate_binding_df(cf["binding_data"], rng, binding_theta_df)
        binding_df.to_csv(out_path("binding"), index=False)
        print(f"Wrote: {out_path('binding')}")

    if "base_growth_data" in cf:
        base_growth_df = generate_base_growth_df(cf["base_growth_data"], parameters_df, rng)
        base_growth_df.to_csv(out_path("base_growth"), index=False)
        print(f"Wrote: {out_path('base_growth')}")

    if "presplit_data" in cf:
        print("\nGenerating presplit data...", flush=True)
        presplit_df = generate_presplit_df(combined_sample_df,
                                           combined_counts_df,
                                           cf, rng)
        presplit_df.to_csv(out_path("presplit"), index=False)
        print(f"Wrote: {out_path('presplit')}")

    with open(config_out, "w") as fh:
        yaml.dump(cf, fh, default_flow_style=False, sort_keys=False)
    print(f"Wrote: {config_out}")


def main():
    return generalized_main(run_simulation_from_config,
                            manual_arg_types={"seed": int})
