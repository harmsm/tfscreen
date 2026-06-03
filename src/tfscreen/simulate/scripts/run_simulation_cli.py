import os
import numpy as np
import pandas as pd
import jax

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.sample_theta import sample_theta_prior
from tfscreen.genetics import standardize_genotypes
from tfscreen.process_raw import counts_to_lncfu
from tfscreen.util.cli.generalized_main import generalized_main


def _generate_binding_data(cf, library_df, binding_cfg, rng):
    """
    Generate simulated binding curve data for specific genotypes.

    Samples theta from the same prior used by library_prediction (same
    theta_component and theta_rng_seed) but evaluated at the binding
    concentrations.  Adds Gaussian noise to the ground-truth theta values.

    Parameters
    ----------
    cf : dict
        Full run configuration (already read from YAML).
    library_df : pandas.DataFrame
        Library returned by library_prediction (has a 'genotype' column).
    binding_cfg : dict
        The 'binding_data' sub-dict from the config. Must contain:
          genotypes   : list of genotype strings
          titrant_name: str, name of the titrant (e.g. 'iptg')
          titrant_conc: list of concentrations (mM)
          noise       : float, sigma for Gaussian noise on theta_obs
    rng : numpy.random.Generator

    Returns
    -------
    pandas.DataFrame
        Columns: genotype, titrant_name, titrant_conc, theta_obs, theta_std
    """
    genotypes = list(standardize_genotypes(binding_cfg["genotypes"]))
    titrant_name = binding_cfg["titrant_name"]
    titrant_conc = list(binding_cfg["titrant_conc"])
    noise = float(binding_cfg.get("noise", 0.0))

    # Build a SimData for just the binding concentrations
    binding_sample_df = pd.DataFrame({
        "titrant_conc": titrant_conc,
        "condition_pre": "binding",
        "condition_sel": "binding",
    })
    sim_data = build_sim_data(
        library_df=library_df,
        sample_df=binding_sample_df,
        thermo_data=cf.get("thermo_data"),
    )

    # Sample theta from the prior using the same seed as the main run
    theta_rng_key = jax.random.PRNGKey(cf.get("theta_rng_seed", 0))
    theta_gc, _ = sample_theta_prior(
        component_name=cf["theta_component"],
        sim_data=sim_data,
        rng_key=theta_rng_key,
        priors_overrides=cf.get("theta_priors"),
    )

    # Build lookup: genotype string → row index in library_df (sim_data order)
    all_genotypes = library_df["genotype"].tolist()
    geno_to_idx = {g: i for i, g in enumerate(all_genotypes)}

    # Sorted unique concentrations (same order as sim_data.titrant_conc)
    sorted_concs = np.sort(np.unique(titrant_conc))
    conc_to_col = {float(c): i for i, c in enumerate(sorted_concs)}

    rows = []
    for g in genotypes:
        if g not in geno_to_idx:
            raise ValueError(
                f"Genotype '{g}' in binding_data.genotypes is not in the library."
            )
        row_idx = geno_to_idx[g]
        for conc in titrant_conc:
            col_idx = conc_to_col[float(conc)]
            theta_true = float(theta_gc[row_idx, col_idx])
            if noise > 0:
                theta_obs = float(np.clip(theta_true + rng.normal(0, noise), 0, 1))
            else:
                theta_obs = theta_true
            rows.append({
                "genotype": g,
                "titrant_name": titrant_name,
                "titrant_conc": conc,
                "theta_obs": theta_obs,
                "theta_std": noise,
            })

    return pd.DataFrame(rows)


def run_simulation_from_config(
    config_file,
    output_dir,
    output_prefix="tfs_sim_",
    num_replicates=1,
    seed=None,
):
    """
    Simulate a TF selection experiment from a YAML configuration file.

    Runs library_prediction once to establish ground-truth phenotypes, then
    simulates num_replicates independent experimental replicates using
    selection_experiment. Writes library, phenotype, genotype_theta, and
    analysis-ready growth CSV files. If the config contains a 'binding_data'
    block, also writes a simulated binding curve CSV.

    Parameters
    ----------
    config_file : str
        Path to the YAML run configuration file.
    output_dir : str
        Directory to write output CSV files into (created if absent).
    output_prefix : str
        Prefix for all output filenames. Default 'tfs_sim_'.
    num_replicates : int
        Number of independent experimental replicates to simulate. Default 1.
    seed : int, optional
        Random seed. Overrides random_seed in the config file when provided.
    """
    cf = tfscreen.util.read_yaml(config_file)
    if seed is not None:
        cf["random_seed"] = seed

    os.makedirs(output_dir, exist_ok=True)

    def out_path(name):
        return os.path.join(output_dir, f"{output_prefix}{name}.csv")

    output_names = ["library", "parameters", "genotype_theta", "growth"]
    if "binding_data" in cf:
        output_names.append("binding")

    existing = [n for n in output_names if os.path.exists(out_path(n))]
    if existing:
        paths = ", ".join(out_path(n) for n in existing)
        raise FileExistsError(f"Output files already exist: {paths}")

    base_seed = cf.get("random_seed", None)
    rng = np.random.default_rng(base_seed)

    # -------------------------------------------------------------------------
    # Ground-truth library and phenotypes (deterministic given the config)

    library_df, phenotype_df, genotype_theta_df, parameters_df = library_prediction(cf)

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
        rep_cf["random_seed"] = (
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
        binding_df = _generate_binding_data(cf, library_df, cf["binding_data"], rng)
        binding_df.to_csv(out_path("binding"), index=False)
        print(f"Wrote: {out_path('binding')}")


def main():
    return generalized_main(run_simulation_from_config,
                            manual_arg_types={"seed": int})
