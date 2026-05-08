import os
import numpy as np
import pandas as pd

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.simulate.setup_observable import setup_observable
from tfscreen.genetics import combine_mutation_effects, standardize_genotypes
from tfscreen.process_raw import counts_to_lncfu
from tfscreen.util.cli.generalized_main import generalized_main


def _generate_binding_data(cf, genotype_ddG_df, binding_cfg, rng):
    """
    Generate simulated binding curve data for specific genotypes.

    Reads binding_data config block, calls the thermodynamic model at the
    requested titrant concentrations, adds Gaussian noise, and returns a
    long-form DataFrame.

    Parameters
    ----------
    cf : dict
        Full run configuration (already read from YAML).
    genotype_ddG_df : pandas.DataFrame
        Per-genotype ddG values returned by library_prediction (has a
        'genotype' column plus one column per thermodynamic species).
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

    # Build a minimal sample_df whose sole purpose is setting e_total in
    # the observable calculator (one row per concentration requested).
    minimal_sample_df = pd.DataFrame({
        "replicate": 1,
        "library": "binding",
        "titrant_name": titrant_name,
        "condition_pre": "binding",
        "t_pre": 0,
        "condition_sel": "binding",
        "titrant_conc": titrant_conc,
        "t_sel": 0,
    })

    # set up observable with the binding concentrations
    theta_fcn, ddG_df = setup_observable(
        cf["observable_calculator"],
        cf["observable_calc_kwargs"],
        cf["ddG_spreadsheet"],
        minimal_sample_df,
    )
    species_cols = list(ddG_df.columns)

    # Build lookup from genotype_ddG_df (genotype column + species columns)
    ddG_lookup = (genotype_ddG_df.set_index("genotype")
                  if "genotype" in genotype_ddG_df.columns
                  else genotype_ddG_df)

    in_library = [g for g in genotypes if g in ddG_lookup.index]
    not_in_library = [g for g in genotypes if g not in ddG_lookup.index]

    geno_ddG = {}
    for g in in_library:
        geno_ddG[g] = ddG_lookup.loc[g, species_cols].to_numpy(dtype=float)

    if not_in_library:
        extra = combine_mutation_effects(
            unique_genotypes=not_in_library,
            single_mutant_effects=ddG_df,
        )
        for g in not_in_library:
            geno_ddG[g] = extra.loc[g].to_numpy(dtype=float)

    rows = []
    for g in genotypes:
        theta_true = theta_fcn(geno_ddG[g])
        if noise > 0:
            theta_obs = np.clip(
                theta_true + rng.normal(0, noise, size=len(theta_true)), 0, 1
            )
        else:
            theta_obs = theta_true.copy()
        for conc, t_obs in zip(titrant_conc, theta_obs):
            rows.append({
                "genotype": g,
                "titrant_name": titrant_name,
                "titrant_conc": conc,
                "theta_obs": float(t_obs),
                "theta_std": noise,
            })

    return pd.DataFrame(rows)


def run_simulation_from_config(
    config_file,
    output_dir,
    output_prefix="tfscreen_",
    num_replicates=2,
):
    """
    Simulate a TF selection experiment from a YAML configuration file.

    Runs library_prediction once to establish ground-truth phenotypes, then
    simulates num_replicates independent experimental replicates using
    selection_experiment. Writes library, phenotype, genotype_ddG, and
    analysis-ready growth CSV files. If the config contains a 'binding_data'
    block, also writes a simulated binding curve CSV.

    Parameters
    ----------
    config_file : str
        Path to the YAML run configuration file.
    output_dir : str
        Directory to write output CSV files into (created if absent).
    output_prefix : str
        Prefix for all output filenames. Default 'tfscreen_'.
    num_replicates : int
        Number of independent experimental replicates to simulate. Default 2.
    """
    cf = tfscreen.util.read_yaml(config_file)
    if cf is None:
        raise RuntimeError("Aborting simulation due to configuration error.")

    os.makedirs(output_dir, exist_ok=True)

    def out_path(name):
        return os.path.join(output_dir, f"{output_prefix}{name}.csv")

    output_names = ["library", "phenotype", "genotype_ddG", "growth"]
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

    library_df, phenotype_df, genotype_ddG_df = library_prediction(cf)

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
    phenotype_df.to_csv(out_path("phenotype"), index=False)
    genotype_ddG_df.to_csv(out_path("genotype_ddG"), index=False)
    growth_df.to_csv(out_path("growth"), index=False)
    print(f"\nWrote: {', '.join(out_path(n) for n in ['library', 'phenotype', 'genotype_ddG', 'growth'])}")

    if "binding_data" in cf:
        binding_df = _generate_binding_data(cf, genotype_ddG_df, cf["binding_data"], rng)
        binding_df.to_csv(out_path("binding"), index=False)
        print(f"Wrote: {out_path('binding')}")


def main():
    return generalized_main(run_simulation_from_config)
