import os
import yaml
import numpy as np
import pandas as pd
import jax

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.simulate.selection_experiment import _sim_index_hop
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

    # Sample theta using the same seed as the main library_prediction run so
    # binding theta values are consistent with the ground-truth phenotypes.
    seed = cf.get("seed", None)
    theta_rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
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


def _generate_presplit_data(combined_sample_df, combined_counts_df, cf, rng):
    """
    Generate simulated pre-split (t = -t_pre) data from selection-experiment
    outputs.

    For each unique ``(replicate, condition_pre)`` pair, this function draws a
    synthetic sequencing sample from the initial library-frequency distribution
    (encoded as ``ln_cfu_0`` in ``combined_counts_df``).  The resulting counts
    are converted to ``ln_cfu`` and ``ln_cfu_std`` using the same
    variance-propagation formula as ``counts_to_lncfu``, yielding a DataFrame
    that can be passed directly to ``tfs-configure-model --presplit_df``.

    Parameters
    ----------
    combined_sample_df : pandas.DataFrame
        Concatenated sample metadata across all replicates (index = sample ID).
        Must contain columns ``replicate`` and ``condition_pre``.
    combined_counts_df : pandas.DataFrame
        Concatenated genotype counts across all replicates.  Must contain
        columns ``sample``, ``genotype``, and ``ln_cfu_0`` (the ground-truth
        initial log-CFU per genotype computed by the simulation).
    cf : dict
        Full run configuration (already read from YAML).  Used keys:
        ``cfu0``, ``total_num_reads``, ``prob_index_hop``.  The optional ``presplit_data`` sub-dict may
        contain a scalar ``noise`` key (default 0) to add extra Gaussian noise
        to ``ln_cfu`` on the log scale.
    rng : numpy.random.Generator
        Seeded random-number generator shared with the rest of the simulation.

    Returns
    -------
    pandas.DataFrame
        Columns: ``replicate``, ``condition_pre``, ``genotype``,
        ``ln_cfu``, ``ln_cfu_std``, ``ln_cfu_0_true``.
        ``ln_cfu_0_true`` records the simulation ground truth for validation.
        Rows with zero initial frequency (genotypes absent from the
        transformation pool) receive ``ln_cfu = NaN``.
    """
    total_cfu0        = float(cf["cfu0"])
    presplit_cfg      = cf.get("presplit_data", {})
    extra_noise       = float(presplit_cfg.get("noise", 0.0)) if presplit_cfg else 0.0
    pseudocount       = 1

    # Reads per presplit sample: use the same budget as the main experiment
    # (total reads / total selection samples).
    total_num_reads    = int(cf["total_num_reads"])
    total_num_samples  = len(combined_sample_df)
    reads_per_sample   = max(1, int(round(total_num_reads / total_num_samples)))

    # Attach condition_pre and replicate to each (genotype, sample) row so we
    # can group by (replicate, condition_pre) below.
    sample_meta = (combined_sample_df
                   .reset_index()[["sample", "replicate", "condition_pre"]]
                   .drop_duplicates("sample"))
    counts_meta = pd.merge(combined_counts_df, sample_meta, on="sample", how="left")

    # One row per (replicate, condition_pre, genotype), keeping the first
    # occurrence of ln_cfu_0 (same value for all selection samples within a
    # library group).
    source = (counts_meta
              .groupby(["replicate", "condition_pre", "genotype"], observed=True)
              .first()
              .reset_index()[["replicate", "condition_pre", "genotype", "ln_cfu_0"]])

    rows = []
    for (rep, cp), grp in source.groupby(["replicate", "condition_pre"],
                                          observed=True):
        genos    = grp["genotype"].values
        ln_cfu0  = grp["ln_cfu_0"].values

        # Convert ground-truth ln_cfu_0 to initial frequencies.
        # Genotypes with ln_cfu_0 = -inf (absent from transformation pool)
        # get frequency 0.
        cfu0_raw  = np.where(np.isfinite(ln_cfu0), np.exp(ln_cfu0), 0.0)
        total_cfu_group = cfu0_raw.sum()
        if total_cfu_group == 0:
            continue
        freqs = cfu0_raw / total_cfu_group

        # Simulate multinomial sequencing draw from the initial distribution.
        counts = rng.multinomial(reads_per_sample, freqs)

        # Optionally apply index hopping (same as the selection samples).
        counts = _sim_index_hop(counts, cf.get("prob_index_hop"), rng)

        # ---------- counts → ln_cfu (mirrors counts_to_lncfu logic) ----------
        sample_cfu     = total_cfu0

        total_adjusted = counts.sum() + len(counts) * pseudocount
        adj_counts     = counts + pseudocount
        freq_est       = adj_counts / total_adjusted
        cfu_est        = freq_est * sample_cfu

        # Variance from binomial frequency uncertainty only
        var_freq      = freq_est * (1.0 - freq_est) / total_adjusted
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_var_freq  = np.where(freq_est > 0,
                                     var_freq / (freq_est ** 2), 0.0)
        ln_cfu_var        = rel_var_freq

        # Optional additional noise on the log scale
        if extra_noise > 0.0:
            ln_cfu_var = ln_cfu_var + extra_noise ** 2
            ln_cfu_shift = rng.normal(0.0, extra_noise, size=len(genos))
        else:
            ln_cfu_shift = np.zeros(len(genos))

        with np.errstate(divide="ignore"):
            ln_cfu_vals = np.log(cfu_est) + ln_cfu_shift

        # Zero-count entries become NaN
        ln_cfu_vals = np.where(cfu_est > 0, ln_cfu_vals, np.nan)
        ln_cfu_var  = np.where(cfu_est > 0, ln_cfu_var,  np.nan)
        ln_cfu_std_vals = np.sqrt(ln_cfu_var)

        for i, geno in enumerate(genos):
            rows.append({
                "replicate":     rep,
                "condition_pre": cp,
                "genotype":      geno,
                "ln_cfu":        float(ln_cfu_vals[i]),
                "ln_cfu_std":    float(ln_cfu_std_vals[i]),
                "ln_cfu_0_true": float(ln_cfu0[i]),
            })

    return pd.DataFrame(rows)


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
    simulated binding curve CSV.

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
        binding_df = _generate_binding_data(cf, library_df, cf["binding_data"], rng)
        binding_df.to_csv(out_path("binding"), index=False)
        print(f"Wrote: {out_path('binding')}")

    if "presplit_data" in cf:
        print("\nGenerating presplit data...", flush=True)
        presplit_df = _generate_presplit_data(combined_sample_df,
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
