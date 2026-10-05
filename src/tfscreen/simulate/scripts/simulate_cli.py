import os
import yaml
import numpy as np
import pandas as pd

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.simulate.binding_data import generate_binding_df
from tfscreen.simulate.library_binding_data import generate_library_binding_df
from tfscreen.simulate.presplit_data import generate_presplit_df
from tfscreen.simulate.base_growth_data import generate_base_growth_df, generate_k_ref_df
from tfscreen.simulate.growth_parameters_output import generate_growth_parameters_df
from tfscreen.simulate.transformation_lam_output import generate_transformation_lam_df
from tfscreen.process_raw import counts_to_lncfu
from tfscreen.simulate.raw_output import write_raw_experiment
from tfscreen.simulate.build_sample_dataframes import (
    CONDITION_COLUMNS,
    read_design,
)
from tfscreen.util.cli.generalized_main import generalized_main


def run_simulation_from_config(
    config_file,
    out_prefix="tfs_sim",
    num_replicates=2,
    seed=None,
    write_raw=True,
    write_growth=True,
):
    """
    Simulate a TF selection experiment from a YAML configuration file.

    Runs library_prediction once to establish ground-truth phenotypes, then
    simulates num_replicates independent experimental replicates using
    selection_experiment. Writes library, parameters, genotype_theta (long-form:
    genotype/titrant_name/titrant_conc/theta), growth, growth_parameters
    (per-condition condition_growth ground truth; see
    tfscreen.simulate.growth_parameters_output.generate_growth_parameters_df),
    and transformation_lam (single-row congression Poisson-rate ground truth;
    see tfscreen.simulate.transformation_lam_output.generate_transformation_lam_df)
    CSV files. If the config contains a 'binding_data' block, also writes a
    simulated binding curve CSV (see
    tfscreen.simulate.binding_data.generate_binding_df). If it contains a
    'base_growth_data' block, also writes a simulated direct growth-rate
    calibration CSV and a single-row k_ref ground-truth CSV (see
    tfscreen.simulate.base_growth_data.generate_base_growth_df and
    .generate_k_ref_df). If it contains a 'presplit_data' block, also writes
    a simulated presplit CSV (see
    tfscreen.simulate.presplit_data.generate_presplit_df). If it contains an
    'od600' block, also writes one OD600 reading per tube for every
    replicate, sequenced or not ('od600'; see tfscreen.simulate.od600), and
    simulates od600.num_od_only_replicates extra replicates that get OD600
    but no reads. With shared_transformation, every replicate (OD-only ones
    included) draws its cells from one library assembly and transformation.

    Parameters
    ----------
    config_file : str
        Path to the YAML run configuration file.
    out_prefix : str
        Output prefix: files are written as ``{out_prefix}_{name}.csv``. A
        directory part (``sim/tfs_sim``) is created if absent.
    num_replicates : int
        Number of independent experimental replicates to simulate. Default 2.
    seed : int, optional
        Random seed. Overrides seed in the config file when provided.
    write_raw : bool
        Also write the experiment in the raw formats real data come in (one
        count file per tube in ``{out_prefix}_counts/``, the tube table
        ``{out_prefix}_tubes.csv`` and, with an ``od600`` block, the OD600
        table and the calibration), so ``tfs-process-counts`` processes it
        exactly as it would a lab's data. The command is printed. See
        ``tfscreen.simulate.raw_output``.
    write_growth : bool
        Also write ``{out_prefix}_growth.csv``, the growth table built
        directly from the simulated counts (with each row's true values).
        ``tfs-process-counts`` builds the same table from the raw files, so a
        full-size simulation can skip it: on a real library's size it is
        several GB.
    """
    cf = tfscreen.util.read_yaml(config_file)
    if seed is not None:
        cf["seed"] = seed

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def out_path(name):
        return f"{out_prefix}_{name}.csv"

    config_out = f"{out_prefix}_input-config.yaml"

    output_names = ["library", "parameters", "genotype_theta",
                    "growth_parameters", "transformation_lam"]
    if write_growth:
        output_names.append("growth")
    if "binding_data" in cf:
        output_names.append("binding")
        if cf["binding_data"].get("library_binding") is not None:
            output_names.append("library_binding")
    if "presplit_data" in cf:
        output_names.append("presplit")
    if "base_growth_data" in cf:
        output_names.append("base_growth")
        output_names.append("k_ref")
    if cf.get("od600") is not None:
        output_names.append("od600")

    if write_raw:
        output_names.append("tubes")
    existing = [out_path(n) for n in output_names if os.path.exists(out_path(n))]
    if os.path.exists(config_out):
        existing.append(config_out)
    if existing:
        paths = ", ".join(existing)
        raise FileExistsError(
            f"Output files already exist: {paths}\n"
            f"Delete them or choose a different out_prefix "
            f"before re-running."
        )

    base_seed = cf.get("seed", None)
    rng = np.random.default_rng(base_seed)

    # -------------------------------------------------------------------------
    # Ground-truth library and phenotypes (deterministic given the config)

    library_df, phenotype_df, genotype_theta_df, parameters_df, binding_theta_df = library_prediction(cf)
    growth_parameters_df = generate_growth_parameters_df(cf["growth"])
    transformation_lam_df = generate_transformation_lam_df(cf)

    # -------------------------------------------------------------------------
    # Simulate independent replicates

    all_sample_parts = []
    all_counts_parts = []
    od_only_parts = []
    sample_id_offset = 0

    # One library assembly and transformation for every replicate when
    # shared_transformation is set (selection_experiment ignores it otherwise).
    shared_state = {}

    od_cf = cf.get("od600") or {}
    num_od_only = int(od_cf.get("num_od_only_replicates") or 0)

    # A design tube table fixes the replicates and each one's tubes.
    design = None
    if cf.get("design") is not None:
        design = read_design(cf["design"])
        if num_od_only:
            raise ValueError("od600.num_od_only_replicates cannot be combined "
                             "with a design; add the OD-only tubes to the "
                             "design instead.")
        rep_ids = sorted(design["replicate"].unique().tolist())
        if num_replicates != len(rep_ids):
            print(f"Design has {len(rep_ids)} replicate(s) {rep_ids}; "
                  f"num_replicates ({num_replicates}) is ignored.", flush=True)
        num_replicates = len(rep_ids)
    else:
        rep_ids = list(range(1, num_replicates + num_od_only + 1))

    for rep_num, rep in enumerate(rep_ids, start=1):
        sequenced = rep_num <= num_replicates
        label = ("" if sequenced else " (OD600 only)")
        print(f"\n--- Replicate {rep} of {len(rep_ids)}{label} ---",
              flush=True)

        # Give each replicate a distinct (but reproducible) random seed so
        # that replicates differ even when a base seed is set.
        rep_cf = dict(cf)
        rep_cf["seed"] = (
            base_seed * num_replicates + rep_num if base_seed is not None else None
        )

        rep_phenotype_df = phenotype_df.copy()
        if design is not None:
            tubes = design.loc[design["replicate"] == rep, CONDITION_COLUMNS]
            rep_phenotype_df = rep_phenotype_df.merge(tubes, on=CONDITION_COLUMNS)
        rep_phenotype_df["replicate"] = rep

        sample_df_rep, counts_df_rep = selection_experiment(
            rep_cf, library_df, rep_phenotype_df,
            shared_state=shared_state, sequence=sequenced
        )
        if not sequenced:
            od_only_parts.append(sample_df_rep.assign(sequenced=False))
            continue

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

    if cf.get("od600") is not None:
        od_df = pd.concat([combined_sample_df.assign(sequenced=True)]
                          + od_only_parts, ignore_index=True)
        keep = [c for c in ["replicate", "library", "condition_pre", "t_pre",
                            "condition_sel", "t_sel", "titrant_name",
                            "titrant_conc", "sequenced", "od600",
                            "od600_detectable", "od600_in_range",
                            "sample_cfu_true", "sample_cfu", "sample_cfu_std"]
                if c in od_df.columns]
        od_df[keep].to_csv(out_path("od600"), index=False)
        print(f"Wrote: {out_path('od600')}")

    if design is not None:
        # The design's tube names, for the raw output.
        key = ["replicate"] + CONDITION_COLUMNS
        named = combined_sample_df[key].merge(
            design[key + ["sample"]].rename(columns={"sample": "design_sample"}),
            on=key, how="left")
        combined_sample_df = combined_sample_df.copy()
        combined_sample_df["design_sample"] = named["design_sample"].to_numpy()

    # The in-memory growth table: written unless skipped, and needed by the
    # in-library binding selection (survivors).
    need_growth = write_growth or (
        "binding_data" in cf
        and cf["binding_data"].get("library_binding") is not None)
    growth_df = None
    if need_growth:
        growth_df = counts_to_lncfu(
            combined_sample_df.drop(columns="design_sample", errors="ignore"),
            combined_counts_df)

    if write_raw:
        raw = write_raw_experiment(combined_sample_df, combined_counts_df,
                                   library_df["genotype"], out_prefix,
                                   od600_config=cf.get("od600"))
        print(f"\nWrote the raw experiment: {raw['tubes']}, "
              f"{raw['counts_dir']}/"
              + (f", {raw['od600']}, {raw['calibration']}"
                 if "od600" in raw else ""), flush=True)
        if raw["dropped"]:
            print(f"Left out of the tube table (OD600 below the detection "
                  f"threshold): {raw['dropped']}", flush=True)
        print(f"Process it as real data with:\n  {raw['command']}", flush=True)

    # -------------------------------------------------------------------------
    # Write outputs

    library_df.to_csv(out_path("library"), index=False)
    parameters_df.to_csv(out_path("parameters"), index=False)
    genotype_theta_df.to_csv(out_path("genotype_theta"), index=False)
    if write_growth:
        growth_df.to_csv(out_path("growth"), index=False)
    growth_parameters_df.to_csv(out_path("growth_parameters"), index=False)
    transformation_lam_df.to_csv(out_path("transformation_lam"), index=False)
    written = ["library", "parameters", "genotype_theta", "growth_parameters",
               "transformation_lam"] + (["growth"] if write_growth else [])
    print(f"\nWrote: {', '.join(out_path(n) for n in written)}")

    if "binding_data" in cf:
        binding_cfg = cf["binding_data"]
        # Spiked (clean) binding measurements from the pre-sim binding_theta_df.
        if binding_theta_df is not None:
            binding_df = generate_binding_df(binding_cfg, rng, binding_theta_df)
        else:
            binding_df = pd.DataFrame(columns=["genotype", "titrant_name",
                                               "titrant_conc", "theta_obs", "theta_std"])
        # In-library binding measurements (post-sim: selection from survivors).
        lb = binding_cfg.get("library_binding")
        if lb is not None:
            spiked_names = list(pd.unique(
                library_df.loc[library_df["library_origin"] == "spiked", "genotype"]))
            lib_binding_df, lib_manifest = generate_library_binding_df(
                lb,
                binding_cfg["titrant_name"],
                binding_cfg["titrant_conc"],
                binding_cfg.get("noise", 0.0),
                parameters_df,
                growth_df,
                spiked_genotypes=spiked_names,
                rng=rng,
                clip_theta_obs=bool(binding_cfg.get("clip_theta_obs", False)),
            )
            binding_df = pd.concat([binding_df, lib_binding_df], ignore_index=True)
            lib_manifest.to_csv(out_path("library_binding"), index=False)
            print(f"Wrote: {out_path('library_binding')}")
        binding_df.to_csv(out_path("binding"), index=False)
        print(f"Wrote: {out_path('binding')}")

    if "base_growth_data" in cf:
        base_growth_df = generate_base_growth_df(cf["base_growth_data"], parameters_df, rng)
        base_growth_df.to_csv(out_path("base_growth"), index=False)
        print(f"Wrote: {out_path('base_growth')}")

        k_ref_df = generate_k_ref_df(cf["base_growth_data"])
        k_ref_df.to_csv(out_path("k_ref"), index=False)
        print(f"Wrote: {out_path('k_ref')}")

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
