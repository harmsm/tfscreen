"""
Build a full-size simulate config that follows the dev-data screen's design.

    python make_sim_config.py [--out_dir full_size_sim] [--seed 1]
                              [--noise realistic|poisson]
                              [--phenotype_model <emp_phenotype_model.json>]
                              [--transition instant|memory]
                              [--calibrate_from <pilot run directory>]

Run from planning/dev-data (gitignored; the lab's data never enter the
repository, C10). Reads processed/ (written by prep_dev_data.py) and writes
<out_dir>/design.csv, <out_dir>/simulate_config.yaml and a copy of the OD600
calibration, all inside planning/dev-data.

What comes from the real experiment, and how:

- design.csv: the real tube table (processed/sample_df.csv): every
  sequenced tube with its real name, replicate, conditions, IPTG and
  selection time. tfs-simulate follows it exactly (the ``design`` key), so
  the simulated tube table has the real layout.
- Library: the genetics keys, transform_sizes and library_mixture of
  processed/library_config.yaml (the nine-spike config).
- Growth (linear, rate = b + m * theta): for each selection condition the
  wt monoculture rates (processed/wt_control_rate_summary.csv) at 0 and
  1 mM, with wt taken as theta 1 at 0 mM and 0 at 1 mM (the monokan rule):
  b = rate at 1 mM, m = rate at 0 minus b. Control conditions: b = their
  mean rate over IPTG, m = 0.
- cfu0 (cells per tube at -t_pre): first guess from the pre-split culture's
  OD600 (0.35, diluted 0.2 into 15.2 mL; prep_dev_data.py) through the real
  calibration, times the 5 mL tube. That guess left the simulated tubes about
  half as dense as the real ones (pheS 0.13 against 0.26 median OD600;
  2026-10-05), so the cfu0 is calibrated per library from a pilot run
  (--calibrate_from): with the phenotypes fixed by the seed, a tube's total
  scales with cfu0, so each library's cfu0 is multiplied by the geometric
  mean of real / simulated tube totals over its tubes. The script prints
  what is left by condition; a per-library level cannot fix a spread that
  comes from the growth rates.
- Reads: total_num_reads = the sum of the real tubes' called reads, spread
  evenly over the tubes (the simulator does not vary depth per tube).
- OD600: the real calibration, totals estimated from OD600 as the lab does
  (sample_cfu_from_od600).
- Congression: transformation_poisson_lambda 0.357, the measured value.

Chosen, not measured: the noise model (``realistic``: founder sampling,
demographic growth, one shared transformation, PCR templates at reads per
tube / 10 with CV 0.5, the setting that matched the real counts' 5-18x
overdispersion, planning/studies/noise-anatomy/; or ``poisson``), the
phenotypes (hill_geno prior draws, or resampled from a tfs-build-empirical
model with --phenotype_model), dk_geno's hyperparameters (the example
config's), tube_noise_sigma (0.002 per minute) and the growth transition
(instant; ``memory`` with the example config's parameters as a test arm).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import yaml

from tfscreen.process_raw.od600 import od600_to_cfu_per_mL
from tfscreen.simulate.build_sample_dataframes import read_design

PROCESSED = "processed"
TUBE_VOLUME_ML = 5.0
PRESPLIT_OD600 = 0.35
PRESPLIT_DILUTION = 0.2 / 15.2
LAMBDA = 0.357

GENETICS_KEYS = ("reading_frame", "first_amplicon_residue", "wt_seq",
                 "degen_sites", "tiles", "expected_5p", "expected_3p",
                 "tile_combos", "spiked_seqs", "transform_sizes",
                 "library_mixture")


def growth_block(rates):
    """Linear growth (b, m) per condition from the wt monoculture rates."""
    out = {}
    for cond, g in rates.groupby("condition_sel"):
        r = g.set_index("titrant_conc")["rate_mean"]
        if "+" in cond:
            b = float(r.loc[1.0])
            m = float(r.loc[0.0]) - b
        else:
            b, m = float(r.mean()), 0.0
        out[cond] = {"b": round(b, 6), "m": round(m, 6)}
    return out


TUBE_KEY = ["replicate", "library", "condition_pre", "t_pre", "condition_sel",
            "t_sel", "titrant_name", "titrant_conc"]


def calibrated_cfu0(pilot_dir, real_tubes):
    """
    Per-library cfu0 that brings a pilot run's tube totals to the real ones.

    Matches each real tube to the pilot's simulated tube by design, takes
    log(real total / simulated true total), and multiplies each library's
    pilot cfu0 by exp(mean log ratio). Prints the residual log ratio by
    library and selection condition after the correction.
    """
    sim = pd.read_csv(os.path.join(pilot_dir, "tfs_sim_od600.csv"))
    with open(os.path.join(pilot_dir, "tfs_sim_input-config.yaml")) as fh:
        pilot_cfu0 = yaml.safe_load(fh)["cfu0"]
    real = real_tubes.rename(columns={"rep": "replicate"})
    m = real[TUBE_KEY + ["sample_cfu"]].merge(
        sim[TUBE_KEY + ["sample_cfu_true"]], on=TUBE_KEY, how="inner")
    if len(m) < len(real):
        print(f"warning: {len(real) - len(m)} real tubes have no pilot tube")
    m["log_ratio"] = np.log(m["sample_cfu"] / m["sample_cfu_true"])
    out = {}
    for lib, g in m.groupby("library"):
        base = (float(str(pilot_cfu0[lib]).replace("_", ""))
                if isinstance(pilot_cfu0, dict)
                else float(str(pilot_cfu0).replace("_", "")))
        out[lib] = base * float(np.exp(g["log_ratio"].mean()))
    m["residual"] = m["log_ratio"] - m.groupby("library")["log_ratio"].transform("mean")
    print("cfu0 calibration: log(real / pilot) total by library, then the "
          "residual by selection condition (mean, SD over tubes):")
    print(m.groupby("library")["log_ratio"].agg(["mean", "std", "count"]).round(3)
          .to_string())
    print(m.groupby(["library", "condition_sel"])["residual"]
          .agg(["mean", "std", "count"]).round(3).to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="full_size_sim")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", choices=("realistic", "poisson"),
                    default="realistic")
    ap.add_argument("--phenotype_model", default=None)
    ap.add_argument("--transition", choices=("instant", "memory"),
                    default="instant")
    ap.add_argument("--calibrate_from", default=None,
                    help="a pilot run's directory (tfs_sim_od600.csv and "
                         "tfs_sim_input-config.yaml) to calibrate cfu0 from")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tubes = pd.read_csv(os.path.join(PROCESSED, "sample_df.csv"))
    design = read_design(tubes)
    design.to_csv(os.path.join(args.out_dir, "design.csv"), index=False)

    with open(os.path.join(PROCESSED, "library_config.yaml")) as fh:
        lib = yaml.safe_load(fh)
    cal_src = os.path.join(PROCESSED, "od600_calibration.yaml")
    cal_dst = os.path.join(args.out_dir, "od600_calibration.yaml")
    shutil.copyfile(cal_src, cal_dst)

    rates = pd.read_csv(os.path.join(PROCESSED,
                                     "wt_control_rate_summary.csv"))
    presplit = float(od600_to_cfu_per_mL(np.array([PRESPLIT_OD600]),
                                         cal_src)[0][0])
    cfu0 = presplit * PRESPLIT_DILUTION * TUBE_VOLUME_ML
    if args.calibrate_from:
        cfu0 = calibrated_cfu0(args.calibrate_from, tubes)
    total_reads = int(tubes["called_reads"].sum())
    reads_per_tube = total_reads / len(design)

    cf = {k: lib[k] for k in GENETICS_KEYS if k in lib}
    cf.update({
        "design": "design.csv",
        "growth": growth_block(rates),
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": 1.0,
        "dk_geno_hyper_shift": 0.02,
        "activity_wt": 1.0,
        "activity_mut_scale": 0.0,
        "lib_assembly_skew_sigma": 1.25,
        "transformation_poisson_lambda": LAMBDA,
        "congression_theta_rule": "homodimer",
        "congression_dk_rule": "dilution",
        "cfu0": ({k: float(round(v, -3)) for k, v in cfu0.items()}
                 if isinstance(cfu0, dict) else float(round(cfu0, -3))),
        "tube_noise_sigma": 0.002,
        "total_num_reads": total_reads,
        "prob_index_hop": 0.0,
        "seed": args.seed,
        # bare file names: tfs-simulate runs from inside out_dir
        "od600": {"calibration": os.path.basename(cal_dst),
                  "tube_volume_mL": TUBE_VOLUME_ML,
                  "sample_cfu_from_od600": True},
    })
    if args.phenotype_model:
        cf["phenotype_source"] = "empirical"
        cf["empirical"] = {"phenotype_model": os.path.abspath(args.phenotype_model)}
    else:
        cf["theta_component"] = "hill_geno"
    if args.noise == "realistic":
        cf.update({"founder_sampling": True, "demographic_growth": True,
                   "shared_transformation": True,
                   "pcr_template_molecules": int(round(reads_per_tube / 10)),
                   "pcr_amplification_cv": 0.5})
    if args.transition == "memory":
        cf["growth_transition"] = [
            {"condition_pre": "pheS-4CP", "model": "memory",
             "tau0": 90.0, "k1": 5.0, "k2": 0.2},
            {"condition_pre": "kanR-kan", "model": "memory",
             "tau0": 60.0, "k1": 2.0, "k2": 0.2}]

    path = os.path.join(args.out_dir, "simulate_config.yaml")
    with open(path, "w") as fh:
        fh.write("# Written by planning/studies/full-size-sim/make_sim_config.py\n"
                 f"# from {os.path.abspath(PROCESSED)}; noise {args.noise}, "
                 f"transition {args.transition}, seed {args.seed}.\n")
        yaml.safe_dump(cf, fh, sort_keys=False)
    print(f"{path}: {len(design)} tubes in {design['replicate'].nunique()} "
          f"replicates; cfu0 {cf['cfu0']}; {total_reads:.3g} reads "
          f"({reads_per_tube:.3g} per tube); growth {cf['growth']}")


if __name__ == "__main__":
    main()
