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
  unassigned_read_fraction, per library, is the share of the real tubes'
  reads in the __unknown__ row (kanR about 0.66, pheS 0.20: the kanR
  library's structural variants), so a simulated genotype gets the reads a
  real one does.
- Pool composition: library_mixture is the realized one, estimated from the
  real pre-split (initial) composition with estimate_library_mixture and
  averaged over the two libraries (they agree within ~0.05), not the
  design's (which gives doubles 83% of the pool against about 54%
  realized). wt is about 11% of the real pool, ~90 times what the spike
  design gives it, for reasons not yet known (being worked on at the bench,
  2026-10-08). The simulation reproduces it by repeating the wt sequence in
  the simulated spiked_seqs (a repeated sequence is degeneracy within the
  spiked origin) and scaling the spiked origin's mixture and transformants
  so the other spikes keep their share. This is simulation-only: the fit is
  configured with the real library config (copied to library_config.yaml),
  as the real data's fit is.
- OD600: the real calibration, totals estimated from OD600 as the lab does
  (sample_cfu_from_od600).
- Congression: transformation_poisson_lambda 0.357, the measured value.
- Pool evenness (2026-10-08), fitted to the real initial composition and
  the screen's dropout: doubles are very uneven before any selection (CV 5
  to 7 in the initial sample, 23-30% at zero reads out of a mean of 20-50),
  yet only 2.7% of real doubles end below 10 reads over the screen, and
  their abundance is shared between the separately transformed libraries
  (log-count correlation 0.69). That is a wide assembly skew with many
  clones per genotype: doubles skew sigma 2.0 and ~300 clones per double
  (transform_sizes double-1-2 = 300 x its genotypes, about 6.7e7); singles
  are even (CV 0.8-1.2), skew 0.8. The simulator merges identical clones
  into weighted rows, so this costs one row per genotype, and holds the
  congressed cells to max_congressed_cells (2e6).
- dk_geno: the real fit's well-measured genotypes (singles, doubles above
  1e4 reads) spread over about 0.016 per minute (1st-99th percentile), with
  none below -0.03; the old hyperparameters (scale 1.0) put 31% of
  genotypes below -0.03 and killed them. Scale 0.15 keeps the median
  (-0.010) and gives a 1st-99th percentile range of about -0.023 to -0.001.

Chosen, not measured: the noise model (``realistic``: founder sampling,
demographic growth, one shared transformation, PCR templates at assigned
reads per tube / 10 with CV 0.5, the setting that matched the real counts' 5-18x
overdispersion, planning/studies/noise-anatomy/; or ``poisson``), the
phenotypes (hill_geno prior draws, or resampled from a tfs-build-empirical
model with --phenotype_model), dk_geno's location and shift (the example
config's; its scale is fitted, above), tube_noise_sigma (0.002 per minute) and the growth transition
(instant; ``memory`` with the example config's parameters as a test arm).
"""

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import yaml

from tfscreen.genetics import UNKNOWN_GENOTYPE
from tfscreen.genetics.library_design import (
    estimate_library_mixture,
    library_composition_table,
)
from tfscreen.genetics.library_manager import LibraryManager
from tfscreen.process_raw.od600 import od600_to_cfu_per_mL
from tfscreen.simulate.build_sample_dataframes import read_design

PROCESSED = "processed"
TUBE_VOLUME_ML = 5.0
PRESPLIT_OD600 = 0.35
PRESPLIT_DILUTION = 0.2 / 15.2
LAMBDA = 0.357
ASSEMBLY_SKEW = {"double-1-2": 2.0, "single-1": 0.8, "single-2": 0.8,
                 "spiked": 0.5}
CLONES_PER_DOUBLE = 300
MAX_CONGRESSED_CELLS = 2_000_000
DK_HYPER_SCALE = 0.15

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


def unassigned_fractions(tubes):
    """Share of each library's reads in the real count files' __unknown__ row."""
    out = {}
    for lib, g in tubes.groupby("library"):
        unknown = total = 0
        for name in g["sample"]:
            path = os.path.join(PROCESSED, "counts", f"counts__{name}__.csv")
            c = pd.read_csv(path)
            unknown += c.loc[c["genotype"] == UNKNOWN_GENOTYPE, "counts"].sum()
            total += c["counts"].sum()
        out[lib] = round(float(unknown / total), 4)
    return out


def realized_pool(lib):
    """
    The realized library_mixture (mean over libraries of
    estimate_library_mixture on the initial composition) and wt's share of
    the pool.
    """
    library_df = LibraryManager(lib).build_library_df()
    ic = pd.read_csv(os.path.join(PROCESSED, "initial_composition.csv"))
    ic = ic[ic["genotype"] != UNKNOWN_GENOTYPE]
    mixes, wt_shares = [], []
    for _, g in ic.groupby("library"):
        mix, _, _ = estimate_library_mixture(
            library_df, g.set_index("genotype")["counts"])
        mixes.append(mix)
        wt_shares.append(g.loc[g["genotype"] == "wt", "counts"].sum()
                         / g["counts"].sum())
    mixture = {k: float(np.mean([m[k] for m in mixes])) for k in mixes[0]}
    return mixture, float(np.mean(wt_shares))


def with_wt_excess(lib, mixture, wt_share):
    """
    Genetics keys, mixture and transform sizes with wt repeated in the
    spiked origin so wt makes up wt_share of the pool, the other spikes
    keeping theirs (simulation only).

    The spiked origin's n sequences share its mixture by degeneracy. Giving
    wt d copies and scaling the origin's mixture and transformants by
    (n - 1 + d) / n leaves every other spike's mass unchanged; d is the
    smallest integer that brings wt's pool fraction (spiked plus bulk, from
    library_composition_table) up to wt_share.
    """
    wt_seq = lib["wt_seq"]
    spiked = list(lib["spiked_seqs"])
    library_df = LibraryManager(lib).build_library_df()
    n_doubles = int((library_df["library_origin"] == "double-1-2").sum())
    n = len(spiked)
    if spiked.count(wt_seq) != 1:
        raise ValueError("expected wt exactly once in spiked_seqs")

    def build(d):
        cf = dict(lib)
        cf["spiked_seqs"] = spiked + [wt_seq] * (d - 1)
        scale = (n - 1 + d) / n
        cf["library_mixture"] = dict(mixture, spiked=mixture["spiked"] * scale)
        sizes = dict(lib["transform_sizes"])
        sizes["double-1-2"] = CLONES_PER_DOUBLE * n_doubles
        sizes["spiked"] = int(round(sizes["spiked"] * scale))
        cf["transform_sizes"] = sizes
        comp = library_composition_table(cf)
        wt = float(comp.loc[comp["genotype"] == "wt", "pool_fraction"].iloc[0])
        return cf, wt

    lo, hi = 1, 2
    while build(hi)[1] < wt_share:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if build(mid)[1] < wt_share:
            lo = mid
        else:
            hi = mid
    cf, wt = build(hi)
    return cf, hi, wt


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
    # the fit is configured with the real library config, as the real data's
    shutil.copyfile(os.path.join(PROCESSED, "library_config.yaml"),
                    os.path.join(args.out_dir, "library_config.yaml"))
    mixture, wt_share = realized_pool(lib)
    sim_lib, wt_copies, wt_pool = with_wt_excess(lib, mixture, wt_share)
    unassigned = unassigned_fractions(tubes)

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

    cf = {k: sim_lib[k] for k in GENETICS_KEYS if k in sim_lib}
    cf.update({
        "design": "design.csv",
        "growth": growth_block(rates),
        "dk_geno_hyper_loc": -3.5,
        "dk_geno_hyper_scale": DK_HYPER_SCALE,
        "dk_geno_hyper_shift": 0.02,
        "activity_wt": 1.0,
        "activity_mut_scale": 0.0,
        "lib_assembly_skew_sigma": dict(ASSEMBLY_SKEW),
        "max_congressed_cells": MAX_CONGRESSED_CELLS,
        "transformation_poisson_lambda": LAMBDA,
        "congression_theta_rule": "homodimer",
        "congression_dk_rule": "dilution",
        "cfu0": ({k: float(round(v, -3)) for k, v in cfu0.items()}
                 if isinstance(cfu0, dict) else float(round(cfu0, -3))),
        "tube_noise_sigma": 0.002,
        "total_num_reads": total_reads,
        "unassigned_read_fraction": unassigned,
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
                   # templates = assigned reads per tube / 10, the ratio
                   # that matched the real counts' overdispersion
                   # (noise-anatomy, measured on called reads)
                   "pcr_template_molecules": int(round(
                       reads_per_tube * (1 - np.mean(list(unassigned.values())))
                       / 10)),
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
    print(f"realized library_mixture {cf['library_mixture']} (design "
          f"{lib['library_mixture']}); wt {wt_share:.3f} of the real pool, "
          f"{wt_copies} wt copies in spiked_seqs give {wt_pool:.3f}; "
          f"unassigned_read_fraction {unassigned}")


if __name__ == "__main__":
    main()
