"""
Reduced-library factorial for the |m| bias (planning/deep-coverage.md, step 1b).

    python make_factorial.py --pilot <pilot run dir> [--out_dir m_bias] [--seeds 1 2]

Run from planning/dev-data (needs processed/; the lab's data stay there,
C10). Builds the full-size calibrated configs with
planning/studies/full-size-sim/make_sim_config.py (realistic and Poisson
noise, cfu0 calibrated from --pilot), then shrinks the library to 5
degenerate sites per tile (tile 1 residues 32, 37, 42, 47, 52; tile 2
residues 74, 79, 84, 88, 93, every spike site kept: about 5,000 doubles
instead of 223,000) with genetics.library_design.scale_library_design, so
every sequence keeps its reads, cells, transformants and PCR templates. The
fit is configured with the real library config shrunk the same way.

Arms (each switches off one feature of the simulation the fit does not
model):

- base: the calibrated realistic simulation (congression lambda 0.357,
  per-tube rate noise 0.002 per minute, founder/demographic/PCR noise)
- no_cong: transformation_poisson_lambda 0 (no co-transformed cells)
- no_tube: tube_noise_sigma 0
- poisson: Poisson counts (the Poisson config: no founder, demographic or
  PCR noise)
- clean: all three off

Each arm directory gets simulate_config.yaml, library_config.yaml and a
copy of growth_priors_loose.csv; run_arm.sh runs the chain in it.
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys

import yaml

from tfscreen.genetics.library_design import scale_library_design
from tfscreen.genetics.library_manager import LibraryManager

HERE = os.path.dirname(os.path.abspath(__file__))
MAKE_SIM = os.path.join(HERE, "..", "full-size-sim", "make_sim_config.py")
LOOSE = os.path.join(HERE, "..", "full-size-sim", "growth_priors_loose.csv")
KEEP_RESIDUES = {32, 37, 42, 47, 52, 74, 79, 84, 88, 93}
ARMS = {
    "base": ("realistic", {}),
    "no_cong": ("realistic", {"transformation_poisson_lambda": 0.0}),
    "no_tube": ("realistic", {"tube_noise_sigma": 0.0}),
    "poisson": ("poisson", {}),
    "clean": ("poisson", {"transformation_poisson_lambda": 0.0,
                          "tube_noise_sigma": 0.0}),
}


def shrink_degen(cfg):
    """degen_sites with only KEEP_RESIDUES left degenerate."""
    d, r0 = cfg["degen_sites"], int(cfg["first_amplicon_residue"])
    codons = [d[i:i + 3] for i in range(0, len(d), 3)]
    out = [c if (c == "..." or r0 + i in KEEP_RESIDUES) else "..."
           for i, c in enumerate(codons)]
    return "".join(out)


def shrink(sim_cf, real_lib):
    """The reduced-library simulate config and library config."""
    cf = copy.deepcopy(sim_cf)
    cf["degen_sites"] = shrink_degen(cf)
    ref = LibraryManager(sim_cf).build_library_df()
    tgt = LibraryManager(cf).build_library_df()
    scaled = scale_library_design(ref, tgt, sim_cf["library_mixture"],
                                  sim_cf["transform_sizes"],
                                  total_num_reads=sim_cf["total_num_reads"])
    ratio = scaled["total_num_reads"] / sim_cf["total_num_reads"]
    cf["library_mixture"] = scaled["library_mixture"]
    cf["transform_sizes"] = scaled["transform_sizes"]
    cf["total_num_reads"] = scaled["total_num_reads"]
    if isinstance(cf["cfu0"], dict):
        cf["cfu0"] = {k: float(v) * ratio for k, v in cf["cfu0"].items()}
    else:
        cf["cfu0"] = float(cf["cfu0"]) * ratio
    if cf.get("pcr_template_molecules"):
        cf["pcr_template_molecules"] = max(
            1, int(round(cf["pcr_template_molecules"] * ratio)))
    if cf.get("max_congressed_cells"):
        cf["max_congressed_cells"] = max(
            1000, int(round(cf["max_congressed_cells"] * ratio)))
    lib = copy.deepcopy(real_lib)
    lib["degen_sites"] = shrink_degen(lib)
    n = (tgt["library_origin"] == "double-1-2").sum()
    return cf, lib, ratio, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out_dir", default="m_bias")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    full = {}
    for noise in ("realistic", "poisson"):
        d = os.path.join(args.out_dir, f"_full_{noise}")
        if not os.path.exists(os.path.join(d, "simulate_config.yaml")):
            subprocess.run([sys.executable, MAKE_SIM, "--out_dir", d,
                            "--seed", "1", "--noise", noise,
                            "--calibrate_from", args.pilot], check=True)
        with open(os.path.join(d, "simulate_config.yaml")) as fh:
            full[noise] = (d, yaml.safe_load(fh))
    with open(os.path.join("processed", "library_config.yaml")) as fh:
        real_lib = yaml.safe_load(fh)

    for arm, (noise, changes) in ARMS.items():
        d_full, sim_cf = full[noise]
        cf, lib, ratio, n = shrink(sim_cf, real_lib)
        cf.update(changes)
        for seed in args.seeds:
            d = os.path.join(args.out_dir, f"{arm}_s{seed}")
            os.makedirs(d, exist_ok=True)
            cf["seed"] = seed
            with open(os.path.join(d, "simulate_config.yaml"), "w") as fh:
                fh.write(f"# make_factorial.py arm {arm}: {noise} noise, "
                         f"changes {changes}, seed {seed}\n")
                yaml.safe_dump(cf, fh, sort_keys=False)
            with open(os.path.join(d, "library_config.yaml"), "w") as fh:
                yaml.safe_dump(lib, fh, sort_keys=False)
            for f in ("design.csv", "od600_calibration.yaml"):
                shutil.copyfile(os.path.join(d_full, f), os.path.join(d, f))
            shutil.copyfile(LOOSE, os.path.join(d, "growth_priors_loose.csv"))
        print(f"{arm}: {n} doubles, scale {ratio:.4f}, reads "
              f"{cf['total_num_reads']:.3g}, changes {changes}")


if __name__ == "__main__":
    main()
