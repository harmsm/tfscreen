"""
Simulate configs for the joint-fit simulation test (README).

    python make_sim.py BASE_CONFIG OUT_DIR --arm joint|null [--seed 1]

BASE_CONFIG is a full-size realistic simulate config
(planning/studies/full-size-sim/make_sim_config.py; on the cluster
full-sized-sims-3/sim_realistic_s2/simulate_config.yaml). Writes
OUT_DIR/simulate_config.yaml and OUT_DIR/spike_hill.csv: the base config
with its relative paths made absolute, plus

- binding for every spike at the lab's 2 uM IPTG grid, from Hill curves
  in spike_hill.csv (wt at the simulator's reference; the others drawn
  here, seeded), noise 0.04 (the scatter within one real curve);
- arm joint: kanR+kan's growth a switch-like function of occupancy
  (warp a = 8, x0 = 0.5), pheS+4CP's graded (a = 1.5, x0 = 0.4), growth
  seeing IPTG at s = 0.5 times nominal, and one binding offset per spike
  (SD 0.1);
- arm null: none of those (linear growth, s = 1, no offsets).

The spikes' curves come from a params file so that their Hill truth is
the curve simulated (the stratified path pastes pool curves over the
spikes without patching tfs_sim_parameters.csv).
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from tfscreen.genetics import library_composition_table

# The lab's 2 uM binding grid (mM), from the 2026-10-15 binding table.
BINDING_IPTG = [0.0, 1e-6, 6e-6, 1e-5, 6e-5, 1e-4, 6e-4, 1e-3, 6e-3, 0.01,
                0.06, 0.1, 0.6, 1.0, 3.0, 5.0]
BINDING_NOISE = 0.04

ARMS = {
    "joint": dict(warp={"kanR+kan": (8.0, 0.5), "pheS+4CP": (1.5, 0.4)},
                  growth_titrant_scale=0.5, genotype_offset_sd=0.1),
    "null": dict(warp={}, growth_titrant_scale=None, genotype_offset_sd=0.0),
}

# The simulator's hill_geno wt reference (get_sim_hyperparameters).
WT = dict(theta_low=0.99, theta_high=0.01, log_hill_K=-4.1, hill_n=2.0)


def absolute_paths(cf, base_dir):
    for keys in (("design",), ("od600", "calibration"), ("library_genotypes",),
                 ("thermo_data",)):
        d = cf
        for k in keys[:-1]:
            d = d.get(k) or {}
        v = d.get(keys[-1]) if isinstance(d, dict) else None
        if isinstance(v, str) and not os.path.isabs(v):
            d[keys[-1]] = os.path.abspath(os.path.join(base_dir, v))
    return cf


def spike_curves(spikes, seed):
    """wt at the reference; each other spike a random repressor-like curve."""
    rng = np.random.default_rng([seed, 2026])
    rows = []
    for g in spikes:
        if g == "wt":
            rows.append(dict(genotype=g, **WT))
            continue
        low = rng.uniform(0.3, 0.99)
        rows.append(dict(genotype=g, theta_low=low,
                         theta_high=rng.uniform(0.01, low),
                         log_hill_K=rng.normal(-4.1, 1.0),
                         hill_n=rng.uniform(1.0, 3.0)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("out_dir")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(args.base_config) as fh:
        cf = yaml.safe_load(fh)
    cf = absolute_paths(cf, os.path.dirname(os.path.abspath(args.base_config)))
    cf["seed"] = args.seed
    os.makedirs(args.out_dir, exist_ok=True)

    lib = library_composition_table(args.base_config)
    spikes = sorted(set(lib.loc[lib["in_spiked_origin"], "genotype"].astype(str)))
    hill_path = os.path.abspath(os.path.join(args.out_dir, "spike_hill.csv"))
    spike_curves(spikes, args.seed).to_csv(hill_path, index=False)

    arm = ARMS[args.arm]
    for cond, (a, x0) in arm["warp"].items():
        if cond not in cf["growth"]:
            raise ValueError(f"{cond} is not a growth condition of the base config")
        cf["growth"][cond]["warp_a"] = a
        cf["growth"][cond]["warp_x0"] = x0
    if arm["growth_titrant_scale"] is not None:
        cf["growth_titrant_scale"] = arm["growth_titrant_scale"]
    cf["binding_data"] = {
        "titrant_name": "iptg",
        "titrant_conc": BINDING_IPTG,
        "noise": BINDING_NOISE,
        "genotype_offset_sd": arm["genotype_offset_sd"],
        "spiked_binding": {"choose_by": hill_path},
    }

    out = os.path.join(args.out_dir, "simulate_config.yaml")
    with open(out, "w") as fh:
        fh.write(f"# joint-fit simulation test, arm {args.arm}, seed {args.seed} "
                 f"(planning/studies/joint-fit-sim)\n")
        yaml.safe_dump(cf, fh, sort_keys=False)
    print(f"wrote {out} and {hill_path} ({len(spikes)} spikes: {', '.join(spikes)})")


if __name__ == "__main__":
    main()
