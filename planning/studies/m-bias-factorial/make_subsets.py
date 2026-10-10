"""
Fit-side subsets of a full-size simulation (m-bias factorial, round 2).

    python make_subsets.py sim_realistic_s2 --out_dir m_subsets

Each arm refits the same simulated growth table with only some genotypes.
Given the shared parameters (k, m, tube offsets, population hypers) each
genotype's count likelihood is independent, so a subset is a valid fit of
the same data; what changes is which genotypes inform k and m.

    no_doubles       wt, spikes and singles
    deep_doubles     no_doubles plus every double with >= DEEP total reads
    shallow_doubles  no_doubles plus as many doubles below DEEP, at random

Each arm directory gets the filtered tfs_growth.csv, links to the
simulation's tfs_sim_* files (truth, and so run_arm.sh skips simulate and
process), and copies of simulate_config.yaml (its seed), library_config.yaml
and growth_priors_loose.csv. Submit run_arm.srun from inside each.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOOSE = os.path.join(HERE, "..", "full-size-sim", "growth_priors_loose.csv")
DEEP = 1000
CHUNK = 5_000_000


def genotype_class(g):
    g = g.astype(str)
    return np.select([g == "wt", g.str.count("/") == 1], ["wt", "double"],
                     "single")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--out_dir", default="m_subsets")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    growth = os.path.join(args.sim_dir, "tfs_growth.csv")

    tot = None
    for chunk in pd.read_csv(growth, usecols=["genotype", "counts"],
                             chunksize=CHUNK):
        s = chunk.groupby("genotype")["counts"].sum()
        tot = s if tot is None else tot.add(s, fill_value=0)
    cls = pd.Series(genotype_class(tot.index.to_series()), index=tot.index)
    doubles = tot[cls == "double"]
    deep = doubles.index[doubles >= DEEP]
    shallow_all = doubles.index[doubles < DEEP]
    rng = np.random.default_rng(args.seed)
    shallow = rng.choice(shallow_all, size=min(len(deep), len(shallow_all)),
                         replace=False)
    core = tot.index[cls != "double"]
    arms = {"no_doubles": set(core),
            "deep_doubles": set(core) | set(deep),
            "shallow_doubles": set(core) | set(shallow)}
    print(f"{len(core)} wt/spike/single genotypes, {len(deep)} doubles >= "
          f"{DEEP} reads, {len(shallow)} of {len(shallow_all)} below")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {}
    for arm in arms:
        d = os.path.join(args.out_dir, arm)
        os.makedirs(d, exist_ok=True)
        out[arm] = os.path.join(d, "tfs_growth.csv")
        if os.path.exists(out[arm]):
            os.remove(out[arm])
    for chunk in pd.read_csv(growth, chunksize=CHUNK):
        for arm, keep in arms.items():
            sub = chunk[chunk["genotype"].isin(keep)]
            sub.to_csv(out[arm], mode="a", index=False,
                       header=not os.path.exists(out[arm]))

    sim = os.path.abspath(args.sim_dir)
    for arm in arms:
        d = os.path.join(args.out_dir, arm)
        for f in os.listdir(sim):
            if f.startswith("tfs_sim_"):
                link = os.path.join(d, f)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(sim, f), link)
        for f in ("simulate_config.yaml", "library_config.yaml"):
            shutil.copyfile(os.path.join(sim, f), os.path.join(d, f))
        shutil.copyfile(LOOSE, os.path.join(d, "growth_priors_loose.csv"))
        n = pd.read_csv(out[arm], usecols=["genotype"])["genotype"].nunique()
        print(f"{arm}: {n} genotypes")


if __name__ == "__main__":
    main()
