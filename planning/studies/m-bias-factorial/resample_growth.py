"""
Filter or resample a growth table by tube (planning/deep-coverage.md).

    python resample_growth.py tfs_growth.csv --library kanR --out kanR/tfs_growth.csv
    python resample_growth.py tfs_growth.csv --drop_frac 0.2 --seed 3 --out boot_03/tfs_growth.csv

--library keeps one library's tubes (the kanR-only check). --drop_frac
drops that share of the tubes at random, stratified by condition_sel and
titrant_conc and keeping at least one tube per stratum: a delete-d
jackknife. Tubes cannot be resampled with replacement, since a duplicated
tube lands on the same tensor cell as the original. For a delete-d
jackknife with n tubes, d dropped and N subsamples, the variance of an
estimate is (n - d) / d times the mean squared deviation of the
subsample estimates from their mean (boot_summary.py applies it).

Writes the kept tubes' rows and prints n, d and the dropped tubes.
Reads in chunks (a full-size table is 8 GB).
"""

import argparse
import os

import numpy as np
import pandas as pd

CHUNK = 5_000_000


def tube_table(path):
    """One row per tube: sample, library, condition_sel, titrant_conc."""
    cols = ["sample", "library", "condition_sel", "titrant_conc"]
    parts = [c.drop_duplicates() for c in
             pd.read_csv(path, usecols=cols, chunksize=CHUNK)]
    return pd.concat(parts).drop_duplicates("sample").reset_index(drop=True)


def choose_drops(tubes, frac, seed):
    """Tubes to drop: frac of each condition_sel x titrant_conc stratum,
    rounded, leaving at least one."""
    rng = np.random.default_rng(seed)
    drop = []
    for _, g in tubes.groupby(["condition_sel", "titrant_conc"]):
        k = min(int(round(frac * len(g))), len(g) - 1)
        if k > 0:
            drop.extend(rng.choice(g["sample"].to_numpy(), size=k,
                                   replace=False))
    return set(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("growth_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--drop_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    tubes = tube_table(args.growth_csv)
    if args.library is not None:
        if args.library not in set(tubes["library"]):
            raise SystemExit(f"no library {args.library!r}: "
                             f"{sorted(set(tubes['library']))}")
        tubes = tubes[tubes["library"] == args.library]
    drop = choose_drops(tubes, args.drop_frac, args.seed) if args.drop_frac else set()
    keep = set(tubes["sample"]) - drop

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    for chunk in pd.read_csv(args.growth_csv, chunksize=CHUNK):
        chunk[chunk["sample"].isin(keep)].to_csv(
            args.out, mode="a", index=False, header=not os.path.exists(args.out))
    print(f"n {len(tubes)} tubes, dropped d {len(drop)}, kept {len(keep)}"
          + (f" (library {args.library})" if args.library else ""))
    if drop:
        print("dropped:", " ".join(sorted(drop)))


if __name__ == "__main__":
    main()
