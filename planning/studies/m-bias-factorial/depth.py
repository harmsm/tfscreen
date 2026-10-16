"""
Total reads per genotype from a growth table (for score_x.py's depth bins).

    python depth.py sim_realistic_s2/tfs_growth.csv --out depth.csv

Reads the table in chunks (a full-size one is 8 GB).
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("growth_csv")
    ap.add_argument("--out", default="depth.csv")
    args = ap.parse_args()
    tot = None
    for chunk in pd.read_csv(args.growth_csv, usecols=["genotype", "counts"],
                             chunksize=5_000_000):
        s = chunk.groupby("genotype")["counts"].sum()
        tot = s if tot is None else tot.add(s, fill_value=0)
    tot.rename("reads").to_csv(args.out)
    print(f"{len(tot)} genotypes -> {args.out}")


if __name__ == "__main__":
    main()
