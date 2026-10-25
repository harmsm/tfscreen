"""
Extract a compact, analysis-ready copy of the 2026-07-23 real-data snapshot
for the roadmap's step 0 studies (planning/analysis-roadmap.md).

growth.csv in the snapshot is 8.2 GB (one row per sample x genotype, with
every sample-level column repeated on every row). This streams it straight
out of the tarball, without unpacking, and writes:

    counts.parquet   sample, genotype, counts   (one row per sample x genotype)
    samples.parquet  one row per sample: design columns, per-tube OD600 and
                     CFU columns as processed, total reads over the listed
                     genotypes, and the frequency denominator implied by the
                     processed columns (adjusted_counts / frequency), which
                     includes the __unknown__ reads and pseudocounts
    binding.csv, base_growth.csv, presplit.csv   copied as-is

Usage:
    python extract_snapshot.py SNAPSHOT_TAR OUT_DIR
"""

import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "2026-07-23_data-snapshot/data/"

SAMPLE_COLS = ["sample", "replicate", "condition_pre", "t_pre",
               "condition_sel", "t_sel", "titrant_name", "titrant_conc",
               "seq_run", "od600", "library", "cfu_per_mL", "cfu_per_mL_std",
               "detectable", "sample_cfu", "sample_cfu_std"]
ROW_COLS = ["sample", "genotype", "counts", "adjusted_counts", "frequency"]


def _extract_growth(tar, out_dir, chunksize=2_000_000):

    member = tar.getmember(PREFIX + "growth.csv")
    fh = tar.extractfile(member)

    usecols = sorted(set(SAMPLE_COLS + ROW_COLS))
    count_parts = []
    sample_rows = {}
    denom = {}
    total = {}

    reader = pd.read_csv(fh, usecols=usecols, chunksize=chunksize,
                         dtype={"sample": "string", "genotype": "string"})
    n = 0
    for chunk in reader:
        n += len(chunk)
        chunk["condition_pre"] = chunk["condition_pre"].str.strip()
        chunk["condition_sel"] = chunk["condition_sel"].str.strip()

        first = chunk.drop_duplicates("sample")
        for _, r in first[SAMPLE_COLS].iterrows():
            sample_rows.setdefault(r["sample"], r.to_dict())

        d = (chunk["adjusted_counts"] / chunk["frequency"]).groupby(chunk["sample"]).median()
        for s, v in d.items():
            denom.setdefault(s, v)
        t = chunk.groupby("sample")["counts"].sum()
        for s, v in t.items():
            total[s] = total.get(s, 0) + int(v)

        count_parts.append(pd.DataFrame({
            "sample": chunk["sample"].astype("category"),
            "genotype": chunk["genotype"].astype("category"),
            "counts": chunk["counts"].astype(np.int64),
        }))
        print(f"  {n:,} rows", flush=True)

    counts = pd.concat(count_parts, ignore_index=True)
    counts["sample"] = counts["sample"].astype(str).astype("category")
    counts["genotype"] = counts["genotype"].astype(str).astype("category")
    counts.to_parquet(out_dir / "counts.parquet", index=False)

    samples = pd.DataFrame(list(sample_rows.values()))
    samples["reads_listed"] = samples["sample"].map(total)
    samples["freq_denominator"] = samples["sample"].map(denom)
    samples.to_parquet(out_dir / "samples.parquet", index=False)

    print(f"{len(counts):,} count rows, {len(samples)} samples, "
          f"{counts['genotype'].nunique():,} genotypes")


def main(tar_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for name in ["binding.csv", "base_growth.csv", "presplit.csv"]:
            data = tar.extractfile(tar.getmember(PREFIX + name)).read()
            (out_dir / name).write_bytes(data)
        _extract_growth(tar, out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
