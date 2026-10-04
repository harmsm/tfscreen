"""
Write a synthetic experiment in the formats tfs-process-counts reads.

    python make_example_data.py
    tfs-process-counts tube_table.csv counts \
        --od600_file od600.csv \
        --od600_calibration_file ../od600/od600_calibration.yaml \
        --tube_volume_mL 5 --out_prefix growth

It writes:

- ``tube_table.csv``: one row per sequenced tube. ``sample`` names the tube
  (and its count file); ``library`` is the transformed library it came from
  (kanR and pheS are transformed and grown up separately); the rest is the
  design: ``replicate``, ``condition_pre``, ``t_pre``, ``condition_sel``,
  ``t_sel`` (minutes), ``titrant_name``, ``titrant_conc`` (mM).
- ``od600.csv``: one OD600 reading per tube (``sample``, ``od600``).
- ``counts/counts_<sample>.csv``: ``genotype``, ``counts``, the format
  tfs-process-fastq writes, including the ``__unknown__`` row of reads that
  matched no library genotype.

The numbers are made up: a few dozen genotypes of the library in
``library_config.yaml`` growing at random rates, read at about 50,000 reads
per tube. They show the formats; they are not a realistic screen.
"""

import os

import numpy as np
import pandas as pd

from tfscreen.genetics import library_composition_table

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261003)

# Genotypes: wt, the spikes and a random handful of library members.
lib = library_composition_table(os.path.join(HERE, "library_config.yaml"))
spikes = list(lib.loc[lib["in_spiked_origin"], "genotype"])
others = [g for g in lib["genotype"] if g != "wt" and g not in spikes]
genotypes = ["wt"] + spikes + list(rng.choice(others, size=30, replace=False))
rate_shift = dict(zip(genotypes, rng.normal(0.0, 0.004, len(genotypes))))
rate_shift["wt"] = 0.0

# Design: two libraries, each with a selection and a control condition, two
# IPTG concentrations and three time points, one replicate.
rows = []
for library, marker, agent in (("kanR", "kanR", "kan"), ("pheS", "pheS", "4CP")):
    for sel in (f"{marker}+{agent}", f"{marker}-{agent}"):
        for conc in (0.0, 1.0):
            for t_sel in (0, 60, 120):
                name = f"{library}_{sel.replace('+', 'p').replace('-', 'm')}" \
                       f"_{conc:g}mM_t{t_sel}"
                rows.append({"sample": name, "library": library,
                             "replicate": 1,
                             "condition_pre": f"{marker}-{agent}", "t_pre": 30,
                             "condition_sel": sel, "t_sel": t_sel,
                             "titrant_name": "iptg", "titrant_conc": conc})
tubes = pd.DataFrame(rows)
tubes.to_csv(os.path.join(HERE, "tube_table.csv"), index=False)

# OD600 grows with time; every reading inside the example calibration's range.
od = 0.15 * np.exp(0.006 * tubes["t_sel"].to_numpy()) \
    * rng.lognormal(0.0, 0.03, len(tubes))
pd.DataFrame({"sample": tubes["sample"], "od600": np.round(od, 4)}).to_csv(
    os.path.join(HERE, "od600.csv"), index=False)

# Counts: each genotype's share grows by its own rate offset; reads are a
# multinomial draw with 5% of reads matching no genotype.
start = rng.dirichlet(np.full(len(genotypes), 5.0))
counts_dir = os.path.join(HERE, "counts")
os.makedirs(counts_dir, exist_ok=True)
shift = np.array([rate_shift[g] for g in genotypes])
for _, tube in tubes.iterrows():
    w = start * np.exp(shift * (tube["t_pre"] + tube["t_sel"]))
    w = 0.95 * w / w.sum()
    reads = rng.multinomial(50_000, np.append(w, 0.05))
    pd.DataFrame({"genotype": genotypes + ["__unknown__"],
                  "counts": reads}).to_csv(
        os.path.join(counts_dir, f"counts_{tube['sample']}.csv"), index=False)

print(f"Wrote tube_table.csv, od600.csv and {len(tubes)} count files.")
