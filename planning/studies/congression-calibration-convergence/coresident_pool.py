"""
Composition of the co-resident pool the mixture draws congressed classes
from, and wt's share of simulated cells.

The pool is the bulk share (pool_fraction * bulk_fraction) of genotypes with
growth data, __unknown__ excluded, as in
ModelOrchestrator._coresident_pool. Also reports how many wt entries each
sub-library enumerates (tfs_sim_library.csv) and wt's share of presplit
cells per replicate.

Run from this directory. GRID names a pulled grid directory (default: the
anchored grid inside the congression-calibration study); the argument is a
run-number prefix (default: run_0002). The pool depends only on the library,
so any run of the grid gives the same answer.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.environ.get("GRID", os.path.join(
    HERE, "..", "congression-calibration", "congression_calibration_anchor"))
RUN = sys.argv[1] if len(sys.argv) > 1 else "run_0002"

run_dir = glob.glob(os.path.join(GRID, RUN + "_*"))[0]


def path(name):
    return os.path.join(run_dir, name)


lib = pd.read_csv(path("tfs_configure_library.csv"))
lib["genotype"] = lib.genotype.astype(str)
grown = set(pd.read_csv(path("tfs_sim_growth.csv"),
                        usecols=["genotype"]).genotype.astype(str))
lib = lib[lib.genotype.isin(grown) & (lib.genotype != "__unknown__")].copy()
lib["share"] = lib.pool_fraction * lib.bulk_fraction
lib["share"] /= lib.share.sum()

print(f"=== {os.path.basename(run_dir)}")
print("Co-resident pool, largest shares:")
print(lib.sort_values("share", ascending=False).head(6)[
    ["genotype", "pool_fraction", "bulk_fraction", "share"]].to_string(index=False))

sets = yaml.safe_load(open(path("tfs_configure_config.yaml")))
sets = sets["components"]["congression_sets"]
slots = sum(num_sets * (n + 1) for n, num_sets in enumerate(sets))
wt = lib.set_index("genotype").share["wt"]
print(f"\nwt share {wt:.3f}: {slots * wt:.1f} of {slots} co-resident slots per "
      f"genotype are wt (congression_sets {sets})")

sim_lib = pd.read_csv(path("tfs_sim_library.csv"))
sim_lib["genotype"] = sim_lib.genotype.astype(str)
totals = sim_lib.groupby("library_origin").degeneracy.sum()
wt_rows = sim_lib[sim_lib.genotype == "wt"].set_index("library_origin")
print("\nwt entries per sub-library (enumerated codon combinations):")
for origin, row in wt_rows.iterrows():
    print(f"  {origin:12s} {row.degeneracy:6d} of {totals[origin]:6d} "
          f"({row.degeneracy / totals[origin]:.2f})")

presplit = pd.read_csv(path("tfs_sim_presplit.csv"))
presplit["genotype"] = presplit.genotype.astype(str)
presplit["cfu"] = np.exp(presplit.ln_cfu_0_true)
keys = ["replicate", "condition_pre"]
share = (presplit[presplit.genotype == "wt"].groupby(keys).cfu.sum()
         / presplit.groupby(keys).cfu.sum())
print("\nwt share of presplit cells (true ln_cfu0):")
print(share.round(3).to_string())
