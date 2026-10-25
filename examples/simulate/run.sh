#!/bin/bash

# Crash on any failure
set -e

# ---------------------------------------------------------------------------
# Template variables (set by tfs-setup-sim-grid from the template: blocks)
#
# NOTE: tfs-setup-sim-grid already applied the simulate: overrides and wrote
# tfs_sim_config.yaml in this directory.  Per-run config choices (noise level,
# thermodynamic model, seed, etc.) live there; they do not appear here.
# ---------------------------------------------------------------------------

NUM_REPLICATES={{ num_replicates }}

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

echo ">>> Running simulation"
# Writes tfs_sim_growth.csv, tfs_sim_genotype_theta.csv, ... in this
# directory. Keep the tfs_sim prefix: tfs-summarize-fit finds the ground
# truth as *_sim_genotype_theta.csv.
tfs-simulate tfs_sim_config.yaml \
    --out_prefix tfs_sim \
    --num_replicates ${NUM_REPLICATES}
