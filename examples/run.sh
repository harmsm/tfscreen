#!/bin/bash

# Crash on any failure
set -e

# ---------------------------------------------------------------------------
# Template variables (set by tfs-setup-sim-grid from the template: blocks)
#
# NOTE: tfs-setup-sim-grid already applied the simulate: overrides and wrote
# tfs_sim_config.yaml in this directory.  Per-run config choices (noise level,
# thermodynamic model, random seed, etc.) live there; they do not appear here.
# ---------------------------------------------------------------------------

NUM_REPLICATES={{ num_replicates }}

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

echo ">>> Running simulation"
tfs-simulate tfs_sim_config.yaml . \
    --output_prefix sim_ \
    --num_replicates ${NUM_REPLICATES}
