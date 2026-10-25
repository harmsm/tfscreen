---
title: Plasmid segregation ("curing") of multiple transformants in low-copy designs
status: idea
filed: 2026-09-13
area: experiment
revisit_when: >-
  When designing or analyzing the next experiment, which moves to a low-copy
  plasmid.
related:
  - planning/congression-physics-plan.md
  - src/tfscreen/simulate/selection_experiment.py
---

**Context:** The congression model assumes co-transformed plasmids persist in
a cell for the whole experiment. That is reasonable for the current high-copy
plasmids over roughly 3 to 7 doublings. The next experimental design switches
to a low-copy plasmid.

**Idea:** Two plasmids with the same origin are incompatible: under no
selection that distinguishes them, a lineage drifts toward carrying only one
type, faster at low copy number. Two consequences:

1. **Modeling:** congression effects would fade over time, so the congressed
   fraction and each cell's variant shares `x_g` become functions of the
   number of generations. The observable-level mixture in the congression plan
   would need time-dependent class weights.
2. **Design tradeoff:** extra rounds of replication before selection would
   "cure" multiple transformants toward single-genotype cells, but outgrowth
   also costs library diversity (bottlenecks, drift, growth-rate bias against
   slow genotypes). There should be an optimal number of pre-selection
   doublings that balances curing against diversity.

**Why not now:** The current dataset uses high-copy plasmids, so the
persistence assumption holds.

**What it would take:**
- A segregation model for plasmid copy number n per cell (for example random
  partitioning at division), giving the probability a co-transformed lineage is
  still mixed after g doublings.
- Simulation: extend `_sim_transform`/`_sim_growth` with per-cell share drift.
- Design calculation: congression remaining vs. library diversity retained, as
  a function of pre-selection doublings, for candidate copy numbers.
