---
title: Which noise source to buy down in the next screen (PCR templates, cells per tube)
status: idea
filed: 2026-10-09
area: experiment
revisit_when: >-
  When the next experiment is being designed (students' simulations for the
  NSF preliminary data, due 2026-11-20), and after the deep genotypes'
  coverage is fixed on this experiment's data, so the sweep's coverage
  numbers mean something.
related:
  - planning/studies/full-size-sim/README.md
  - planning/studies/full-size-sim/make_sim_config.py
  - planning/studies/noise-anatomy/
---

**Context.** The calibrated full-size simulation of the dev-data screen
(2026-10-09, `planning/studies/full-size-sim/`, arms `sim_realistic_s2` and
`sim_poisson_s2`) matches the real screen's read depth. The two arms differ
only in count noise, and the difference is large where most real doubles
sit: at 100-1,000 total reads (about 90,000 doubles) X correlates with the
truth at 0.96 under Poisson counts and 0.62 under the realistic noise; at
100 reads or fewer, 0.72 against 0.26. The realistic noise is founder
sampling plus a PCR template bottleneck, calibrated to the real counts'
5-18x overdispersion (`planning/studies/noise-anatomy/`): templates at a
tenth of the reads, which by itself multiplies count variance about
13-fold.

**Idea.** Sweep the noise sources on this simulation to say what each buys
before the next screen: `pcr_template_molecules` (more template DNA into
PCR), `cfu0` (more cells per tube, so less founder noise), and
`total_num_reads` for comparison. Score X recovery and coverage by depth
bin, especially at 100-1,000 reads, against the cost of each change at the
bench.

**Why not now.** The immediate goal is what this experiment's data support
without new sequencing: first the deep genotypes' coverage, which is lost
to shared k/m errors with no interval. The sweep's coverage numbers inherit
that problem until it is fixed.

**What it would take.** `make_sim_config.py` already writes the realistic
config; add arms with templates at 1x, 3x and 10x the current value, `cfu0`
at 3x and 10x, and reads at 2x. Each arm is one simulation (about 15
minutes locally) plus the cluster fit. Students can run the grid once the
fit's k/m handling is settled. Open: whether the realistic noise model's
split between founder and PCR noise matches the real one (the noise-anatomy
study fit their sum to the counts, not each part).
