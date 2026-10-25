---
title: Estimate the dk_geno combining exponent (alpha) by varying congression
status: idea
filed: 2026-09-13
area: experiment
revisit_when: >-
  After the congression physics work settles on the soft-min dk_geno rule and
  the current single-lambda dataset is published.
related:
  - planning/congression-physics-plan.md
  - src/tfscreen/simulate/selection_experiment.py
  - src/tfscreen/tfmodel/generative/components/transformation/mixture.py
---

**Context:** While fixing the simulator/fit mismatch in how co-resident
plasmids combine growth effects, we chose to combine each physical quantity at
its own level. For the pleiotropic growth cost, a cell carrying variants with
dimer shares `x_g` gets

```
dk_cell = -(1/alpha) * log( sum_g x_g * exp(-alpha * dk_g) )
```

`alpha -> 0` is dilution (share-weighted mean burden); `alpha -> inf` is
dominance (the worst variant sets the cost). Burden that scales with the amount
of misfolded protein points toward small alpha; aggregation or mixed-oligomer
poisoning points toward large alpha.

**Idea:** Estimate alpha empirically with a screen run at several deliberately
different congression levels (lambda). Alpha only acts through co-resident
cells, so a single low lambda (0.36 in our dataset, where 70% of a genotype's
cells have no co-resident) leaves it weakly identified. Varying lambda changes
the co-resident fraction and the burden mixture it sees while the per-genotype
dk_geno values stay fixed.

**Why not now:** Our dataset has a single lambda, so alpha is set from biology
and checked by sensitivity (alpha = 0 vs. large), not fit. The experiment is a
separate project.

**What it would take:**
- Wet lab: the same library transformed at 2 to 3 lambda values (for example
  0.1 / 0.4 / 1.0), with lambda measured independently for each.
- Model: alpha as a shared latent with lambda-indexed transformation parameters.
- Simulation first, to find the lambda spacing that identifies alpha.
