---
title: Estimate a genotype's spike/bulk fraction from ln_cfu curvature
status: idea
filed: 2026-09-13
area: tfmodel
revisit_when: >-
  If design-computed spike/bulk fractions (f_g) look wrong for real data, or
  after the observable-level mixture (congression plan step 3) is in place.
related:
  - future/congression-physics-plan.md
---

**Context:** In the congression plan, each genotype's cells mix clean and
congressed classes, weighted by `f_g`, the fraction of its cells from the bulk
library. We compute `f_g` from the library design (`transform_sizes`,
`library_mixture`, codon degeneracy).

**Idea:** Two subpopulations growing at different rates give a
`ln_cfu(t)` that is not linear in time. A genotype that is genuinely clean
grows along a straight line; a mixed one curves. With enough timepoints that
curvature carries information about `f_g`, and could check or refine the
design-based value for real data where transformation efficiencies are
uncertain.

**Why not now:** `f_g` from the design is good enough to start, the curvature
signal is weak over 3 timepoints per condition, and future runs will
distinguish spikes by codon, which measures purity directly.

**What it would take:** The observable-level mixture from the plan, then a
`f_g` prior centered on the design value (for example a Beta) instead of a
fixed value, plus a simulation to see whether it is identifiable.
