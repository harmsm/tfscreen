---
title: Honest intervals for the well-measured genotypes
status: active
filed: 2026-10-09
area: tfmodel
revisit_when: >-
  Active. Goal: what this experiment's data support, without new
  sequencing (user, 2026-10-09).
related:
  - planning/studies/km-anchor/README.md
  - planning/studies/full-size-sim/README.md
  - planning/experiment-pipeline.md
  - planning/offset-mode-growth-transition.md
  - src/tfscreen/tfmodel/priors_edit.py
  - src/tfscreen/tfmodel/inference/posteriors.py
---

## Why

On the full-size simulation calibrated to the real screen's depth
(`planning/studies/full-size-sim/`, 2026-10-09), genotypes above 1e3 reads
rank well (X r 0.93-0.99) but their 95% intervals cover 0.56 down to 0.08
with depth (realistic noise), and worse under Poisson counts. The error is
shared: X is compressed (|m| 30-45% too large) and shifted (k slid up
against dk_geno), and the arrowhead Laplace holds exactly those directions
at the MAP, so k and m carry no interval. About 77,000 real doubles have
more than 1e3 reads; these are the curves the manuscript can use, and
their intervals have to be honest.

The real fit is off the same way in kanR+kan (k 0.030 and m -0.020 against
the wt monoculture's 0.015 and -0.014, as in the simulation, whose truth
is the monoculture), so the real kanR curves are likely compressed too.

## Assumptions

- Ranking is not the problem; the absolute X scale (k and m) is. Counts
  pin only relative frequencies; the tube totals pin the scale weakly
  through free per-tube offsets.
- The wt monoculture rates are the experiment's independent measurement of
  that scale, with their own error (SE of the replicate mean, at least
  0.002 per minute day to day).
- No new sequencing. New fits, and new simulations only to test.

## Steps

1. [ ] **Anchor k and m with the monoculture rates**
   (`planning/studies/km-anchor/`). Three fit-only arms: the realistic
   simulation with the monoculture priors, the same with the rates
   perturbed by their own error, and the real data with the monoculture
   priors. Read: deep X coverage and slope against the baseline, k/m
   against truth, held directions, and on the real data whether k and m
   move to the monoculture with the offsets staying clean (or whether the
   data fight the anchor, which would point at a real in-pool difference
   such as the selection-onset transient).
2. [ ] **An interval for the held directions.** If directions are still
   held with k and m anchored, floor them at the prior instead of holding
   them, so k and m carry the monoculture's uncertainty into every X
   interval. Check coverage by depth again.
3. [ ] **pheS+4CP.** The real fit's pheS selection departs from the
   monoculture differently (k much higher, |m| smaller). Decide with step
   1's real arm whether the anchor holds there or the condition needs the
   growth-transition work first.
4. [ ] **Then the shallow genotypes' points.** Hold the X population SDs
   (`theta_{X_low,X_delta,log_hill_K}_hyper_scale_fixed`) so genotypes
   below 1e3 reads shrink toward the population instead of scattering;
   their intervals already cover.

## Out of scope

New sequencing or a new screen design (`planning/design-noise-sweep.md`),
and the simulator's remaining depth mismatches (dropout tail, wt's share of
reads), which do not touch the deep genotypes.
