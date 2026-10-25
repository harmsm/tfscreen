---
title: The tube-offset mode and the growth transition at the switch to selection
status: idea
filed: 2026-10-05
area: tfmodel
revisit_when: >-
  After pipeline plan steps 6 and 7 (the simulator's raw formats, the
  documented chain and the offset diagnostic), as part of the science
  prioritization in pipeline step 8 (user, 2026-10-05). Its simulation
  steps should use the raw-format path. Keep the step list current while
  it is worked.
related:
  - planning/studies/staged-map/README.md
  - planning/experiment-pipeline.md
  - planning/analysis-roadmap.md
  - src/tfscreen/tfmodel/inference/staged_map.py
  - src/tfscreen/tfmodel/generative/components/growth_transition/
  - src/tfscreen/tfmodel/generative/components/sample_offset/level.py
  - src/tfscreen/simulate/growth/transition_linkage.py
---

## Why

The staged MAP (`tfs-fit-model --stage_offsets`, pipeline plan step 3) was
built to keep a level-offset fit out of what we called the offset trap, a
mode where the tube offsets carry the population's growth. Run 2 of
`planning/studies/staged-map/`, with every MAP run to convergence on the
exact full-batch loss, showed that the trap is not an optimizer artifact.
It is the best point found so far of the model's objective.

- `ref_refit`, the hand-chained reference fit run to convergence, ended
  7.9e4 nats (log joint) above the staged MAP and 4.5e5 above where it
  started, and it got there by moving into the offset mode: offsets up to
  -4.2, 10-25 prior SDs each at `sigma_fixed` 0.17, with growth k and m
  unphysical (kanR+kan k 0.036, m -0.050; about 0.01 elsewhere).
- The offsets are not noise. In both selection conditions they are a smooth
  function of IPTG, with opposite signs (kanR selection about +2 at 0 mM to
  -3 to -4 at 1 mM; pheS selection -1 to +1.9), nearly the same at all three
  time points, and near 0 in the no-selection conditions.
- The staged MAP's offsets are small and patternless (SD 0.21) and its k
  uniform (about 0.027). It reaches that point only because it stops in a
  local optimum. A longer run, a better optimizer or a multi-start would
  leave it.

An offset that is constant in time and structured by IPTG and selection is
a population-level shift set before the first sampled time point. IPTG is
added at `-t_pre` (user, 2026-10-05): the pre-growth exists so the cells
express their markers before selection starts, so the shift is not
"marker not yet expressed". It points at the selection onset instead: a
transient when the selection agent arrives (a kanamycin lag or die-off of
low-resistance cells, a delay before 4CP takes effect) whose size depends on
the marker level, hence on IPTG. The `instant` growth transition cannot
produce such a transient, so the offsets supply it. This is roadmap step 7b
arriving from the real data.

Pipeline plan step 3 is closed as machinery: the staged MAP and
exact-loss convergence work, and the staged fit is the physical one. This
plan carries the model problem. Until it is solved, a longer or better fit
can leave the physical basin; pipeline step 7's tube-offset diagnostic
flags a fit that does.

## Done when

- One growth-transition model (or `instant` with an explicit reason) makes
  the physical mode the best-scoring one on the dev data: a cold fit, the
  staged fit and a refit from the offset mode all end at the same optimum,
  with tube offsets small and without IPTG structure.
- The same holds on a simulation with a known selection-onset transient,
  and `instant` is kept where the simulation has none.
- Roadmap step 7b's purge is done for the models not kept.

## Steps

1. [ ] **Where the offset mode's likelihood comes from** (local, existing
   runs). Extend `planning/studies/staged-map/score_map.py` to report the
   count log-likelihood by tube and by genotype class, and compare
   `ref_refit` with `staged_auto2`. If the 9e4-nat gain sits in the
   selection tubes, spread across the library and concentrated at the
   first sampled time point, the transient explanation stands; if it sits
   in a few genotypes or tubes, look there instead.
2. [ ] **Simulation with a known transient.** Simulate a small library with
   a selection-onset transient (`two_pop` or `baranyi` in
   `simulate/growth/transition_linkage.py`) whose size depends on theta.
   Fit `instant` with level offsets: does the IPTG-structured offset mode
   appear and score best? Fit the matching transition: does it vanish? A
   control simulation with no transient should keep `instant`.
3. [ ] **Dev-data grid.** Transition `instant`, `two_pop` and one more if
   step 2 argues for it, each fit three ways: staged, cold, and refit from
   `ref_refit`'s point (`--init_from`). Score every end point with
   `score_map.py`; report offsets by tube design and k/m. About 9 runs of
   about 4 h. Fix `two_pop`'s fallback for `g_pre - g_sel - k_trans <= 0`
   first if it is a candidate (roadmap 7b).
4. [ ] **Decide and purge** (roadmap 7b, D12, D13). Keep `instant` and at
   most one transition model; remove the rest from the fit, the simulator,
   examples, grid `auto` enumeration, tests, docs and CLAUDE.md.
5. [ ] **Confirm the staged MAP under the chosen model.** Rerun the
   staged-map comparison: the staged MAP should reach the best-scoring
   mode, and cold and refit starts should not beat it. Then compare spike
   K and n between converged fits.
6. [ ] **Revisit the growth-only defaults** (moved from pipeline step 2):
   the held log(n), X_low, X_delta and log K population SDs and the offset
   SD 0.17, under the chosen transition model.

## Also open

- **The slow tail.** Every converged MAP stopped on the `loss_rtol` floor
  (61 nats per window on the dev data) while still descending about 40
  nats per window at step size 1e-6, after 1-1.2 M steps (4 h). The floor
  sets where they stop. If that matters for the decision above, a final
  full-batch second-order polish (L-BFGS on `full_batch_loss`) is the
  proper fix; decide after step 3 shows whether end points move.
- **The offsets' own freedom.** An i.i.d. N(0, 0.17) prior cannot stop
  offsets from absorbing any pattern shared by every genotype in a tube,
  because each tube's likelihood outweighs its one prior term by orders of
  magnitude. If no transition model removes the pattern, constrain the
  offsets structurally instead (no IPTG or time structure) or let the
  population model (roadmap step 6) carry the tube totals.

## Out of scope

- The orchestrator (pipeline plan step 5), deferred.
