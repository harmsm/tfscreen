---
title: Return to a joint fit of growth and binding, with per-marker maps
status: active
filed: 2026-10-16
area: tfmodel
revisit_when: >-
  Agreed 2026-10-16. Supersedes the open steps of planning/relative-m.md and is roadmap
  step 9 (planning/analysis-roadmap.md) with step 8's map measured.
related:
  - planning/growth-only-relative-fit.md
  - planning/analysis-roadmap.md
  - planning/analysis-roadmap-summary.md
  - planning/deep-coverage.md
  - planning/two-stage-fit.md
  - planning/relative-m.md
  - planning/studies/real-data-fit/README.md
  - planning/studies/m-bias-factorial/README.md
  - planning/studies/two-stage-validation/README.md
  - planning/studies/x-vs-binding/README.md
  - planning/studies/within-tube-m/README.md
  - planning/studies/relative-m/README.md
  - src/tfscreen/tfmodel/generative/components/growth/linear.py
  - src/tfscreen/tfmodel/generative/components/theta/hill_geno.py
  - src/tfscreen/tfmodel/model_orchestrator.py
---

## Review, 2026-10-18

Step 6's first run and a review of the planning files and git log
(`planning/findings-ledger.md`) changed the footing of this plan:

- The 2 uM agreement in "The arc" item 5 was in a gauge built from wt's own
  binding endpoints, which absorbs any affine level and scale difference
  between binding and growth. In the absolute model the difference
  reappears (step 6, run 1: every binding offset negative).
- The arc starts on 2026-09-26 and misses July's work: about 20 in-library
  binding anchors already pinned m then, and lambda already ran away under
  a model conflict.
- The binding-to-growth link has failed three times (the 2026-09 joint fit,
  the 2026-10-01 "no map" with M42I's inverted K, and step 6 run 1), so
  binding cannot yet set theta's absolute scale.

Steps 6-8 are paused until the link is decided (the ledger's Open item 1).
The simulation results (step 5) stand as a test of the model under its own
assumptions.

## Why

We split binding from growth on 2026-09-26 because the joint fit asserted
more than the data could support, and step 8 of the roadmap then found "no
map" between growth and binding. Three weeks of work since have removed
most of the reasons for that verdict and found the pieces the original plan
said would be needed. Absolute occupancy is also what the congression
correction (the partition-function rules) and the long-term goal (learning
thermodynamic ensembles from these data) need. This plan brings binding
back, at its proper weight, with the map we have now measured.

## The arc

**1. The split (2026-09-26).** The joint fit asserted that in vitro binding
equals in vivo occupancy and that growth is linear in it, and its default
binding weight (`N_growth_rows / N_binding_rows`) made binding count as much
as the whole growth dataset, so any mismatch distorted growth
(`planning/growth-only-relative-fit.md`). The plan: fit a wt-relative X from
growth alone (3f628bf7 the idea, 98fb6db0 growth-only models, 5820c19f
`hill_relative`), then measure the map from X to binding (roadmap step 8:
identity, affine, monotone f, with a global IPTG scale s; spikes and bulk
apart), then rejoin (step 9: learnable f and s, binding at weight 1). It
kept one X shared by both markers (decision D5) and named the test: "across
conditions it is testable (kan and 4CP responses should be affine in each
other per genotype)".

**2. "No map" (to 2026-10-05).** The first real relative fits
(`planning/studies/real-data-fit/`, c5de9f24) found growth curves broader
than binding (n 0.6-1.2 against 1.6-2.5), M42I's K on the wrong side of
wt's, a common mutant floor near X -0.5, and no shared protein
concentration or monotone map carrying binding onto growth. Step 8 was
recorded as "no map". The tube-offset mode (27109e26) and the full-size
simulation (bf4a1821: X undercovers at depth, |m| 20-40% too large, k off
along the k/dk_geno slide) followed.

**3. The m bias (2026-10-09 to 10-12).** Anchoring k and m on the wt
monoculture failed (34b8143a): the fit preferred an m about 40% too steep.
The m-bias study (7fcef80c) found the cause: each double pulls m a little
and the pulls add, so the full-library MAP compresses X about 1.47-fold.
The fix is the two-stage fit: k and m from wt, spikes and singles (with the
congression mixture), pinned for the full library (58b83b28, b8be0d2b; the
real X stretches a uniform 1.47). So the "no map" fits had a compressed X.

**4. The two-stage command and its validation (2026-10-13 to 10-15).**
`tfs-fit-two-stage` (3ce04cc5) reproduced the study, but its k-shift
refits failed: they leave the MAP and slide along the dk_geno population
(7d01721a). The marker-consistency flag fired on 10.6% of deep real
genotypes against 1.9% on simulations, and pheS alone compressed X against
kanR, even with wt's curve held (b6655671).

**5. Binding again (2026-10-15).** With the new binding data (nine
genotypes, 2 and 20 uM repressor dimer, normalized to day-of controls) and
the two-stage X: **kanR-alone X matched 2 uM binding with no correction**
(rms 0.12 against 0.30 on 2026-10-01), pheS did not
(`planning/studies/x-vs-binding/`).

**6. Why pheS disagreed (2026-10-15 to 10-16).** In order
(`planning/studies/within-tube-m/`, `planning/studies/relative-m/`):
- within-tube contrasts (tube totals cancel) disagreed with stage 1's m in
  both markers, but freeing the totals did not move m (e5501aba): not OD600;
- a log-linear IPTG term shared by every genotype helped kanR a little;
  a free one was unidentified (aea7f6f4, d8060513);
- the markers read occupancy through differently shaped maps: kanR
  switch-like (n at 4), pheS graded (n about 1), wt alike (6aa9ae4b). This
  is D5's test, and it fails: one shared linear X blends a steep readout
  with a broad one, which alone can make "growth broader than binding";
- one row of pheS tubes was swapped (3d5fbc43; the lab agrees it is
  probable; `tfs-check-tubes` 76ce30d3, `tfs-correct-tubes` f15f5b4c);
- a logistic warp of kanR's growth in occupancy lowers the loss by 8,000
  nats and is nearly linear on a simulation (d540edfc, a1670702);
- relative information alone cannot fix the scale of X: each estimate of m
  rests on a different anchor (tube totals, priors and Hill shape, or weak
  binders being flat), and they differ 2.5x in kanR (ba1752ba, 9a0a2f80,
  b86c0ecb). Binding at 2 uM supports the larger kanR m.

## What we know now

- kanR growth is close to linear in 2 uM binding theta for the spikes,
  once X is not compressed.
- pheS growth is a graded, differently shaped function of the same
  occupancy; one shared linear map cannot serve both markers.
- The scale of the growth variable needs an external anchor; binding is
  the natural one.
- Still open: M42I grows 0.3-0.4 below its 2 uM binding at low IPTG; wt's
  IPTG midpoint matches 20 uM binding while the mutants' levels match 2 uM
  (an in vivo IPTG scale s, or an effective protein concentration).

## Carry over

- The count likelihood and the staged MAP.
- The two-stage fit for the doubles' m bias (stage 1: wt, spikes and
  singles; stage 2: full library with stage 1's growth parameters pinned).
- Per-marker maps from occupancy to growth instead of one shared linear X
  (`linear`'s `m_warp`).
- The corrected tube table (`tfs-correct-tubes`) and the tube QC
  (`tfs-check-tubes`).
- Binding at weight 1: each binding observation counts as one observation.
  The default `N_growth_rows / N_binding_rows` goes.

## Design

**theta.** Absolute occupancy per genotype from a Hill curve in IPTG
(`hill_geno`), shared by both markers and observed by binding.

**Growth.** g = k_c + dk_g + m_c f_c(theta) per condition, with f_c the
per-marker map: start from `m_warp`'s logistic warp (f(0) = 0, f(1) = 1),
for both selection markers, now identifiable because binding fixes theta.
Whether kanR needs a warp at all is a result, not an assumption.

**Binding.** theta_obs at 2 uM against the genotype's theta at an IPTG
scaled by a global s (in vivo IPTG = s x nominal), weight 1. A per-genotype
deviation between in vivo and in vitro (a hierarchical offset on the
binding genotypes' logit theta or log K, with a learned scale) lets a
genotype like M42I disagree without moving the rest. Binding sets the
scale of theta and the maps; the 200,000 growth genotypes set their own
curves.

**Congression.** With absolute theta, the partition-function rules
(`homodimer`, the physical default) return to stage 1's mixture.

**Two stages.** Stage 1 (wt, spikes, singles, binding) fits k, m, the
maps, s and the deviation scale; stage 2 (full library) pins them. The
k/dk_geno slide and the intervals for k and m are revisited once m is
anchored by binding (the k-shift refits failed and are not reused).

## Steps

1. [x] **Agree this plan** (user, 2026-10-16).
2. [x] **Binding input.** `tfs-process-binding` (2026-10-16): the lab's
   table to `binding_df` at one protein concentration (default 2 uM),
   replicate means, `theta_std` from a pooled replicate SD. On the
   2026-10-15 table, 2 uM gives 110 points over 7 genotypes (D88A and
   V95N were measured only at 20 uM). Only wt is replicated there (five
   days), so the pooled SD (0.13, 31 dof) is wt's day-to-day scatter, about
   three times the point-to-point scatter within one curve. A day effect
   is shared by a whole curve, so treating the points as independent
   overstates how well a curve's level is known; step 4(b)'s per-genotype
   deviation is where that shared error should go. At 20 uM the pooled SD
   is 0.059 (150 dof).
3. [x] **Weight 1** (2026-10-16). `binding_weight` None now resolves to 1;
   an explicit weight is kept. Stored configs carry their resolved weight,
   so none change; the count-likelihood, congression-calibration,
   svi-overconfidence and real-data-fit studies already passed 1.
4. [ ] **Model pieces**, each with tests.
   - (a) [x] The global IPTG scale s between binding and growth
     concentrations (2026-10-16): `hill_geno`'s
     `theta_growth_log_conc_shift_scale` > 0 samples log s; growth sees the
     curve at s times nominal IPTG, binding at nominal. Tested on the smoke
     data through MAP, Laplace, extraction and growth prediction, single
     and mixture. `tfs-predict-theta` reports the curve in binding units;
     an in vivo option can follow if step 6 wants it.
   - (b) [x] The per-genotype in vivo/in vitro deviation for binding
     genotypes (2026-10-16): `theta_binding_noise: genotype_offset`, one
     additive theta offset per binding genotype, constant over IPTG,
     tau * z_g with a learned tau. Additive in theta because a curve's
     day-normalization error and M42I's disagreement are level shifts; a
     log K deviation can follow if step 6's residuals show midpoint shifts.
   - (c) [x] `m_warp` and the mixture's partition-function rules with
     `hill_geno` (2026-10-16): nothing refused them; the warp acts on each
     cell class's theta after the mixture combines plasmids.
     `tests/tfscreen/tfmodel/generative/test_joint_warp.py` traces every
     piece together and checks a near-linear warp reproduces the unwarped
     model through the mixture; `tfs-extract-params` now writes
     `warp_log_a`/`warp_x0` per warped condition. End to end on the smoke
     data (`--growth_priors` with an `m_warp` column, homodimer mixture,
     s, offsets): MAP, Laplace, extraction, chunked prediction.
5. [ ] **Simulation first.** Add a per-condition occupancy warp to the
   simulator's growth linkage (it has linear, power and saturation, no
   warp), simulate kanR switch-like and pheS graded with binding for the
   spikes at known s and deviations, and check stage 1 recovers m, the
   maps, s and theta, and that binding at weight 1 does not distort growth.
   - [x] Simulator keys (2026-10-16): `warp_a`/`warp_x0` on a linear
     growth entry, `growth_titrant_scale`, `binding_data.genotype_offset_sd`,
     with truth files.
   - [ ] The study (`planning/studies/joint-fit-sim/`): joint and null
     simulations from the full-size realistic config, fitted with the full
     and plain joint stage 1 (three runs).
     Run 1 (2026-10-17): on the joint simulation the full model recovers
     kanR's m (2%) and warp and s (16%) and halves the singles' theta error
     of the plain model; on the null simulation it finds a wrong mode (the
     same steep warp on both markers, s 0.25, large offsets, theta off by
     0.3). A warp shared by both markers is pinned only by the binding
     genotypes, which the offsets loosen. Next: is that a trap or a
     degeneracy (null fitted with the plain model; compare losses).
     Runs 2-3 (2026-10-17): it was a trap; the full model started from
     the plain fit's MAP recovers the null (near-linear warps, s covering
     1, offsets covering 0) and the joint truth (kanR's switch, s, the
     offsets). Open before step 6: m 15-25% steep even with the true model,
     and theta intervals covering 14-17%.
     Runs 4-7 (2026-10-17): the m bias is an affine squeeze of the
     singles' theta (k absorbs the constant, m the slope) that only binding
     genotypes resist, pushed by the logit population prior and by the
     congression mixture's approximation. Twenty binding singles cut it from
     14% to 6% on the no-congression simulation (weak pooling, 14% to 11%;
     both, 5%). Experimental implication: binding on about twenty singles
     spanning theta. Still open: the remaining 5%, the congression part
     with binding singles, and theta coverage (0.38).
     Run 8: with congression, twenty binding singles halve the bias (21% to
     12%), leaving the mixture's approximation as the largest piece; fifty
     singles take the no-congression bias only to 5%; the full recipe on
     the joint truth with twenty gets kanR's m within 1%, pheS's within 7%,
     s covered. Next: the mixture's co-resident draws (`congression_sets`).
     Run 9: 75 co-resident sets instead of 16 take the congression null's
     bias from 12% to 9% with twenty binding singles; 430 sets only to 8.6%,
     against 6% without congression. The remaining 3 points are the
     mixture's form, not its draws. Many sets run the Laplace Hessian out of
     GPU memory.
6. [ ] **Real stage 1.** wt, spikes and singles with binding at 2 uM, the
   swapped row corrected; report m against the within-tube values, the
   maps, s, the deviations (M42I), and binding residuals per genotype;
   compare likelihoods with and without each piece.
   Set up 2026-10-18 (`planning/studies/joint-fit-real/`): plain, then
   full from plain, 75 co-resident sets, the current 2 uM binding (seven
   genotypes), the swapped row dropped. Expect m 17-21% steep and theta
   intervals too narrow (joint-fit-sim runs 8-10); rerun when the updated
   binding (and any binding singles) arrive.
   Run 1 (2026-10-18): the 2 uM binding sits below growth's theta for every
   genotype (offsets -0.07 to -0.28, all negative) and wt's binding
   midpoint is about 10x lower in IPTG than growth's; s (0.53) does not
   absorb it, lambda runs to 0.8-2.1, and m and the warps are not
   interpretable. The binding-to-growth link at 2 uM needs more than s and
   per-genotype offsets.
7. [ ] **Real stage 2 and intervals.** Full library pinned; theta
   intervals; the marker-consistency flag rate against the simulations'.
8. [ ] **Into `tfs-fit-two-stage`.** A joint mode replacing the relative
   one's k shift; docs.

## Open questions

- Binding at 2 uM only, or both concentrations through a binding model in
  protein concentration (which the thermodynamic-ensemble goal will want)?
  Start with 2 uM.
- s alone, or s plus an effective protein concentration? Start with s.
- The binding data are still being replicated; when do they freeze for
  step 6?
- Bulk-only binding genotypes (G65H, V95N) carry congression; fit them
  through the mixture like any bulk genotype.

## Out of scope

- The selection-onset transient (`planning/offset-mode-growth-transition.md`).
- Thermodynamic ensembles (long term; this plan supplies the absolute
  theta they need).
