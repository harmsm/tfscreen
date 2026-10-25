---
title: Take m from relative information only (no tube totals in X)
status: idea
filed: 2026-10-15
area: tfmodel
revisit_when: >-
  Proposed 2026-10-15 for the user's review. If agreed it becomes the
  plan for replacing the two-stage fit's k shift
  (planning/two-stage-fit.md step 4).
related:
  - planning/studies/within-tube-m/README.md
  - planning/studies/two-stage-validation/README.md
  - planning/studies/x-vs-binding/README.md
  - planning/two-stage-fit.md
  - planning/offset-mode-growth-transition.md
  - src/tfscreen/tfmodel/generative/components/sample_offset/level.py
  - src/tfscreen/tfmodel/generative/observe/growth_counts.py
  - src/tfscreen/tfmodel/analysis/shared_shift.py
---

## Why

The within-tube study (`planning/studies/within-tube-m/`, runs 1-3) found
that the fit's growth slopes disagree with the part of the data that does
not depend on tube totals:

- Against flat pure spikes in the same tube, wt's swing from 0 to 1 mM
  IPTG is -0.005 per minute in kanR+kan and +0.018 in pheS+4CP. Stage 1
  pins -0.0148 and +0.0070.
- Across 903 deep singles the pheS swing over the kanR swing is -2.2 to
  -2.5 (wt -3.9, wide); stage 1's ratio is -0.47. wt sits with the
  library, so the problem is the fit's m, not wt.
- Stage 1 does not reproduce the within-tube contrast (predicted 0.004 in
  pheS against 0.018 observed); on a simulation it does.

The fit's absolute scale comes from the OD600-derived tube totals, which
tie each condition's k and m to the population's growth. A conversion
from OD600 to cells that drifts with IPTG or cell state under selection
would bias m, and the monoculture (also OD-based) with it. Even with that
map measured, OD600 is a noisy input. X does not need it.

## The idea

The count likelihood is `log mu = ln_sample_reads + ln_cfu_pred -
sample_ln_cfu + offset_tube`. With the per-tube level offset free (a flat
prior), it absorbs each tube's total, and the likelihood keeps only the
relative abundances of genotypes within each tube. Then:

- **k drops out.** Every tube is one time point, so `k (t_pre + t_sel)` is
  common to every genotype in it and the tube's offset absorbs it. k can be
  pinned anywhere; X does not depend on it.
- **m is still identified** through the gauge: X_wt goes from 1 at 0 IPTG
  to 0 at 1 mM, so wt's contrast with the rest of the library sets m. This
  is the within-tube statistic, used by the fit itself.
- **dk_geno stays identified** relative to wt (dk_wt = 0) through the
  no-selection tubes.
- **A growth shift shared by every genotype in a tube** (the IPTG term of
  run 2) is absorbed by the offsets, so the log-linear titrant k is not
  needed.
- **k's uncertainty no longer reaches X**, so the shared k shift and its
  refits (which failed, `planning/studies/two-stage-validation/`) go.
  m's uncertainty replaces it, and needs no refits: with the data's
  contrasts fixed, scaling m by (1 + e) scales every X_g - X_wt by
  1 / (1 + e), so the shared shift per draw is -e (X_g - X_wt) at each
  concentration, through the existing `--shared_shift`.
- **The OD600 tables become optional** for X. They still give absolute
  growth rates.

Everything needed for the first test exists: `sample_offset: level` with
`sigma_fixed` large holds the offsets' prior nearly flat, and
`--growth_priors` with `k_pinned` = 1, `m_pinned` = 0 pins k and leaves m
free.

## What could go wrong

- **Precision.** m now rests on contrasts against wt. wt's own within-tube
  swing has an SE of about 0.002-0.004 per minute (10-40% of m), though the
  fit also uses every other genotype's contrast with wt. The jackknife will
  say what m's SE is. If it is too wide, binding-anchored spikes could add
  scale information later.
- **The offset mode.** Freed offsets went badly before
  (`planning/offset-mode-growth-transition.md`: IPTG-structured offsets,
  k and m unphysical). There k was free and the full library's doubles
  pulled m (the 1.45x bias). Here k is pinned and stage 1 has no doubles,
  so the offsets are meant to carry the totals. If m still runs away (the
  old fit reached -0.05 in kanR), the relative information is not what we
  think, and step 1 shows it.
- **The dk_geno population direction** that stage 2's Laplace held (the
  population location against wt) may stay soft. Watch the held
  directions.
- **Simulations have consistent totals,** so relative and absolute fits
  should agree there; they test that relative-only is unbiased and say
  what it costs in width, not that it fixes an OD bias. A simulation with
  an IPTG-dependent OD bias would test that directly (optional, step 5).

## Steps

1. [x] **Relative stage 1** (no code). Run 1 (2026-10-15,
   `planning/studies/relative-m/`): the seeds pass, but on the real data
   m does not move (pheS/kanR -0.49 against -0.47 absolute and -2.2
   within tube); the fit discards pheS's IPTG response because the X
   shared by both markers cannot follow their different IPTG curves.
   Steps 2-6 wait on a per-marker midpoint test. Setup as planned: The two-stage fit's stage 1 with
   `sigma_fixed=10` and k pinned at the current stage 1's values, m free,
   on the real data, seed 1 and seed 2 (3 runs of about 30 minutes).
   Pass:
   - seeds: m within about 5% of the truth, as the absolute stage 1 is;
   - real: m near the within-tube values (kanR+kan about -0.005, pheS+4CP
     about +0.012 to +0.018, ratio about -2.3);
   - the within-tube posterior-predictive check (`ppc_within_tube.py`)
     reproduces the observed contrast;
   - no held direction in the Laplace beyond what stage 1 has now.
   Also report the offsets by tube design (they now carry the totals and
   any shared IPTG term).
2. [ ] **m's precision.** The stage 1 jackknife (30 subsamples) on the
   relative stage 1, real data: m's SE and its direction across the two
   markers.
3. [ ] **Relative stage 2** on seed 1, seed 2 and the real data, k and m
   pinned at the relative stage 1, offsets free. Compare with run 1 of the
   two-stage validation:
   - seeds: deep-double X coverage (run 1, absolute: 0.90 / 0.76 without
     the k shift);
   - real: X against binding at 2 uM for kanR alone and pheS alone (both
     should now match if m was the problem), and the marker-consistency
     flag rate (10.6% deep in run 1; it should fall toward the
     simulations' 1.9%).
4. [ ] **m's uncertainty in X.** The analytic scale shift from step 2's
   jackknife through `--shared_shift`; coverage on the seeds.
5. [ ] *(Optional)* **A simulation with an IPTG-dependent OD bias**, if
   the simulator's `od600` block can be given one: the absolute fit should
   fail and the relative one should not.
6. [ ] **Fold into `tfs-fit-two-stage`** if steps 1-4 pass: a relative
   mode (free offsets, k pinned, m jackknife, scale shift) that replaces
   the k shift; retire `kshift`; update `planning/two-stage-fit.md`.

Independent of these, the bench test: OD600 against plate counts under
each selection across IPTG. It would explain the absolute fits' m and
give absolute growth rates, but X would not depend on it.

## Out of scope

- The full-library m bias from the doubles (stage 2 keeps m pinned).
- The selection-onset transient (`planning/offset-mode-growth-transition.md`),
  unless step 1's offsets point at it.
