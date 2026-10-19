---
title: Findings ledger for the screen's model (established, refuted, open)
status: active
filed: 2026-10-18
area: tfmodel
revisit_when: >-
  Read before filing a new plan or study on the growth/binding model,
  lambda, the k/dk_geno slide or the slope m; add a line whenever a study
  establishes, refutes or reopens something.
related:
  - planning/joint-fit-return.md
  - planning/analysis-roadmap-summary.md
  - planning/growth-only-relative-fit.md
  - planning/congression-physics-plan.md
  - planning/two-stage-fit.md
  - planning/absolute-abundance-measurements.md
  - planning/studies/joint-fit-sim/README.md
  - planning/studies/joint-fit-real/README.md
---

## Why this file

On 2026-10-18 a review of the planning files, the studies and the git log
found that a week of joint-fit work had re-found several results from July
and early October (listed under "Re-found" below). Plans had cited history
only back to their own start. This page collects what is known, so a new
plan starts from it. Each entry names its source; `planning/dev-data/` is
gitignored and is cited by file. Dates are when a result was found.

## Established

**Growth identifies occupancy only up to an affine map.** k and m absorb any
affine change of theta (`g = k + dk + m theta`), so growth alone cannot fix
theta's scale (CLAUDE.md, Relative-X fit; roadmap step 5). Every "squeeze"
below is this direction: the doubles' pull on m, the singles' squeeze in
an absolute-theta stage 1 (joint-fit-sim run 6: theta_fit = 0.048 + 0.90
theta_true), and the slide of X against k.

**Binding genotypes are what resist the squeeze, as about 1/sqrt(n).**
July: about 20 in-library binding anchors with m free gave a theta slope of
0.91 and m ratios 0.95/1.14, robust to a wrong lambda (memory
`project_theta_recovery_congression_confound`, `out11/libexp`, 2026-07-08).
Quantified 2026-10-18 (joint-fit-sim run 10): without congression the
growth-only singles' theta RMSE is 0.010 + 0.053 / sqrt(n + 2) for n = 0-50
binding singles, insensitive to which singles; coverage 0.11 to 0.51.
This holds only if binding and in vivo occupancy share one scale (see Open).

**The full-library MAP steepens m about 1.45x through the doubles** (not
congression: still 1.37 without it); the two-stage fit (k and m from wt,
spikes and singles, pinned for the full library) removes most of it
(m-bias-factorial rounds 2-5, 7fcef80c, b8be0d2b, 2026-10-10/12). Stage 1
itself keeps a smaller bias (2-6% relative; 14% absolute without binding
singles, joint-fit-sim run 5).

**The markers read occupancy through differently shaped maps:** kanR
switch-like (Hill n at the bound of 4 on X), pheS graded (n about 1); one
shared linear X blends them; a logistic warp of kanR gains about 8,000 nats
(relative-m runs 2-4, 6aa9ae4b, d540edfc). In an absolute-theta model the
warp, s and per-genotype offsets are recoverable on simulations when the
full model starts from the plain one (joint-fit-sim runs 1-3).

**lambda absorbs misfit.** Converged fits recover it on simulations
(0.004 at 0, 0.98 at 1; count likelihood 0.02-0.06 at 0), but its profile
is flat over 0.6-1.4, seeds spread it 30-50 posterior SDs, and it runs away
when the model conflicts with the data: 0.6-2.4 in July (2.41 with m
clamped against binding anchors), 0.82-2.10 in joint-fit-real run 1. Real
fits that did not conflict gave 0.17-0.37 (congression-calibration README;
congression-physics-plan step 3.5; m-bias-factorial lines ~1226-1236;
dev-data real_fit 2026-10-02).

**The congression theta rule is not identifiable from the screen**
(cross-rule grid, 3d60dfd2): homodimer is the physical default;
`hill_relative` needs `max` (the only rule invariant to an increasing map).
The fit's fixed co-resident draws leave about 3 points of m bias however
many there are (joint-fit-sim run 9); more draws (75) are cheap except in
Laplace memory.

**The k/dk_geno/k_ref slide is a symmetry, not a bug** (`k += C, dk -= C`).
Per-condition k priors (128cd477), base_growth/k_ref (7e083271) and the wt
monoculture anchor (34b8143a) each failed to close it on its own; pinning
k and m from stage 1 nearly closes it (median dk error -0.0057 to -0.0008);
what remains is the dk_geno population location against wt, which the
stage 2 Laplace holds (two-stage-validation, 7d01721a).

**dk_geno is not rankable from the screen:** the two control conditions
disagree (r = 0.03-0.04) and the noise exceeds the spread (memory
`project_dk_geno_not_rankable`; deep-coverage step 3).

**The binding weight was not the driver of the joint fit's distortion.**
At the old default (about 251 x SD 0.001) binding dominated the objective;
at weight 1 theta coverage rose but the mixture's drift stayed
(congression-calibration second diagnosis, grid_bw1, 2026-09-24). The
default is 1 since 45ffb7fe.

**Theta intervals undercover because of the shared parameters' bias, not
the per-genotype uncertainty:** at the true k and m without congression
they cover 0.91; with congression 0.70 (the mixture adds per-genotype error
the model does not carry); fitted, 0.1-0.5 (joint-fit-sim runs 5, 10).

## The binding-to-growth link (the central open problem)

Facts, each found more than once:

- **wt's IPTG midpoint in growth is about 10x its 2 uM binding midpoint:**
  growth near 30 uM IPTG, 2 uM binding near 3 uM; 20 uM binding matches
  wt's midpoint (x-vs-binding, 2026-10-15; joint-fit-real run 1). An
  earlier form with 10 uM binding: dev-data real_fit, 2026-09-29.
- **wt is not fully induced in vivo:** about 70% at 1 mM where binding says
  93% (2026-10-01; roadmap summary), a floor of 0.15-0.24 in run 1.
- **Growth curves are broader than binding curves:** spike growth n
  0.6-1.2 against binding n 1.6-2.5 (2026-10-01). Rechecked on the
  kanR-only two-stage X: n 0.7-0.95 for most genotypes against 1.9-3.2 at
  20 uM (x-vs-binding run 2).
- **M42I's order inverts:** binding K about 10x wt's at both 2 and 20 uM;
  growth K about half of wt's (6.4 against 12 uM IPTG). Protein
  concentration does not move that ratio, so no single concentration
  removes it; per genotype, wt prefers about 12 uM protein and M42I about
  1.9 (dev-data real_fit 2026-10-01, `scan_protein.py`). Rechecked on
  the two-stage, kanR-only X (x-vs-binding run 2, 2026-10-18): M42I's K is
  0.18x wt's in growth against 3.6x (2 uM) and 5.3x (20 uM) in binding, a
  factor of 20-30; M42I/H74A keeps the order but sits 10x lower in growth;
  D88A and V95N agree within 2x.
- **No single protein concentration or shared monotone map carries binding
  onto growth** (best shared concentration 4.5 uM, rms 0.285; isotonic
  0.21-0.25; departures 0.3-1.0 against interval widths 0.15-0.35;
  2026-10-01, "no map").
- **The 2 uM agreement that motivated rejoining (kanR X against binding,
  rms 0.12) was in a gauge built from wt's own binding endpoints**, which
  absorbs any affine level and scale difference; in an absolute model the
  difference reappears (joint-fit-real run 1: every binding offset negative,
  -0.07 to -0.28).
- **Westerns put total repressor in the 100s of uM,** so an "apparent
  protein concentration" is a lumped mapping, not a concentration
  (dev-data README, binding plan).

So binding cannot yet set theta's absolute scale: every route that uses
it that way (the 2026-09 joint fit, roadmap step 8's maps, the 10 uM
interpolation, joint-fit-real run 1) has met these facts.

## Refuted or retired

- Up-weighting binding (N_growth/N_binding) as needed for binding to
  matter; and as the cause of the joint fit's distortion (see above).
- One shared X for both markers (D5's test failed; relative-m).
- OD600 tube totals as the cause of the within-tube/fit m gap (freeing the
  totals left the ratio at -0.49; relative-m run 1).
- A free per-concentration IPTG k (unidentified; within-tube-m run 2).
- Tighter or moved theta population priors as a fix for the squeeze
  (joint-fit-sim run 6: moving the location changes nothing; holding the
  SDs at the truth makes it worse).
- The k-shift refits for k's uncertainty in X (they leave the MAP and
  slide; two-stage-validation).
- "The prefit zeros dk_geno, so k is low" (July; disconfirmed: wt anchors
  k).

## Open

1. **The binding-to-growth link** (above). Candidates on record: cell-to-cell
   variation in IPTG uptake or repressor level (broadens and shifts in vivo
   curves; testable on wt first); a binding model in protein concentration
   with a fitted in vivo concentration; an affine calibration between the
   assays. The bench tests on record: monoculture growth of the spikes
   across the titration (`planning/absolute-abundance-measurements.md`);
   binding on more singles helps only once the link holds.
2. ~~Whether M42I's inversion and the broad growth curves survive on the
   two-stage, kanR-only X.~~ Both survive (x-vs-binding run 2, 2026-10-18).
   Both genotypes carrying M42I respond at lower IPTG in vivo than in vitro
   relative to wt; a genotype-specific in vivo factor (repressor level?)
   is a hypothesis, testable with expression data.
3. **lambda's prior:** fixing it from the measurement was proposed
   (2026-09-26, congression-physics-plan step 3.5) and never done; later
   decisions loosened it ("a readout, not a measurement", 2026-10-02;
   "keep it fitted", 2026-10-11). Unreconciled. wt purity (excess wt, about
   11-16% of the pool) moves fitted lambda 2x (0.365 cfg against 0.174
   wtspiked) and is still open at the bench.
4. **The dk_geno population location in stage 2:** holding it at stage 1's
   value is proposed, not run.
5. **Honest theta intervals:** the shared-scale error (carry its uncertainty
   into each genotype, as `--shared_shift` does for k) and congression's
   per-genotype error (the mixture's form).
6. **The theta_low/dk_geno ridge at 0 IPTG** (July): proposed fixes (drop 0
   IPTG from epistasis inputs; base_growth on genotypes other than the
   binding ones) never tested.

## Contradictions to keep in mind

- Study 0a (2026-09-26, buggy data, 22 uM protein) put growth's response at
  30-100x lower IPTG than binding; every later comparison puts wt's at
  about 10x higher IPTG than its 2 uM binding.
- Within-tube pheS m 0.018 (within-tube-m run 3) predates the swapped-row
  fix; without the row it is 0.0125.
- July "wt alone anchors k" against September/October "wt's pin is too
  weak against the slide" (different models and scales).

## Re-found on 2026-10-17/18 (joint-fit-sim, joint-fit-real)

- Binding singles pin m (July's in-library anchors).
- The singles' affine squeeze (the documented gauge symmetry).
- lambda running away under a model conflict (July).
- wt's 10x midpoint offset and its in vivo floor (2026-10-01, 2026-10-15).
- New in that work: the joint model's pieces recover on simulations; the
  full model must start from the plain one; the 1/sqrt(n + 2) scaling; the
  mixture's fixed-draw floor; intervals honest at the true k and m without
  congression.
