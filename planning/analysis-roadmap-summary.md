---
title: What the analysis roadmap branch did and found
status: done
filed: 2026-10-02
area: tfmodel
revisit_when: >-
  A record. Revisit when picking up any item under "Where to go from here",
  or before a methods section is written from it.
related:
  - planning/analysis-roadmap.md
  - planning/studies/count-likelihood/README.md
  - planning/studies/relative-fit/README.md
  - planning/studies/svi-overconfidence/README.md
  - planning/studies/noise-anatomy/README.md
  - planning/studies/od600-population/README.md
  - planning/studies/growth-binding-map/README.md
  - planning/dev-data/real_fit/README.md
---

Branch `claude/physics-improvements-roadmap-03a273`, 2026-09-26 to
2026-10-02. The plan is `planning/analysis-roadmap.md`; this file is the
outcome. Part 1 is written as a methods narrative, with the reasons and the
decisive numbers. Part 2 lists the bugs caught on the way. Part 3 is where
to go next, and part 4 is what it takes to reproduce the work and what goes
into the merge.

Numbers about the real screen are summary statistics from
`planning/dev-data/` (gitignored). The data themselves never enter the
repository.

# 1. Methods

## Measurement model

**Reads and totals.** Each tube is one timepoint: it is pulled, read for
OD600, spun down and frozen, and sequenced later. Reads measure
frequencies, and OD600 measures the tube's total. A tube that grew a little
faster than its neighbors scales every genotype and the total by the same
factor, so frequencies do not see it. We therefore modeled reads against
the tube total instead of attaching per-tube noise to every genotype.

**Count noise.** We measured it before choosing a likelihood
(`planning/studies/noise-anatomy/`). Doubles are 97% of rows, with a median
of 3-5 reads per tube and 23-33% zeros. Per-tube variance was 5-18 times
Poisson at 100-3,000 reads, which points to a bottleneck upstream of the
reads (founder cells or PCR templates) rather than a constant-CV floor.
Today's `ln_cfu_var` understated the noise 5-10 fold. Low counts were
mostly low abundance, not crashes: 88% of double rows under 5 reads in
selective tubes were already under 5 reads without the drug. This study ran
on an older processing snapshot and has not been rerun on the new one.

**Count likelihood.** We chose to observe reads directly, as
`reads ~ NegBin(mu, mu (1 + phi) + mu^2 inv_r)`, with no pseudocount
(`planning/studies/count-likelihood/`):
- The linear `phi` term matches the bottleneck signature.
- numpyro's own negative binomial lost several nats per observation to
  float32 cancellation at 1e4-1e6 reads, so we implemented Loader's
  algorithm and tested it against scipy at high depth.
- We paired it with one constant-prior level offset per tube, so that an
  error in a tube's supplied total does not move every genotype in it.

We tested it against the `ln_cfu` Student-t likelihood on 24 simulations:
2 noise models x 3 seeds x 4 fits. On the final run, theta 95% coverage
was 0.83 / 0.84 for counts (Poisson / realistic noise) against 0.72 / 0.83
for `ln_cfu`. Theta RMSE was 0.035 / 0.072 against 0.100 / 0.109, and
dk_geno coverage was 0.71-0.86 against 0.53-0.55. At a true congression
lambda of 0, counts found 0.02-0.06, where `ln_cfu` found about 0.12, so
most of the spurious congression came from the read floor. Counts with
level offsets became the `tfs-configure-model` default.

**OD600 calibration.** We ported the lab's OD600-to-CFU notebook into a
generic tool, `tfs-calibrate-od600`. It fits a weighted polynomial and
keeps the full coefficient covariance, because the curve error is shared
by every tube and must not be treated as per-tube noise. The lab's data and
constants stay out of the repository. The port reproduced the notebook's
coefficients to 2e-9 and found that the notebook never divided by the
plated volume, so its totals were 10x low. Growth rates and frequencies are
unaffected; absolute levels are.

**Simulator realism.** We added the noise sources the data showed, each
switchable so its effect on a fit can be measured: founder sampling,
demographic growth, a shared transformation across replicates, a PCR
template bottleneck and OD600 through an inverse calibration. Matching the
real 5-18x over-Poisson noise needs templates on the order of reads per
tube / 10.

## The growth-only relative fit

**Why a relative variable.** A first, model-free look at five spiked
genotypes rejected a single linear map from in vitro binding theta to
growth (reduced chi-square 6.7 in kan; `planning/studies/growth-binding-map/`,
provisional). Growth alone fixes an occupancy-like variable only up to an
affine map, so we fit a wt-relative growth variable X instead of theta
(`theta: hill_relative`). It has Hill shapes with real-valued baselines,
and wt is gauged to X = 1 at 0 IPTG and X = 0 at 1 mM. One X per genotype
and IPTG concentration is shared by kan and 4CP, and each condition's k and
m map it to growth. Then k is wt's rate at 1 mM and m is wt's change
between 0 and 1 mM. The orchestrator refuses combinations the gauge cannot
support: binding data, learned activity, logit rescaling, non-linear growth
and partition-function congression rules.

**ln_cfu0.** We fit one starting abundance per replicate x pre-condition x
genotype (`ln_cfu0: hierarchical`), because the kanR and pheS libraries are
transformed and grown separately. The factored model, which shares a
genotype's abundance across libraries, pushed the real between-library
difference (SD 0.17-0.29 in simulation) into a confident X error. At more
than 1000 reads, 95% coverage was 0.09 against 0.80, and RMSE was 0.101
against 0.027. The orchestrator now refuses the factored model across
libraries.

## Inference: what we tested and chose

**The m·X ridge.** Growth constrains mostly `m (X_g - X_wt)`. The posterior
is therefore a curved ridge along m·X that only wt's absolute growth change
pins, through the tube totals. Every inference choice below follows from
that.

**Guides on simulations.** We compared five inference arms on 30
simulations, after fixing the start, scale and Laplace bugs in Part 2
(`planning/studies/relative-fit/`, run 4).

| Arm | X 95% coverage, Poisson / realistic | X RMSE, Poisson / realistic |
|---|---|---|
| Mean-field component guide | 0.06 / 0.38 | 0.30 / 0.29 |
| Low-rank guide (any rank from 20 to numpyro's default) | 0.81 / 0.76 | 0.07 / 0.12 |
| MAP + full Laplace | 0.98 / 0.94 | 0.11 / 0.29 |
| Two-stage fit (refit at Laplace draws of k and m) | 0.91 / 0.78 | 0.08 / 0.11 |

- The component guide slides along the ridge.
- MAP + full Laplace covered best, with the noisiest point and the widest
  intervals.
- Every guide collapsed k and m: k coverage was 0, and m coverage 0-0.25.
- A dense full-covariance guide ran its loss to -1e23 through float32 error
  in its own log density.
- NUTS stayed at the maximum tree depth with R-hat 3.5 under every mass
  matrix and in float64, so we used calibration over many seeds as the
  reference instead.

**Real data rejected the guides.** On the full library (217,945
genotypes), both the component and the low-rank guide fit the counts
3.3-3.4e5 nats worse than the MAP, again by sliding along the ridge.
A full Laplace is out of reach there, with about 2.4M parameters.

**Per-genotype Laplace.** Given the shared parameters (k, m,
hyperparameters, tube offsets), genotypes do not couple. On a simulation's
full Hessian every cross-genotype entry was exactly 0, wt included. So the
Hessian splits into one 9-11 parameter block per genotype, and we get all
the blocks from B Hessian-vector products per genotype chunk. The code
refuses a model whose blocks leak. With the shared parameters held at the
MAP, this block Laplace covered X 0.93 / 0.84 on simulations, with k/m
coverage 0.

**Arrowhead Laplace.** We then added each genotype's coupling to the shared
parameters and the shared block, at one more Hessian-vector product per
shared parameter. We draw the shared parameters from their marginal, the
Schur complement, and each genotype from its conditional given them. That
is the full Laplace, and on a toy model it matched the dense inverse
Hessian exactly, chunked or not.

A MAP stopped short of an optimum needed two rules:
- A floored direction of a genotype block carries no coupling. Kept, the
  couplings of floored blocks subtracted 6e10 from m's curvature of 1e10.
- A negative direction of the Schur complement is held at the MAP. Floored
  at the prior instead, it gave k its prior SD and X widths of 1-3.

On the simulations it covered X 0.97 / 0.94 (0.95-0.98 above 1000 reads)
and k/m 0.92 / 1.0 (Poisson) and 0.50 / 0.42 (realistic). That is close to
the full Laplace's 0.92 / 0.92 and 0.58 / 0.50, so the realistic-noise
shortfall is not the approximation's. **MAP + arrowhead Laplace is the
full-library route.** On the real library it ran in 20 minutes.

**Laplace floor.** Each Hessian eigenvalue is floored at the prior's
curvature along its eigenvector, so no direction is wider than the prior.
The old flat 1e-3 floor turned negative eigenvalues into variances of 1000.

## Fitting the real screen

**Data.** The screen had 118 tubes after dropping three failed ones, and
22.9M count rows over 217,945 genotypes.
- Tube totals came from OD600 through the calibration. Every screen
  reading lay inside the calibrated range.
- Binding curves came from one joint fit of anisotropy to theta and Hill
  curves, with per-day levels and robust residuals, interpolated to 10 µM
  protein.
- The library has nine spikes. The processing config listed three, and its
  M42I codon differed from the spikes'; we corrected both.
- The initial kan and phe samples were swapped, and we swapped them back.

**Starting point.** A level-offset MAP from a cold start landed in a ±2.8
offset mode with unphysical k and m. Adam moves every parameter about one
step size per step, so the fast parameters were carried by whole units in
the first window, and the step-size cuts froze them there. We started
instead from the no-offset MAP plus small per-tube offsets
(`tfs-fit-model --init_from`), which was 8.6e5 nats better. The offsets'
SD is held at 0.17, the per-tube OD600 scatter. Learned, it grew until each
tube's offset absorbed the population's growth.

**Population SDs.** The learned log(hill_n) population SD ran to about 19,
giving n from 0.01 to 13. In 42% of genotypes the curve then changed more
between 0 and 0.1 µM IPTG than across the whole measured range, and the
counts show no such step. We hold that SD at 0.5. Holding X_low, X_delta
and log K as well, at 0.5 / 0.5 / 1.0 against learned 3.6 / 4.3 / 11, is a
sensitivity check (below).

**Growth priors.** k ~ N(0.015, 0.01) and m ~ N(0, 0.01) per condition,
written by hand, since the pre-fit refuses growth-only models. The wt
monoculture rates as kan priors changed m in the fifth digit.

**Final fit.** The final fit is `rel_off_n05`. It is growth-only, with the
count likelihood, level offsets at SD 0.17, the n population SD held at
0.5, loose priors and nine spikes, followed by the arrowhead Laplace. It fits
the counts 2.0e4 nats better than the best joint MAP, which used the same
offsets. Without offsets, the growth-only fit was likewise 2.2e4 nats ahead
of the joint one. Its
median X 95% width by genotype read depth is:

| Reads | 95% width |
|---|---|
| ≤ 100 | 2.6 |
| 1e2-1e3 | 1.2 |
| 1e3-1e4 | 0.69 |
| 1e4-1e5 | 0.42 |
| > 1e5 | 0.33 |

## Assumptions we checked

- **Genotype blocks are uncoupled given the shared parameters,** checked on
  a full Hessian and guarded by a leak check on every run.
- **The simulator matches the fit model:** growth k + dk + m·A·θ to 2e-16,
  θ to the Hill truth to 1.4e-5.
- **Every per-genotype latent is mini-batch safe,** by shape and by order,
  registry-wide.
- **The level offsets are not the unassigned-read share** (correlation
  -0.21 / -0.41). Closure of the frequencies improved with them (SD 0.16
  to 0.11).
- **The kanR and pheS libraries differ in composition** (Spearman about
  0.1), which supports per-library ln_cfu0.
- **Unselected kan shows no IPTG effect** (wt 1 mM / 0 mM log-ratio about
  0).
- **The split density from the protocol agrees** with extrapolation of the
  unselected tubes within 1.5-fold.
- **wt's within-series growth in the screen matches its monoculture** in
  kan (IPTG effect 0.013 vs 0.014). In pheS+4CP it does not: library wt
  keeps growing at high IPTG where the monoculture stops.

## What the results are insensitive to

- **The spike curve shapes, K and n, stay inside their arrowhead Laplace
  intervals across:**
  - holding all four Hill population SDs (`rel_off_hs`);
  - adding the congression mixture (`rel_off_hs_mix`, `max` rule, lambda
    prior matched to the measured 0.357 with likelihood-ratio 95% CI
    0.16-0.67);
  - the k/m uncertainty.
- **The k and m priors** (loose against monoculture).
- **The SVI guide's rank** (20 against numpyro's default).

**Sensitive, and under-constrained: m and the absolute X scale.** Holding
the population SDs made every |m| 38% larger and scaled X by 0.77, and the
mixture moved |m| another 21%. These shifts are far outside m's Laplace
interval of ±4-5%. Each step fit the counts about 2.2e4 nats better and
grew the tube offsets. The offsets follow IPTG, in opposite directions in
kan and 4CP. A per-tube offset moves every genotype's predicted frequency
in that tube, so a library-wide IPTG trend can sit in the offsets instead
of in m·X. The Laplace intervals on k and m are therefore conditional on
where the MAP sits on the ridge. We report curve shapes and baselines
relative to wt, not rates.

## What the fit says about growth and binding

- **Growth curves are broader than binding curves.** Spike growth n is
  0.6-1.2 (every 95% upper bound ≤ 1.17), against binding n 1.6-2.5. This
  holds for clean spikes and bulk-sourced genotypes alike and survived the
  congression mixture, so it is not congression.
- **M42I induces below wt in growth,** with K 6.4 µM (95% 4.2-9.6) against
  wt 12 µM (8.8-16.7). In binding its K is about 10x wt's, at both 2 and
  20 µM protein.
- **Every mutant falls to a common floor near X -0.5 by 1 mM,** including
  weak binders whose binding plateaus sit above it. Taking the floor as
  zero occupancy, growth says wt is about 70% induced at 1 mM, where
  binding says 93%.
- **No single apparent protein concentration and no shared monotone map
  carries binding onto growth.** The best shared concentration left an RMS
  of 0.285 in X, and isotonic maps 0.21-0.25 at every concentration. The
  departures, 0.3-1.0 in X, are well outside the 0.15-0.35 interval
  widths.

# 2. Bugs caught and fixed

## Inference and optimization

- **SVI started far from the pre-MAP point.** The guide's initial scale was
  0.1 in each site's units, plus a 0.1 multiplicative jitter. SVI began
  about 1000x above the pre-MAP loss and fell into wrong optima: mirror
  modes, halved m. This invalidated the absolute coverage of every earlier
  SVI grid. Fix: scale 1e-4, jitter 0 (c57d9ee5).
- **That fix froze the hyperparameter scales.** It put them on their
  `greater_than(1e-4)` bound, which is -inf unconstrained, so they never
  moved. This affected every component-guide fit from 2026-09-27 to
  2026-09-29, which we reran as relative-fit run 4 and the
  svi-overconfidence fix grid. Fix: start above the bound (9db28286).
- **Step-size cuts came mid-descent.** The cut rule judged each window
  alone. 20 of 23 first cuts on the count-likelihood v2 grid came during a
  descent that pooled to t = 4-14, and one cut led into a wrong optimum.
  Fix: pool the stalled windows before any cut or stop (740ebc19).
- **A runaway loss (-8e23) was reported as converged.** Fix: a `diverged`
  stop (32a7ec12).
- **NUTS chains started at random,** because no site matched the start
  values. Fix (8e244cf0).

## Laplace and posteriors

- **The pre-fit took its Hessian at exp(value) of every positive site.**
  Constrained and unconstrained values were mixed. phi overflowed and 8
  count fits crashed, and on `ln_cfu` fits the written k scale was 4-9x too
  loose. We reran the whole first count-likelihood grid. Fix (c65390aa).
- **The flat 1e-3 Laplace floor** turned negative eigenvalues into
  variances of 1000. It caused 5 of 6 Laplace blowups in relative-fit runs
  2-3 and spread m 50x on the two-stage grid. Fix: the prior-curvature
  floor (9db28286).
- **The Laplace kept the mini-batch likelihood scale at full batch,** 53x
  on the real screen at batch 4096. No result was affected. Fix (cd208d15).

## Model components and numerics

- **The factored ln_cfu0 assumed one library.** It moved the kanR/pheS
  difference into confident theta errors in every study grid. Fix: refuse
  it across libraries and rerun the grids with `hierarchical` (722393cf).
- **The factored ln_cfu0's centered tube offset had an unbounded density,**
  so no MAP existed. Fix: non-centered offsets (32a7ec12).
- **The negative binomial concentration underflowed** for near-extinct
  genotypes, giving NaN gradients. Fix: a floor (c65390aa).
- **A horseshoe local scale squared to infinity** for low-read doubles.
  Fix: overflow-safe helpers (740ebc19).

## Simulator, processing and I/O

- **Simulated binding observations were clipped to [0, 1]** against an
  unclipped likelihood. That affected 10-17% of anchors. Fix (6e2903d5).
- **`tfs-extract-params` mislabeled ln_cfu0 rows** after any missing
  genotype cell. Correlation with presplit was 0.09-0.25 instead of 0.999.
  The fits themselves were fine. Fix (079680bd).
- **Prediction failed with per-tube offsets.** Prediction now zeroes them,
  giving the typical tube (775f8b79).
- **Absent prior groups reloaded from the priors CSV as NaN.** Fix
  (98fb6db0).
- **`read_configuration` silently dropped guesses for sites outside
  `init_params`.** This is how the cold-start offset mode arose. Fix:
  `--init_from` (faa5b116).

## Data problems found

- The OD600 notebook omitted the plated-volume division.
- The initial kan and phe samples were swapped.
- The processing config listed 3 of 9 spikes, with a wrong M42I codon.
- Sample-name forms differed between files.
- One tube was missing from `sample_df`.
- The old snapshot's `presplit.csv` had residue numbering one lower than
  `growth.csv`.
- That snapshot's FASTQ processing sent all unknown reads to wt, which is
  why the step 0 studies are provisional.

## Fit pathologies that needed new options

These were model behaviors, not code bugs:
- runaway Hill population SDs, now held with
  `theta_*_hyper_scale_fixed`;
- a learned tube-offset SD that absorbed growth, now held with
  `sigma_fixed`;
- a single m prior width for every condition; `m_scale_plus` and
  `m_scale_minus` now accept per-condition arrays.

# 3. Where to go from here

In rough order of payoff for the science:

1. **The binding/growth relationship, as its own study.** The facts to
   explain are firm: broader growth curves, a common floor, wt only about
   70% induced in vivo, and M42I's K inverted. Candidates:
   - cell-to-cell variation in IPTG uptake or repressor level, which
     flattens a steep θ(IPTG), testable first on wt's binding titration;
   - a full thermodynamic binding model with protein concentration.

   The direct bench test is monoculture growth of the spikes across the
   titration (`planning/absolute-abundance-measurements.md`).
2. **The IPTG-patterned tube offsets.** Candidates:
   - an OD600-to-CFU relation that shifts under selection (roadmap D11);
   - a library-wide growth effect the linear growth model lacks.

   A bench check of OD against plate counts under selection would separate
   them, and the population model (roadmap step 6) would carry it.
3. **Pinning the X scale and m, if rates are ever needed.**
   - wt monoculture rates as an observer, not just a prior;
   - a presplit measurement for this screen;
   - a per-tube spike-in counting standard for future screens.
4. **A MAP that reaches its optimum.** Every hierarchical MAP stopped at
   its epoch cap, and on real data it is a saddle along the m·X ridge.
   - The Laplace needs floor and hold rules for that reason.
   - Low-depth genotypes are noisy at the MAP: on simulations, X RMSE was
     1.4 at ≤ 100 reads against 0.22 for the low-rank point.
   - Newton steps with the arrowhead Hessian are one route.
   - Holding the population SDs shrinks the low-depth genotypes, but only
     together with the ridge slide.
5. **Rerun the provisional step 0 studies** on the new processing:
   0a growth-binding map, 0b noise anatomy, 0c OD600 population. Finish
   0c's curve-form decision (roadmap D9).
6. **Remaining roadmap steps:**
   - 2: counts and OD600 through processing;
   - 6: the population model;
   - 7b: growth transitions, test then purge, with 4CP the likely case;
   - 8 and 9: the learned map and the joint fit;
   - the kan-vs-4CP consistency check (D5), which a shared X cannot test
     alone.
7. **Known calibration gaps on simulations:**
   - k/m undercover under realistic noise even with the full Laplace;
   - the joint fit's saturated high plateau undercovers (0.31);
   - the pre-fit under the count likelihood wrote k priors far from truth.

# 4. Reproducing this work, and what goes into the merge

**What reproduces now.** Every package change is committed with tests, and
the CHANGELOG and CLAUDE.md describe it. These studies have tracked scripts
and grids with fixed seeds:
- `count-likelihood`
- `relative-fit`
- `svi-overconfidence`
- `noise-anatomy`
- `od600-population`
- `growth-binding-map`
- `step0-data`

Their grid outputs are gitignored and live only on the cluster and local
disk.

**Gaps in the tracked studies, closed 2026-10-03:**
- Each study README's "Commit" section now names the commit each grid ran
  on. Where the README had not recorded it, the commit is inferred from the
  file times of the downloaded `run.out` files, and the README says so.
- `relative-fit/laplace_blocks.sh block|arrowhead` reproduces the block and
  arrowhead Laplace tables from a finished `relative_fit_v4/`. Rerun on
  2026-10-03, the arrowhead mode matched the README's 95% table exactly
  (50% coverage within 0.01, from new draws).
- The roadmap's step 5 and 7 entries and the relative-fit decision text are
  current, and the relative-fit README opens with the current answer.
- The one-off diagnoses in `svi-overconfidence` (NUTS variants, the plateau
  tests, the truth-pinned fit) are marked in its README as records, not
  reproducible results. Scripting each one is not worth it.

**The real-data fit is the harder part.** Its scripts live in the
gitignored `planning/dev-data/`, mixed with 55 GB of data and run outputs:
- Processing: `prep_dev_data.py`, `binding_fit.py`.
- Run templates: `run.srun`, `run_relative.srun`, `run_blap.srun`.
- Start points: `make_init_npz.py`, `make_warm_start.py`,
  `make_init_hs.py`, `check_warm.py`.
- Priors: `set_growth_priors.py`.
- Scoring: `score_counts.py`, `residual_compare.py`.
- Scans and figures: `scan_protein.py`, `scan_monotone.py`,
  `plot_x_vs_binding.py`, `resid/small_offsets.py`,
  `resid/profile_offsets.py`.

The final fit's start depends on a chain of six earlier runs: `rel_loose_b`
and `map_off_n05`, and through it `map_nooffset_n05`, `map_warm`,
`map_sig017b` and `map_nooffset_b`. About twenty other run directories and
the smoke tests are dead ends.

Proposal:
1. Keep the data where they are, out of the repository (C10).
2. Collect the scripts the final results need into one place, with a
   README that gives the chain as commands: inputs, each run's script and
   settings, and the helper that builds the next start. Drop the dead
   scripts: `compare_arms.py`, `resume_posterior.srun`, `fetch.sh`,
   `local_offsets/`.
3. Shorten the chain if possible. A recipe of no-offset relative MAP, then
   offsets warm-started from it, then the final fit would need one rerun to
   confirm it lands at the same point. Otherwise the chain is documented as
   it ran.
4. Archive the final MAP parameters and starting npz files with the lab's
   data, since they are derived from it.

Two decisions here are Mike's:
- **Where the real-data scripts live.** They hold no raw data, but they do
  encode the lab's protocol: library design, spike codons, tube volume,
  presplit dilution and file names. One option is tracked, in a study
  directory such as `planning/studies/dev-data-fit/`, reading data from the
  gitignored `planning/dev-data/`. The other is the lab's own analysis
  repository, pinned to a tfscreen commit.
- **Whether to rerun the shortened chain** to make the final fit one recipe
  long.

**The merge itself.** `main` has not moved since 2026-09-26, so the merge
is mechanical. The full suite, slow and smoke tests
included, ran on 2026-10-03: 5,325 passed and 8 smoke tests failed. The 8
built mock data with `ln_cfu` only and had not been run since the count
likelihood became the configure default (2620a5b3). They now ask for the
`ln_cfu` likelihood their data imply, and all 44 smoke tests pass; the
count path keeps its smoke coverage through the fixture-based tests. The
stale entries are fixed (above).
Still open: set the roadmap to `done`, or keep it `active` for steps 2, 6,
7b, 8 and 9 (Mike's choice).
