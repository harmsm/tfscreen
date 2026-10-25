---
title: Honest intervals for the well-measured genotypes
status: active
filed: 2026-10-09
area: tfmodel
revisit_when: >-
  Active. Goal: what this experiment's data support, without new
  sequencing (user, 2026-10-09).
related:
  - planning/two-stage-fit.md
  - planning/studies/m-bias-factorial/README.md
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

1. [x] **Anchor k and m with the monoculture rates**
   (`planning/studies/km-anchor/`). Three fit-only arms: the realistic
   simulation with the monoculture priors, the same with the rates
   perturbed by their own error, and the real data with the monoculture
   priors. Read: deep X coverage and slope against the baseline, k/m
   against truth, held directions, and on the real data whether k and m
   move to the monoculture with the offsets staying clean (or whether the
   data fight the anchor, which would point at a real in-pool difference
   such as the selection-onset transient).
   Done 2026-10-09: the data reject the anchor in the simulation and the
   real data alike. k moves only along the k/dk_geno slide; m stays about
   40% too large in kanR (sim truth -0.0138, fit -0.018 to -0.020 even with
   the truth as a 0.002-SD prior; real fit -0.020 against the monoculture's
   -0.014). The bias is in the fitted likelihood, a misspecification
   (`planning/studies/km-anchor/`).
1b. [x] **Find the misspecification behind |m|.** A reduced-library
   factorial on the simulation, local and fast: the same design and
   conditions, switching off one of congression, per-tube rate noise and
   realistic count noise at a time (and the dk_geno prior's shape), and
   measuring the m bias. Whatever removes it is what the fit must model or
   correct before step 2 is worth doing. Set up 2026-10-09 as
   `planning/studies/m-bias-factorial/` (5 arms x 2 seeds, 5,256 doubles,
   cluster: the staged MAP was too slow on a laptop).
   Round 1 (2026-10-09) was inconclusive. The reduced `base` arm gave an m
   ratio of 1.03-1.14 against 1.47 at full size, because shrinking the
   library left wt at 81% of each tube's reads (31% at full size) and
   dropped 25 tubes below OD600 detection. Every arm fell within seed
   noise. Round 2 refits subsets of the full-size simulation's genotypes
   instead: no doubles, plus deep doubles, plus as many shallow doubles.
   Round 2 (2026-10-10) found it: m is right without the doubles (kanR
   1.03, pheS 0.89), about 1.2 with 81,000 doubles of either depth, and
   1.47 with all 208,000. Each double pulls m a little, and the pulls add.
   That fits the joint MAP's Neyman-Scott bias, or a per-genotype
   misspecification of the doubles. Next: the same no-doubles fit on the
   real data (does kanR m reach the monoculture's -0.014?), and on the
   simulation the full library with k and m held at the no-doubles values
   (round 3, `run_held.srun`).
   Real data without doubles (2026-10-10): kanR+kan k 0.0160, m -0.0141,
   on the wt monoculture (0.0150, -0.0141) with loose priors; the real
   kanR curves fitted with all genotypes are compressed by about 1.45.
   pheS+4CP moves away from the monoculture (m 0.0061 against 0.0144):
   step 3.
   Round 3 (2026-10-10), the simulation's full library with k and m held
   at its no-doubles MAP: above 1e4 reads X's slope 0.95 (was 1.36),
   RMSE 0.05-0.07 (0.16), 95% coverage 0.90 / 0.65 at 1e4-1e5 / >1e5
   (0.29 / 0.10); the k/dk_geno slide is gone. The remaining gap is k and
   m's own error, which the held fit leaves out: step 2.
   Round 4 (set up 2026-10-10): the real fit holds kanR only (tight prior
   at the real no-doubles MAP; pheS loose until step 3), checked first on
   the simulation, since X is shared and pheS's m stays free.
   Round 4 result: holding kanR alone is enough (the free pheS m follows
   to 1.04 of the truth), but the tight prior leaked about 11% of the
   doubles' pull (deep coverage 0.82 / 0.42 against the pins' 0.90 /
   0.65). Real: kanR m -0.0133, pheS m 0.0064, half the monoculture's.
   Round 5 repeats both arms with `linear`'s new per-condition pins.
   Round 5 result (2026-10-11): the pins hold exactly and pheS's free m is
   right, but pheS's free k reopens the k/dk_geno slide: deep X shifts up
   by 0.04-0.09, coverage 0.67 / 0.20. Pinning every condition at the
   no-doubles fit (round 3) closes it, and the real pheS no-doubles m is
   what the data give with X set by kanR, so the real fit should pin all
   four conditions.
   Round 6 (2026-10-11), the real fit with all four pinned: every
   genotype's X stretches by a uniform 1.47 against the recipe fit (r
   0.99-1.00 above 100 reads), the factor the simulation predicted; no k,
   m or slide direction is held. The two-stage fit (k and m from wt,
   spikes and singles; the full library at those values) is the route.
   Still open: k/m uncertainty in X (step 2), a second simulation seed for
   stage 1 (pheS m 0.89 of the truth on seed 1; `sim_realistic_s2` is the second calibration, seed 1), pheS (step 3), and
   moving the two stages into the pipeline once step 2 settles how.
2. [ ] **Carry k and m's error into X.** The held fit (step 1b, round 3)
   gives X intervals conditional on k and m. Take k and m's uncertainty
   from the no-doubles fit (its Laplace), refit the full library held at a
   handful of draws, and pool the draws' posteriors, as the two-stage fit
   of `planning/studies/svi-overconfidence/two_stage.py` did. A soft prior
   at the no-doubles SD will not do: the doubles pulled m through a 0.002
   prior (`planning/studies/km-anchor/`). Check coverage by depth against
   round 3 (0.90 / 0.65 above 1e4 reads).
   Set up 2026-10-11 on the seed-1 simulation (`km_draws.py`,
   `run_km_draws.srun`, `run_pool.srun`; 10 draws), with a second
   simulation seed run through both stages (`run_two_stage.srun`) to test
   stage 1's own error, which a stage-1 Laplace may understate
   (`planning/studies/m-bias-factorial/`, "Step 2", "Second seed").
   Found 2026-10-11: the stage-1 Laplace is far too narrow (m SD about
   0.6%), and stage 1's m error is a bias that repeats across seeds
   (kanR 2% steep, pheS 10% shallow; 12% under Poisson counts), so the
   draws were cancelled. k's error is noise several times its Laplace SD.
   Congression is the pheS bias's cause: without it stage 1's pheS m is
   1.035 of the truth (0.895 with it); kanR does not move. Next: a stage
   1 of wt and the spikes (little congression, few genotypes), on both
   seeds and the real data. Step 2 now needs (a) the stage-1 m bias
   fixed or corrected and (b) k's uncertainty from resampling stage 1,
   not its Laplace.
   (a) done 2026-10-12 in the simulation. Spikes alone worked on the
   simulation but not the real data (nine genotypes cannot pin X's
   scale), and adding the 300 deepest singles brought the pheS bias back.
   Congression is not the doubles' pull, which persists without it. A
   stage 1 with the congression mixture (max rule, lambda prior 0.357 +-
   0.15) removes the pheS bias (pheS m 1.02-1.04 of the truth). The recipe
   is a mixture stage 1 and a `single` stage 2: deepest-genotype coverage
   0.76 / 0.67 on seeds 1 / 2 (0.65 / 0.51 before); a mixture stage 2
   adds nothing for the deep genotypes. Left: a small steep m bias
   (2-6%), k's uncertainty (b), and, for the real data, wt's purity (the
   X gauge), which the mixture needs.
   Real mixture stage 1 (2026-10-12): congression barely moves the real
   pheS m (0.0061 to 0.0068), so the real pheS gap (about 2x the
   monoculture) is mostly not congression (step 3). The data lean toward
   wt mostly bulk (the real config as written: 25 nats better, fitted
   lambda 0.365 on the measured 0.357). kanR within 4% of the monoculture.
   The real stage 2 at that stage 1 (`held_mix_real`, 2026-10-12)
   converged with no k, m or slide direction held; its X is round 6's
   compressed about 6% (r >= 0.998), 1.39 times the original recipe's.
   Next (decided 2026-10-12, user): (i) k and m's uncertainty from a
   delete-d jackknife of the mixture stage 1 (drop 20% of tubes,
   stratified by condition and IPTG; tubes cannot be duplicated, so no
   classic bootstrap), checked on the simulation against stage 1's known
   errors on both seeds (`run_jackknife.srun`, `jackknife_summary.py`);
   (ii) then carry it into X. Refitting stage 2 at every draw costs
   2-5 GPU hours each; k and m mostly shift and stretch X, so refit
   stage 2 at two or three draws, measure that map per genotype, and
   apply it to all the jackknife draws. Check coverage by depth on the
   simulation.
   Jackknife result (seed 1, 2026-10-12): k's jackknife SE is about five
   times its Laplace SD and covers k's errors (0.5-0.8 SE); m's agrees
   with its Laplace and its error (6-9 SE) is the accepted bias. 94% of
   k's variance is one direction, so two stage 2 refits along it (plus
   and minus one SD) measure X's k sensitivity; `k_band.py` widens X's
   intervals with it. Scored on seed 1 (2026-10-13): 95% coverage 0.94 at
   1e4-1e5 reads and 0.91 above 1e5 (0.90 and 0.76 without), the rest
   being the accepted m bias. k's uncertainty is a shift shared by every
   non-wt genotype (one SD moves X by -0.032 to -0.038, genotype-to-
   genotype SD 0.005), so the pipeline should add it as one shared
   offset per posterior draw, not as independent width.
   Accepted as a limitation (user, 2026-10-12): stage 1's small steep m
   bias (2-6% in the simulation, an X slope of 1.03-1.05). Resampling
   does not cover a bias; quantify it from the simulation for the paper,
   and correct it from the simulation only if the methods need it.
3. [ ] **pheS+4CP.** Accepted as a limitation for now, possibly for the
   paper (user, 2026-10-12): the real pool's pheS selection is about half
   the wt monoculture's (m 0.0068 against 0.0144) and congression explains
   little of it. The two-stage fit takes pheS's m from the pool data with X
   set through kanR, so the gap need not distort X's scale. It could
   distort the curves' shapes if its cause depends on genotype (a
   selection-onset transient, `planning/offset-mode-growth-transition.md`).
   Check: stage 2 on the kanR data alone, against the both-marker fit, on
   the real data and on the seed-1 simulation (which shows what agreement
   looks like when pheS is fine). If they agree, the gap is an observation
   about the pool, not a problem with the curves.
   Result (2026-10-12): the median real genotype agrees across markers as
   in the simulation, but 8% of the deep doubles disagree beyond 2 SD
   (2.5% in the simulation). Keep the gap as a limitation and flag those
   genotypes (marker-consistency z between the kanR-only and both-marker
   fits). Candidate cause: dk_geno shared across markers while real
   fitness effects may differ in kan and 4CP; a marker-specific dk_geno
   would test it. Tested 2026-10-12 without a refit (`control_dk_check.py`):
   the tail's X offsets track the difference in control-condition growth
   between libraries no more than on the simulation, where dk is shared
   (r 0.40 in both; slopes about 6, against about 34 for a real effect),
   and the real controls barely agree on any genotype's dk (r 0.04). Not
   supported; the tail stays unexplained and flagged.
4. [ ] **Then the shallow genotypes' points.** Hold the X population SDs
   (`theta_{X_low,X_delta,log_hill_K}_hyper_scale_fixed`) so genotypes
   below 1e3 reads shrink toward the population instead of scattering;
   their intervals already cover.
5. [ ] **The pipeline.** Make the two-stage fit one documented recipe (or
   one command): the stage-1 subset (wt, spikes, singles) fitted with the
   congression mixture, stage 2 on the full library with every
   condition's k and m pinned at it, and the resampled intervals of step
   2. The students' design simulations and the NSF preliminary data use
   it. Step 2's method is settled (2026-10-13): a delete-d jackknife of
   stage 1 (20% of tubes, about 30 refits of about 950 genotypes), two
   stage 2 refits at plus and minus one SD along k's main direction
   (started from the stage 2 MAP), and the X shift between them added to
   the posterior draws as one shared offset per draw. Pieces to build:
   the stage-1 subset and its mixture fit, the jackknife, the two
   refits, and the shared-offset draws in `tfs-sample-posterior` or a
   post-processing step; plus the marker-consistency flag (step 3).
   Planned in `planning/two-stage-fit.md` (active, 2026-10-13).

## Out of scope

New sequencing or a new screen design (`planning/design-noise-sweep.md`),
and the simulator's remaining depth mismatches (dropout tail, wt's share of
reads), which do not touch the deep genotypes.
