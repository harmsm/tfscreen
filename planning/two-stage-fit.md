---
title: The two-stage fit as one command, with honest intervals
status: active
filed: 2026-10-13
area: tfmodel
revisit_when: >-
  Active (decisions settled by the user, 2026-10-13). Students' design
  simulations for the NSF preliminary data (due 2026-11-20) are the first
  users. Keep the step list current.
related:
  - planning/deep-coverage.md
  - planning/experiment-pipeline.md
  - planning/design-noise-sweep.md
  - planning/studies/m-bias-factorial/README.md
  - planning/studies/full-size-sim/make_sim_config.py
  - src/tfscreen/tfmodel/scripts/configure_model_cli.py
  - src/tfscreen/tfmodel/scripts/fit_model_cli.py
  - src/tfscreen/tfmodel/scripts/predict_theta_cli.py
  - src/tfscreen/tfmodel/analysis/prediction.py
---

## Why

The m-bias study (`planning/studies/m-bias-factorial/`) found the fit
that gives honest X for this screen, but it runs as a dozen study
scripts and hand-copied cluster commands:

- The full-library MAP compresses X about 1.45-fold: each double pulls m
  steeper, and the pulls add. A stage 1 without the doubles gets k and m
  right; stage 2 fits the full library with them pinned.
- Congression flattens pheS's m about 10% in any stage 1 large enough to
  pin X's scale; stage 1 must fit the congression mixture (`max` rule,
  lambda fitted with the lab's measurement as its prior). Stage 2 does not
  need it.
- Stage 1's Laplace understates k's error about fivefold. A delete-d
  jackknife of stage 1 measures it. Its effect on X is a shift shared by
  every non-wt genotype (one SD moves X by -0.032 to -0.038 on the
  simulation; genotype-to-genotype SD 0.005).
- With that shift in, the simulation's 95% X coverage is 0.94 at 1e4-1e5
  reads and 0.91 above 1e5 (0.90 and 0.76 without), the rest being stage
  1's accepted steep m bias (2-6%).

Students running design simulations, and the NSF preliminary data, need
this as one documented, tested command that reproduces those numbers.

## The fit, step by step

1. **Stage 1 genotypes.** wt, every spiked genotype and every single
   (from the library config: `in_spiked_origin`, or one mutation), taken
   from the growth table in memory. About 950 genotypes here.
2. **Stage 1 fit.** The configured model plus `transformation: mixture`,
   `congression_theta_rule: max` and the lambda prior; staged MAP. Keeps
   every condition's k and m (and the fitted lambda, for the record).
3. **Jackknife.** N subsamples (default 30), each dropping a share of the
   tubes (default 20%) at random, stratified by condition and IPTG level,
   keeping one tube per stratum; each refits stage 1. Tubes cannot be
   duplicated (a duplicate lands on its original's tensor cell), hence no
   classic bootstrap. Output: k and m SEs (variance factor (n - d) / d),
   k's covariance and its main direction.
4. **Stage 2.** The full library, `single`, every condition's k and m
   pinned at stage 1 (`--growth_priors` pin columns); staged MAP; arrowhead
   Laplace.
5. **The k shift.** Two stage 2 refits with k at plus and minus one SD
   along k's main direction (m at stage 1's), started from stage 2's MAP
   (`--init_from`); X predicted at each. The shift is half their
   difference, per genotype and concentration.
6. **Posterior draws with the shift.** Each posterior draw gets one shared
   standard normal z, and every non-wt genotype's X in that draw moves by
   z times its shift; then quantiles. Rankings and differences between
   genotypes mostly cancel it; epistasis on X keeps the one copy it
   should (wt does not shift).
7. **Marker-consistency flag** (on by default; `--no_consistency` skips
   it). Stage 2 refit on one library's
   tubes (kanR), against the full stage 2; per genotype, the z-score of
   the level offset between them. About 8% of the real deep doubles
   disagree beyond 2 SD against 2.5% simulated (cause unknown; not a
   library-specific dk_geno).

## Interface

One new command, `tfs-fit-two-stage`, that chains the existing CLI
functions (it does not reimplement them). Each step writes its own
subdirectory (`stage1/`, `jackknife/jk_NN/`, `stage2/`, `kshift/plus/`,
`kshift/minus/`, `consistency/`) and is skipped when its outputs exist, so
a run resumes and a step reruns by deleting its directory.

- `tfs-fit-two-stage config.yaml` runs every step in order on one machine.
- `--step NAME` runs one step (`stage1`, `jackknife`, `stage2`, `kshift`,
  `predict`, `consistency`), and `--jackknife_task i` one jackknife
  subsample, so a SLURM array can run the jackknife in parallel. The
  command never writes cluster settings; the docs show an example array
  script.
- The YAML holds what `tfs-configure-model` takes now (data, library
  config, model choices, held priors) plus the two-stage settings: the
  lambda prior (required with congression), the jackknife N and drop
  share. The consistency step runs unless `--no_consistency` (or the
  YAML) turns it off.
- Prediction takes the shift as an input: `tfs-predict-theta` and
  `tfs-predict-epistasis` get `--shared_shift kshift.csv`, applied to the
  draws before quantiles (`prediction.predict` already works from joint
  draws). The two-stage command writes `kshift.csv`.

This is a stage of the deferred orchestrator (`planning/experiment-
pipeline.md` step 5, `tfs-run-experiment`): built so the orchestrator can
call it later, without waiting for it.

**Cost per experiment** (one GPU, full-size library): stage 1 15-30
minutes; jackknife 30 x 15-30 minutes (about 10 hours serial, about one
hour as an array of 10); stage 2 2-5 hours plus the Laplace; the two
shift refits 1-2 hours each from the MAP; consistency about 5 hours.
About a day serial.

**Design simulations do not need the intervals.** A design sweep compares
points (slope, RMSE, r by depth) across noise settings; `--step stage1`,
`--step stage2` and `predict` without the shift cost about 3-6 hours per
simulation. Intervals matter for the final analysis.

## Steps

1. [x] **Library functions, with tests.** Port from the study scripts:
   stage 1 genotype selection; stratified tube subsampling
   (`resample_growth.py`); the jackknife summary, k's main direction and
   the plus/minus pin tables (`jackknife_summary.py`, `held_priors.py`);
   the shift table (`k_band.py`). Done 2026-10-13: `tfmodel/two_stage.py`,
   `tests/tfscreen/tfmodel/test_two_stage.py`; on the seed-1 jackknife and
   k-band refits it reproduces the study's k and m SEs and the shift
   exactly (k's step is 1.7% smaller: the covariance now divides by N, as
   the jackknife variance does, where the study script divided by N - 1).
2. [x] **The shared shift in prediction.** `--shared_shift` for
   `tfs-predict-theta` and `tfs-predict-epistasis`. Tests: one z per draw
   shared across genotypes; wt unshifted; quantiles widen by the shift
   in quadrature when the posterior is otherwise narrow; differences
   between genotypes unchanged. Done 2026-10-13:
   `tfmodel/analysis/shared_shift.py`, wired into `extract_theta_curves`
   and `extract_theta_epistasis` and both CLIs (refused for out-of-training
   genotypes and for a single MAP draw);
   `tests/tfscreen/tfmodel/analysis/test_shared_shift.py`.
3. [x] **`tfs-fit-two-stage`**, with step skipping, `--step` and
   `--jackknife_task`; the YAML schema; `docs/source/cli.rst`. Done
   2026-10-13 (`tfmodel/scripts/fit_two_stage_cli.py`,
   `examples/tfmodel/two_stage.yaml`). The YAML's top-level keys are
   `tfs-configure-model`'s arguments (paths relative to the YAML); the
   `two_stage` block holds transformation_lambda (required), seed,
   jackknife_subsamples, jackknife_drop_frac, num_posterior_samples,
   consistency, consistency_library, predict_epistasis, fit and posterior
   (extra arguments). It refuses a non-relative or non-linear model,
   unshared replicates, transformation keys at the top level and pins in
   set_priors. Each step runs in its own directory with absolute input
   paths; the stage-1 and one-library tables are written by chunked
   filters (a full-size table is 8 GB). Tests mock the chained functions
   (`test_fit_two_stage_cli.py`); a real end-to-end run on round 1's small
   table (6,500 genotypes, fits cut to 40 steps) finished every step in
   4.7 minutes, stage 2's k equal to stage 1's, wt's intervals untouched,
   and a rerun skipped everything.
4. [ ] **Validate by reproduction.** Rerun the seed-1 and seed-2
   simulations through the command and match the study's numbers
   (stage 1 m ratios; coverage 0.94 / 0.91 on seed 1); rerun the real
   data and match `held_mix_real`.
5. [ ] **Scoring for simulations.** X against the truth by read depth
   (`score_x.py`) as a `tfs-summarize-fit` option or its own CLI, so the
   students score design sweeps without study scripts.
6. [x] **Consistency flag** step (on by default, `--no_consistency`) and
   its output column. Built with step 3: `consistency/consistency.csv`
   (per genotype: level offset after the affine map, its SD, z,
   `inconsistent` beyond |z| 2; `two_stage.marker_consistency`). Its
   threshold should be calibrated against the simulation in step 4.
7. [ ] **A simulation config from a real experiment**, for the students'
   design sweeps: `tfs-build-sim-config`, the port of
   `planning/studies/full-size-sim/make_sim_config.py`. Inputs: the tube
   table (the simulation follows it as its `design`), the per-tube counts
   (total reads and each library's unassigned share), the initial (pre-
   split) composition (the realized `library_mixture` via
   `estimate_library_mixture`, and wt's share of the pool), the library
   config, the OD600 calibration, the wt monoculture rates (k and m per
   selection condition at the gauge concentrations), and optionally a
   pilot run to calibrate cfu0 per library from real against simulated
   tube totals. The values the study fitted to this screen by hand become
   explicit arguments, not constants: pool skew by origin (doubles 2.0,
   singles 0.8, spikes 0.5 here), clones per double (300), the dk_geno
   spread (0.15), the pre-split OD600 and volumes, lambda. A documented
   example (with this screen's values, no lab data) shows them. Validate:
   the built config reproduces `sim_realistic_s2`'s read depth (median
   double about 470 reads) and composition.
8. [ ] **Docs.** A recipe page (what each step does and why, with the
   study's numbers), an example SLURM array script, the students'
   simulate-fit-score loop, CLAUDE.md, CHANGELOG.

## Decisions (user, 2026-10-13)

1. A standalone command, `tfs-fit-two-stage`, callable later by the
   deferred orchestrator (`tfs-run-experiment`).
2. The shift is a prediction input (`--shared_shift`), not written into
   the posterior file.
3. Defaults: stage 1 = wt, spikes and singles; jackknife 30 subsamples,
   20% of tubes dropped; the lambda prior required whenever congression is
   fitted, with no default value.
4. wt purity: the library config as written (wt mostly bulk), which the
   real data favored.
5. The consistency step is on by default; `--no_consistency` skips it.
6. The students' simulation-config builder is part of this plan (step 7).

## Out of scope

The accepted limitations: stage 1's small steep m bias (quantified in the
study; correctable from simulation if the methods need it) and the real
pheS gap (`planning/deep-coverage.md` step 3). Marker-specific dk_geno
(tested, not supported).
