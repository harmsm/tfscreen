---
title: Close out the 0.5.0 release (X with honest intervals, runnable by the lab)
status: active
filed: 2026-10-19
area: release
revisit_when: >-
  Decisions settled by the user, 2026-10-19. Keep the checklist current;
  0.5.0 and its 0.5.x bugfix releases are what Natalie's manuscript uses.
related:
  - planning/two-stage-fit.md
  - planning/deep-coverage.md
  - planning/relative-m.md
  - planning/findings-ledger.md
  - planning/studies/two-stage-validation/README.md
  - planning/studies/relative-m/README.md
  - CHANGELOG.md
---

## Goal

Someone in the lab other than Mike (Natalie first) takes a screen from
per-tube counts to per-genotype X curves with reasonable 95% intervals,
using one documented command and no study scripts. The release is that
route, validated, documented and tagged.

**The deliverable is X, not theta.** X is the wt-relative occupancy-like
variable of `hill_relative` (wt X = 1 at the lowest IPTG, 0 at the
highest). Comparing X with binding happens after the fit and is not part
of the release.

## Where things stand (2026-10-19)

- **The route.** `tfs-fit-two-stage` (`planning/two-stage-fit.md`).
  Stage 1 fits wt, the spikes and the singles with the congression
  mixture (`max` rule). Stage 2 fits the full library with every
  condition's k and m pinned. A jackknife of stage 1 gives k's error, and
  a marker-consistency flag runs at the end. Validation run 1 reproduced
  the study's stage 1, jackknife, stage 2 and flag
  (`planning/studies/two-stage-validation/`).
- **Coverage.** On the simulations, deep doubles' 95% X coverage without
  the k shift is 0.90 / 0.76 (seed 1, 1e4-1e5 / >1e5 reads) and
  0.91 / 0.81 (seed 2). The study's hand-built shift reached 0.94 / 0.91.
- **The k-shift step does not work as built.** Its refits leave stage 2's
  MAP and slide along the dk_geno population against wt (real X +0.32,
  the same with k unchanged). So the command's shift measures where the
  optimizer stopped. It got 0.915 / 0.858.
- **Congression in X** needs the `max` rule. The partition-function rules
  cannot be used with X, and stage 2 is `single` anyway.
- **The kanR warp** (`m_warp`) is real on the real stage 1: about 8,000
  nats for two parameters, and nearly linear on the simulation. It closes
  little of the pheS gap. A pheS warp collapsed to a step. Warping both
  markers was not tested. The warp is not carried through
  `tfs-fit-two-stage`: its pin table holds only k and m.
- **Monoculture rates.** The only monoculture data are wt's k and m
  (`2025-10-02_control-experiments.xlsx`). As k/m priors the data
  rejected them (`planning/deep-coverage.md` step 1). They are now a
  check only: the real stage 1's kanR is within 4% of them, and pheS's m
  is half. No mutant monoculture dk_geno data are in the dev data. The
  `base_growth_df` input exists and the two-stage YAML accepts it, but it
  has never been run with `hill_relative`.
- **Docs.** `docs/source/fitting.rst` (last edited 2026-10-05) describes a
  single `hill_relative` fit. The two-stage route appears only in
  `cli.rst`. The quickstart is about simulations.
- **Tests.** The full suite passed on 2026-10-18 (6,109 passed).
  `__version__` already reads 0.5.0. The last tag is v0.4.4.

## Decisions (user, 2026-10-19)

1. [x] **k's uncertainty in X.** Repair the shift first: start the two
   refits at the final step size with the dk_geno population location
   held at stage 1's (ledger open item 4), then one validation round. If
   that fails, ship without the shift and report the no-shift coverage
   (0.90 / 0.76-0.81). Replacing it with k and m free under the jackknife
   covariance is out of scope.
   Outcome (2026-10-20): the repair failed validation round 2 (the shift a
   third or less of the true one on simulations;
   `planning/studies/two-stage-validation/`, run 2), so the release ships
   without it (user: 0.76-0.90 coverage is fine for now).
   `two_stage.kshift` defaults to false and skips the jackknife too.
2. [x] **The kanR warp is out of 0.5.0.** Its gain (about 8,000 nats on
   the real stage 1) is recorded for 0.6. Stage 2's X stays validated
   without it.
3. [x] **Drop pheS+4CP replicate 2, t_sel 82.** Data: the IPTG tubes look
   reversed with the 0 tube in place (the tube labeled 0.1 uM is a 1 mM
   tube, the one labeled 1 uM a 30 uM tube), but the 10 uM slot does not
   fit and the low tubes' order cannot be recovered
   (`planning/studies/relative-m/`, run 7). The lab found no mechanism
   for a three-way swap, and Natalie rules out a flip of the IPTG tubes
   with the 0 tube in place. The row goes in a corrections file as an
   exclusion, with that evidence.
4. [x] **The lambda prior** stays at the measured value (0.38 +- 0.15).
   The docs say it is required and where it comes from.
5. [x] **Usable genotypes.** The docs state one default read-depth
   threshold and one policy for the consistency flag. Set both from
   validation round 2's coverage by depth and the flag's no-signal rate.
   Set 2026-10-20: 1,000 reads (seed 1: slope 0.88 and RMSE 0.17 at
   1e3-1e4 reads, against 0.39 and 0.54 below); flagged genotypes are
   reported separately, not dropped.
6. [x] **kanR's m scale** (`planning/studies/relative-m/`, run 6: the
   absolute m against the within-tube m, about 2.5x apart). The two-stage
   fit takes m from the tube totals, which binding at 2 uM supports and
   the weak binders' flat kanR response does not. Decided: ship the
   tube-totals m and state the range as a limitation in the
   interpretation section.

## Code

6. [x] Implement decision 1 in `tfs-fit-two-stage` and its tests.
   Done 2026-10-19: `kshift_step_size` (1e-5), the dk_geno population
   held at stage 2's MAP (not stage 1's: holding stage 1's would change
   the validated stage 2), and a k-unchanged control refit with
   `kshift/drift.csv`. Validation round 2 (item 12) is set up in
   `planning/studies/two-stage-validation/`, run 2.
7. [x] The dev data's corrections file: exclude pheS+4CP replicate 2,
   t_sel 82 (decision 3). Document
   `tfs-correct-tubes` -> `tfs-process-counts` as the route, since the
   two-stage fit reads the processed growth table.
   Done 2026-10-19: `planning/dev-data/tube_corrections.yaml` (gitignored,
   C10) excludes the row's six pheS+4CP tubes (the pheS-4CP control tubes
   of that replicate and time stay), checked with `tfs-correct-tubes` (118
   tubes to 112). `planning/studies/real-data-fit/fit/run_recipe.srun` runs
   `tfs-correct-tubes` before `tfs-process-counts` (`CORRECTIONS`, default
   `../../tube_corrections.yaml`). The recipe page (item 15) documents the
   route.
8. [x] Refuse or warn on the experimental joint-fit pieces in a two-stage
   config: `theta_growth_log_conc_shift_scale`, `genotype_offset` and a
   binding table. Mark them experimental in their docstrings.
   Done 2026-10-19: `read_two_stage_config` refuses `binding_df`, theta
   noise other than `zero`, the joint-fit and warp priors in `set_priors`,
   and an `m_warp` column with any 1 in `growth_priors` (stage 1 would fit
   the warp and stage 2's pins drop it). `genotype_offset`, hill_geno's
   titrant shift and linear's `m_warp` say they are experimental.
9. [x] Make the outputs easy to read: one top-level table per genotype
   and concentration with X quantiles, read depth, `in_training_data` and
   the consistency flag, so users never open the step directories.
   Done 2026-10-19: the `summary` step writes `{out_dir}/tfs_x.csv`
   (also `k_shift`, `num_mutations` and reads per library). On run 1's
   real outputs: 217,945 genotypes, 1.7 M rows, about 40 s to count the
   reads. A depth flag waits on decision 5's threshold.
10. [x] Turn epistasis on X on in the recipe (`predict_epistasis`), or
    document how to run `tfs-predict-epistasis --shared_shift` afterward.
    Natalie's paper needs it.
    Done: the screen page's YAML sets `predict_epistasis: true` and says
    when to (the command's default stays false). With the k shift off
    there is no `--shared_shift` to document.
11. [x] Pin or floor the dependency versions that broke CI (pandas 3).
    Check a fresh install on the cluster.
    Done locally 2026-10-20 (`planning/studies/fresh-install/`): floors in
    `pyproject.toml` at the maintainer's passing environment (no caps; CI
    tests the newest) and a `cuda12` extra for GPU jax on Linux; a fresh
    venv on the newest releases (pandas 3.0.6, numpy 2.5.4, jax 0.12.0)
    passed the full suite. Install docs (README, quickstart, screen page)
    updated. The cluster install works (user, by hand, 2026-10-20). No
    constraints file in the repo (user): the manuscript records its own
    versions (item 23). CI's 3.11 jobs then failed (jax 0.11 has no 3.11
    wheel), so tfscreen needs Python 3.12+ and CI tests 3.12-3.14.

## Validation

12. [x] Validation round 2 with decision 1 in: seed 1, seed 2 and the
    real data through the command. Done 2026-10-20: the repair failed
    (shift too small on the simulations), so decision 1's fallback
    applies; seed 1's coverage by depth without the shift is recorded in
    `planning/studies/two-stage-validation/`, run 2.
13. [x] A dry run by Natalie on the dev data from a clean install,
    following only the docs. Every question she asks becomes a docs fix.
    Done 2026-10-22 to 2026-10-24: the user and Natalie both ran it. Docs
    fixes: run processing on a compute node, `consistency_library` is a
    convention, run the submit script or the command (not both), and the
    growth-table section rewritten as numbered steps (the inputs list no
    longer implies a corrections file already exists).
14. [x] `two-stage-fit.md` step 4 marked done, and the plan's status set.
    Done 2026-10-24: steps 4 and 8 done; the plan stays active for steps
    5 and 7 (the design-sweep tools, after the release).

## Docs

15. [x] A real-data recipe page: counts, `tfs-process-counts`,
    `tfs-check-tubes` and `tfs-correct-tubes`, `tfs-configure-model`'s
    inputs, `tfs-fit-two-stage`, the outputs. Include cost and wall time,
    and an example SLURM script with the jackknife as an array.
16. [x] An interpretation section:
    - what X is, and why it is not binding (the affine map; the M42I
      case);
    - what the intervals include: k's shift, if decision 1 lands;
    - what they leave out: stage 1's 2-6% m bias, the pheS gap,
      congression error per genotype, and kanR's m scale (decision 6);
    - that shallow genotypes are prior-dominated;
    - what the consistency flag means.
17. [x] `fitting.rst`: make the two-stage route the default for
    growth-only screens. Mark the single-MAP `hill_relative` fit, the
    joint absolute fit and the thermodynamic components as advanced or
    experimental.
18. [x] `README.rst` and `quickstart.rst`: point to the recipe page.
   Drafted 2026-10-19: `docs/source/screen-analysis.rst` ("Analyzing a
   screen") covers 15 and 16, with `examples/tfmodel/two_stage_submit.sh`,
   `two_stage_step.srun` and `growth_priors_loose.csv`; `fitting.rst`
   points to it and marks the joint model experimental (17); the index,
   quickstart and README point to it (18). Updated 2026-10-20 for the
   fallback: no k shift, the 1,000-read threshold, the coverage-by-depth
   table, and k's uncertainty listed among what the intervals leave out.

19. [x] Regenerate `cli.rst`, and update CLAUDE.md.
    Done 2026-10-19: `cli.rst` was current (its test passes). CLAUDE.md:
    the relative-X section names the two-stage fit as the route (it said
    the choice was open), and the planning section leads with this plan.
    `pipeline.rst` points to the screen page and to `tfs-check-tubes` /
    `tfs-correct-tubes` (it sent tube cleanup to the lab's own script).

## Release mechanics

20. [ ] CHANGELOG: fold `[Unreleased]` into the existing `[0.5.0]`
    section and change its date (2026-10-03) to the release date. 0.5.0
    was never tagged or shared in the lab (user, 2026-10-19): the bump
    came before the two-stage work, so the release stays 0.5.0 and
    `__version__` is already right. Add an upgrade note: the removed
    commands and the configs that no longer load.
21. [ ] Full test suite, with `--runslow` and the smoke tests. Run lint.
22. [ ] Tag v0.5.0 and write GitHub release notes. Install that tag on
    the cluster.
23. [ ] Record the commit and version Natalie's analysis uses in her
    manuscript folder, with the environment's `pip freeze`.

## Out of scope (after 0.5.0)

- The official analysis for the paper, including X against binding
  (`planning/studies/x-vs-binding/`) and the M42I re-prep.
- Thermodynamic models with a floating protein concentration (NSF).
- Experimental design: `tfs-build-sim-config` and the scoring CLI
  (`two-stage-fit.md` steps 5 and 7), and
  `planning/design-noise-sweep.md`.
  - The kanR warp (decision 2), the relative-m plan,
    the k/m-under-jackknife-prior alternative (decision 1), the joint fit
    (`planning/joint-fit-return.md`, paused), the X population SDs held for
    shallow genotypes (`deep-coverage.md` step 4), and mutant monoculture
    dk_geno data as `base_growth_df`.
