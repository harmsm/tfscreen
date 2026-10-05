---
title: One experiment file from raw data to posterior
status: active
filed: 2026-10-03
area: tfmodel
revisit_when: >-
  Agreed 2026-10-03 for the next branch, after
  claude/physics-improvements-roadmap-03a273 merges. Keep the step list
  current while it is worked. Re-ordered 2026-10-05: steps 6 and 7 next,
  step 5 (the orchestrator) deferred, then a science prioritization.
related:
  - planning/analysis-roadmap.md
  - planning/analysis-roadmap-summary.md
  - planning/studies/real-data-fit/README.md
  - src/tfscreen/process_raw/scripts/process_counts_cli.py
  - src/tfscreen/tfmodel/scripts/configure_model_cli.py
  - src/tfscreen/tfmodel/scripts/prefit_calibration_cli.py
  - src/tfscreen/tfmodel/scripts/fit_model_cli.py
  - src/tfscreen/simulate/empirical/
---

## Why

The CLI chain was built for the old flow: `tfs-process-fastq`, then
`tfs-process-counts`, then `tfs-configure-model`, then
`tfs-prefit-calibration`, then `tfs-fit-model`. The analysis-roadmap
branch changed what goes in and how a fit has to be run. It added the
count likelihood, OD600 tube totals, the growth-only relative fit, held
population SDs, staged MAP starts and the arrowhead Laplace. Each gap
between the old chain and the new needs was patched by hand. The real fit
(`planning/studies/real-data-fit/`) took three hand steps:

1. **Processing.** `prep_dev_data.py` turned OD600 into tube totals through
   the calibration, fixed the library config and cleaned up lab problems.
   `tfs-process-counts` then ran once per library, and the outputs were
   joined by hand.
2. **Priors.** `set_growth_priors.py` wrote k/m priors, because
   `tfs-prefit-calibration` refuses growth-only models, and `sed` set the
   held population SDs and the tube-offset SD in the generated priors CSV.
3. **The staged MAP.** A no-offset MAP, then `make_init_npz.py` and
   `check_warm.py`, then `tfs-fit-model --init_from`, over a chain of six
   runs.

Nobody else can run that, and it cannot be rerun on a full-size simulation
without redoing it by hand.

## Approach

Fix the three seams first, each as a small, tested change to an existing
CLI. Then add a thin orchestrator that only calls the CLIs in order. An
orchestrator built first would wrap the hacks and freeze them.
`tfs-build-empirical` is the precedent for a CLI that runs others.

## Done when

Revised 2026-10-05 (user), with step 5 deferred and the offset mode found
to be a model problem (`planning/offset-mode-growth-transition.md`):

- The real fit reruns through the documented CLI chain (process, configure,
  fit, posterior, extract, predict) with no hand-edited intermediate file
  and no hand-built start. The chain's commands are the official recipe
  for the real fit, replacing the `real-data-fit` study's six-run chain.
- A full-size simulation of the same design, written by `tfs-simulate` in
  the raw formats (step 6), runs through the same chain with only the data
  paths changed. That is the start of the reviewer simulations in
  `planning/studies/real-data-fit/README.md`, "Next", and the footing for
  every model study after this plan: a user can reproduce one end to end.
- Both report the tube-offset diagnostic (step 7), so a fit in the offset
  mode is flagged, not accepted silently.

The original criterion, every spike's K and n inside the `rel_off_n05_alap`
intervals, was dropped: run to convergence, the start behind
`rel_off_n05` moves into the offset mode, so its intervals are not a
reference.

## Constraints

- **P1. Lab data and calibrations stay out of the repository** (roadmap
  C10). The experiment YAML names files; the repository holds formats,
  methods and synthetic examples.
- **P2. Breaking changes are fine** (roadmap D12). Old processed files and
  configs need not load.
- **P3. Leave room for the population model** (roadmap step 6). Processing
  supplies per-tube totals from OD600 today. Step 6 will instead fit a
  population curve to every OD600 reading inside the model. Keep the OD600
  table a first-class input, so that change replaces the totals step
  without changing the experiment file's shape.
- **P4. Lab-specific cleanup stays the lab's.** Swapped samples, pooled
  resequencing and failed tubes are the lab's business. The pipeline takes
  a clean tube table and documents its format. It may offer a hook for a
  per-experiment cleanup script, but it encodes none of that cleanup.
- **P5. Every run records its provenance.** Each CLI writes the tfscreen
  version and git commit into its outputs, and the orchestrator also
  records input checksums. The study READMEs had to infer commits from
  file times (2026-10-03).

## Steps

0. [x] **Provenance in every CLI** (P5). `generalized_main` prints, and
   each writer records, the version, the commit (with a dirty flag) and the
   command line. Do this first and small; it pays off at once.
   Done in 0.5.0 (`util/provenance.py`): every run prints it and writes
   `{out_prefix}_provenance.json`; the configure YAML and fit checkpoints
   embed it. Posterior `.h5` files do not yet; their run's JSON sits
   beside them.
1. [x] **Processing takes OD600 directly** (seam 1).
   - `tfs-process-counts` takes the per-tube OD600 table and a
     `tfs-calibrate-od600` file. It computes `sample_cfu` and its SD
     itself, keeping the shared curve error apart from the per-reading
     noise (C11). Supplied `sample_cfu` stays available.
   - It processes several libraries in one call, from one tube table with
     a `library` column. That replaces the per-library runs and the join.
   - Document the tube-table and OD600 formats with a synthetic example in
     `examples/`.
   - The spike list and the M42I codon problem came from a processing
     config that disagreed with the real spikes. Check at configure time
     that every spike named in the library YAML is present in the counts.
   Done in 0.5.0: `--od600_file`, `--od600_calibration_file`,
   `--tube_volume_mL` (`od600.tube_totals_from_od600`); several libraries
   already worked in one call (per-library filter and fill), now tested;
   `examples/process_raw/`; `tfs-configure-model` refuses a missing spike
   unless `--allow_missing_spikes`. `tfs-process-presplit` became
   `tfs-process-counts --presplit`.
2. [x] **Configure exposes everything that was hand-edited** (seam 2).
   - Flags for `sigma_fixed` on level offsets and for every
     `theta_*_hyper_scale_fixed`.
   - A growth-prior step for growth-only models, replacing
     `set_growth_priors.py`. Per-condition k/m loc and scale come from
     defaults, a table, or wt monoculture rates (the `monokan` rule:
     k = rate at the high gauge concentration, m = the difference). The
     pre-fit stays for joint models.
   - Decide whether the growth-only defaults change: n held at 0.5, X_low,
     X_delta and log K held, offsets at 0.17. The real fit and the
     sensitivity runs say how much each matters
     (`planning/analysis-roadmap-summary.md`).
   Flags done in 0.5.0: `--set_priors name=value ...` (any scalar prior,
   by full name or unique suffix), `--growth_priors` (per-condition table)
   and `--growth_priors_wt_rates` (the monokan rule, hill_relative only),
   in `tfmodel/priors_edit.py`. Closed 2026-10-05. The defaults did not
   change (user, 2026-10-04); the decision moved to
   `planning/offset-mode-growth-transition.md`, since the held SDs and the
   offset SD are likely to change with the growth-transition model.
3. [x] **The staged MAP inside `tfs-fit-model`** (seam 3).
   - With level offsets on, the fit stages itself. First a MAP with the
     offsets held at 0. Then each tube's offset alone, with everything else
     held (what `small_offsets.py` did, cheap). Then the joint MAP from
     there at a small step size.
   - Each stage writes its own checkpoint and convergence record, so a
     stage can be inspected or resumed.
   - This retires `make_init_npz.py`, `make_warm_start.py` and
     `check_warm.py`. Keep `--init_from` for experiments.
   - Test on a simulation: does a cold level-offset MAP reproduce the ±2.8
     offset trap, and does the staged one avoid it? Then on the real data.
   Built after 0.5.0: `tfs-fit-model --stage_offsets` (default `auto`),
   `inference/staged_map.py`, with unit and smoke tests. The trap was only
   ever seen on the full dev data, so the test runs there first
   (`planning/studies/staged-map/`); a full-size simulation waits on step
   6. Retire the hand scripts once the study passes.
   Study result (2026-10-05): the staged MAP avoids the offset mode, but run
   to convergence on the exact full-batch loss (also added) the offset mode
   scores best: it is the model's preferred optimum, not a trap, and its
   IPTG-structured offsets point at a missing selection-onset transient
   (roadmap 7b).
   Closed 2026-10-05 as machinery (user): the staged MAP and exact-loss
   convergence work, and the staged fit is the physical one (3.7e5 nats
   better than `rel_off_n05` as it was used). The open problem is the
   model, carried by `planning/offset-mode-growth-transition.md`: under
   the current model the offset mode is the better optimum, so a longer
   or better fit can still leave the physical basin. Step 7's offset
   diagnostic guards against that until the model is fixed. The hand
   scripts retire in step 7.
4. [x] **Posterior defaults for large libraries.** Choose the Laplace
   automatically from the library size: the full Laplace below a
   parameter threshold, the arrowhead (`--laplace_blocks --laplace_shared`)
   above it. Report the held Schur directions in a file as well as the
   log, since they say which k/m directions have no interval.
   Done in 0.5.0: `tfs-sample-posterior --laplace auto|full|arrowhead|blocks|point`,
   auto switching at `--laplace_max_params` (20,000 MAP parameters, about
   1.6 GB of float32 Hessian); `{out_prefix}_held_directions.csv`. A model
   whose genotypes couple (hill_mut) cannot use the arrowhead, so auto
   fails on a large one of those; there is no full-library route for it.
5. [ ] **The orchestrator.** `tfs-run-experiment experiment.yaml`.
   Deferred 2026-10-05 (user): a convenience over the CLI chain, which
   steps 6 and 7 document and test first.
   - The YAML names the inputs (counts directory, tube table, OD600 table,
     calibration, library YAML, optional binding and monoculture rates)
     and the choices (model, priors, staging, posterior, predictions).
   - It runs processing, configure, priors, fit, posterior, extract and
     predict in order, calling the existing CLI functions.
   - Each stage writes into its own subdirectory and is skipped if its
     outputs are already there, so a run is resumable and a stage can be
     rerun by deleting it.
   - It writes a manifest: commit, input checksums, the resolved YAML.
   - It can emit a SLURM script for the fit and posterior stages.
     Cluster-specific settings (account, partition, memory) come from the
     YAML, never from code.
   - Scoring (`score_counts.py`) becomes a CLI if the orchestrator reports
     a fit's count likelihood; otherwise it stays a study script.
6. [ ] **The simulator writes the same raw formats** (next).
   - `tfs-simulate` writes per-tube count files in `tfs-process-fastq`'s
     format (`counts_<sample>.csv`: `genotype`, `counts`, with the
     `__unknown__` row), the tube table and the OD600 table in step 1's
     formats, so a simulated experiment enters at `tfs-process-counts`,
     exactly where real data do. Ground truth stays in its own files.
   - Keep the current direct outputs (`tfs_sim_growth.csv` and the rest)
     until every study grid and example has moved to the raw path, then
     decide whether to drop them.
   - Test: `tfs-process-counts` on the simulator's raw output reproduces
     the simulator's own growth table (same counts, tube totals from the
     simulated OD600 through the same calibration).
   - Then the full-size simulation of the real design (library, tube grid,
     depth, OD600), as a study.
   Raw formats done 2026-10-05 (`simulate/raw_output.py`, written by
   `tfs-simulate` by default): the round trip matched the direct table
   exactly on the simulate-and-analyze example (51,000 rows) and, through
   OD600, on the simulate example; `examples/simulate-and-analyze/run.sh`
   uses it. Raw presplit is not written yet (the presplit table is still
   direct). Left: the full-size simulation of the real design.
   Full-size simulation set up 2026-10-05: `planning/studies/full-size-sim/`
   (config derived from the dev data by `make_sim_config.py`; the
   simulator's new `design` key copies the real 118-tube layout), to run on
   the cluster. A local pilot ran (85 min, 68 GB peak); `cfu0` is now
   calibrated per library from a pilot. The calibrated simulation still
   misses the real tube totals in a structured way (IPTG, time), which is
   recorded for step 8.
7. [ ] **Validate and document** (after step 6). Without the orchestrator:
   - Run the real data and the full-size simulation through the CLI chain
     with no hand step, and write the commands down as the recipe
     (`planning/studies/real-data-fit/`, and a page in `docs/`).
   - Add the tube-offset diagnostic: `tfs-summarize-fit` reports each
     level-offset fit's offsets in prior SDs and their correlation with
     titrant within each selection condition, and flags a fit whose
     offsets carry structure (the offset mode). Pass test for this step:
     both runs reproduce, and the diagnostic is clean or flagged.
   - Update `docs/` and CLAUDE.md, and retire `make_init_npz.py`,
     `make_warm_start.py` and `check_warm.py` (frozen in the study; no
     longer part of any recipe).
   Code and docs done 2026-10-05: `tfs-extract-params` writes the tube
   offsets labeled by tube (`sample_offset/_tubes.py`), and
   `tfs-summarize-fit` reports them in prior SDs with a Spearman trend
   against titrant and time per condition, flagging BH q < 0.05
   (`tfmodel/analysis/tube_offsets.py`; JSON `tube_offsets`). The recipe is
   `planning/studies/real-data-fit/fit/run_recipe.srun` and
   `docs/source/pipeline.rst` (step 7, "Check the fit", added). One seam
   found and closed: `prep_dev_data.py`'s tube table carried OD600 and
   totals both, which `tfs-process-counts` refuses, so it now also writes
   `tube_table.csv` and `tube_od600.csv`. The hand scripts are marked
   retired in the study README. Left: the two cluster runs (recipe, and
   `full-size-sim`), each checked for the diagnostic's verdict.
8. [ ] **Then: prioritize the science** (user, 2026-10-05). With steps 6
   and 7 in, any model change can be tested end to end on simulations a
   user can reproduce. Step back and list the model work (the growth
   transition, the offsets' freedom, the population model, the
   growth-binding map, and others), how the changes interact, and what
   order to do them in. Not to be decided before steps 6 and 7.

## Out of scope

- The population model and the OD600 likelihood (roadmap step 6), and the
  growth-transition decision (7b, now
  `planning/offset-mode-growth-transition.md`). This plan only keeps room
  for them (P3).
- The growth-binding relationship study (summary, Part 3, item 1).
- Workflow engines (Snakemake, Nextflow). A Python orchestrator over the
  CLI functions is enough at this size, and adds no dependency.

## Open

- Whether the orchestrator replaces `tfs-build-empirical`'s internal
  configure and pre-fit, or calls it.
- How binding data enter a growth-only experiment file. Today they are only
  compared after the fit; roadmap step 9 brings them back into the model.
