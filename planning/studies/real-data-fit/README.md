# Real-data fit of the dev-data screen

**Question.** Does the growth-only relative fit (`theta: hill_relative`),
with the count likelihood and tube offsets, run on a full screen? What do
its curves say about the relationship between in vitro binding and in vivo
growth?

**Decisions it fed.**
- Roadmap step 5: the full-library inference route. The answer is the MAP
  plus the arrowhead Laplace; the SVI guides slid along the m·X ridge.
- Roadmap step 8: whether a simple growth-binding map exists. It does not.
- The new options added along the way: `--init_from`, `sigma_fixed`, the
  held Hill population SDs, per-condition m prior scales, and the block
  and arrowhead Laplace.

The results and their reasoning are in `planning/analysis-roadmap-summary.md`.
The full lab notebook, with every run including the dead ends, is
`planning/dev-data/README.md` and `planning/dev-data/real_fit/README.md`.
Those are gitignored, because they sit with the data.

## Layout: tracked scripts, untracked data

The lab's data never enter the repository (roadmap C10). They live in
`planning/dev-data/` (gitignored), and these scripts expect that layout:

```
planning/dev-data/
  counts/                    tfs-process-fastq output, one dir per tube
  counts/run_config.yaml     processing library config (3 spikes; see below)
  sample_df.csv, all_reps_screen_od600.xlsx, 2025-10-02_control-experiments.xlsx
  binding/                   anisotropy tables
  od600-to-cfu/              calibration spreadsheets
  processed/                 written by processing/prep_dev_data.py
  real_fit/inputs/           growth.csv.gz, library_config.yaml,
                             wt_control_rate_summary.csv, init_*.npz
  real_fit/<run>/            one directory per fit
```

In `planning/dev-data/` the scripts are symlinks to the copies here, so they
run in place: `prep_dev_data.py` from `dev-data/`, the `fit/` and
`analysis/` scripts from `real_fit/`, as `../<script>` from inside a run
directory. On the cluster, copy `fit/*` into `real_fit/` by hand. The data
directory is gitignored and is not pulled. Check the copy with
`grep -c HS_FIXED run_relative.srun`.

- `processing/`
  - `process_fastq.srun`: `tfs-process-fastq` per tube pair.
  - `prep_dev_data.py`: the tube table, OD600 to tube totals through
    `tfs-calibrate-od600`, the swapped initial samples, pooling of
    resequenced tubes, failed tubes dropped, the nine-spike library config
    (`SPIKE_EDITS`), and the wt monoculture rates.
  - `binding_fit.py`, `binding_plot.py`: the joint anisotropy-to-theta and
    Hill fit, interpolated to 10 µM protein.
- `fit/`
  - `run.srun`: the joint fits.
  - `run_relative.srun`: the growth-only relative fits.
  - `run_blap.srun`: the block Laplace, or the arrowhead Laplace with
    `SHARED=1`.
  - `set_growth_priors.py`: k/m priors for growth-only models, which the
    pre-fit refuses.
  - `make_warm_start.py`, `make_init_npz.py`, `make_init_hs.py`: build
    `--init_from` start points.
  - `check_warm.py`: stops a job whose start is not where it should be.
- `analysis/`
  - `score_counts.py`: the count log-likelihood by tube, traced from the
    model, so it works for any model.
  - `residual_compare.py`: two single-transformation MAPs compared
    observation by observation.
  - `small_offsets.py`: the best per-tube offset with everything else held.
  - `profile_offsets.py`: SVI against MAP.
  - `plot_x_vs_binding.py`, `scan_protein.py`, `scan_monotone.py`: the
    binding comparison.

## How to run

Processing, from `planning/dev-data/`, with `tfs-process-fastq` already
run:

```bash
python prep_dev_data.py
```

```bash
python binding_fit.py
```

Then `tfs-process-counts` per library on `processed/sample_df_{kanR,pheS}.csv`,
concatenated into `real_fit/inputs/growth.csv.gz`.
`processed/library_config.yaml` becomes `real_fit/inputs/library_config.yaml`.
`counts/run_config.yaml` lists only three spikes, with the wrong M42I codon,
so it must not be used for configure.

**The final fit and its start chain, as it ran.** A level-offset MAP
started cold lands in a ±2.8 offset mode (see "Starting point" in the
summary). Each offset fit therefore started from an earlier point. All
commands run from `real_fit/`, each in its own new run directory:

| Run | Command (`sbatch --export=ALL,...`) | Start |
|---|---|---|
| `map_nooffset_b` | `ARM=map,OFFSET_MODEL=zero,BATCH_SIZE=4096 ../run.srun` | cold |
| `map_sig017b` | `ARM=map,SIGMA_FIXED=0.17,BATCH_SIZE=4096 ../run.srun` | cold; only its config is used |
| `map_warm` | `ARM=map,WARM=1,ADAM_STEP=1e-4 ../run.srun` | `make_warm_start.py map_sig017b map_nooffset_b map_warm`, after `residual_compare.py map_sig017b map_nooffset_b resid/sig_vs_nooff` and `resid/small_offsets.py` |
| `map_nooffset_n05` | `ARM=map,OFFSET_MODEL=zero,BATCH_SIZE=4096,N_SCALE=0.5 ../run.srun` | cold |
| `map_off_n05` | `ARM=map,SIGMA_FIXED=0.17,N_SCALE=0.5,BATCH_SIZE=4096,INIT_FROM=../inputs/init_n05_warm.npz,INIT_REF=../map_nooffset_n05,ADAM_STEP=1e-4 ../run.srun` | `make_init_npz.py map_nooffset_n05 map_warm inputs/init_n05_warm.npz` |
| `rel_loose_b` | `PRIORS=loose ../run_relative.srun` | cold |
| **`rel_off_n05`** | `PRIORS=loose,N_SCALE=0.5,OFFSET_MODEL=level,SIGMA_FIXED=0.17,INIT_FROM=../inputs/init_rel_off.npz,ADAM_STEP=1e-4 ../run_relative.srun` | `make_init_npz.py rel_loose_b map_off_n05 inputs/init_rel_off.npz` |
| `rel_off_n05_blap` | `MAP_DIR=../rel_off_n05 ../run_blap.srun` | |
| **`rel_off_n05_alap`** | `MAP_DIR=../rel_off_n05,SHARED=1 ../run_blap.srun` | |
| `rel_off_hs` | `rel_off_n05`'s settings plus `HS_FIXED=X_low=0.5:X_delta=0.5:log_hill_K=1.0` and `INIT_FROM=../inputs/init_rel_off_hs.npz` | `make_init_hs.py rel_off_n05 inputs/init_rel_off_hs.npz X_low=0.5 X_delta=0.5 log_hill_K=1.0` |
| `rel_off_hs_mix` | `rel_off_hs`'s settings plus `TRANSFORMATION=mixture` and `INIT_FROM=../inputs/init_rel_off_hs_map.npz` | a copy of `rel_off_hs/tfs_fit_model_params.npz` |

The chain is long because each step fixed a problem found by the one
before. The pipeline plan (`planning/experiment-pipeline.md`) builds the
staged start into `tfs-fit-model`, and its done criterion is this fit
rerun from one experiment file, so the shortened recipe comes from there
rather than from a rerun of this chain.

Every other run directory in `real_fit/` is a dead end recorded in the
notebook: the first offset fits, the SVI arms, `rel_monokan`,
`rel_loose_n05` (stale script), `local_offsets/` and the `smoke_*` tests.

## Inputs

Only the gitignored files above. No input path points outside
`planning/dev-data/`.

## Commit

The runs do not log a commit. Each run below needed the option its commit
added, and its `run.out` start time falls after that commit and before the
next one it would need.

| Runs | Commit |
|---|---|
| `map_nooffset_b`, `map_sig017b` | d451faa9 |
| `map_warm` | faa5b116 |
| `rel_loose_b` | e4320c7f |
| `map_nooffset_n05`, `map_off_n05`, `rel_off_n05` | beacad0e |
| `rel_off_n05_blap` | cd208d15 |
| `rel_off_n05_alap`, `rel_off_hs`, `rel_off_hs_mix` | a874ac38 |

## Results

See `planning/analysis-roadmap-summary.md`, Part 1, "Fitting the real
screen" onward. In short:
- The growth curves are broader than binding (n 0.6-1.2 against 1.6-2.5).
- M42I's K is below wt's in growth (6.4 against 12 µM) and about 10x above
  it in binding.
- Every mutant falls to a common floor near X -0.5.
- No shared protein concentration or monotone map carries binding onto
  growth.
- K and n held up when the Hill population SDs were held, when the
  congression mixture was added, and with the k/m uncertainty included.
- m and the absolute X scale are under-constrained.

## Next

- **Full-size simulations of this design,** to answer reviewers: the same
  library size, read depth, noise and fit recipe, with known truth, to show
  that the curves, the intervals and the binding comparison come out
  right. This should run through the new pipeline's experiment file once
  the simulator writes the same raw formats (the pipeline plan, step 6).
- Rerun the fit from one experiment file when the pipeline lands. This
  makes the shortened recipe the official one.
