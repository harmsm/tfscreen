# What makes the fit's m too steep

**Question.** On the calibrated full-size simulation the fit's growth slope
m is about 40% too steep in kanR+kan (-0.018 to -0.020 against a true
-0.0138), even with the truth as a 0.002-SD prior
(`planning/studies/km-anchor/`); the real fit has the same m against the
wt monoculture. The fitted likelihood prefers the wrong m, so something
the simulation does is missing from the fit. Which of the simulation's
features the fit does not model causes it?

**Decision it feeds.** `planning/deep-coverage.md`, step 1b: what the fit
has to model, or correct for, before the deep genotypes' intervals can be
honest.

## Design

A reduced library (5 degenerate sites per tile, every spike site kept:
5,256 doubles instead of 223,000), scaled with
`genetics.library_design.scale_library_design` so each sequence keeps its
reads, cells, transformants and PCR templates, on the real design (118
tubes) with the calibrated simulation of `planning/studies/full-size-sim/`.
The fit is the real recipe's (growth-only relative model, level offsets at
SD 0.17, loose k/m priors), MAP only; the slope does not need the Laplace.

| arm | changes from the calibrated simulation |
|---|---|
| `base` | none |
| `no_cong` | no co-transformed cells (`transformation_poisson_lambda` 0) |
| `no_tube` | no per-tube rate noise (`tube_noise_sigma` 0) |
| `poisson` | Poisson counts (no founder, demographic or PCR noise) |
| `clean` | all three off |

Two seeds each. `score_arms.py` puts the truth on the fit's X scale (wt's
true curve at the gauge concentrations) and reports k, m, the ratio
m_fit / m_true, and the median dk_geno error.

`base` must reproduce the bias first; if it does not, the reduced library
changes the problem and the factorial says nothing.

A local test (2026-10-09) ran the simulation, processing and configure in
minutes, but the staged MAP's first stage was still descending (about 1.8
nats per window, t = 12) at step 80,000 after an hour on a laptop CPU, so
the arms run on the cluster.

## How to run

On the cluster, after pulling this commit, from the full-size simulation's
working directory (it holds `processed/` and `pilot_s1/`):

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-v3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_factorial.py --pilot pilot_s1 --out_dir m_bias
```

It builds the two full-size configs (a few minutes) and prints one line per
arm: 5256 doubles, scale 0.1612, reads 8.17e+08. Then:

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/{run_arm.sh,run_arm.srun,score_arms.py} m_bias/
```

```bash
cd m_bias && for d in base_s1 base_s2 no_cong_s1 no_cong_s2 no_tube_s1 no_tube_s2 poisson_s1 poisson_s2 clean_s1 clean_s2; do (cd $d && sbatch ../run_arm.srun); done
```

Each arm ends with `>>> Done` in its `run.out` and a
`tfs_params_growth_m.csv`. Then, in `m_bias/`:

```bash
python score_arms.py *_s1 *_s2
```

It prints the table and writes `factorial_scores.csv`.

## Inputs

`processed/` (the dev data, via make_sim_config.py) and the full-size
pilot (`pilot_s1/`, for the cfu0 calibration).

## Commit

The commit that adds this study.

## Results

Pending.
