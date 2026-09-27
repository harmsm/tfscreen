# Count-likelihood study

**Question.** Does observing read counts directly (a negative binomial with
learned dispersion, no pseudocount; `growth_likelihood: counts`) fit better
and give better-calibrated theta than observing `ln_cfu`, both on data whose
counts are Poisson and on data with the real data's extra count noise? And,
at a true lambda of 0, does the congression mixture still find lambda
around 0.12 once the read-count floor is modeled?

**Decisions it feeds.**
- Roadmap step 7 (`planning/analysis-roadmap.md`): whether the count
  likelihood becomes the default for real data.
- Congression plan step 6 (`planning/congression-physics-plan.md`, on main):
  whether the mixture's theta gain at a true lambda of 0 was the read-count
  floor. That step also re-runs `grid_baseline.yaml` and `grid_dkrule.yaml`
  on the count likelihood; this grid covers the lambda-0 question directly.

## Design

Base config: `../congression-calibration/simulate_config.yaml` (483
genotypes, hill_mut theta, instant growth transitions, kanR and pheS, 20
in-library binding anchors), at lambda 0 and the wide dk spread, with the
anchored fit settings of that study's baseline (binding weight 1, prefit
`k_scale_ceiling` 0.005).

[`grid.yaml`](grid.yaml): 2 noise x 3 seeds x 4 fits = 24 runs.

| axis | levels |
|---|---|
| simulated noise | `poisson` (the simulator's old behavior: Poisson counts); `realistic` (founder sampling and demographic growth with about 300 founders per genotype per tube, `cfu0` 150,000; shared transformation; 150,000 template molecules for about 830k reads per tube with amplification CV 0.5, so count variance about 8x Poisson, the middle of the 5-18x measured on real data in `../noise-anatomy/`) |
| seed | 1, 2, 3 |
| fit | `lncfu`/`zero` (today's fit: Student-t on `ln_cfu`, `normal_kt` growth noise); `lncfu`/`level` (plus a per-tube level offset); `counts`/`level` (the count likelihood; growth noise off, as it requires); `counts`/`level`/`mixture` with the measured lambda prior (0.357 +/- 0.05) |

Each run ([`run.srun`](run.srun)) is the congression-calibration pipeline
with `--growth_likelihood` and `--sample_offset_model` from the fit arm.

## How to run

On the cluster, from a scratch directory, with this repository checked out:

```bash
tfs-setup-sim-grid grid.yaml --out_prefix count_likelihood
for d in count_likelihood/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration count_likelihood --out_prefix calib/count_likelihood \
    --baseline likelihood=lncfu sample_offset=zero fit=single \
    --facet_by founder_sampling
```

## What to look at

- theta (`theta_test`): coverage, calibration bias, width and RMSE by
  `purity` x `has_binding`, paired against `lncfu`/`zero`, separately for
  the two noise levels. Under `realistic` noise the `lncfu` fits treat 5-18x
  too little variance as known, so their coverage should fall; the count
  fit should hold it.
- `growth_phi` and `growth_inv_r`: near 0 under `poisson`; `phi` near the
  simulated multiplier (about 8) under `realistic`.
- The `mixture` arm's `lam` against the true 0.
- Convergence (every run should stop on the noise-referenced rule, not the
  epoch cap).

## Inputs

Repository paths only: `grid.yaml`, `run.srun`,
`../congression-calibration/simulate_config.yaml` and its
`hill_params.csv`. Seeds fixed in the grid.

## Commit

Not yet run. A local smoke run of one `realistic`/`counts`/`mixture` run with
3 epochs completed every pipeline step (2026-09-26); it checks plumbing, not
results.

## Results

Pending.
