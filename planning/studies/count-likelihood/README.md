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

First run: commit 775f8b7 (2026-09-26/27). Rerun after the two fixes below
(this branch, 2026-09-27); give the rerun its own `--out_prefix`.

## Results

**First run (775f8b7): 11 of the 12 `counts` runs failed; not usable for
the comparison.** All 12 `lncfu` runs and one `counts` run
(`poisson`/seed 2/`mixture`) finished.

- 8 `counts` runs crashed at the fit on a NaN `k_scale` written by the
  pre-fit. Cause: `RunInference.compute_hessian_sigmas` took the constrained
  MAP values it was given as unconstrained, so every positive site was
  evaluated at `exp(value)`; the count likelihood's `growth_phi` (MAP 49
  from the calibration model's misfit, dk_geno fixed at 0) became e^49 and
  overflowed. Reproduced from `run_0003`'s saved MAP; fixed. The same bug
  loosened the `lncfu` arms' pre-fit `k_scale` (0.0028 written against
  k sigmas of 0.0003-0.0007 at the right point, which the 0.002 floor
  now overrides), so the `lncfu` arms need the rerun too.
- 3 `counts` runs (`poisson` seeds 2 and 3) went NaN within 250 SVI steps
  after the pre-MAP. Cause: the negative binomial's concentration `mu / phi`
  underflows float32 for genotypes predicted near extinction (all-zero
  genotypes drift there), and the gradient goes NaN. Reproduced by
  replaying `run_0007`'s pre-MAP hand-off (NaN by step 250); with the
  concentration floor (`c + e^-30`) the same replay runs normally.
- The one finished `counts` run (true lambda 0, prior 0.357 +/- 0.05)
  found lambda 0.027 (95% 0.026-0.028): far from the prior and well below
  the ~0.12 the `lncfu` mixture found at true lambda 0 in the congression
  study, as the floor hypothesis predicts; one run, and the interval is
  overconfident. Its theta RMSE was 0.054 against 0.10 for the `lncfu`
  arms at the same noise.
- `lncfu` arms, pooled theta 95% coverage (3 seeds each): `poisson`
  0.76 (`zero`) and 0.55 (`level`); `realistic` 0.82 (`zero`) and 0.68
  (`level`). The `level` offset narrows the theta intervals by about half
  at the same RMSE, so it makes `lncfu` more overconfident, not less.
  `growth_k` coverage is about 0 in every arm (intervals ~1e-4 wide, RMSE
  0.001): the pre-fit's k prior dominates.
- Counts runs start SVI at a much higher ELBO than `lncfu` (about 3e8 at
  the pre-MAP point here): at 1e4-1e5 reads the likelihood is sharp and the
  component guide's initial location scales (0.1) cost ~(0.1 mu)^2 / var per
  observation. It falls as the scales shrink; a smaller
  `--guide_init_scale` may suit counts fits if this slows convergence.
