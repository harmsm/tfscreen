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
tfs-setup-sim-grid grid.yaml --out_prefix count_likelihood_v4
for d in count_likelihood_v4/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration count_likelihood_v4 --out_prefix calib/count_likelihood_v4 \
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

First run: commit 775f8b7 (2026-09-26/27), results in `count_likelihood/`.
Second run: commit c65390aa (2026-09-27), results in `count_likelihood_v2/`
(neither is committed).

## Results

**Fourth run (pending, 2026-09-28): rerun with `ln_cfu0: hierarchical`.**
Every earlier run fit `ln_cfu0: hierarchical_factored`. That model shares each
genotype's starting abundance across the kanR and pheS pre-conditions,
but the simulator (like the real experiment) grows those libraries up
separately. The fit pushed the difference into a confident per-genotype
theta error, which made coverage fall with read depth
(`../svi-overconfidence/`: 95% coverage above 1000 reads 0.09 with the
factored model, 0.80 with `hierarchical`). The earlier runs' absolute
coverage and RMSE carry that mismatch. Their between-arm comparisons were
made under the same handicap. `run.srun` now uses `hierarchical`; outputs go
to `count_likelihood_v4/`.

**Third run (`count_likelihood_v3`, commit 740ebc19, 2026-09-27): the
answer.** All 24 runs finished; no NaN, no wrong optimum (run_0012's theta
RMSE is 0.035, was 0.51). 16 stopped on the convergence rule. 8 reached
the 100k-epoch cap, all poisson: the loss was flat at the floor step size
and one global noise parameter whose true value is 0 (`growth_inv_r` for
counts, `growth_noise` sigma for `lncfu`/`level`) kept sliding toward its
boundary just past the movement tolerance (excess 0.05-0.08) -- a benign,
honestly reported non-convergence.

Pooled theta (test genotypes), mean over 3 seeds:

| noise | fit | 95% coverage | 95% width | RMSE |
|---|---|---|---|---|
| poisson | lncfu / zero | 0.76 | 0.156 | 0.101 |
| poisson | lncfu / level | 0.56 | 0.073 | 0.100 |
| poisson | counts / level / single | 0.65 | 0.067 | 0.041 |
| poisson | counts / level / mixture | 0.65 | 0.067 | 0.043 |
| realistic | lncfu / zero | 0.82 | 0.207 | 0.113 |
| realistic | lncfu / level | 0.69 | 0.139 | 0.110 |
| realistic | counts / level / single | 0.79 | 0.152 | 0.073 |
| realistic | counts / level / mixture | 0.78 | 0.154 | 0.074 |

- **Accuracy: counts wins everywhere.** Theta RMSE falls 60% (poisson) and
  35% (realistic); dk_geno RMSE 85% (poisson) and 40% (realistic); Hill K
  and n improve too (log K RMSE 0.41 against 0.68 for `lncfu`/`zero`,
  poisson). By stratum the gain holds for bulk, spiked and binding-anchored
  genotypes; spiked genotypes gain the most under realistic noise (RMSE
  0.13 against 0.26, coverage 0.93 against 0.50).
- **Calibration: counts is about as overconfident as `lncfu`/`zero`, not
  less.** Coverage 0.79 against 0.82 (realistic), 0.65 against 0.76
  (poisson): intervals shrink with the error but not as far. The shortfall
  is concentrated where the data say least: `mixed`-purity genotypes (0.41
  poisson, 0.55 realistic) and theta near saturation (0.46 poisson, 0.69
  realistic). So the remaining overconfidence is the fit's (mean-field
  guide, the k pin), not the likelihood's; `growth_k` coverage is ~0 in
  every arm (95% width 1e-4 against RMSE 6e-4-1.3e-3).
- **Lambda at a true 0** (prior 0.357 +/- 0.05): 0.019-0.027 (poisson),
  0.055-0.064 (realistic), against ~0.12 for the `lncfu` mixture in the
  congression study. Most of that spurious lambda was the read-count floor;
  the rest remains, and its intervals (width 0.002-0.006) exclude 0. The
  mixture costs nothing on theta against `single`.
- **Longer runs changed little.** Run for run against v2 (early cuts),
  theta RMSE moved by at most 0.0015 and 95% coverage by at most 0.03
  (poisson counts runs lost ~0.025), except run_0012, which the early cut
  had broken. The pooled check mattered for robustness, not for the
  typical run.
- Dispersion and per-tube offsets as in v2: poisson phi ~0.05, realistic
  phi ~28 (the realistic arm is ~29x Poisson, more than intended);
  `sample_offset` SD ~0.36 absorbing the simulator's tube growth noise.

**Second run (`count_likelihood_v2`, commit c65390aa, 2026-09-27): the
count likelihood is clearly more accurate; two failures remain, neither in
the likelihood itself.** 23 of 24 runs finished; 18 stopped on the
convergence rule, 5 at the epoch cap (3 `counts`, 2 `lncfu`/`level`).

Pooled theta (test genotypes), mean over seeds:

| noise | fit | 95% coverage | 95% width | RMSE |
|---|---|---|---|---|
| poisson | lncfu / zero | 0.76 | 0.157 | 0.101 |
| poisson | lncfu / level | 0.56 | 0.076 | 0.100 |
| poisson | counts / level / single (2 runs) | 0.70 | 0.068 | 0.037 |
| poisson | counts / level / mixture (seeds 1, 2) | 0.66 | 0.072 | 0.047 |
| realistic | lncfu / zero | 0.81 | 0.207 | 0.114 |
| realistic | lncfu / level | 0.68 | 0.140 | 0.110 |
| realistic | counts / level / single | 0.78 | 0.153 | 0.073 |
| realistic | counts / level / mixture | 0.78 | 0.155 | 0.074 |

- **Accuracy.** Counts cut theta RMSE by a third (realistic) to two thirds
  (poisson) against either `lncfu` arm, with coverage close to `lncfu`/
  `zero` and better than `lncfu`/`level`. dk_geno RMSE also falls
  (poisson single: 0.0003 against 0.002). Every arm still under-covers
  (0.66-0.81 at 95%): the overconfidence is not the likelihood's.
- **Lambda at a true 0** (prior 0.357 +/- 0.05): 0.023 and 0.027
  (poisson), 0.054-0.064 (realistic), against ~0.12 for the `lncfu`
  mixture in the congression study. Counts removes most of the spurious
  lambda but not all, and the intervals (width ~0.01) exclude 0. The
  mixture arms lose nothing against `single` on theta.
- **Dispersion.** poisson: phi 0.03-0.06, inv_r 0.0014 (Poisson, as
  simulated). realistic: phi 27-29, inv_r 0.007, i.e. ~29x Poisson,
  well above the ~8x the arm was designed for (real data: 5-18x); the
  realistic arm is noisier than intended.
- **`sample_offset` level SD ~0.36 in every count run**, poisson included:
  the base config's `tube_noise_sigma` (0.002/min over ~230 min, ~0.4 ln
  units) shifts each tube's genotypes and its supplied total together, and
  the model's predictions carry no tube term; the offset absorbs it, as
  intended until step 6. The same explains `lncfu`/`level` under-covering:
  in `lncfu`/`zero` the `normal_kt` row noise soaked up this tube noise and
  with it the understated `ln_cfu` noise; the structural offset removes that
  cover and the intervals halve at the same RMSE.
- **`growth_k` coverage ~0 in every arm** (95% widths ~1e-4 against RMSE
  ~1e-3): the pre-fit's k pin, not the likelihood.
- **Failure 1, `run_0007` (poisson, seed 2, counts, single): NaN at step
  ~12,400.** Not the likelihood: hill_mut's horseshoe local scales for the
  lowest-read doubles (200-800 reads over all tubes, mostly zeros under
  selection; typical doubles 3,600-200,000) widened (guide LogNormal scale
  up to 11.5 on 41 elements; every other run stays at ~1.5), because beyond
  saturation theta, and so the likelihood, no longer changes, while the
  half-Cauchy tail is heavy. A draw of ~e^90 overflows float32 and the
  epistasis term, theta, growth and the likelihood go NaN. Reproduced by
  replaying from the step-12,000 checkpoint.
- **Failure 2, `run_0012` (poisson, seed 3, counts, mixture): converged to
  a wrong answer** (theta RMSE 0.51; m shrunk to ~40% and 4CP's m of the
  wrong sign; ELBO 5.4e5 against 2.6e5 for the same-sized seed-2 mixture;
  its `single` twin on the same data is fine). The step size was cut at
  step 20,000 during a noisy but steady descent: window-median losses fell
  1.5e5-1.9e5 per window, but each window's own trend was t = 1.9, 3.0,
  1.0, so three windows counted as a plateau. It also showed some lambda
  widening (scales to 4.9), perhaps from the same low-read doubles.
- **Both fixed on this branch (2026-09-27).** Horseshoe: overflow-safe
  `regularized_scale` and `HalfCauchy` (`components/_horseshoe.py`); a
  replay of run_0007 from its step-12,000 checkpoint, which went NaN at step
  12,370 before, ran 3,000 steps clean. Convergence: before a cut or stop,
  the stalled windows are pooled into one trend (`pooled_loss_trend`).
  Replaying every v2 run's loss trace up to its first cut, 20 of 23 first
  cuts were made mid-descent (pooled t 4.4-14.2; run_0012: 14.2) and would
  now continue; the other 3 (pooled t -0.5 to 1.4) are still cut. So the
  early cut was the norm, not a one-off, and the v2 fits (and the earlier
  congression grids) mostly finished their fast phase at too small a step.
  A third run with both fixes is needed; runs will be longer.

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
  **Update (2026-09-27):** it did more than slow convergence. SVI lost the
  pre-MAP point and settled into other optima (`../relative-fit/`,
  Results), and every arm here, `lncfu` included, started 2e6-5e8 above
  its pre-MAP. The default is now 1e-4 with no jitter. The comparisons
  between arms ran under the same handicap. The absolute coverage numbers
  above, `growth_k` included, need a rerun.
