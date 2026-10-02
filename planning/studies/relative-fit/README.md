# Relative-X fit study

**Question.** Does a growth-only fit on the wt-relative scale X (theta
component `hill_relative`: no binding data, no prefit, wt's X pinned to 1 at
0 mM and 0 at 1 mM IPTG) recover X, the Hill K and n, and each condition's
`k` and `m`, about as well as the anchored joint fit recovers theta? Does
that hold under realistic count noise?

**Decisions it feeds.**
- Roadmap step 5 (`planning/analysis-roadmap.md`): whether the relative fit
  is sound enough to run on real data and build the growth-binding map on
  (step 8).
- C4: whether `k_c`/`m_c` are recovered from growth alone, with totals
  supplied (step 6 later replaces the supplied totals).

## Design

Base config: `../congression-calibration/simulate_config.yaml` (483
genotypes, hill_mut theta, instant growth transitions, kanR and pheS, 20
in-library binding anchors), at lambda 0 and the wide dk spread, like
`../count-likelihood/`. Lambda 0 keeps congression out of the comparison;
both fits use `single`.

[`grid.yaml`](grid.yaml): 2 noise x 4 fit/inference arms x 3 seeds = 24
runs, all on the count likelihood (the default after `../count-likelihood/`;
the `lncfu` arm of the first design was dropped, 2026-09-27). Run 1 had only
the two component-guide arms. Run 2 added the `low_rank` and `map` arms for
the relative fit after run 1 found the mean-field guide biased there
(Results).

| axis | levels |
|---|---|
| simulated noise | `poisson`; `realistic` (founder sampling, demographic growth, shared transformation, 150,000 PCR templates with CV 0.5; designed for ~8x Poisson, fitted at ~29x in `../count-likelihood/` v3) |
| fit | `joint` (`hill_geno`, binding at weight 1, prefit with `k_scale_ceiling` 0.005); `relative` (`hill_relative`, growth only, no prefit) |
| inference | `component` (mean-field component guide, the default); `low_rank` (`auto_low_rank_multivariate_normal`, numpyro's default rank); `map` (MAP, then a Laplace posterior in `tfs-sample-posterior`). The joint fit runs `component` only. |
| seed | 1, 2, 3 |

Both fits use `growth_likelihood: counts` with `sample_offset: level` and
per-genotype Hill curves (the simulation's `hill_mut` truth
is fit by neither), activity fixed and no theta noise (the relative fit
refuses it; the joint fit leaves it off to match). Each run
([`run.srun`](run.srun)) simulates, configures, prefits (joint only), fits,
samples the posterior and summarizes.

## How to run

On the cluster, from a scratch directory, with this repository checked out:

```bash
tfs-setup-sim-grid grid.yaml --out_prefix relative_fit_v3
for d in relative_fit_v3/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration relative_fit_v3 --out_prefix calib/relative_fit_v3 \
    --baseline fit=joint inference=component --facet_by founder_sampling
```

## What to look at

- `theta_test`: for the relative arms `tfs-summarize-fit` puts the truth on
  the X gauge, `X = (A theta - A_wt theta_wt(1 mM)) / (A_wt (theta_wt(0) -
  theta_wt(1 mM)))`, and the rows carry `theta_regime = X`. Coverage, PIT,
  width by `purity` x `has_binding` (has_binding is held-out for the
  relative fit: it saw no binding). Coverage and PIT are scale-free and
  compare across arms; width and RMSE do not (X spans about 1 where theta
  spans about 0.98 for the base config's wt, so close).
- `log_hill_K`, `hill_n` (`*_params_theta_*`): the same quantity in both
  fits (the gauge leaves them unchanged).
- `growth_k`, `growth_m`: for the relative arms the truth is mapped to X
  (`k_X = k + m theta_wt(1 mM)`, `m_X = m (theta_wt(0) - theta_wt(1 mM))`).
  C4 asks for these explicitly: the joint fit gets them partly from the
  prefit's binding anchors; the relative fit only from wt and the totals.
- `dk_geno`: unaffected by the gauge; the X baselines trade against it
  (idea file, "what stays assumed"), so compare its recovery across arms.
- Genotypes whose true theta does not depend on IPTG (the simulator's
  stuck-bound and never-binds classes): internal standards across columns
  (C4).
- Convergence: no prefit on the relative arm, so check that the fit stops
  on the noise-referenced rule, not the epoch cap.

## Not covered here

- The kan-vs-4CP consistency check (D5): X is shared by construction, so a
  fit cannot test it by itself. It needs either a variant with one X per
  condition family or a model-free comparison of kan and 4CP wt-relative
  slopes per genotype (as in `../growth-binding-map/`); decide when the
  real-data run is set up.
- Real data: waits for the new processing batch and step 2's smoothed total.

## Inputs

Repository paths only: `grid.yaml`, `run.srun`,
`../congression-calibration/simulate_config.yaml` and its
`hill_params.csv`. Seeds fixed in the grid.

## Commit

Not yet run. A local smoke run of one `realistic`/`counts`/`relative` run
(SVI stopped after 3 epochs past the pre-MAP) completed every pipeline step
(2026-09-26): the theta test file carries `theta_scale = X` with truth on the
gauge, and `growth_k`/`growth_m` carry X-scale truth. It checks plumbing,
not results.

## Results

**Run 3 (`relative_fit_v3`, 2026-09-28): `ln_cfu0: hierarchical`.** All
24 runs finished. Theta/X test set, 95% coverage / RMSE, mean over 3 seeds
(run 2 in brackets):

| noise | arm | run 3 | run 2 |
|---|---|---|---|
| poisson | joint component | 0.36 / 0.071 | 0.43 / 0.068 |
| poisson | relative component | 0.07 / 0.254 | 0.17 / 0.130 |
| poisson | relative low_rank | 0.78 / 0.065 | 0.64 / 0.057 |
| realistic | joint component | 0.61 / 0.094 | 0.58 / 0.097 |
| realistic | relative component | 0.38 / 0.288 | 0.43 / 0.270 |
| realistic | relative low_rank | 0.78 / 0.122 | 0.78 / 0.115 |

- **This library barely moved, unlike the small one.** Most of its
  genotypes are low-read doubles (median 3-5 reads per tube), where the
  ln_cfu0 mismatch was never the main error. The low-rank relative fit
  gained coverage under Poisson noise (0.64 to 0.78).
- **The relative component guide got worse** (RMSE 0.13 to 0.25 under
  Poisson noise). The per-library ln_cfu0 adds free parameters per
  genotype, which flattens the m·X ridge the mean-field guide already
  slides along. The low-rank guide is unaffected, which keeps the
  recommendation.
- **MAP + Laplace:** m covers in every run (1.0). Theta is still unusable
  because of the Laplace blowups.

**Run 1 (2026-09-27): inconclusive; the SVI start is broken.** All 12 runs
finished. Three hit the 100,000-epoch cap (0002, 0003, 0004). run_0011's
summary came down incomplete, so we regenerated it locally from the run's
CSVs.
Pooled with `tfs-summarize-calibration` into `calib/` (not committed).

The final fits looked bad in both arms:

| | theta/X test coverage (95%) | r | notes |
|---|---|---|---|
| joint, Poisson | 0.37 | 0.29 | run 0005 in the mirror mode (m and theta sign-flipped); 0003 capped mid-descent |
| joint, realistic | 0.59 | 0.97 | |
| relative, Poisson | 0.08 | 0.99 | X shifted 0.2-0.3 below truth, library-wide |
| relative, realistic | 0.33 | 0.96 | run 0010 fine (0.58), the others shifted |

The relative fit's growth rates matched the truth within each tube to 1e-4.
wt did not: the fit moved the whole library against wt, missing wt's
ln_cfu by 0.1-0.27 per tube, while the joint fit held it within 0.03.

**The pre-MAP was right in every run.** The pre-MAP recovered each
condition's m to within 0.0002 of the X-scale truth (relative) and 0.001 of
the theta-scale truth (joint), and put the relative fit's X on truth (slope
1.00, intercept -0.02 to -0.08). SVI then lost it. It halved m in 0003,
flipped it in 0005, and shifted X in the relative runs.

**Cause: the pre-MAP to component-guide hand-off.**
- **Starting loss.** SVI started at a loss of about 4e8 against the
  pre-MAP's 2.5e5, then spent about 40,000 epochs descending from there
  into whatever optimum it reached.
- **The scale cap is in each site's own units.** `component_guide_start`
  caps every guide scale at `guide_init_scale` (0.1). Evaluated at run
  0002's hand-off (scratch script, one site at 0.1 and the rest at 1e-5,
  base loss 2.8e5), SD 0.1 added:
  - 6e8 on `dk_geno_hyper_shift`
  - 4e7 on `condition_growth_k`
  - 1e7 on `condition_growth_m`
  - 9e6 on `ln_cfu0_tube_offset`
  - 2e6 on `dk_geno_hyper_loc`

  These are growth rates per minute or ln_cfu levels, where SD 0.1 is
  enormous: 0.1/min over a 170-minute selection is 17 ln units.
- **The jitter makes it worse.** `init_param_jitter` (0.1) is
  multiplicative, so it moves locations with large values (ln_cfu0 about
  10-18) by 1-2 ln units. At scale 0.001 it raised the starting loss from
  1.0e6 to 1.6e7.
- **Every earlier SVI grid has the same start.** The count-likelihood
  v1/v2 runs, lncfu included, began at 2e6-5e8. Their between-arm
  comparisons ran under the same handicap, but their absolute coverage
  numbers are suspect.

**Separately:** the joint arm's prefit wrote per-condition priors far from
truth under the count likelihood (kanR+kan `k_loc` -0.0006 and `m_loc`
+0.0007, against 0.0107 and -0.0099, at scales 0.002 and 0.001). The
pre-MAP overrode them here. This needs its own look before the prefit is
trusted with counts.

**Fix (2026-09-27).** `DEFAULT_GUIDE_INIT_SCALE` is now 1e-4, applied to
the component guide and to fresh autoguides, and `init_param_jitter`
defaults to 0. At run 0002's hand-off the starting loss fell from 5.9e8 to
2.81e5, level with scales at 1e-5.

**The start was not the only cause.** We checked run 0002 locally in two
continuations (scratch scripts, not committed):
- **SVI.** 8000 SVI steps from the fixed start held the loss near the
  pre-MAP's but moved to the same shifted solution. The kanR+kan m went
  from -0.0098 to -0.0087, and X to slope 1.12 and intercept -0.25.
- **MAP.** Continuing the pre-MAP (AutoDelta) for 18,000 more epochs
  stayed on truth: flat loss, m -0.00973 against an X-scale truth of
  -0.00973.

The tube-offset spread was the same, about 0.08, in both. So the mean-field
ELBO, not the MAP optimum, prefers the shift. The likely mechanism: growth
alone constrains mostly the product m·(X_g - X_wt). The posterior is
therefore a curved ridge along m·X = constant. Only wt's absolute growth
change across IPTG sets the position along it, and that reaches the fit
only through the tube totals. A mean-field Gaussian cannot follow a curved
ridge, and it slides toward smaller |m|. The joint fit escapes because
binding observes theta for the anchors.

**MAP + Laplace works on run 0002, once the MAP has converged.** We
continued the pre-MAP at a fixed step size of 1e-3. The result sat far
from a stationary point, with gradient norm 1.4e7. Its Hessian had 9
negative eigenvalues, down to -767, all on per-genotype offsets.
Clamping them made the Laplace X intervals useless (median 95% width
2.6e10). Continuing with step-size cuts, the fit reached 1e-4 after about
128,000 epochs in total. The loss was still falling about 3 nats per
2000-epoch window. At that point:
- **Gradient and Hessian.** The gradient norm was 2.8e4 and one tiny
  negative eigenvalue remained (-5e-5, on `ln_cfu0_tube_scale`).
- **m.** m stayed on truth (-0.00976).
- **Laplace X.** X was unbiased: RMSE 0.109, slope 1.00, intercept
  -0.01. 95% coverage was 0.99 and 50% coverage 0.77, so the intervals
  are conservative. The run-1 SVI covered 8%.

Run 2 therefore adds a `map` arm with `--max_num_epochs 200000` and a
`low_rank` arm (`auto_low_rank_multivariate_normal`).

**Run 2 (2026-09-28): `relative_fit_v2`, 24 runs, all finished.** Pooled
in `calib/relative_fit_v2_*` (not committed).

- **Start fix confirmed.** Every SVI run began within about 45k of its
  pre-MAP loss (run 1 began about 4e8 above). No mirror modes. All
  component-guide runs, joint included, hit the 100,000-epoch cap at step
  size 1e-6 without being called converged. Five of the six `low_rank`
  runs converged. All `map` runs hit the 200,000 cap at step size 1e-4 or
  1e-5.
- **Joint fit (component).** Much better than run 1: theta r 0.99/0.97
  and RMSE 0.07/0.10 (Poisson/realistic), against run 1's 0.29 r under
  Poisson. It still undercovers: 95% coverage 0.43/0.58.

X test set (`theta_test`), per-arm means over 3 seeds, Poisson / realistic:

| relative arm | 95% coverage | 50% coverage | RMSE | median 95% width |
|---|---|---|---|---|
| component | 0.17 / 0.43 | 0.06 / 0.18 | 0.13 / 0.27 | 0.045 / 0.23 |
| low_rank | 0.64 / 0.78 | 0.29 / 0.39 | 0.058 / 0.115 | 0.034 / 0.18 |
| map + Laplace | 1 of 6 runs usable | | | |

- **`low_rank` removes most of the relative fit's bias.** Its X RMSE is
  at or below the joint fit's (0.058 vs 0.068 Poisson), with coverage at
  or above the joint fit's. The component guide stays biased, as in
  run 1.
- **`map` + Laplace is not usable as is.** Every MAP recovered m with
  calibrated intervals (95% coverage 1.0/1.0). The Laplace covariance on
  the genotype parameters blew up in 5 of 6 runs, though: 36-126
  genotypes had `hill_n` 95% widths above 100, and X widths reached
  1e8-1e14 even for the other genotypes. Only run 0004 (seed 1, Poisson)
  gave sensible X (95% coverage 0.91).
- **Every SVI arm is overconfident.** 50% coverage of X/theta is 0.2-0.4.
  `growth_k`/`growth_m` intervals are about 1e-4 wide with coverage near
  0, in the joint and relative arms alike. The MAP's Laplace intervals
  on k and m cover (0.92-1.0). Remaining gap: the guides' variance, not
  their location.

> **Caveat (2026-10-01).** Runs 2 and 3 ran with two bugs fixed on
> 2026-09-29 (`9db28286`). From `c57d9ee5` (2026-09-27 16:48) every
> component-guide hyperparameter scale started on its `greater_than(1e-4)`
> bound and never moved, so the `component` arms ran crippled while the
> `low_rank` autoguide did not; the comparison between them is lopsided.
> The Laplace floor clamped negative eigenvalues to 1e-3 (variance 1,000),
> which is the `map` arm's blowups; the learned log(hill_n) spread, which
> ran away on real data, was also free here. The svi-overconfidence
> study's relative arm, in the same window on a smaller library, found
> component and low-rank about equal (X 95% coverage 0.68 vs 0.74, RMSE
> 0.071 vs 0.072). So the guide recommendation below is not established.
> The post-fix rerun is run 4 (`grid_v4.yaml`, below). The k/m collapse
> under every guide holds after the fixes.

**Decision fed (step 5).** The relative fit is sound. Its point estimates
match the anchored joint fit's without binding data. Fit it with
`--guide_type auto_low_rank_multivariate_normal`, not the component guide.
This is documented, not made the default: the default waits on studies
with real data (user, 2026-09-28).
Its intervals, like every SVI fit's here, are too narrow. That is a
separate problem shared with the joint fit, to take up before trusting
coverage from any arm.

## Run 4: post-fix guide comparison (2026-10-01)

**Question.** With the frozen-scale and Laplace-floor fixes in, which
inference gives honest X intervals on the relative fit: the component
guide, the low-rank guide (numpyro's default rank, and rank 20 as on the
real-data run `planning/dev-data/real_fit/rel_off_n05_svi`), MAP + floored
Laplace, or the two-stage fit?

**Design** ([`grid_v4.yaml`](grid_v4.yaml), template
[`run_v4.srun`](run_v4.srun)): runs 2-3's simulations (same base config,
noise levels and seeds 1-3), relative fit only, 5 inference arms, so 2
noise x 3 seeds x 5 = 30 runs. Every arm holds the log(hill_n) population
SD at 0.5, as the real-data fits now do. `run_v4.srun` is run.srun plus
the svi-overconfidence study's two-stage path and guide diagnostics
(`../svi-overconfidence/two_stage.py`, `guide_diagnostics.py`, staged
into the grid's inputs) and a `guide_rank` variable. 12 h per run.

```bash
tfs-setup-sim-grid grid_v4.yaml --out_prefix relative_fit_v4
for d in relative_fit_v4/run_*/; do (cd "$d" && sbatch run_v4.srun); done
```

Afterwards:

```bash
tfs-summarize-calibration relative_fit_v4 --out_prefix calib/relative_fit_v4 \
    --facet_by founder_sampling
```

Compare X coverage by depth, RMSE and width across arms, k and m coverage,
and the guide diagnostics (log-weight SD, k-hat) for the two SVI arms. The
low-rank rank-20 arm decides whether the real-data run's rank was enough.

**Results (2026-10-01).** All 30 runs finished. Every MAP and the
Poisson two-stage MAPs hit their caps (200,000 / 100,000 epochs); every
other SVI fit converged. Pooled into `calib/relative_fit_v4_*` (not
committed). Means over 3 seeds per arm.

X test set:

| noise | arm | 95% cov | 50% cov | median 95% width | RMSE |
|---|---|---|---|---|---|
| poisson | component | 0.06 | 0.02 | 0.16 | 0.297 |
| poisson | low_rank | 0.81 | 0.38 | 0.11 | 0.068 |
| poisson | low_rank rank 20 | 0.81 | 0.38 | 0.11 | 0.065 |
| poisson | map + Laplace | 0.98 | 0.63 | 0.22 | 0.105 |
| poisson | two_stage | 0.91 | 0.51 | 0.13 | 0.075 |
| realistic | component | 0.38 | 0.14 | 0.31 | 0.289 |
| realistic | low_rank | 0.76 | 0.38 | 0.27 | 0.126 |
| realistic | low_rank rank 20 | 0.76 | 0.38 | 0.27 | 0.119 |
| realistic | map + Laplace | 0.94 | 0.45 | 0.64 | 0.293 |
| realistic | two_stage | 0.78 | 0.32 | 0.27 | 0.110 |

95% coverage of the shared and per-genotype parameters (Poisson /
realistic):

| arm | growth_k | growth_m | dk_geno | log_hill_K | hill_n |
|---|---|---|---|---|---|
| component | 0 / 0 | 0 / 0.08 | 0.20 / 0.63 | 0.81 / 0.84 | 0.87 / 0.87 |
| low_rank | 0 / 0 | 0.25 / 0.08 | 0.41 / 0.71 | 0.80 / 0.85 | 0.83 / 0.87 |
| low_rank rank 20 | 0 / 0 | 0.25 / 0.08 | 0.38 / 0.70 | 0.79 / 0.85 | 0.82 / 0.86 |
| map + Laplace | 0.92 / 0.58 | 0.92 / 0.50 | 0.98 / 0.88 | 0.92 / 0.86 | 0.91 / 0.85 |
| two_stage | 0.75 / 0.25 | 0.83 / 0.33 | 0.75 / 0.88 | 0.85 / 0.84 | 0.87 / 0.86 |

- **The component guide is biased even with the fixes**: X RMSE 0.29-0.30
  against 0.07-0.13 for the low-rank guide; 95% coverage 0.06 / 0.38. The
  frozen scales were not the cause, and runs 2-3's guide verdict stands.
- **Rank 20 matches numpyro's default rank** in every number, so the
  real-data run's rank is enough.
- **The low-rank guide undercovers X moderately** (0.81 / 0.76 at 95%,
  0.38 at 50%).
- **Two-stage is the best calibrated under Poisson noise** (0.91 / 0.51)
  but no better than low-rank under realistic noise (0.78 / 0.32). It is
  the only SVI route that covers k and m at all, and only partly under
  realistic noise.
- **MAP + floored Laplace now works**: it covers X (0.98 / 0.94) and k/m
  best, but with the widest intervals and the noisiest point (RMSE 0.105 /
  0.293). The run-2 blowups were the old floor. Every MAP hit its 200,000
  epoch cap.
- **Every guide collapses k and m** (coverage 0-0.25).

**Decision fed.** Fit the relative model with the low-rank guide (rank 20
is enough). Expect X 95% intervals to cover about 0.76-0.81, and treat k
and m intervals from any guide as too narrow. A full-library Laplace is
out of reach (about 2M parameters), so on real data the k/m uncertainty
needs the two-stage fit with the Laplace taken on a subset.
