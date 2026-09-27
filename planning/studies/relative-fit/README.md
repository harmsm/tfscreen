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

[`grid.yaml`](grid.yaml): 2 noise x 2 fits x 3 seeds = 12 runs, all on
the count likelihood (the default after `../count-likelihood/`; the
`lncfu` arm of the first design was dropped, 2026-09-27).

| axis | levels |
|---|---|
| simulated noise | `poisson`; `realistic` (founder sampling, demographic growth, shared transformation, 150,000 PCR templates with CV 0.5; designed for ~8x Poisson, fitted at ~29x in `../count-likelihood/` v3) |
| fit | `joint` (`hill_geno`, binding at weight 1, prefit with `k_scale_ceiling` 0.005); `relative` (`hill_relative`, growth only, no prefit) |
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
tfs-setup-sim-grid grid.yaml --out_prefix relative_fit
for d in relative_fit/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration relative_fit --out_prefix calib/relative_fit \
    --baseline fit=joint --facet_by founder_sampling
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

Next: test a guide that can follow the ridge (`auto_low_rank_multivariate_normal`) and
MAP + Laplace on the relative arm, then rerun the grid.
