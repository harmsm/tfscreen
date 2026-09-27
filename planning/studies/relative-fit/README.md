# Relative-X fit study

**Question.** Does a growth-only fit on the wt-relative scale X (theta
component `hill_relative`: no binding data, no prefit, wt's X pinned to 1 at
0 mM and 0 at 1 mM IPTG) recover X, the Hill K and n, and each condition's
`k` and `m`, about as well as the anchored joint fit recovers theta? Does
that hold under realistic count noise, and on either growth likelihood?

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

[`grid.yaml`](grid.yaml): 2 noise x 2 likelihood x 2 fits x 3 seeds = 24
runs.

| axis | levels |
|---|---|
| simulated noise | `poisson`; `realistic` (founder sampling, demographic growth, shared transformation, 150,000 PCR templates with CV 0.5: about 8x Poisson), as in `../count-likelihood/` |
| likelihood | `lncfu` (Student-t on `ln_cfu`, `normal_kt` growth noise); `counts` (negative binomial on reads, `sample_offset: level`) |
| fit | `joint` (`hill_geno`, binding at weight 1, prefit with `k_scale_ceiling` 0.005); `relative` (`hill_relative`, growth only, no prefit) |
| seed | 1, 2, 3 |

Both fits use per-genotype Hill curves (the simulation's `hill_mut` truth
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

Pending.
