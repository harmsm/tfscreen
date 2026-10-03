# SVI overconfidence study

**Question.** Every SVI fit in the relative-fit grid (`../relative-fit/`,
run 2) undercovered theta, and the gap grew with read depth. At fewer
than 5 reads per tube, 95% coverage was 0.85-0.94. Above 1000 reads it
was 0.24-0.50. `growth_m` sat 10-300 of its own SDs off truth, in the joint
fit and in every relative arm. The MAP's Laplace intervals on `m` did cover.
Is the gap the variational approximation, meaning the exact posterior
covers and the guides do not? Or is it the model, meaning the exact
posterior undercovers too?

**Decision it feeds.** What to fix: the guide (richer guides, or a
correction to the ELBO) or the model and simulation. Every calibration
result since the count-likelihood study depends on it.

## What we knew going in

From `../relative-fit/relative_fit_v2` (Poisson arm, run 0001 joint and
0003 relative low-rank):
- **The error is a shared scale.** Above 1000 reads, the joint fit's theta
  error is proportional to theta: -0.034 near theta = 1, about 0 near 0.
  The median |z| there is 8-9. The joint fit's `m` is 3-7% too large in
  every run.
- **SVI collapses `m`'s uncertainty.** Its 68% intervals on `m` are about
  0.05% wide. The Laplace intervals are a few percent wide and cover.
- So each genotype's interval reflects its own counts but not the shared
  scale's uncertainty. Once a genotype's counts are precise, the shared
  error dominates.

That points at the variational approximation, but the Laplace arm's 50%
coverage at high depth was only 0.04, so its centers were off too. A
reference posterior settles it.

## Design

[`simulate_config.yaml`](simulate_config.yaml) is the congression
config, cut to singles and spikes: 50 genotypes, 120 tubes, median 1000
reads per genotype per tube. Doubles are dropped, reads and cfu0 are
scaled so singles keep their full-library depth, and 5 in-library
binding anchors are used. The relative fit has 575 latents and 6,140
observations. Poisson noise only.

The first design used NUTS as the reference posterior, but NUTS did not mix
on this model (Results, NUTS pilot). The design now calibrates each arm
against truth over many simulations. MAP + Laplace is the arm with a
full covariance.

[`grid.yaml`](grid.yaml): 10 seeds x 2 fits x 3 inference arms = 60 runs,
on the count likelihood with `sample_offset: level`, as in
`../relative-fit/`.

| axis | levels |
|---|---|
| fit | `joint` (`hill_geno`, binding at weight 1, prefit); `relative` (`hill_relative`, growth only) |
| inference | `component`; `low_rank` (`auto_low_rank_multivariate_normal`); `map` (MAP with `--max_num_epochs 200000`, then a Laplace posterior) |
| seed | 1-10 |

Each run ([`run.srun`](run.srun)) simulates, configures, prefits (joint
only), fits, samples the posterior and summarizes. SVI runs also run
[`guide_diagnostics.py`](guide_diagnostics.py): the Pareto k-hat and
log-weight SD of importance weights `p(theta, y) / q(theta)` over 2000
guide draws, and growth_k/growth_m under the guide and reweighted. Where
k-hat is small, the reweighted numbers are a reference. For mean-field
guides in ~600 dimensions it is not expected to be small, so it mainly
ranks the guides.

NUTS changes made for the pilot, kept (2026-09-28):
- **Start.** `run_nuts` used to start every chain at a uniform draw in
  [-2, 2] on the unconstrained scale. It handed `initialize_model`'s
  `ParamInfo` to `init_to_value`, and no site matched. It now starts at the
  pre-MAP point.
- **Batch.** It passes the full batch through `get_batch`.
- **Diagnostics.** It prints the worst split R-hat and the smallest n_eff.
- **Mass matrix.** `--nuts_dense_mass` is new.

## How to run

On the cluster, from this directory:

```bash
tfs-setup-sim-grid grid.yaml --out_prefix svi_overconfidence_v2
for d in svi_overconfidence_v2/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration svi_overconfidence_v2 --out_prefix calib/svi_overconfidence_v2 --baseline inference=map
python coverage_by_depth.py svi_overconfidence_v2 --out_prefix calib/svi_overconfidence_v2
```

## What to look at

[`coverage_by_depth.py`](coverage_by_depth.py) reports theta/X coverage by
each genotype's median reads per tube, k and m error in the arm's own SDs,
and the guide diagnostics.
- **Coverage by depth.** In `../relative-fit/relative_fit_v2`, 95%
  coverage fell with depth in every arm:
  - joint component: 0.75 at 5 reads or fewer, 0.37 above 1000
  - relative low-rank: 0.94 to 0.50
  - relative component: 0.85 to 0.05

  If MAP + Laplace holds its coverage across depth here while the guides
  do not, the gap is the approximation. If it falls too, the model is
  (part of) the problem.
- **k and m.** In the same grid, the SVI arms sat a median 10-90 of their
  own SDs off truth (95% coverage 0-0.29). MAP + Laplace sat at 0.2 SDs and
  covered (0.96-1.0).
- **Guide diagnostics.** k-hat and log-weight SD, low-rank against
  component, joint against relative.
- **Laplace robustness.** In the large-library grid Laplace blew up in 5 of
  6 runs. It needs to hold on this library before it can be the reference.

## Inputs

`simulate_config.yaml`, `hill_params.csv` (copied from
`../congression-calibration/`), `grid.yaml`, `run.srun`,
`guide_diagnostics.py`, `coverage_by_depth.py`. Seeds fixed in the grid.

## Commit

Grid runs do not log a commit. Where the README did not record one, the
commit below is inferred: the last commit before the grid's runs finished,
from the file times of the downloaded `run.out` files (2026-10-03).

- Run 1 (local, `run_local.sh`, 2026-09-27/28): factored ln_cfu0, before
  722393cf. The current `run.srun` uses `hierarchical`, so rerunning it
  needs `--ln_cfu0_model hierarchical_factored` and an older commit.
- Run 2 (`svi_overconfidence_v2`, 2026-09-28): 722393cf (inferred).
- Two-stage grid (`svi_overconfidence_two_stage`, 2026-09-28): beaab7bb
  (inferred).
- Fix grid (`svi_overconfidence_fix`, 2026-09-29): 9db28286 (inferred).
- Unclipped-binding grid (`svi_overconfidence_noclip`, 2026-09-29):
  6e2903d5 (inferred).
- The NUTS pilots, the reference MAP and full-covariance pilots, the
  continuation runs and the high-plateau diagnosis (Student-t population,
  flat-in-theta prior, pinned noise, precise binding, truth-pinned fit) were
  local one-offs with scratch scripts or temporary code edits that were not
  committed. Their numbers here are a record, not a reproducible result.

None of the result directories are committed (gitignored).

## Results

**Run 2 (`svi_overconfidence_v2`, cluster, 2026-09-28):
`ln_cfu0: hierarchical`.** All 60 runs finished. 95% theta/X coverage by
median reads per tube, mean over 10 seeds (run 1 in brackets):

| arm | <=5 | 21-100 | 101-1000 | >1000 | RMSE |
|---|---|---|---|---|---|
| joint component | 0.84 (0.78) | 0.81 (0.61) | 0.63 (0.47) | 0.36 (0.30) | 0.029 (0.052) |
| joint low_rank | 0.88 (0.80) | 0.87 (0.65) | 0.75 (0.49) | 0.31 (0.27) | 0.027 (0.053) |
| relative component | 0.86 (0.84) | 0.81 (0.62) | 0.75 (0.43) | 0.55 (0.30) | 0.060 (0.080) |
| relative low_rank | 0.90 (0.88) | 0.81 (0.64) | 0.77 (0.43) | 0.63 (0.29) | 0.057 (0.078) |

- **The ln_cfu0 fix halved the joint fit's RMSE** and lifted coverage at
  every depth. It was less than the seed-1 refit suggested (0.80 above 1000
  reads there, 0.31-0.63 across 10 seeds here).
- **The depth gradient remains, and it tracks k and m.** SVI arms sit a
  median 5-125 of their own SDs off truth on k and m (95% coverage
  0-0.25). MAP + Laplace sits at 0.3-0.5 SDs and covers (0.95-1.0). With
  the model mismatch gone, what remains is the guides dropping the shared
  parameters' uncertainty. It matters most for genotypes whose own counts
  are precise. The low-rank guide is closer to its posterior (log-weight SD
  21-22 against 750-780 for the component guide) but still collapses k and
  m.
- **MAP + Laplace on theta is still unusable in the relative fit**
  (Laplace blowups). In the joint fit it covers 0.53 above 1000 reads, the
  best of any arm there.

Next: carry k and m uncertainty into the per-genotype posteriors. Options:
a guide that pairs the per-condition k and m with the genotype parameters
(block structure), or a two-stage approach that samples k and m from their
Laplace posterior and fits the rest conditionally.

**NUTS pilot (2026-09-28, local, seed 1, relative fit): no usable
reference yet.** Every attempt ran at the maximum tree depth (1023
leapfrog steps) with a step size of 1e-6 to 1e-3. Over 500 warmup and 500
draws, split R-hat was 3.5-3.7 and n_eff 3 on essentially every latent,
with no divergences:

| attempt | outcome |
|---|---|
| diagonal mass matrix | tree-depth ceiling through warmup (stopped at 41%) |
| dense mass matrix | R-hat 3.6, n_eff 3 |
| dense, `ln_cfu0_tube_scale` lifted off its collapsed MAP value (5e-5 against a prior median of 0.34) | R-hat 3.7, n_eff 3 |
| dense, every hierarchical scale held at its value (removes the non-centered funnels) | R-hat 3.5, n_eff 3 |
| the same in float64 | the same step sizes at the ceiling (stopped at 30% of warmup) |

So the cause is not the funnels, the collapsed scale or float32 precision.
It is the posterior's own curvature at this depth. Candidates are the
curved m·X ridge and Hill parameters that are sharp for some genotypes and
flat for others. A dense mass matrix cannot remove curvature, and this is
the same geometry that defeats the mean-field guide.

Options: reparameterize (centered per-genotype latents for well-measured
genotypes) or try another sampler, with open-ended cost. Or drop the exact
reference: simulation-based calibration of Laplace at a converged MAP over
many seeds, plus diagnostics of how well each SVI guide matches the
posterior. The relative-fit grid already hints at the answer without a
reference: Laplace intervals on `m` covered in every run, where SVI's missed
by 10-300 SDs.

**Reference pilots (2026-09-28, local, seed 1, relative fit).** The small
library reproduces the gap with the component guide. 95% X coverage is 1.0
at 5 reads or fewer and 0.12 above 1000. growth_k and growth_m sit 20-100
of their own SDs off truth. Both candidate references failed on it:

- **MAP + Laplace.** The MAP hit its 200,000-epoch cap at step size 1e-5.
  The Hessian had 5 negative eigenvalues, down to -66. The Laplace X
  intervals came out about 4e10 wide, with 50% coverage 0-0.04 at every
  depth. Its k/m intervals were sensible: |z| at most 0.6 for the
  selective m, but up to 6.8 for the control conditions.
- **`auto_multivariate_normal` (full-covariance guide).** The loss
  diverged to about -1e23 by epoch 5,000 and ended near -8e23. The
  convergence monitor then called it converged, which is a bug in the
  monitor. X intervals were about 0.01 wide, with 95% coverage 0.05-0.25.

**The common cause is a model bug, not the references.** `hierarchical_factored`
draws `ln_cfu0_tube_offset ~ Normal(0, ln_cfu0_tube_scale)`, centered, over
only 4 offsets (replicate x pre-condition). The offsets trade against the
per-genotype baselines, so they can sit near 0. There the joint density
grows like `tube_scale^-4` as the scale goes to 0, and the HalfNormal prior
does not stop it. The posterior density is unbounded:
- no MAP exists (the pre-MAP puts the scale at 5e-5, against a prior median
  of 0.34);
- a full-covariance guide chases it to -inf loss;
- NUTS started there sits in the funnel.

The mean-field guide survives only because it is too rigid to follow.

Next: a non-centered tube offset (`offset = tube_scale * z`, `z ~ N(0, 1)`)
removes the singularity. Then retry the full-covariance guide and
MAP + Laplace as the reference on this library. Separately, the convergence
monitor should refuse to call a run converged when its loss has run off to
about -1e23.

**Fixes (2026-09-28), then the grid.**

1. **Tube offset.** The `hierarchical_factored` tube offset is now
   non-centered. On seed 1 the MAP's `ln_cfu0_tube_scale` is now 0.26 (was
   5e-5). A new test checks that the density stays bounded as the scale
   goes to 0.
2. **Runaway loss.** The convergence monitor ends a run whose loss falls
   below -1000 times its starting magnitude as `diverged`.
3. **Retried references (seed 1, relative), neither usable yet:**
   - **Full-covariance guide.** Still ran away, to -2.8e22, and was stopped
     at step 2000 as diverged. The model's log density at its draws is
     finite (about -5e4). The runaway is the guide's own `log q` (-4.2e22):
     its `scale_tril` is badly conditioned (all entries at most 6e-4),
     and float32 triangular solves recover wrong standard-normal values
     that the optimizer exploits. A numerical failure of the dense guide,
     not a model singularity.
   - **MAP + Laplace.** The MAP still ends at the 200,000-epoch cap,
     improving (t = 17 at step size 1e-6), with 5 negative Hessian
     eigenvalues. Laplace X is useless: widths about 2e10, mean error 5.7.
     Its k and m intervals are sensible for the selective conditions
     (|z| at most 0.4) but not the controls (|z| up to 9).

The grid ran anyway (user, 2026-09-28): 60 runs locally, 3 at a time
([`run_local.sh`](run_local.sh)), on the fixed model. It measures the
component and low-rank guides' coverage by depth across 10 seeds. The MAP
arm is expected to fail on theta but give k and m.

**Grid results (2026-09-28): 60 runs, all finished locally.** No divergences.
Pooled in `calib/` (not committed). 95% theta/X coverage fell with depth in
every arm:

| arm | <=5 reads | 21-100 | 101-1000 | >1000 |
|---|---|---|---|---|
| joint component | 0.78 | 0.61 | 0.47 | 0.30 |
| joint low_rank | 0.80 | 0.66 | 0.49 | 0.27 |
| joint map | 0.80 | 0.73 | 0.54 | 0.43 |
| relative component | 0.84 | 0.62 | 0.43 | 0.30 |
| relative low_rank | 0.88 | 0.64 | 0.43 | 0.29 |
| relative map | 0.72 | 0.20 | 0.13 | 0.17 (Laplace blew up, as expected) |

- **k and m.** SVI arms sat a median 13-62 of their own SDs off truth
  (95% coverage 0-0.05). MAP + Laplace sat at 0.5-2.4 SDs (coverage
  0.48-0.78).
- **Guide diagnostics.** The low-rank guide is far closer to its posterior
  than the component guide (log-weight SD 15 against 290; k-hat 4.4-4.5
  against 7.4-7.6) and covers no better.
- **The misses are per-genotype, not a shared scale.** Above 1000 reads the
  relative fits are unbiased on average (slope 1.00, mean error +0.003).
  Their RMSE of 0.042 is about 5 times the posterior SD, and the error is
  nearly the same at every concentration of a genotype: a per-genotype
  offset in X.

**Cause: the ln_cfu0 model does not match the simulation.** Every study
grid (congression-calibration, count-likelihood, relative-fit, this one),
and `examples/simulate-and-analyze/run.sh`, fit with
`ln_cfu0: hierarchical_factored`. That model gives each genotype one
starting abundance shared by every pre-condition, plus one offset per
tube. The simulator transforms kanR and pheS as separate libraries, so each
genotype's starting abundance differs between them. Here the per-genotype
kanR minus pheS difference has SD 0.17-0.29 ln units, up to 0.9. The fit
has to put that difference into the growth rates, and so into a
per-genotype X offset it is confident about.

Refitting seed 1 (relative, low-rank) with `ln_cfu0: hierarchical`, one
starting abundance per replicate x pre-condition x genotype and the
configure default, on the same data:

| | factored | hierarchical |
|---|---|---|
| X RMSE | 0.101 | 0.027 |
| mean X error | -0.079 | -0.002 |
| 95% coverage, 101-1000 reads | 0.19 | 0.81 |
| 95% coverage, >1000 reads | 0.09 | 0.80 |
| growth_m median abs z | ~80 | ~9 |

**Conclusions.**
1. **Most of the theta undercoverage in these grids is the ln_cfu0
   mismatch.** It is not the variational approximation. Whether real
   data need `hierarchical` or `hierarchical_factored` depends on whether
   kanR and pheS come from one stock (the factored model's assumption) or
   from separate transformations (the simulator's). That is a protocol
   question, and the choice matters a great deal.
2. **The guides do still collapse k and m.** On the `hierarchical` refit,
   growth_k and growth_m remain 1-200 of their own SDs off truth with SDs
   near 1e-5, and 95% theta coverage stays about 0.80 rather than 0.95.
   That part is the approximation, and it is now small enough to study on
   its own.

Next: rerun this grid (or seed subsets) with `ln_cfu0: hierarchical` to
measure what remains, once the ln_cfu0 question is settled for the real
protocol. The earlier grids' calibration numbers carry the same mismatch.

**Two-stage fit (2026-09-28).** The SVI guides collapse the uncertainty on
the per-condition k and m, and Laplace at the MAP covers them. So the
`two_stage` arm ([`two_stage.py`](two_stage.py)) works in two stages:
- **Stage 1.** Fit the MAP and take a Laplace posterior.
- **Stage 2.** Take 10 draws of (k, m) from it. For each, refit everything
  else by SVI with k and m pinned at the draw (`linear`'s new `k_pinned`
  and the existing `m_pinned`).
- **Pool.** Combine the 10 conditional posteriors with equal weight. With
  the draws from p(k, m | y), the pool approximates the marginal posterior
  of the genotype parameters, and their intervals carry the k/m
  uncertainty.

Local test, seed 1, relative fit, only 2 draws (same data as v2):

| | two-stage | v2 component | v2 low_rank |
|---|---|---|---|
| X RMSE | 0.027 | 0.030 | 0.027 |
| 95% coverage, 101-1000 reads | 0.88 | 0.70 | 0.81 |
| 95% coverage, >1000 reads | 0.91 | 0.60 | 0.79 |
| growth_m abs z | 0.2-1.1 | 0.8-7.4 | 0.9-29 |

With 2 draws growth_k is still 2-19 SDs off; the grid uses 10. Each
conditional fit took about 5 minutes locally. Each conditional SVI fit
ended at its 100,000-epoch cap, because the parameter test reads an
infinite drift on `dk_geno_hyper_loc_scale`. That is a monitor quirk to fix
separately.

[`grid_two_stage.yaml`](grid_two_stage.yaml): the same simulations as
`grid.yaml` (same seeds), `two_stage` only, 20 runs. On the cluster:

```bash
tfs-setup-sim-grid grid_two_stage.yaml --out_prefix svi_overconfidence_two_stage
for d in svi_overconfidence_two_stage/run_*/; do (cd "$d" && sbatch run.srun); done
```

When they finish, pool and compare with `svi_overconfidence_v2`:

```bash
tfs-summarize-calibration svi_overconfidence_two_stage --out_prefix calib/svi_overconfidence_two_stage
python coverage_by_depth.py svi_overconfidence_two_stage --out_prefix calib/svi_overconfidence_two_stage
```

### Two-stage grid (2026-09-28)

`svi_overconfidence_two_stage`: 20 runs, the v2 simulations, 10 Laplace
draws per run. All 20 finished. Pooled with the commands above; compared
with `svi_overconfidence_v2`.

Theta (joint) or X (relative), mean over 10 seeds:

| fit | inference | 95% coverage | >1000 reads | 95% width | RMSE |
|---|---|---|---|---|---|
| joint | component | 0.57 | 0.36 | 0.062 | 0.037 |
| joint | low_rank | 0.59 | 0.31 | 0.057 | 0.034 |
| joint | two_stage | 0.86 | 0.81 | 0.136 | 0.046 |
| relative | component | 0.68 | 0.55 | 0.10 | 0.071 |
| relative | low_rank | 0.74 | 0.63 | 0.11 | 0.072 |
| relative | two_stage | 0.94 | 0.97 | 0.52 | 0.071 |

growth_k and growth_m 95% coverage rose from 0-0.25 under the guides to
0.78-1.0, median |z| 0.4-0.7. dk_geno coverage rose from 0.85-0.86 to
0.92-0.94 and log_hill_K from 0.76-0.88 to 0.91-0.93.

The averages hide a split by seed. Every MAP hit its 200,000-epoch cap,
and every Laplace Hessian had 2-7 negative eigenvalues, clamped to 1e-3.
Where the clamped directions touch m, the Laplace draws of m spread widely
(seed 8 joint: m for kanR+kan from -0.049 to +0.017, SD 0.018, truth
-0.010), and the conditional fits' final losses differ by thousands of
nats. So the loss range across a run's draws flags a broken Laplace:

- Joint seeds with a sound Laplace (loss range under 170 nats; seeds 1, 6,
  7, 10): k and m are covered (|z| at most 1.9), but theta above 1000
  reads still covers only 0.44-0.63, and widths stay at 0.04-0.10.
- Joint seeds with a broken Laplace (range 440-7,600 nats): coverage
  0.93-1.0, bought with widths of 0.10-0.30. Seed 8's RMSE tripled
  (0.033 to 0.113).
- Relative: coverage above 1000 reads is 0.83-1.0 in every seed, sound
  Laplace or not. Seeds 3 and 5 have broken Laplaces and X widths of 2.7
  and 1.2. Seed 2's RMSE of 0.22 matches every other method on that seed.

Reading: for the relative fit, carrying k and m uncertainty closes the
coverage gap. For the joint fit it does not: with k and m honestly
covered, the conditional fit still undercovers theta at high depth, so
something inside the conditional component-guide fit collapses too. The
Laplace stage is fragile everywhere, because the MAP never converges.

Each conditional fit again ran to its 100,000-epoch cap on the
`dk_geno_hyper_loc_scale` infinite-drift quirk.

### Fixes after the two-stage grid, and the fix grid (2026-09-29)

Two bugs, both fixed in the code:

- **Frozen hyperparameter scales.** The component guide's hyperparameter
  scales are constrained `greater_than(1e-4)`, and the guide start capped
  every scale at 1e-4, so they started on the bound, `-inf` unconstrained,
  and never moved. Every component-guide SVI fit since 2026-09-27 kept each
  hyperparameter's guide SD at 1e-4, v2 and the two-stage grid included.
  The conditional fits' "infinite drift" on `dk_geno_hyper_loc_scale` was
  this, not a monitor quirk. It may explain part of the joint fit's
  undercoverage.
- **Laplace floor.** The 1e-3 eigenvalue floor is now the prior's
  curvature along each eigenvector. Rerun locally on the grid's own MAPs,
  the Laplace SD of m fell from 0.018 to 0.0005 on seed 8 joint, and m
  and k now sit within about 3 SD of truth on seeds 1, 3 and 8 joint and
  3 and 5 relative. The MAPs still stop at the cap with 4-7 negative
  eigenvalues.

`two_stage.py` now records each conditional fit's final loss in
`draws.csv` and prints their span, and takes `--max_num_epochs`.
`run.srun` has a `two_stage_low_rank` arm: the same pipeline with the
low-rank autoguide in the conditional fits.

[`grid_fix.yaml`](grid_fix.yaml): joint fit, seeds 1 and 7 (sound
Laplace, still undercovered) and 3 and 8 (broken Laplace), each with
`component`, `two_stage` and `two_stage_low_rank`. 12 runs.

```bash
tfs-setup-sim-grid grid_fix.yaml --out_prefix svi_overconfidence_fix
for d in svi_overconfidence_fix/run_*/; do (cd "$d" && sbatch run.srun); done
```

Then:

```bash
tfs-summarize-calibration svi_overconfidence_fix --out_prefix calib/svi_overconfidence_fix
python coverage_by_depth.py svi_overconfidence_fix --out_prefix calib/svi_overconfidence_fix
```

### Fix grid results, and clipped binding (2026-09-29)

`svi_overconfidence_fix`, 12 runs, all finished. Joint theta, mean over
seeds 1, 3, 7 and 8:

| grid | inference | 95% coverage | >1000 reads | 95% width | RMSE |
|---|---|---|---|---|---|
| v2 | component | 0.57 | 0.34 | 0.060 | 0.030 |
| v2 | low_rank | 0.61 | 0.33 | 0.055 | 0.029 |
| two-stage (old Laplace) | two_stage | 0.84 | 0.76 | 0.177 | 0.051 |
| fix | component | 0.61 | 0.37 | 0.061 | 0.029 |
| fix | two_stage | 0.67 | 0.46 | 0.060 | 0.028 |
| fix | two_stage_low_rank | 0.67 | 0.46 | 0.055 | 0.031 |

- The unfrozen hyperparameter scales barely moved plain SVI (0.34 to 0.37
  above 1000 reads).
- With a sound Laplace, the two-stage fit helps only modestly (0.46). The
  old grid's 0.76 came from the broken Laplace's inflated k/m draws.
- The low-rank conditional guide adds nothing.
- growth_k and growth_m cover 0.63-0.69 in two_stage (median |z| 1.7 and
  0.8): the floored Laplace is somewhat narrow, since the MAP still stops
  short of its optimum.

Where the high-depth misses are: in the two_stage runs, every genotype
above 1000 reads, wt included, has theta_low (about 0.99) and theta_high
(about 0.01) about 0.01 low, median z of -3.3 and -3.2, in all four seeds.
log_hill_K, hill_n and dk_geno cover 0.90-0.96. Theta misses are worst at
the plateaus (coverage 0.2 at 0-0.001 and 1 mM) and fine at the
transition (0.82-0.92 at 0.01-0.03 mM). So the remaining undercoverage is
a shared offset in the absolute theta scale, which the joint fit takes
from the binding data. The relative fit, with no binding, covers.

One candidate: the simulator clipped noisy binding observations to [0, 1]
(8-14 of 81 per run), while the fit's binding likelihood is an unclipped
Normal. Clipping near 1 pulls the high plateau down, which fits
theta_low's bias, but clipping near 0 would pull theta_high up, not down,
so it may not be the whole story. `tfs-simulate` no longer clips
(`binding_data.clip_theta_obs`, default false); regenerated with it,
seed 1's binding table differs in exactly its 8 clipped rows and the
growth data match to round-off.

[`grid_noclip.yaml`](grid_noclip.yaml): the same 4 seeds, component and
two_stage, 8 runs.

```bash
tfs-setup-sim-grid grid_noclip.yaml --out_prefix svi_overconfidence_noclip
for d in svi_overconfidence_noclip/run_*/; do (cd "$d" && sbatch run.srun); done
```

### Unclipped-binding grid (2026-09-29)

`svi_overconfidence_noclip`, seeds 1, 3 and 7 scored (seed 8 still
copying). Clipping is not the cause. Median error of the two plateaus
for genotypes above 1000 reads, clipped (fix grid) against unclipped:

| seed | inference | high-IPTG plateau | low-IPTG plateau |
|---|---|---|---|
| 1 | component | -0.0102 / -0.0102 | -0.0106 / -0.0100 |
| 1 | two_stage | -0.0095 / -0.0093 | -0.0089 / -0.0109 |
| 3 | component | -0.0066 / -0.0068 | -0.0170 / -0.0132 |
| 3 | two_stage | -0.0055 / -0.0059 | -0.0119 / -0.0089 |
| 7 | component | -0.0087 / -0.0088 | -0.0106 / -0.0097 |
| 7 | two_stage | -0.0076 / -0.0076 | -0.0095 / -0.0090 |

Unclipped binding stays the simulator default: it matches the fit's
likelihood and real anisotropy-derived theta.

**Truth-pinned fit** (seed 1, joint, local, k and m pinned at their true
values, component guide, 40,000 steps). The simulator's growth follows
k + dk + m A theta exactly (to 2e-16) and its theta is the Hill truth
(to 1.4e-5), so this is not a simulator-model mismatch. Above 1000
reads:

- low-IPTG plateau (theta about 0.99): error -0.003, z -1.1, 95%
  coverage 0.85. Its bias came mostly from k and m.
- high-IPTG plateau (theta about 0.01): error -0.010, SD 0.002, z -4.9,
  coverage 0.04. It stays biased with k and m exact.
- log_hill_K covers 0.96.

So the high plateau is a separate problem. Near theta = 0 growth barely
sees theta: with |m| about 0.01 per min, 0.01 of theta is about 0.02 ln
units over a whole selection. Yet the fit reports an SD of 0.002, which
must come from the logit-scale hierarchy (theta_high = sigmoid(logit_low +
logit_delta), pooled across genotypes), not from the data. This is the
saturated regime `in_regime` already treats as model-conditional.

**Full noclip grid (8 of 8).** Joint theta coverage above 1000 reads:
component 0.39 (fix grid 0.37), two_stage 0.57 (0.46). k and m in
two_stage cover 0.63 and 0.81.

**Split by true theta** (points above 1000 reads, seeds 1, 3, 7, 8;
saturated = true theta below 0.02 or above 0.98, 46% of the points):

| regime | grid | inference | 95% coverage | RMSE |
|---|---|---|---|---|
| resolvable | v2 | component | 0.56 | 0.012 |
| resolvable | fix | component | 0.59 | 0.012 |
| resolvable | fix | two_stage | 0.68 | 0.010 |
| resolvable | noclip | component | 0.64 | 0.011 |
| resolvable | noclip | two_stage | 0.80 | 0.009 |
| saturated | v2 | component | 0.09 | 0.014 |
| saturated | fix | two_stage | 0.21 | 0.011 |
| saturated | noclip | component | 0.11 | 0.011 |
| saturated | noclip | two_stage | 0.31 | 0.010 |

Where growth can see theta, the two-stage fit with the fixes covers
0.80. The saturated plateaus, which growth barely sees, carry the
remaining overconfidence: the logit hierarchy in `hill_geno` gives them
intervals the data do not support. That is the next thing to fix.

### High-plateau diagnosis (2026-09-29, local, seed 1 joint, k and m pinned at truth)

All fits component-guide SVI from the same pre-MAP, 40,000 steps, scored
above 1000 reads. The warm-up MAP puts every deep genotype's high-IPTG
plateau near truth (H74S logit -4.19 against a true -4.53); every SVI
variant slides it to about 0.0005 (logit -7 to -8) with too-narrow
intervals:

| variant | high plateau error | coverage | width |
|---|---|---|---|
| baseline | -0.0102 | 0.31 | 0.0076 |
| Student-t population (df 3) on the plateau offsets | -0.0102 | 0.31 | 0.0077 |
| no per-tube offsets (misspecified: loss 11,000 worse) | -0.0102 | 1.00 | 0.029 |
| count noise held near the MAP by a tight prior | -0.0096 | 0.15 | 0.0065 |

The fix grid's low-rank conditional guide is also low (-0.0070).

Profiles of the full log density over one genotype's high plateau,
everything else held fixed:

- at the MAP: H74S's data pin it near 0.015; pushing it to 0.0001 costs
  61 nats (K84D 31, M42Y 6).
- at the SVI posterior median: flat from 0 to 0.003 and almost the same
  shape for all three genotypes, so something shared has changed.

SVI's shared parameters differ from the MAP's: count overdispersion
`growth_inv_r` 5x larger (0.0005 against 0.0001; about 0 is right for
this Poisson-only simulation), and the per-tube offsets of one condition
shifted by about +0.006 at low IPTG and -0.005 at high IPTG, the size
that absorbs a common shift of about 0.01 in theta. Pinning either one
alone did not remove the slide.

Ruled out: binding clipping, k and m, the population's tail, the guide
family. Not yet tested: the plateau parameterization itself. On the
logit scale a plateau at 0.0005 has as much prior room as one at 0.01,
and growth, linear in theta, separates them only weakly once the shared
parameters loosen. A plateau parameterized on the theta scale, or
bounded away from 0, would test that. The Student-t option was reverted.

Further tests, same setup (all SVI, 100,000 steps at a constant step
size unless noted; high plateau error above 1000 reads):

| variant | high plateau error | coverage |
|---|---|---|
| prior flat in theta for both plateaus (log-Jacobian factor) | +0.0105 | 0.23 |
| constant step size 1e-3, no cuts | -0.0106 | 0.08 |
| + count noise hard-pinned at the MAP values | -0.0102 | 0.04 |
| + binding data 5x more precise (noise 0.005) | -0.0103 | 0.27 |

- Flat-in-theta prior: the plateaus follow whatever prior they get
  (the low plateau went to -0.027), so in the SVI solution they are
  prior-dominated; the parameterization is not the cause.
- Constant step size: the loss levels off, so this is SVI's optimum, not
  a transient frozen by early step-size cuts. `growth_inv_r` rises to
  0.00098 there, 10x the MAP.
- Precise binding pins the shared level (wt's high plateau 0.0092 against
  0.0100), yet the non-binding genotypes still collapse: H74S, 65,000
  reads, 0.0002 against 0.0107, with its low plateau, K, n and dk_geno all
  on truth. So it is per genotype, not a shared shift.

Found along the way: `tfs-extract-params` mislabeled `ln_cfu0` rows
whenever a genotype was missing from a block (fixed; see CHANGELOG). The
model's own `ln_cfu0` matches the presplit data at 0.999; only the
extraction was wrong.

Open: SVI's optimum puts the high plateau of well-measured genotypes near
0 while the MAP puts it near truth, and none of prior shape, population
tail, tube offsets, count noise, step size or binding precision changes
that. Next candidates: pin the tube offsets and the noise together at the
MAP, or compare SVI with a reference that needs no guide (importance
reweighting of the Laplace, or NUTS on a smaller library).
