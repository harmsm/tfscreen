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

[`grid.yaml`](grid.yaml): 3 seeds x 2 fits x 4 inference arms = 24 runs,
on the count likelihood with `sample_offset: level`, as in
`../relative-fit/`.

| axis | levels |
|---|---|
| fit | `joint` (`hill_geno`, binding at weight 1, prefit); `relative` (`hill_relative`, growth only) |
| inference | `nuts` (the reference: 4 chains, 1000 warmup, 1000 draws, dense mass matrix, chains started at the pre-MAP point); `component`; `low_rank` (`auto_low_rank_multivariate_normal`); `map` (MAP + Laplace) |
| seed | 1, 2, 3 |

Each run ([`run.srun`](run.srun)) simulates, configures, prefits (joint
only), fits, samples the posterior and summarizes.

NUTS changes this study needed (2026-09-28):
- **Start.** `run_nuts` used to start every chain at a uniform draw in
  [-2, 2] on the unconstrained scale. It handed `initialize_model`'s
  `ParamInfo` to `init_to_value`, and no site matched. It now starts at the
  pre-MAP point.
- **Batch.** It passes the full batch through `get_batch`.
- **Diagnostics.** It prints the worst split R-hat and the smallest n_eff.
- **Mass matrix.** `--nuts_dense_mass` is new. With a diagonal mass matrix
  the relative fit ran at NUTS's maximum tree depth (1023 steps) through
  warmup.

## How to run

On the cluster, from this directory:

```bash
tfs-setup-sim-grid grid.yaml --out_prefix svi_overconfidence
for d in svi_overconfidence/run_*/; do (cd "$d" && sbatch run.srun); done
```

When the runs finish:

```bash
tfs-summarize-calibration svi_overconfidence --out_prefix calib/svi_overconfidence --baseline inference=nuts
python compare_to_nuts.py svi_overconfidence --out_prefix calib/compare_to_nuts
```

## What to look at

- **NUTS against truth.** Does NUTS cover, overall and above 1000 reads
  (`tfs-summarize-calibration`)? First check that its R-hat is at most
  about 1.01, its n_eff is adequate and it has no divergences (`run.out`).
  - If NUTS covers, the gap is the approximation.
  - If NUTS undercovers the same way, the gap is the model or the
    simulation-to-fit mismatch.
- **Each arm against NUTS** ([`compare_to_nuts.py`](compare_to_nuts.py)),
  with no truth needed:
  - `width_ratio`: the arm's 68% width over NUTS's. Below 1 means
    overconfident.
  - `shift_z`: the arm's median minus NUTS's, in NUTS SDs.
  - Per quantity (`growth_k`, `growth_m`, `dk_geno`, `log_hill_K`,
    `hill_n`, theta/X). The `growth_m` rows are the ones to watch.
- **Joint against relative.** Whether binding anchors change the picture.

## Inputs

`simulate_config.yaml`, `hill_params.csv` (copied from
`../congression-calibration/`), `grid.yaml`, `run.srun`,
`compare_to_nuts.py`. Seeds fixed in the grid.

## Commit

Not run on the cluster: NUTS does not mix on this model (below).

## Results

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
