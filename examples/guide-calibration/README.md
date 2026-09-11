# guide-calibration

A grid for Phase 3 of the SVI calibration work. Every run simulates a library
(so the true parameters and θ are known), fits it with one variational guide,
and writes calibration diagnostics against that truth. Comparing the runs
separates three possible causes of posterior miscalibration:

- the variational family (mean-field vs. covariance);
- how the component guides are built (parameterization, initialization,
  constraints);
- genotype mini-batching.

## Contents

| File | Purpose |
|------|---------|
| `simulate_config.yaml` | Base simulation. It matches `examples/simulate-and-analyze`: congression at λ = 0.3572, 20 stratified in-library binding genotypes, and 20 random base-growth genotypes disjoint from the binding sets. The 3-site library gives about 480 genotypes and about 9k latent parameters (Tier 1). |
| `hill_params.csv` | Hill parameters for the spiked binding controls. |
| `grid.yaml` | The `tfs-setup-sim-grid` grid: 5 simulation seeds × 6 guides × 2 batch sizes × 2 `theta_growth_noise` models × 2 fit seeds = 240 runs. |
| `run.sh` | Jinja2 template rendered into each run directory. The pipeline is simulate → configure → prefit → fit (`--guide_type`) → posterior → predictions → `tfs-summarize-fit`. |

## Usage

```bash
cd examples/guide-calibration
tfs-setup-sim-grid grid.yaml --out_prefix guide_grid
for d in guide_grid/*/; do (cd "$d" && sbatch run.sh); done
```

Each run directory holds its own `tfs_sim_config.yaml` (the base config plus
that run's simulation seed), a `combo.json` recording its grid variables, and
a `summary/` directory from `tfs-summarize-fit`. That directory contains the
PIT and coverage outputs (`*_calibration_curve.csv`, `*_pit.csv`) for θ, the
fitted parameters and growth.

## Summarizing the grid

`tfs-summarize-calibration` collects every finished run. It is safe to run
while the grid is still in progress: unfinished runs are listed in
`*_run_status.csv`.

```bash
tfs-summarize-calibration guide_grid --out_prefix calib/guide \
    --baseline guide_type=component guide_rank=None \
    --facet_by batch_size theta_growth_noise_model
```

| Output | Contents |
|--------|----------|
| `calib/guide_runs.csv` | Metrics for each run × quantity × stratum |
| `calib/guide_arms.csv` | Mean and std over the 10 replicates of each arm (5 simulation seeds × 2 fit seeds) |
| `calib/guide_paired_summary.csv` | Each arm minus `component` on the same simulated data: mean, std and SE of the change in calibration error, bias, 95% coverage, 95% width, RMSE and KS statistic |
| `calib/guide_theta_test_calibration_curves.pdf` | Pooled θ calibration curves, one panel per batch size × θ-noise model |

The metrics are:
- `coverage_<a>`: fraction of true values inside the central `a` interval.
- `calibration_bias`: negative means intervals are too narrow.
- `width_<a>`: interval width.
- `rmse`: error of the posterior median.

Strata are `has_binding` × `origin` (spiked/bulk), plus `theta_regime`
(resolvable/saturated) for θ. `all` marks a pooled row.

## Design notes

- **Paired comparisons.** Every fit arm sees the same five simulated data
  sets, so the guides can be compared data set by data set.
- **Guides.** `component` and `auto_normal` are both mean-field, so the
  difference between them comes from how the component guides are built.
  `auto_diagonal_normal` should agree with `auto_normal` and serves as a
  sanity check. Low-rank (rank 10 and 50) and the dense
  `auto_multivariate_normal` add posterior covariance.
- **Batching.** `batch_size` 65536 is larger than the library, i.e. full
  batch. 160 is about a third of the library, matching production's
  65536 / ~200k ratio.
- **`theta_growth_noise`.** The component guide uses the prior as the guide
  for `logit_normal`'s per-observation ε, whereas every autoguide learns it.
  The `zero` arm separates that difference from the guide-family effect.
- **Fit settings** otherwise match the production fit (`dev/run.sh`), except
  that prefit runs without `--pin_m`.
- **Tier 2.** For production-scale runs, swap in the full-scan `degen_sites`
  line (commented out in `simulate_config.yaml`) and drop the dense guide,
  which needs O(D²) memory.
