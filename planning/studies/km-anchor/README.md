# Anchoring k and m with the wt monoculture rates

**Question.** On the full-size simulation the deep genotypes rank well (X r
0.99 above 1e4 reads) but their intervals do not cover (95% coverage 0.29
realistic, 0.02 Poisson), because k and m are off and held at the MAP: X is
compressed by |m| 30-45% too large, and k slid up against dk_geno. Counts
measure relative frequencies, so only the tube totals pin the absolute
scale, weakly. The experiment has one more measurement of that scale, the
wt monoculture growth rates. Does a k/m prior from them bring deep coverage
up while keeping the ranking? Does the real data accept that anchor?

**Decision it feeds.** `planning/deep-coverage.md`, step 1: whether the
real fit's k and m get their priors from the monoculture, and what the
deep genotypes' intervals can claim.

**Why it matters for the real data.** In kanR+kan the real fit and the
simulation are off the same way against the monoculture rates, which are
the simulation's truth:

| kanR+kan | k | m |
|---|---|---|
| wt monoculture (= sim truth) | 0.0150 | -0.0141 |
| real fit (recipe) | 0.0298 | -0.0204 |
| simulated fit, realistic | 0.0274 | -0.0203 |
| simulated fit, Poisson | 0.0189 | -0.0182 |

In the simulation that is a property of the fit, so the real kanR curves
are likely compressed the same way. pheS+4CP differs: the real fit's k is
0.023 against 0.003 for the monoculture, and its |m| is smaller (0.0097
against 0.0144), which may be the real selection-onset effect
(`planning/offset-mode-growth-transition.md`).

## Arms

All three refit an existing growth table with the recipe's model (growth-
only relative fit, level offsets at SD 0.17, log(n) population SD 0.5,
loose k/m priors in the control conditions), adding k/m priors in the two
selection conditions from the wt monoculture rates
(`--growth_priors_wt_rates`: k = wt rate at 1 mM, m = rate at 0 minus rate
at 1 mM, SD = standard error of the replicate mean floored at 0.002 per
minute).

| arm | growth table | rates |
|---|---|---|
| `anchor_sim` | `sim_realistic_s2` | the monoculture rates (the simulation's truth, at the monoculture's SD) |
| `anchor_sim_perturbed` | `sim_realistic_s2` | the same, each rate moved once by a draw of its SD (seed 1) |
| `anchor_real` | the recipe's `tfs_growth.csv` | the monoculture rates |

Only the selection conditions get monoculture priors (`make_wt_rates.py`):
in the control conditions wt's monoculture rate varies with IPTG in ways
an m of 0 cannot carry, and the simulation sets their m to 0. The rate
tables are derived from lab data, so they are generated on the cluster and
never committed.

## How to run

On the cluster, after pulling this commit. Simulation arms, from the
full-size simulation's working directory (it holds `sim_realistic_s2/` and
the `processed/` link):

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-v3
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/km-anchor/{run_anchor.srun,make_wt_rates.py} .
```

```bash
python make_wt_rates.py processed/wt_control_rate_summary.csv --seed 1
```

It prints the 0 and 1 mM rates and their perturbed values; check that it
wrote `wt_rates_selection.csv` and `wt_rates_selection_perturbed.csv`.
`growth_priors_loose.csv` is already in this directory from the full-size
run.

```bash
mkdir anchor_sim && cd anchor_sim && sbatch --export=ALL,GROWTH=../sim_realistic_s2/tfs_growth.csv,LIBRARY=../sim_realistic_s2/library_config.yaml,WT_RATES=../wt_rates_selection.csv,TRUTH_DIR=../sim_realistic_s2 ../run_anchor.srun && cd ..
```

```bash
mkdir anchor_sim_perturbed && cd anchor_sim_perturbed && sbatch --export=ALL,GROWTH=../sim_realistic_s2/tfs_growth.csv,LIBRARY=../sim_realistic_s2/library_config.yaml,WT_RATES=../wt_rates_selection_perturbed.csv,TRUTH_DIR=../sim_realistic_s2 ../run_anchor.srun && cd ..
```

Real arm, next to the recipe run:

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/km-anchor/{run_anchor.srun,make_wt_rates.py} .
```

```bash
python make_wt_rates.py ../processed/wt_control_rate_summary.csv --seed 1
```

```bash
mkdir anchor_real && cd anchor_real && sbatch --export=ALL,GROWTH=../recipe/tfs_growth.csv,LIBRARY=../../processed/library_config.yaml,WT_RATES=../wt_rates_selection.csv ../run_anchor.srun && cd ..
```

`growth_priors_loose.csv` is already in `real_fit/` from the recipe. Each
arm ends with `>>> Done` in its `run.out` and a
`summary/tfs_summarize_fit_summary.json`.

## Inputs

`sim_realistic_s2` (`planning/studies/full-size-sim/`), the recipe's growth
table (`planning/studies/real-data-fit/`), and
`processed/wt_control_rate_summary.csv` (the wt monoculture rates, from
`prep_dev_data.py`).

## Commit

The commit that adds this study. Every step writes its provenance.

## Results

Pending.
