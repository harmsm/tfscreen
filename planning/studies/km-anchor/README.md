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

Run 2026-10-09 (all three arms converged at every stage; offsets clean in
all). **The data reject the anchor, in the simulation and in the real
data.**

| arm | condition | k prior (SD) | k fit | m prior (SD) | m fit | truth k, m |
|---|---|---|---|---|---|---|
| unanchored sim (`sim_realistic_s2`) | kanR+kan | 0.015 (0.01) | 0.0274 | 0 (0.01) | -0.0203 | 0.0149, -0.0138 |
| `anchor_sim` | kanR+kan | 0.0150 (0.002) | 0.0247 | -0.0141 (0.002) | -0.0184 | 0.0149, -0.0138 |
| `anchor_sim_perturbed` | kanR+kan | 0.0156 (0.002) | 0.0255 | -0.0140 (0.002) | -0.0199 | 0.0149, -0.0138 |
| `anchor_sim` | pheS+4CP | 0.0031 (0.002) | 0.0065 | 0.0144 (0.002) | 0.0183 | 0.0033, 0.0141 |
| recipe (real) | kanR+kan | 0.015 (0.01) | 0.0298 | 0 (0.01) | -0.0204 | |
| `anchor_real` | kanR+kan | 0.0150 (0.002) | 0.0249 | -0.0141 (0.002) | -0.0202 | |
| recipe (real) | pheS+4CP | 0.015 (0.01) | 0.0226 | 0 (0.01) | 0.0097 | |
| `anchor_real` | pheS+4CP | 0.0031 (0.002) | 0.0172 | 0.0144 (0.002) | 0.0097 | |

- k moves by a pure slide: on the real data it fell by 0.005 in every
  condition, the unanchored controls included, with dk_geno up by 0.0048
  everywhere; it still ends five prior SDs above the monoculture.
- m does not move. The real kanR m stays at -0.020 against a prior of
  -0.014 +/- 0.002 (pheS 0.0097 against 0.0144), so the likelihood pins it.
  In the simulation, given the truth as its prior, the fit still puts m at
  -0.018 to -0.020 against -0.0138: the fitted model prefers the wrong m.
- Deep coverage barely improves (`anchor_sim`, X 95% coverage 0.65 /
  0.35 / 0.22 at 1e3-1e4 / 1e4-1e5 / >1e5 reads, against 0.56 / 0.29 /
  0.08 unanchored); X slope 1.08-1.23 against 1.19-1.37.
- The real fit is otherwise unchanged (X shifts by 0.03, log K and n by
  under 0.01, 9 held directions).

So the |m| inflation is a misspecification of the fit against how the
simulation makes its data, not a weakly pinned ridge, and the simulation
reproduces the real kanR numbers, so the real kanR X is most likely
compressed by the same factor (~1.4). Ruled out: the OD-derived tube totals
(unbiased in the simulation, drift <= 0.001 per minute, 96% of tubes in the
calibrated range) and the 2026-10-08 simulator features (the bias is in the
s1b arms, which had neither unassigned reads nor the excess wt). Suspects:
congression (the simulation's lambda 0.357 with the homodimer rule; the fit
uses `single`), per-tube rate noise (the fit has level offsets), and the
dk_geno prior's shape. Next: a reduced-library factorial that switches each
off (`planning/deep-coverage.md`, step 1b).
