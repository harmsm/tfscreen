# What makes the fit's m too steep

**Question.** On the calibrated full-size simulation the fit's growth slope
m is about 40% too steep in kanR+kan (-0.018 to -0.020 against a true
-0.0138), even with the truth as a 0.002-SD prior
(`planning/studies/km-anchor/`); the real fit has the same m against the
wt monoculture. The fitted likelihood prefers the wrong m, so something
the simulation does is missing from the fit. Which of the simulation's
features the fit does not model causes it?

**Decision it feeds.** `planning/deep-coverage.md`, step 1b: what the fit
has to model, or correct for, before the deep genotypes' intervals can be
honest.

## Round 1: reduced-library simulations

A reduced library (5 degenerate sites per tile, every spike site kept:
5,256 doubles instead of 223,000), scaled with
`genetics.library_design.scale_library_design` so each sequence keeps its
reads, cells, transformants and PCR templates, on the real design (118
tubes) with the calibrated simulation of `planning/studies/full-size-sim/`.
The fit is the real recipe's (growth-only relative model, level offsets at
SD 0.17, loose k/m priors), MAP only; the slope does not need the Laplace.

| arm | changes from the calibrated simulation |
|---|---|
| `base` | none |
| `no_cong` | no co-transformed cells (`transformation_poisson_lambda` 0) |
| `no_tube` | no per-tube rate noise (`tube_noise_sigma` 0) |
| `poisson` | Poisson counts (no founder, demographic or PCR noise) |
| `clean` | all three off |

Two seeds each. `score_arms.py` puts the truth on the fit's X scale (wt's
true curve at the gauge concentrations) and reports k, m, the ratio
m_fit / m_true, and the median dk_geno error.

`base` must reproduce the bias first; if it does not, the reduced library
changes the problem and the factorial says nothing.

A local test (2026-10-09) ran the simulation, processing and configure in
minutes, but the staged MAP's first stage was still descending (about 1.8
nats per window, t = 12) at step 80,000 after an hour on a laptop CPU, so
the arms run on the cluster.

### How to run round 1

On the cluster, after pulling this commit, from the full-size simulation's
working directory (it holds `processed/` and `pilot_s1/`):

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_factorial.py --pilot pilot_s1 --out_dir m_bias
```

It builds the two full-size configs (a few minutes) and prints one line per
arm: 5256 doubles, scale 0.1612, reads 8.17e+08. Then:

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/{run_arm.sh,run_arm.srun,score_arms.py} m_bias/
```

```bash
cd m_bias && for d in base_s1 base_s2 no_cong_s1 no_cong_s2 no_tube_s1 no_tube_s2 poisson_s1 poisson_s2 clean_s1 clean_s2; do (cd $d && sbatch ../run_arm.srun); done
```

Each arm ends with `>>> Done` in its `run.out` and a
`tfs_params_growth_m.csv`. Then, in `m_bias/`:

```bash
python score_arms.py *_s1 *_s2
```

It prints the table and writes `factorial_scores.csv`.

Inputs: `processed/` (the dev data, via make_sim_config.py) and the
full-size pilot (`pilot_s1/`, for the cfu0 calibration).

### Round 1 results (2026-10-09)

**Inconclusive: `base` did not reproduce the bias.** m_fit / m_true on the
X scale (truth from wt's true curve at the gauge):

| arm | kanR+kan s1 / s2 | pheS+4CP s1 / s2 |
|---|---|---|
| full size, realistic (`sim_realistic_s2`) | 1.47 | 1.43 |
| full size, Poisson (`sim_poisson_s2`) | 1.32 | 1.16 |
| `base` | 1.03 / 1.14 | 0.96 / 1.06 |
| `no_cong` | 0.99 / 0.96 | 1.03 / 0.99 |
| `no_tube` | 1.05 / 1.21 | 1.00 / 1.12 |
| `poisson` | 1.13 / 1.14 | 1.01 / 1.02 |
| `clean` | 1.04 / 1.05 | 1.04 / 1.05 |

Every arm sits between 0.96 and 1.21. The seed-to-seed spread (`base` 1.03
against 1.14) is as large as any difference between arms. Every fit
converged except `clean`, whose stage 1 and joint stage both reached the
epoch cap while still dropping a few nats per window.

The reduced library changed the problem in two ways:

- **Composition.** `scale_library_design` keeps each sequence's abundance,
  so the spiked origin (wt repeated 271 times) keeps its mass while the
  bulk shrinks. wt starts at 55% of the reduced pool against 11% at full
  size, and doubles at 6% against 44% (`library_composition_table`). In
  the sequenced tubes wt is 81% of reads against 31%, and doubles are 3%
  against 35%.
- **Detection.** cfu0 scales with the pool, so each tube's OD600 falls by
  the same factor. 25 of 118 tubes read below the detection threshold and
  dropped out (93 sequenced against 118).

The bias weakens when wt dominates the tubes. That fits the tube totals
pinning the absolute scale through wt when wt is most of the population,
and the library's genotypes setting it when they are. It does not say
which feature is misspecified.

## Round 2: fit-side subsets of the full-size simulation

Given the shared parameters (k, m, tube offsets, population hypers), each
genotype's count likelihood is independent: the count model observes each
genotype's reads against its predicted share of the tube's cells, with no
constraint that the shares sum to one. So refitting the full-size
`sim_realistic_s2` growth table (m ratio 1.47) with only some genotypes is a
valid fit of the same simulated data. It changes only which genotypes
inform k and m. `make_subsets.py` writes three arms:

| arm | genotypes |
|---|---|
| `no_doubles` | wt, spikes and singles (952) |
| `deep_doubles` | those plus every double with at least 1,000 total reads (80,915) |
| `shallow_doubles` | those plus 80,915 of the 126,684 doubles below 1,000 reads, at random (seed 1) |

Same chain as round 1 (`run_arm.sh` skips simulate and process because the
arm directory links the simulation's `tfs_sim_*` files and has its own
`tfs_growth.csv`). Readout: if `no_doubles` recovers m (ratio near 1), the
doubles drive the bias; `deep_doubles` against `shallow_doubles` says
whether it is the well-measured doubles or the noisy ones. If `no_doubles`
is still at about 1.4, the bias comes from how wt, spikes and singles are
modeled (or from the design), and congression and the dk_geno prior's
shape become the next arms at full size.

### How to run round 2

On the cluster, after pulling this commit, from the full-size simulation's
working directory (it holds `sim_realistic_s2/`):

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py sim_realistic_s2 --out_dir m_subsets
```

It reads the 8 GB growth table twice (about 6 minutes locally) and prints
`952 wt/spike/single genotypes, 80915 doubles >= 1000 reads, 80915 of
126684 below`, then one genotype count per arm (952, 81867, 81867).

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/{run_arm.sh,run_arm.srun,score_arms.py} m_subsets/
```

```bash
cd m_subsets && for d in no_doubles deep_doubles shallow_doubles; do (cd $d && sbatch ../run_arm.srun); done
```

When each `run.out` ends with `>>> Done`, in `m_subsets/`:

```bash
python score_arms.py no_doubles deep_doubles shallow_doubles
```

Inputs: `sim_realistic_s2/` (`planning/studies/full-size-sim/`).

## Round 3: the full library with k and m held

If the doubles bias m and wt, spikes and singles do not, then fitting the
full library with k and m held at the no-doubles MAP should recover X for
the doubles. `held_priors.py` turns an arm's fitted k and m into a
`--growth_priors` table, and `run_held.srun` runs the full-size chain
(staged MAP, arrowhead Laplace, predict, `tfs-summarize-fit`) with both
clamped (`--set_priors m_pinned=1 k_pinned=1`). The intervals then leave
out k and m's own uncertainty, so this checks X's slope and the coverage
given k and m, against `sim_realistic_s2`'s summary (X slope 1.19-1.37,
95% coverage 0.56 / 0.29 / 0.08 at 1e3-1e4 / 1e4-1e5 / >1e5 reads). A
20-epoch smoke test of the chain with both pins ran locally (configure, MAP,
arrowhead Laplace, extract).

### How to run round 3

On the cluster, from the full-size simulation's working directory:

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir held_s2 && cd held_s2
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets/no_doubles --out growth_priors_held.csv
```

It prints the four conditions' k and m; kanR+kan should read k 0.0182,
m -0.0142. Then link the simulation's data and truth, and submit:

```bash
ln -s ../sim_realistic_s2/tfs_growth.csv ../sim_realistic_s2/tfs_sim_* . && cp ../sim_realistic_s2/library_config.yaml .
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun . && sbatch run_held.srun
```

It ends with `>>> Done` in `run.out` and a `summary/` directory, like the
full-size runs (several hours: the full library and the Laplace).

## Round 4: hold kanR only

The real pheS+4CP departs from its monoculture with or without the doubles
(`planning/deep-coverage.md` step 3), so the real fit holds kanR only:
kanR+kan and kanR-kan at the no-doubles MAP by a tight prior (SD 1e-4;
the doubles' pull has a likelihood SD near 0.0014, so this gives up about
0.5% of it), pheS at the recipe's loose priors, and no pins (they act on
every condition). X is shared across markers, so the simulated arm checks
that holding kanR alone fixes X while pheS's m stays free: does pheS's m
follow the kanR-pinned X, or do the doubles pull it, and X with it, steep
again? The real arm is read only if the simulated one passes.

| arm | data | held |
|---|---|---|
| `held_kanR_s2` | `sim_realistic_s2` | kanR at the simulated `no_doubles` MAP |
| `held_kanR_real` | the recipe's growth table | kanR at the real `no_doubles` MAP |

### How to run round 4

`held_priors.py --hold` has since changed to write pin columns (round 5);
round 4 ran it at commit 75611b08, where `--hold` held the named
conditions by a tight prior (`--scale`, default 1e-4).

Simulated arm, in the full-size simulation's working directory:

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir held_kanR_s2 && cd held_kanR_s2
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets/no_doubles --hold kanR --loose ../growth_priors_loose.csv --out growth_priors_held.csv
```

Check: kanR+kan k 0.018182, m -0.014221 at scale 0.0001; pheS rows 0.015,
0.01, 0, 0.01.

```bash
ln -s ../sim_realistic_s2/tfs_growth.csv ../sim_realistic_s2/tfs_sim_* . && cp ../sim_realistic_s2/library_config.yaml .
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun . && sbatch --export=ALL,PINS= run_held.srun
```

Real arm, next to the recipe run:

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
mkdir held_kanR_real && cd held_kanR_real
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets_real/no_doubles --hold kanR --loose ../growth_priors_loose.csv --out growth_priors_held.csv
```

Check: kanR+kan k 0.016018, m -0.014136 at scale 0.0001.

```bash
ln -s ../recipe/tfs_growth.csv . && cp ../../processed/library_config.yaml .
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun . && sbatch --export=ALL,PINS= run_held.srun
```

The log's configure line reads `pins: none`. Each run ends with `>>> Done`
in `run.out` and a `summary/` directory (about 2 hours, as round 3).

## Round 5: kanR pinned per condition

Round 4's tight prior leaked (see its result below), so round 5 repeats it
with `linear`'s per-condition pins: `held_priors.py --hold kanR` now writes
`k_pinned`/`m_pinned` columns (1 for kanR, 0 for pheS) into the growth
priors table, which `tfs-configure-model --growth_priors` turns into
per-condition clamps. kanR's k and m are then exact and leave the
Laplace; pheS's are fitted. It needs the commit that adds per-condition
pins on the cluster.

### How to run round 5

Simulated arm:

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir held_kanR_pin_s2 && cd held_kanR_pin_s2
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets/no_doubles --hold kanR --loose ../growth_priors_loose.csv --out growth_priors_held.csv
```

Check: the table has `k_pinned` and `m_pinned` columns, 1 for both kanR
rows and 0 for pheS.

```bash
ln -s ../sim_realistic_s2/tfs_growth.csv ../sim_realistic_s2/tfs_sim_* . && cp ../sim_realistic_s2/library_config.yaml .
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun . && sbatch --export=ALL,PINS= run_held.srun
```

Real arm:

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
mkdir held_kanR_pin_real && cd held_kanR_pin_real
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets_real/no_doubles --hold kanR --loose ../growth_priors_loose.csv --out growth_priors_held.csv
```

```bash
ln -s ../recipe/tfs_growth.csv . && cp ../../processed/library_config.yaml .
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun . && sbatch --export=ALL,PINS= run_held.srun
```

In `tfs_configure_priors.csv`, `growth.condition_growth.m_pinned` has one
row per condition (1, 1, 0, 0 by name). After the fit,
`tfs_params_growth_m.csv` reads kanR's held values exactly.

## Round 6: the real data with every condition pinned

Round 5 showed a free k on either marker reopens the k/dk_geno slide, and
the real pheS m with X set by kanR matches pheS's own no-doubles fit. So
the real fit pins all four conditions at the real no-doubles MAP, round
3's setup (`held_priors.py` without `--hold`, `run_held.srun` with its
default pins), in `real_fit/held_all_real`. The pheS curves stay
provisional until `planning/deep-coverage.md` step 3: the pool's pheS
selection is about half the wt monoculture's, and the simulation cannot
say whether that is biology or a misspecification it lacks.

## Step 2: k and m's uncertainty in X

`planning/deep-coverage.md` step 2. A held fit gives X intervals given k
and m. To carry their uncertainty, take joint draws of every condition's k
and m from the stage-1 fit's arrowhead Laplace, refit the full library
pinned at each draw (round 3's setup, starting from round 3's MAP with
`--init_from`), and pool the draws' posteriors into one equal-weight
mixture. The pooled intervals should cover the deepest genotypes better
than round 3 (0.65 above 1e5 reads) if stage 1's own error is what they
miss. The Laplace's k/m spread is only as honest as stage 1: on seed 1
its pheS m was 11% off the truth, so if the spread comes out near 1%, the
draws cannot cover that and the second seed (below) says whether it was
luck.

`km_draws.py` stops if the stage-1 Laplace holds a k or m direction at
zero width (it would give those draws no spread). `score_x.py` scores
any run directory or pooled table by depth; `depth.py` makes its read
table.

### How to run step 2 (simulation, `sim_realistic_s2`)

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_km_draws.srun
```

`run_km_draws.out` prints each condition's k and m SD across the stage-1
posterior and ends with `wrote 10 draws to km_draws/draw_*/`. If it stops
on held k/m directions instead, send me the table it prints. Then submit
the draws, each starting from round 3's MAP:

```bash
cd km_draws && for d in draw_*; do (cd $d && ln -s ../../sim_realistic_s2/tfs_growth.csv ../../sim_realistic_s2/tfs_sim_* . && cp ../../sim_realistic_s2/library_config.yaml . && sbatch --export=ALL,INIT=../../held_s2/tfs_fit_model_params.npz /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun); done
```

When every `draw_*/run.out` ends with `>>> Done`, pool them (in
`km_draws/`):

```bash
sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_pool.srun
```

Then score `km_draws/pooled_aggregate.csv` against round 3 with
`score_x.py` and round 3's truth table.

## Second seed

Stage 1's pheS m came out 11% low on the one simulation so far.
`sim_realistic_s2` is seed 1 (the "s2" is the second calibration of the
simulator). A fresh seed with the same calibration, run through both
stages by `run_two_stage.srun`, says whether that was noise and whether the
two-stage fit's X coverage holds on new data. The stage-1 Laplace is part
of the job, so step 2 can run on it too.

### How to run the second seed

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/full-size-sim/make_sim_config.py --out_dir sim_realistic_seed2 --seed 2 --calibrate_from pilot_s1
```

Check that `sim_realistic_seed2/simulate_config.yaml` reads `seed: 2`.

```bash
cd sim_realistic_seed2 && sbatch --export=ALL,SEED=2 /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_two_stage.srun
```

The simulation alone takes about 85 minutes; the whole job ends with
`>>> Done` in `run_two_stage.out` and `stage2/summary/`. To also get this
seed's unheld full-library fit (the bias at 1.45 should reproduce), submit
`run_full_size.srun` in the same directory after it (it reuses
`tfs_growth.csv`):

```bash
sbatch --export=ALL,SEED=2 --dependency=afterok:<the two-stage job id> ../run_full_size.srun
```

## Commit

The commit that adds this study.

## Results

Round 1 above (inconclusive).

### Round 2 results (2026-10-10)

All three arms converged at every stage (about 15 minutes for
`no_doubles`, 1.7 and 2.3 hours for the others on one GPU). m_fit / m_true
on the X scale, `sim_realistic_s2`:

| genotypes in the fit | doubles | kanR+kan | pheS+4CP |
|---|---|---|---|
| `no_doubles` | 0 | 1.03 | 0.89 |
| `deep_doubles` | 80,915, all >= 1e3 reads | 1.23 | 1.16 |
| `shallow_doubles` | 80,915, all < 1e3 reads | 1.20 | 1.10 |
| all (the full-size fit) | 207,599 | 1.47 | 1.43 |

**The doubles cause the bias, in proportion to how many there are, whatever
their depth.** With only wt, spikes and singles the fit recovers m in kanR
(pheS 11% shallow, within about twice round 1's seed spread). Adding 81,000
doubles moves it to about 1.2 whether they are all deep or all shallow, and
all 208,000 move it to 1.47. k follows m along the ridge (kanR k 0.018,
0.022, 0.022 and 0.027 against a true 0.0149). The tube offsets look alike
in every arm (SD 0.31-0.35).

The genotypes' data are independent given the shared parameters, so each
double adds a small, consistent pull on m, and the pulls add. A pull that
does not depend on depth is not plain count noise. Two candidates:

- **The joint MAP over many per-genotype latents.** The MAP optimizes each
  genotype's X, K, n and dk_geno jointly with m, along the m·X ridge. The
  hierarchical priors on those latents add one term per genotype. If a
  steeper m with compressed X fits the population priors slightly better
  per genotype, the preference grows with the number of genotypes, while the
  information on the true m (wt, spikes, the tube totals) stays fixed. This
  is the classic Neyman-Scott bias of joint estimation.
- **A per-genotype misspecification of the doubles' model**, such as their
  shared bulk transformation (congression; the fit uses `single`), whose
  per-genotype effect would add the same way. Full-size Poisson counts
  still give 1.32, so count noise accounts for at most part of it.

**Practical route, whatever the mechanism:** estimate k and m (and the tube
offsets) from wt, spikes and singles alone, then hold them for the full
library. The first step is the same subset of the real data: does the real
kanR m move from -0.020 to the monoculture's -0.014 when the doubles are
dropped?

### How to run the real-data `no_doubles` arm

On the cluster, after pulling, next to the recipe run:

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py recipe --out_dir m_subsets_real --arms no_doubles --library_config ../processed/library_config.yaml
```

Check that it prints about 950 wt/spike/single genotypes, then:

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/{run_arm.sh,run_arm.srun} m_subsets_real/
```

```bash
cd m_subsets_real/no_doubles && sbatch ../run_arm.srun
```

It ends with `>>> Done` in `run.out` (about 15 minutes) and writes
`tfs_params_growth_k.csv` and `tfs_params_growth_m.csv`. There is no truth,
so compare by hand against the recipe and the monoculture (kanR+kan k
0.0150, m -0.0141; pheS+4CP k 0.0031, m 0.0144).

### Real-data `no_doubles` result (2026-10-10)

The recipe's growth table without the doubles (15 minutes on one GPU).
Stage 1 and the joint stage reached the 100,000-epoch cap still descending
about 1 nat per window in the joint stage, led by an ln_cfu0 hyper scale
(the hierarchical MAP's funnel, which has no finite optimum). The tube
offsets are centered and small (SD 0.23).

| condition | fit | k | m |
|---|---|---|---|
| kanR+kan | wt monoculture | 0.0150 | -0.0141 |
| | recipe, all genotypes | 0.0298 | -0.0204 |
| | recipe, monoculture priors (`anchor_real`) | 0.0249 | -0.0202 |
| | **no doubles** | **0.0160** | **-0.0141** |
| pheS+4CP | wt monoculture | 0.0031 | 0.0144 |
| | recipe, all genotypes | 0.0226 | 0.0097 |
| | recipe, monoculture priors (`anchor_real`) | 0.0172 | 0.0097 |
| | **no doubles** | **0.0117** | **0.0061** |

**In kanR+kan the real data behave as the simulation does.** Without the
doubles, k and m land on the wt monoculture with no prior pulling them
there (loose priors, SD 0.01), so the real kanR curves fitted with all
genotypes are compressed by about 1.45, as in the simulation.

**pheS+4CP does not.** Dropping the doubles lowers m, as in the simulation,
but from a starting point already below the monoculture, so it moves away
from it (0.0061 against 0.0144) and k stays high. Something in the real
pheS data that the simulation lacks sets pheS's scale: the selection-onset
transient (`planning/offset-mode-growth-transition.md`) or the known
mismatch between the 4CP wt in the screen and in monoculture. That is
`planning/deep-coverage.md` step 3.

### Round 3 result (2026-10-10)

`held_s2`: the full `sim_realistic_s2` library with k and m clamped at the
simulated `no_doubles` MAP (kanR+kan k 0.0182, m -0.0142; pheS+4CP k
0.0057, m 0.0126). Every stage converged; about 2 hours on one GPU. X of
the doubles against the truth, by total reads, unheld (the original
full-size fit) against held:

| reads | doubles | r | slope (truth on fit) | RMSE | 95% coverage | median 95% width |
|---|---|---|---|---|---|---|
| <= 1e2 | 37,707 | 0.26 / 0.28 | 0.05 / 0.05 | 2.16 / 2.25 | 0.97 / 0.98 | 4.08 / 5.74 |
| 1e2-1e3 | 89,022 | 0.62 / 0.64 | 0.49 / 0.38 | 0.45 / 0.57 | 0.82 / 0.90 | 0.85 / 1.29 |
| 1e3-1e4 | 64,362 | 0.93 / 0.93 | 1.19 / 0.82 | 0.20 / 0.19 | 0.56 / 0.89 | 0.35 / 0.53 |
| 1e4-1e5 | 15,283 | 0.99 / 0.99 | 1.36 / 0.95 | 0.17 / 0.07 | 0.29 / 0.90 | 0.14 / 0.22 |
| > 1e5 | 1,225 | 0.99 / 1.00 | 1.37 / 0.96 | 0.16 / 0.05 | 0.10 / 0.65 | 0.05 / 0.08 |

**Holding k and m removes the compression.** Above 1e4 reads the slope
goes from 1.36 to 0.95, RMSE falls about threefold and 95% coverage rises
from 0.29 to 0.90 (1e4-1e5) and from 0.10 to 0.65 (above 1e5). The
k/dk_geno slide is gone too: the median dk_geno error falls from -0.0057
to -0.0008, and the held directions shrink from the k/dk_geno slide
(eigenvalue -1.4e10) to a small dk_geno hyperparameter direction
(-2.9e3). Tube offsets are unchanged (SD 0.32).

What is left:

- **The deepest genotypes still undercover (0.65 above 1e5 reads)**,
  because the intervals leave out k and m's own error. The no-doubles m is
  3% steep in kanR and 11% shallow in pheS, and at widths of 0.08 that is
  as large as the interval.
- **Below 1e3 reads the held fit is a little worse** (RMSE 0.57 against
  0.45 at 1e2-1e3, slope 0.38): the shallow points scatter more and their
  intervals widen to cover it (0.90). That is
  `planning/deep-coverage.md` step 4 (hold the X population SDs).
- At 1e3-1e4 a slope below 1 (0.82) is the expected regression dilution
  of a noisy point, not a scale bias.

### Round 4 result (2026-10-10)

Both arms converged at every stage (4.9 and 5.7 hours). The configure
lines read `pins: none`.

| condition | truth (sim) | unheld | all pinned (round 3) | kanR held (round 4) |
|---|---|---|---|---|
| kanR+kan m | -0.0138 | -0.0203 | -0.0142 (held) | -0.0149 |
| pheS+4CP m (free in round 4) | 0.0141 | 0.0202 | 0.0126 (held) | 0.0147 |

| doubles' X (sim) | unheld | all pinned | kanR held |
|---|---|---|---|
| 1e4-1e5 reads: slope / RMSE / 95% coverage | 1.36 / 0.17 / 0.29 | 0.95 / 0.07 / 0.90 | 1.00 / 0.08 / 0.82 |
| > 1e5 reads: slope / RMSE / 95% coverage | 1.37 / 0.16 / 0.10 | 0.96 / 0.05 / 0.65 | 1.01 / 0.07 / 0.42 |

- **Holding kanR alone is enough.** With X fixed through kanR, the free
  pheS m settles at 1.04 of the truth, closer than when it was held at its
  own no-doubles value (0.89).
- **The tight prior leaked.** kanR m moved from the held -0.0142 to
  -0.0149, seven prior SDs and about 11% of the doubles' pull, not the
  0.5% estimated from the k/m anchor study. That costs deep coverage
  against the pinned run, and k and m came back as held Laplace directions
  (eigenvalues near -1e10).
- **Real data.** kanR m leaked shallow (-0.0133 against the held -0.0141,
  k 0.0162), so the real kanR X comes out about 1.5 times wider than in the
  recipe fit (0.0204 / 0.0133). The free pheS m settled at 0.0064 (k
  0.0099), as in the no-doubles fit: with X set by kanR, the real pool's
  pheS selection is about half the wt monoculture's, a property of the
  data (`planning/deep-coverage.md` step 3).

So round 5 pins kanR per condition instead.

### Round 5 result (2026-10-11)

Both arms converged at every stage (about 3.5 and 7 hours). The pins held
exactly: kanR's k and m equal the no-doubles values in both arms.

| sim, doubles above 1e4 reads | all pinned (round 3) | kanR prior (round 4) | kanR pinned (round 5) |
|---|---|---|---|
| 1e4-1e5: slope / RMSE / 95% coverage | 0.95 / 0.07 / 0.90 | 1.00 / 0.08 / 0.82 | 0.99 / 0.10 / 0.67 |
| > 1e5: slope / RMSE / 95% coverage | 0.96 / 0.05 / 0.65 | 1.01 / 0.07 / 0.42 | 0.99 / 0.09 / 0.20 |
| mean X error at 0 / 1 mM IPTG | +0.02 / -0.05 | +0.06 / 0.00 | +0.09 / +0.04 |
| pheS+4CP k (truth 0.0033) | 0.0057 (held) | 0.0005 | -0.0009 |
| pheS+4CP m (truth 0.0141) | 0.0126 (held) | 0.0147 | 0.0140 |

**Freeing pheS's k reopens the k/dk_geno slide.** The free pheS m is
right (1.00 of the truth), and X's scale is right (slope 0.99), but every
deep double's X shifts up by a near-constant amount. A uniform shift in X
is compensated by dk_geno and pheS's k together. kanR's pinned k no longer
closes the slide on its own, because dk_geno is shared and pheS's k can
absorb the difference. Only wt's gauge resists, weakly. The Laplace
holds that direction (pheS+4CP k, eigenvalue -1.4e10) at zero width, so
the intervals do not widen to cover the shift. Round 4 left pheS's k
free too; its MAP stopped at a smaller shift. Round 3, with every
condition pinned, has no slide.

**Real arm:** kanR k 0.0160, m -0.0141 (pinned); pheS+4CP k 0.0093, m
0.0068, against its own no-doubles fit's 0.0117 and 0.0061. pheS's k slid
the same way, and its held direction is again pheS+4CP k (eigenvalue
-4.7e9).

**What the data support for pheS.** In the real data, pheS's m with X
set by kanR (0.0064 in round 4, 0.0068 here) matches pheS's own
no-doubles fit (0.0061). So the half-strength pheS selection is the pool's
property, the same with or without the doubles, not an artifact of the
no-doubles fit. Pinning pheS at its no-doubles values (round 3's setup) is
what the data support, and it is the setup that closes the slide on the
simulation.

### Round 6 result (2026-10-11)

`held_all_real` converged at every stage. k and m sit at the real
no-doubles values in all four conditions; the only held Laplace direction
left is a small one on the dk_geno population spread (eigenvalue -3.6e3;
no k, m or slide direction). Tube offsets SD 0.21.

Against the recipe fit, by total reads (the recipe's `tfs_pred_theta.csv`
and per-genotype reads from its growth table):

| genotypes | reads | n | r with recipe | slope (pinned on recipe) | median 95% width, recipe / pinned |
|---|---|---|---|---|---|
| doubles | <= 1e2 | 43,713 | 0.977 | 1.48 | 1.61 / 2.44 |
| doubles | 1e2-1e3 | 95,717 | 0.990 | 1.47 | 0.63 / 0.94 |
| doubles | 1e3-1e4 | 60,542 | 0.997 | 1.47 | 0.31 / 0.46 |
| doubles | 1e4-1e5 | 15,634 | 0.999 | 1.48 | 0.18 / 0.27 |
| doubles | > 1e5 | 1,387 | 0.999 | 1.48 | 0.13 / 0.21 |
| singles | > 1e5 | 904 | 1.000 | 1.47 | 0.13 / 0.19 |

**The real X stretches by a uniform 1.47**, at every depth, singles and
doubles alike: between kanR's m ratio (0.0204 / 0.0141 = 1.45) and pheS's
(0.0097 / 0.0061 = 1.59), as a single X shared across markers should be,
and the factor the simulation predicted for the recipe's compression
(about 1.45). Ranking is unchanged (r 0.99-1.00 above 100 reads) and the
widths stretch with X. Deep doubles' median X runs from 0.53 at no IPTG to
0.13 at 1 mM, against 0.42 to 0.14 in the recipe.

The intervals still leave out k and m's own uncertainty
(`planning/deep-coverage.md` step 2), and the pheS-derived part of X waits
on step 3.

### Step 2: the stage-1 Laplace (2026-10-11)

`run_km_draws.srun` on `m_subsets/no_doubles` (seed 1). The arrowhead
Laplace held no k or m direction, and its k/m spread is far smaller than
stage 1's actual error:

| condition | stage-1 k (SD) | k truth | stage-1 m (SD) | m truth (X scale) |
|---|---|---|---|---|
| kanR+kan | 0.0182 (0.0008) | 0.0149 | -0.01422 (0.00008) | -0.0138 |
| pheS+4CP | 0.0056 (0.0011) | 0.0033 | 0.01261 (0.00007) | 0.0141 |

m is off by about 5 SDs in kanR and 21 in pheS; k by 4 and 2. So draws
from this Laplace vary m by about 0.6% and cannot cover a 3-11% error.
Either stage 1 is biased, or its error is noise much larger than its
Laplace (the simulation's per-tube rate noise, which the fit's level offsets do not model, is
a likely source). The second seed tells the two apart. If it is noise,
the k/m uncertainty should come from resampling stage 1 (refits over
tubes or replicates), not from its Laplace. The ten draw fits were
cancelled before they ran; `km_draws/` holds only the draw tables.

### Second seed: stage 1 (2026-10-11)

`sim_realistic_seed2/stage1/no_doubles` converged at every stage (about 10
minutes); the Laplace held no k or m direction (k SD 0.0003-0.0006, m SD
under 0.0001).

| | seed 1 | seed 2 |
|---|---|---|
| kanR+kan m / truth | 1.028 | 1.017 |
| pheS+4CP m / truth | 0.895 | 0.897 |
| kanR+kan k - truth | +0.0033 | -0.0024 |
| pheS+4CP k - truth | +0.0023 | -0.0007 |

**Stage 1's m error is a bias, not noise**: the same on both seeds, kanR
about 2% steep and pheS about 10% shallow. Wider intervals cannot fix a
bias, so step 2's draws would not help even with an honest spread. k's
error is mostly noise: it changes sign between seeds, and its seed-to-seed
spread (about 0.003) is several times its Laplace SD. Through the
k/dk_geno slide that noise moves X by a constant. Next: find the pheS bias
with stage-1-only fits (about 10 minutes each), starting with the Poisson
simulation (`sim_poisson_s2`) to test the realistic count noise.

### Stage 1 under Poisson counts (2026-10-11)

`m_subsets_poisson/no_doubles` (the `sim_poisson_s2` growth table, seed
1; the joint stage reached the epoch cap still creeping about 30 nats per
window, led by `growth_phi` drifting toward the Poisson limit): kanR+kan m
1.004 of the truth, pheS+4CP 0.874. The realistic count noise is not the
cause, and neither is founder or demographic noise (the Poisson simulation
has none). The OD600-derived tube totals are not either: against the
simulation's true totals they are within 2% in every condition and IPTG
level, with no trend.

**Suspect: congression's asymmetry between markers.** The homodimer rule
gives a co-transformed cell a θ that leans toward its higher-θ plasmid. In
pheS+4CP (m > 0) higher θ grows faster, so a low-X mutant sharing a cell
with a high-X partner grows like the partner and those cells take over:
low-X genotypes look higher, the response compresses, and m reads
shallow. In kanR+kan (m < 0) the shared cells grow slower and fall
behind, so the clean cells dominate and the bias stays small. The test:
the seed-1 simulation with `transformation_poisson_lambda` 0 (same seed,
so the same phenotypes), stage 1 only (`run_two_stage.srun`, `STAGES=1`).

### How to run the no-congression stage 1

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir sim_realistic_nocong && cp sim_realistic_s2/{simulate_config.yaml,design.csv,od600_calibration.yaml,library_config.yaml} sim_realistic_nocong/
```

```bash
sed -i 's/^transformation_poisson_lambda: .*/transformation_poisson_lambda: 0.0/' sim_realistic_nocong/simulate_config.yaml && grep -n "lambda\|^seed" sim_realistic_nocong/simulate_config.yaml
```

Check: `transformation_poisson_lambda: 0.0` and `seed: 1`.

```bash
cd sim_realistic_nocong && sbatch --export=ALL,SEED=1,STAGES=1 /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_two_stage.srun
```

About 85 minutes to simulate, then about 10 for stage 1. Read
`stage1/no_doubles/tfs_params_growth_m.csv` against the truth with
`score_arms.py`.

### No-congression stage 1 (2026-10-11)

`sim_realistic_nocong` (seed 1, `transformation_poisson_lambda` 0, else
as `sim_realistic_s2`); every stage converged; no k or m direction held.

| stage-1 m / truth | congression on | congression off |
|---|---|---|
| kanR+kan | 1.028 | 1.032 |
| pheS+4CP | 0.895 | 1.035 |

**Congression causes stage 1's pheS bias.** kanR does not move, as the
asymmetry predicts. Both markers keep a small steep bias (about 3%)
without congression.

Fitting congression (`transformation: mixture`) does not fit the
two-stage idea: a mixture needs each co-resident partner's θ, so the
partners, mostly doubles, have to be in the fit, which brings back their
pull on m; and the relative-X fit accepts only the `max` rule, not the
simulation's `homodimer`. A stage 1 of clones with little congression
avoids both: wt and the spikes. In the simulated pool (with the 271
repeated wt sequences), about 17% of a bulk genotype's cells are
congressed (`transformation_poisson_lambda` 0.357); wt is 77% spiked, so
about 4% of its cells are congressed, single spikes about 10%, double
spikes almost none. wt alone sets the X gauge and, through the tube
totals, its growth difference between 0 and 1 mM is m. `make_subsets.py
--arms spikes` keeps wt and the library config's spiked genotypes (about
six with growth data).

### How to run the spikes-only stage 1

Simulation, seed 1 (congression on), in the full-size working directory:

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py sim_realistic_s2 --arms spikes --out_dir m_subsets
```

```bash
cd m_subsets/spikes && sbatch ../run_arm.srun && cd ../..
```

Seed 2 (`run_arm.srun` and `run_arm.sh` are not in its `stage1/` yet):

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py sim_realistic_seed2 --arms spikes --out_dir sim_realistic_seed2/stage1
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/{run_arm.sh,run_arm.srun} sim_realistic_seed2/stage1/ && cd sim_realistic_seed2/stage1/spikes && sbatch ../run_arm.srun
```

Real data, next to the recipe:

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py recipe --arms spikes --out_dir m_subsets_real --library_config ../processed/library_config.yaml
```

```bash
cd m_subsets_real/spikes && sbatch ../run_arm.srun
```

Each `make_subsets.py` reads its 8 GB growth table (about 6 minutes) and
prints `spikes: N genotypes`; each fit takes about 10 minutes. On the
real data, whether pheS m rises toward the monoculture's 0.0144 says how
much of the real pheS gap is congression (the simulation's congression
alone takes about 14% off pheS m; the real gap is about 2x).

### Second seed: stage 2 (2026-10-11)

`sim_realistic_seed2/stage2` (every condition pinned at seed 2's stage 1)
converged at every stage (about 1.7 hours). The doubles' X by total reads
(`score_x.py`, reads from `depth.py` on seed 2's growth table):

| doubles | seed 1 (round 3) | seed 2 |
|---|---|---|
| 1e3-1e4: slope / RMSE / 95% coverage | 0.82 / 0.19 / 0.89 | 0.83 / 0.19 / 0.88 |
| 1e4-1e5: slope / RMSE / 95% coverage | 0.95 / 0.07 / 0.90 | 0.97 / 0.07 / 0.87 |
| > 1e5: slope / RMSE / 95% coverage | 0.96 / 0.05 / 0.65 | 0.98 / 0.06 / 0.51 |
| deep X error at 0 / 1 mM | +0.02 / -0.05 | +0.05 / +0.01 |

The two-stage fit's gain reproduces on a new seed: slope near 1 and RMSE
0.06-0.07 above 1e4 reads, against 1.36 and 0.16 for the one-stage fit.
What is left is an X offset of up to about 0.05 that changes between
seeds (stage 1's k noise, through the k/dk_geno slide) on top of pheS's
congression bias; the deepest genotypes' intervals (widths about 0.08)
do not cover it (0.65, 0.51).

### Spikes-only stage 1 (2026-10-11)

wt plus the spiked genotypes (`make_subsets.py --arms spikes`), about 10
minutes each. Simulation (m / truth):

| | seed 1 | seed 2 |
|---|---|---|
| all singles, kanR+kan | 1.028 | 1.017 |
| all singles, pheS+4CP | 0.895 | 0.897 |
| spikes only, kanR+kan | 1.040 | 0.969 |
| spikes only, pheS+4CP | 1.037 | 0.988 |

**In the simulation, spikes only removes the congression bias** (pheS as
with congression off), at the cost of noise: about ±4% between seeds
against about ±1% with all singles. Seed 1's joint stage reached the
epoch cap still descending about 8 nats per window (a count-noise
parameter moving), so its numbers are provisional.

**On the real data it fails.** All nine spikes are deep (wt 3.9e8 reads,
each spike 1.3-6.2e6), but the fit settled in another solution: kanR+kan
m -0.0051, the control kanR-kan m -0.0123 with that condition's tube
offsets spread to SD 0.64 (the others 0.14-0.23), and pheS+4CP m about 0
(stage 1 at the epoch cap, nearly flat). With nine genotypes the model
can trade slope between the selection and control conditions, and the
loose priors give the controls' m the same SD (0.01) as selection's. Next:
the same fit with the control slopes held near 0 (SD 0.001, `linear`'s
default for control conditions).

### How to run the real spikes-only fit with tight controls

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit/m_subsets_real
```

```bash
mkdir spikes_tightctl && cd spikes_tightctl && ln -s ../spikes/tfs_growth.csv . && cp ../spikes/library_config.yaml .
```

```bash
printf "condition_rep,k_loc,k_scale,m_loc,m_scale\nkanR+kan,0.015,0.01,0.0,0.01\nkanR-kan,0.015,0.01,0.0,0.001\npheS+4CP,0.015,0.01,0.0,0.01\npheS-4CP,0.015,0.01,0.0,0.001\n" > growth_priors_loose.csv && cat growth_priors_loose.csv
```

Check that the two control rows (`-kan`, `-4CP`) have m_scale 0.001.
`run_arm.sh` reads the file by that name.

```bash
sbatch ../run_arm.srun
```

### Real spikes-only with tight controls, and wt's own trajectories (2026-10-11)

`m_subsets_real/spikes_tightctl` (control m SD 0.001): the same solution
as before (kanR+kan m -0.0042, kanR-kan m -0.0093, nine prior SDs from 0,
pheS+4CP m about -0.001; kanR-kan tube offsets SD 0.54). With nine
genotypes there is no population of X values to pin X's scale, so the fit
slides along the m·X ridge to a small m. The spikes-only stage 1 is
dropped for the real data.

Reading wt's growth straight off the growth tables (slope of ln_cfu over
time per condition and IPTG level) is too noisy to use: each replicate
has 2-3 sequenced tubes over about 40 minutes per level, and tube-level
noise of about 0.3 ln units gives a slope error near 0.008 per minute, the
size of m. On the simulation it gives kanR m of +0.039 and -0.026 in the
two replicates against a truth of -0.014, and pooling replicates
confounds their starting levels with time. The real pool's wt happened
to come out near the monoculture (kanR m -0.016, pheS +0.013, pooled over
replicates), but that is not evidence. The real pheS gap stays open.

Options for congression in stage 1: correct the all-singles stage 1's
pheS m by the simulation's factor (about 1.15; assumes the simulation's
lambda and homodimer rule); a middle-sized stage 1 (the spikes plus a few
hundred of the deepest singles); or fit congression (`mixture`, which
needs the partners and the `max` rule).

### Option 2 and a test of whether congression is the doubles' pull

Option 2: `make_subsets.py --arms spikes_deep` keeps wt, the spikes and
the 300 deepest singles (`--n_deep_singles`), on both simulation seeds.

Option 3 is feasible: the relative fit configures and runs with
`--transformation_model mixture --congression_theta_rule max` (40 steps
on a round 1 table, about 1.8 s per step on a laptop). The lab's estimate
of lambda is 0.357 with a likelihood-ratio 95% interval of 0.156-0.674
(log-SD about 0.37; as the (mean, std) that `--transformation_lambda`
takes, about 0.38 0.15). The simulation's 0.357 sits mid-interval, but the
real lambda could be nearly twice that, and congression's pheS bias grows
with lambda.

Before building option 3 into stage 1: round 1's reduced library hinted
that congression also drives the doubles' pull (no_cong m 0.96-0.99
against base 1.03-1.14). If it does, one full-library fit with `mixture`
could fix both and the two stages would not be needed. The test is the
one-stage full-library fit on `sim_realistic_nocong`: m near 1.0 means
the pull is congression; m near 1.45 means it is separate.

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3/sim_realistic_nocong && sbatch --export=ALL,SEED=1 ../run_full_size.srun
```

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py sim_realistic_s2 --arms spikes_deep --out_dir m_subsets && (cd m_subsets/spikes_deep && sbatch ../run_arm.srun)
```

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/make_subsets.py sim_realistic_seed2 --arms spikes_deep --out_dir sim_realistic_seed2/stage1 && (cd sim_realistic_seed2/stage1/spikes_deep && sbatch ../run_arm.srun)
```

### Option 2 result (2026-10-11)

| stage-1 m / truth | all singles | spikes only | spikes + 300 deepest singles |
|---|---|---|---|
| seed 1, kanR+kan | 1.028 | 1.040 | 1.014 |
| seed 1, pheS+4CP | 0.895 | 1.037 | 0.889 |
| seed 2, kanR+kan | 1.017 | 0.969 | 1.024 |
| seed 2, pheS+4CP | 0.897 | 0.988 | 0.908 |

The deepest singles are as congressed as the rest; once they are in, the
pheS bias is back at the all-singles level (seed 1's joint stage at the
epoch cap, seed 2 converged). A stage 1 large enough to pin X's scale
carries congression, so congression has to be modeled (option 3). The
one-stage no-congression fit decides whether that goes into one full
fit or into stage 1.

### One-stage fit without congression (2026-10-11)

`sim_realistic_nocong`, the full library through `run_full_size.srun`;
every stage converged.

| full-library m / truth | congression on | congression off |
|---|---|---|
| kanR+kan | 1.47 | 1.37 |
| pheS+4CP | 1.43 | 1.51 |

**The doubles' pull is not congression.** Two separate problems: the
doubles steepen m about 1.45-fold (fixed by the two stages), and
congression flattens pheS's m about 0.9-fold in any stage 1 large enough to
pin X's scale. So congression is modeled inside the two-stage fit, in
stage 1.

## Option 3: stage 1 with the congression mixture

Stage 1 without doubles, with `--transformation_model mixture
--transformation_lambda 0.357 0.15 --congression_theta_rule max`
(`run_arm.sh` reads extra configure flags from the arm's
`configure_extra.txt`). Two approximations: the fit draws each genotype's
co-resident partners from the genotypes with growth data, so in stage 1
from wt, the spikes and the singles, while the real partners are mostly
doubles (in the simulation singles and doubles draw their phenotypes from
the same distribution); and the fit's `max` rule stands in for the
simulation's `homodimer`. The arm is configured with the simulation's own
composition (`simulate_config.yaml` as the library config: wt's 271
spiked copies, so wt is mostly clean, as in the simulated pool), so the test
is of modeling congression, not of knowing wt's purity. On real data, wt's
purity is unknown (the excess wt is being worked out at the bench).

### How to run option 3 (both seeds, reusing the no-doubles tables)

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3/m_subsets
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_arm.sh . && mkdir no_doubles_mix && cd no_doubles_mix && ln -s ../no_doubles/tfs_growth.csv ../no_doubles/tfs_sim_* . && cp ../no_doubles/{simulate_config.yaml,growth_priors_loose.csv} . && cp simulate_config.yaml library_config.yaml && echo "--transformation_model mixture --transformation_lambda 0.357 0.15 --congression_theta_rule max" > configure_extra.txt && sbatch ../run_arm.srun
```

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3/sim_realistic_seed2/stage1
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_arm.sh . && mkdir no_doubles_mix && cd no_doubles_mix && ln -s ../no_doubles/tfs_growth.csv ../no_doubles/tfs_sim_* . && cp ../no_doubles/{simulate_config.yaml,growth_priors_loose.csv} . && cp simulate_config.yaml library_config.yaml && echo "--transformation_model mixture --transformation_lambda 0.357 0.15 --congression_theta_rule max" > configure_extra.txt && sbatch ../run_arm.srun
```

The `cp run_arm.sh` replaces the copy already in each folder with the one
that reads `configure_extra.txt`. In each arm's `configure.log` the
command line shows the mixture flags.

### Option 3 result: stage 1 (2026-10-11)

| stage-1 m / truth | seed 1 | seed 2 |
|---|---|---|
| without mixture, kanR+kan | 1.028 | 1.017 |
| without mixture, pheS+4CP | 0.895 | 0.897 |
| with mixture, kanR+kan | 1.056 | 1.047 |
| with mixture, pheS+4CP | 1.037 | 1.023 |

**Modeling congression removes stage 1's pheS bias** on both seeds. Both
markers keep a small steep bias (2-6%). Seed 2 converged at every stage;
seed 1's first stage reached its cap, the later ones converged. The
fitted lambda is 0.27 and 0.25 against the simulation's 0.357, inside the
prior: the `max` rule lets the higher-θ plasmid set a shared cell's θ
completely, where `homodimer` only leans toward it, so fewer congressed
cells reproduce the same effect. On real data the fitted lambda is a
model parameter, not the measured one; keep it fitted, with the
measurement as its prior.

## Stage 2 after a mixture stage 1

k and m pinned at the mixture stage 1, on both seeds, with stage 2 fitted
without the mixture (`single`, as rounds 3 and 6) and with it (every
genotype's partners are in the full library; a few cell classes per
genotype, so heavier). `run_held.srun` reads `configure_extra.txt` like
`run_arm.sh`. All four use the simulation's composition as the library
config, as the stage 1 did.

### How to run

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

Seed 1 (`sim_realistic_s2`), stage 2 without and with the mixture:

```bash
for v in single mix; do mkdir stage2_${v}_seed1 && (cd stage2_${v}_seed1 && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets/no_doubles_mix --out growth_priors_held.csv && ln -s ../sim_realistic_s2/tfs_growth.csv ../sim_realistic_s2/tfs_sim_* . && cp ../sim_realistic_s2/simulate_config.yaml library_config.yaml && if [ $v = mix ]; then echo "--transformation_model mixture --transformation_lambda 0.357 0.15 --congression_theta_rule max" > configure_extra.txt; fi && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun); done
```

Seed 2:

```bash
for v in single mix; do mkdir stage2_${v}_seed2 && (cd stage2_${v}_seed2 && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../sim_realistic_seed2/stage1/no_doubles_mix --out growth_priors_held.csv && ln -s ../sim_realistic_seed2/tfs_growth.csv ../sim_realistic_seed2/tfs_sim_* . && cp ../sim_realistic_seed2/simulate_config.yaml library_config.yaml && if [ $v = mix ]; then echo "--transformation_model mixture --transformation_lambda 0.357 0.15 --congression_theta_rule max" > configure_extra.txt; fi && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun); done
```

Each `held_priors.py` prints the pinned table (seed 1 kanR+kan m about
-0.0146). Each `run.out` names the extra flags on its configure line
(`extra: none` for `single`). Score with `score_x.py` against
`held_s2` (seed 1) and `sim_realistic_seed2/stage2` (seed 2).

### Stage 2 after a mixture stage 1: first results (2026-10-12)

`stage2_single_seed1` (k and m pinned at the mixture stage 1, stage 2
`single`) converged; against round 3 (stage 1 without the mixture):

| doubles, seed 1 | round 3 | mixture stage 1, `single` stage 2 |
|---|---|---|
| 1e4-1e5: slope / RMSE / 95% coverage | 0.95 / 0.07 / 0.90 | 1.02 / 0.07 / 0.90 |
| > 1e5: slope / RMSE / 95% coverage | 0.96 / 0.05 / 0.65 | 1.03 / 0.05 / 0.76 |
| deep X error at 0 / 1 mM | +0.02 / -0.05 | +0.005 / -0.02 |

The better stage 1 roughly halves the X offset and raises the deepest
genotypes' coverage from 0.65 to 0.76.

`stage2_mix_seed1`'s MAP converged (162,000 steps, no memory trouble), but
the arrowhead Laplace refuses it: "genotype blocks are coupled (response
on unprobed genotypes 1.10e-02 of the probed ones)". The mixture couples
each genotype to its co-resident partners' X, which the block Laplace
assumes away; the coupling is about 1% of a genotype's own response. Seed
2's mixture run's MAP converged too (102,000 steps); its Laplace ran out
of GPU memory (a 26.6 GB allocation) before reaching the coupling check. `run_point.srun` finishes such a
fit at the MAP point (`--laplace point`, predict, summarize), enough to
compare X points with `single`. If the mixture stage 2 is worth keeping,
its intervals need a Laplace that tolerates (or drops) couplings this small.

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3/stage2_mix_seed1 && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_point.srun
```

Once `stage2_mix_seed2/run.err` shows the same error, the same in
`stage2_mix_seed2`.

`stage2_single_seed2` converged; against seed 2's first two-stage fit
(stage 1 without the mixture):

| doubles, seed 2 | stage 1 without mixture | mixture stage 1, `single` stage 2 |
|---|---|---|
| 1e4-1e5: slope / RMSE / 95% coverage | 0.97 / 0.07 / 0.87 | 1.04 / 0.07 / 0.88 |
| > 1e5: slope / RMSE / 95% coverage | 0.98 / 0.06 / 0.51 | 1.05 / 0.06 / 0.67 |
| deep X error at 0 / 1 mM | +0.05 / +0.01 | +0.03 / +0.03 |

On both seeds the mixture stage 1 raises the deepest genotypes' coverage
(0.65 to 0.76, 0.51 to 0.67). The slope now overshoots slightly
(1.03-1.05): the small steep bias the mixture stage 1 keeps in m (kanR
about 5%, pheS 2-4%) is now the main error in X's scale, plus a constant
offset near 0.03.

### Stage 2 with the mixture, at the MAP point (2026-10-12)

`run_point.srun` on both mixture stage 2 fits. Seed 1 finished; seed 2
ran out of GPU memory at a 4.1 GB allocation even for the one-draw
posterior (seed 1's identical job succeeded, probably on a larger card).
Seed 1, X of the doubles:

| doubles, seed 1 | `single` stage 2 | mixture stage 2 (MAP point) |
|---|---|---|
| 1e4-1e5: slope / RMSE / error at 0, 1 mM | 1.02 / 0.067 / +0.005, -0.020 | 1.02 / 0.070 / -0.002, -0.010 |
| > 1e5: slope / RMSE / error at 0, 1 mM | 1.03 / 0.050 / +0.013, -0.025 | 1.03 / 0.044 / +0.006, -0.013 |
| 1e2-1e3: r / RMSE | 0.65 / 0.51 | 0.50 / 0.86 |

For the deep genotypes the mixture stage 2 trims the offsets by about
0.01 and leaves slope and RMSE unchanged. At low depth it looks worse, but
the `single` column is a Laplace median and the mixture's a bare MAP
point, which scatters more there. Not worth a coupled Laplace.

**The recipe: a mixture stage 1, then a `single` stage 2.** Stage 1 needs
congression modeled to get pheS's m right; with k and m pinned, stage 2
gets X right without it.

## The real data: mixture stage 1 under two assumptions about wt

The mixture needs the pool's composition, and wt's purity (how much of wt
is clean spike rather than bulk) is not known for these data. Two
variants of the real no-doubles stage 1 with the mixture (max rule,
lambda prior 0.38 +- 0.15, the lab's estimate 0.357 with 95% CI
0.156-0.674 in log space):

| variant | library config | wt in the fit's pool |
|---|---|---|
| `cfg` | the real config as written (design mixture 100 / 100 / 1000 / 1) | 1.2% of the pool, 99% bulk (about 17% of its cells congressed) |
| `wtspiked` | the simulation's config (the realized mixture from the initial composition, plus wt repeated in the spiked origin) | 11% of the pool, 23% bulk (about 4% congressed) |

`wtspiked` is the composition the simulation was built to match (the
realized starting pool has wt at about 11%); its config has the same
genetics keys as the real one, so it serves as the library config. Both
stay on the cluster (lab-derived). Readout: kanR+kan m against the
monoculture's -0.0141 (the all-singles stage 1 without the mixture sat on
it; in the simulation the mixture moved kanR about 3% steeper), and pheS+4CP
m against the monoculture's 0.0144. kanR may not separate the variants
strongly, since its congression bias is small.

### How to run the two real variants

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit/m_subsets_real
```

```bash
cp /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_arm.sh . && for v in cfg wtspiked; do mkdir no_doubles_mix_${v} && (cd no_doubles_mix_${v} && ln -s ../no_doubles/tfs_growth.csv . && cp ../no_doubles/growth_priors_loose.csv . && if [ $v = cfg ]; then cp ../no_doubles/library_config.yaml .; else cp /gpfs/projects/harmslab/harms/studies/full-sized-sims-3/sim_realistic_s2/simulate_config.yaml library_config.yaml; fi && echo "--transformation_model mixture --transformation_lambda 0.38 0.15 --congression_theta_rule max" > configure_extra.txt && sbatch ../run_arm.srun); done
```

Check: `grep -c "" no_doubles_mix_wtspiked/library_config.yaml` is about
350 lines (the full simulate config) and `no_doubles_mix_cfg`'s is the
short real config; each `configure.log` shows the mixture flags.

### The two real variants (2026-10-12)

Both joint stages ended at the epoch cap creeping about 1 nat per window
(nuisance parameters moving, not k or m).

| real stage 1 | no mixture | mixture, `cfg` | mixture, `wtspiked` | wt monoculture |
|---|---|---|---|---|
| kanR+kan k / m | 0.0160 / -0.0141 | 0.0167 / -0.0147 | 0.0160 / -0.0139 | 0.0150 / -0.0141 |
| pheS+4CP k / m | 0.0117 / 0.0061 | 0.0120 / 0.0068 | 0.0122 / 0.0062 | 0.0031 / 0.0144 |
| fitted lambda | | 0.365 | 0.174 | measured 0.357 (0.16-0.67) |
| final exact loss | | 856,117 | 856,142 | |

- **Congression barely moves the real pheS m** (0.0061 to 0.0068 or
  0.0062; it moved the simulation's about 15%). The real pheS gap with the
  monoculture (about 2x) is mostly something else: step 3, and the pheS
  curves stay provisional.
- **The data lean toward wt mostly bulk (`cfg`):** about 25 nats better
  on the same data (about 2 of them the lambda prior), and its fitted
  lambda lands on the measurement while `wtspiked` drives it to the low
  end of the interval. Neither is decisive (the `max` rule fit lambda
  about 25% low in the simulation), but both point the same way.
- **kanR stays within about 4% of the monoculture** either way (the
  variants differ by about 6%).

Next: the real stage 2 (`single`, every condition pinned at the `cfg`
mixture stage 1), `held_mix_real`, to compare with round 6
(`held_all_real`):

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
mkdir held_mix_real && cd held_mix_real && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets_real/no_doubles_mix_cfg --out growth_priors_held.csv && ln -s ../recipe/tfs_growth.csv . && cp ../../processed/library_config.yaml . && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun
```

Check: kanR+kan k 0.016722, m -0.014744 in the printed table.

### The real stage 2 under the full recipe (2026-10-12)

`held_mix_real`: every condition pinned at the real mixture stage 1 (wt
mostly bulk), stage 2 `single`. Every stage converged; one small held
direction (dk_geno's population mean, eigenvalue -6.8e3), none on k, m or
the slide; tube offsets SD 0.20.

| real X, full recipe against | round 6 | the original recipe (one stage) |
|---|---|---|
| r, doubles above 100 reads | 0.999-1.000 | 0.991-0.999 |
| slope | 0.94 | 1.39 |
| deep doubles' median X, 0 to 1 mM | 0.53 to 0.13, now 0.53 to 0.15 | 0.42 to 0.14, now 0.53 to 0.15 |

Against round 6 the real X compresses about 6%, as the mixture stage 1's
slightly steeper m predicts (kanR -0.0147 against -0.0141, pheS 0.0068
against 0.0061); ranking and shapes are unchanged and the widths shrink
with X. Against the original recipe the real X stretches about 1.39 (1.47
in round 6). The kanR part of the scale sits within 4% of the monoculture;
the pheS part waits on step 3.

## The kanR-only check

Stage 2 on the kanR tubes alone, with kanR's k and m pinned at the same
mixture stage 1, against the both-marker fit. If the pheS gap's cause
depends on genotype, the pheS data bend the X curves and the two
disagree. The seed-1 simulation shows what agreement looks like when
pheS is fine. `resample_growth.py --library kanR` keeps one library's
tubes; `held_priors.py --only kanR` keeps kanR's pinned conditions.

Real (about 6 minutes to filter the growth table, then about 5 hours):

```bash
cd /gpfs/projects/harmslab/harms/studies/dev-data/real_fit
```

```bash
mkdir kanR_only_real && cd kanR_only_real && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/resample_growth.py ../recipe/tfs_growth.csv --library kanR --out tfs_growth.csv && cp ../../processed/library_config.yaml . && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets_real/no_doubles_mix_cfg --only kanR --out growth_priors_held.csv && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun
```

Simulation, seed 1:

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir kanR_only_seed1 && cd kanR_only_seed1 && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/resample_growth.py ../sim_realistic_s2/tfs_growth.csv --library kanR --out tfs_growth.csv && ln -s ../sim_realistic_s2/tfs_sim_* . && cp ../sim_realistic_s2/simulate_config.yaml library_config.yaml && python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/held_priors.py ../m_subsets/no_doubles_mix --only kanR --out growth_priors_held.csv && sbatch /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_held.srun
```

Each `resample_growth.py` prints the kanR tube count; each
`held_priors.py` prints two rows (kanR+kan, kanR-kan). Compare with
`held_mix_real` and `stage2_single_seed1`.

## Stage 1's k and m uncertainty: delete-d jackknife

`run_jackknife.srun` (an array of 30, at most 10 at once) drops a random
20% of the tubes from the stage-1 growth table, stratified by condition and
IPTG (`resample_growth.py --drop_frac`, seed = task number), and refits
the mixture stage 1 in `jk_NN/` with the same model (it copies the full
stage 1's `library_config.yaml`, `growth_priors_loose.csv` and
`configure_extra.txt`). Tubes cannot be resampled with replacement: a
duplicated tube lands on the original's tensor cell. `jackknife_summary.py`
scales the subsample spread by (n - d) / d and, on the simulation,
compares the full stage 1's error with the jackknife SE. Seed 1 first;
seed 2 the same way for a second check of the SE against a known error.

```bash
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-3
```

```bash
mkdir jk_seed1 && cd jk_seed1 && sbatch --export=ALL,SRC=$PWD/../m_subsets/no_doubles/tfs_growth.csv,CONF=$PWD/../m_subsets/no_doubles_mix /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/run_jackknife.srun
```

Each task writes `jk_NN.out` ending `>>> Done jk_NN` (about 20-30 minutes
each). Then, in `jk_seed1/`:

```bash
python /gpfs/home/harms/tfscreen/planning/studies/m-bias-factorial/jackknife_summary.py . --full ../m_subsets/no_doubles_mix --truth ../sim_realistic_s2
```

