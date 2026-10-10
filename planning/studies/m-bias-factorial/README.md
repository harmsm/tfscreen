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
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-v3
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
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-v3
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
cd /gpfs/projects/harmslab/harms/studies/full-sized-sims-v3
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
