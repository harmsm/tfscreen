# Full-size simulation of the dev-data design

**Question.** Does the documented chain (simulate, process counts,
configure, staged MAP, arrowhead Laplace, extract, summarize) run on a
simulation the size and shape of the real screen, with no hand step, and
how well does the growth-only relative fit recover the truth there?

**Decisions it feeds.**
- Pipeline plan step 6's last item and step 7
  (`planning/experiment-pipeline.md`): this is the full-size simulation
  that runs through the same chain as the real data, with only the data
  paths changed.
- The footing for the science prioritization (pipeline step 8) and for
  `planning/offset-mode-growth-transition.md`: every later model study
  starts from this design, on files a user can regenerate.

## Design

`make_sim_config.py` builds the simulate config from the real experiment;
its docstring says what is taken from the data and what is chosen. In
short:

- **Design:** the real tube table, 118 tubes in 3 replicates, with the real
  names and per-replicate selection times. `tfs-simulate` follows it
  exactly through the `design` key (added for this study), so the
  simulated tube table has the real layout.
- **Library:** the real genetics, `transform_sizes` and `library_mixture`
  (the nine-spike config), about 218,000 genotypes.
- **Growth:** linear, b and m per condition from the wt monoculture rates
  (the monokan rule); control conditions m = 0.
- **Population:** `cfu0` per library, calibrated so the simulated tube
  totals match the real ones on average (below); total reads equal to the
  real experiment's (5.1e9, 4.3e7 per tube); OD600 through the real
  calibration, totals estimated from it.
- **Congression:** lambda 0.357, measured.
- **Chosen:** noise (`realistic`, the default, or `poisson`), phenotypes
  (hill_geno prior draws, or `--phenotype_model` from `tfs-build-empirical`),
  the growth transition (`instant`; `memory` as a test arm), dk_geno's
  hyperparameters and `tube_noise_sigma`.

The fit is the real data's: growth-only `hill_relative`, level tube offsets
held at SD 0.17, log(n) population SD held at 0.5, loose k/m priors
(`growth_priors_loose.csv`), MAP (staged automatically), arrowhead Laplace.

## How to run

A full-size simulation is large: about 85 minutes, 10 GB resident and a
68 GB peak memory footprint on a laptop for one seed (2026-10-05), and
several GB of files. Run it on the cluster, in `planning/dev-data/`
(gitignored, not pulled), after `prep_dev_data.py` has written
`processed/`. Copy the study files there (the staged-map study's
`score_map.py` too):

```bash
cp <repo>/planning/studies/full-size-sim/{make_sim_config.py,run_pilot.srun,run_full_size.srun,growth_priors_loose.csv} <repo>/planning/studies/staged-map/score_map.py .
```

First a pilot at the first-guess `cfu0`, only to calibrate it (no raw
files, no growth table):

```bash
sbatch run_pilot.srun
```

It runs `make_sim_config.py --out_dir pilot_s1 --seed 1`, then
`tfs-simulate` with `--no_write_raw --no_write_growth` inside `pilot_s1`.

Then the calibrated config and the run (same seed, so the same phenotypes):

```bash
python make_sim_config.py --out_dir sim_realistic_s1 --seed 1 --calibrate_from pilot_s1
```

```bash
cd sim_realistic_s1 && sbatch ../run_full_size.srun
```

A Poisson-noise arm for comparison:

```bash
python make_sim_config.py --out_dir sim_poisson_s1 --seed 1 --noise poisson --calibrate_from pilot_s1
```

```bash
cd sim_poisson_s1 && sbatch ../run_full_size.srun
```

## Inputs

`planning/dev-data/processed/`: `sample_df.csv` (the tube table),
`library_config.yaml`, `od600_calibration.yaml`,
`wt_control_rate_summary.csv`. Nothing outside `planning/dev-data`.

## Commit

The commit that adds the simulator's `design` key. Every step writes its
provenance (`*_provenance.json`).

## Results

Local pilot (2026-10-05, the uncommitted working tree, seed 1, first-guess
`cfu0` 1.5e7 per tube): it ran in 85 minutes (10 GB resident, 68 GB peak
footprint) and followed the 118-tube design. The simulated tubes were
about half as dense as the real ones (median OD600 kanR 0.16 against 0.22,
pheS 0.13 against 0.26), and six pheS tubes read below the detection
threshold. Calibrating the level per library from it gave `cfu0` x1.74 for
kanR and x3.58 for pheS (mean log real/simulated total 0.55 and 1.28).

A per-library level leaves a structured residual (log real/simulated
total, SD 0.6-0.7 over tubes), which the calibration does not try to
remove, because it is the model question itself:

- **By IPTG.** kanR+kan: about +1 at low IPTG and -0.3 at 1 mM; pheS+4CP:
  +1 rising to +2.1 at high IPTG. The simulated growth difference across
  IPTG (wt monoculture rates on the theta scale) is stronger than the real
  library's, most of which is mutants.
- **Over time.** Within nearly every condition, controls included, real
  totals fall behind the simulation by about 0.01-0.02 per minute: the
  screen populations grow slower than the wt monoculture rates say.

Both bear on `planning/offset-mode-growth-transition.md` (the tube offsets
carried an IPTG pattern) and on roadmap step 6 (the population model), and
go to the science prioritization (pipeline plan step 8).

Cluster runs, first attempt (2026-10-06): both arms simulated and
processed, then stopped at configure. Seed 1 drew dk_geno -0.1 per minute
for the spike H74A/K84L. At that rate the spike falls about e^-12 behind by
the first sampled tube (30 minutes of pre-growth plus about 90 of
selection), so it has no reads, and configure refused a spike without
growth data. A spike can die in a simulation, so `run_full_size.srun` now
passes `--allow_missing_spikes`. It also skips the simulation and the
processing when their outputs exist.

Rerun from configure (2026-10-06): both arms ran through the whole chain
with no hand step (staged MAP converged at every stage; arrowhead Laplace
held 45 and 55 shared directions, all k/m/dk_geno shift combinations), and
the tube-offset diagnostic found no structure (no condition trends with
IPTG or time; offset SD 1.8 to 1.9 times the held 0.17). The recovery
numbers are not usable: **the simulated wt did not respond to IPTG**.
hill_geno's simulation sampler drew wt's phenotype like any genotype's,
and seed 1 gave it "never binds" (theta 0.0063 at 0 to 0.0060 at 1 mM).
The relative fit gauges X on wt's own change, so the mapped X truth was
about 3,600 times too wide (theta test RMSE about 2,700), and the fit
itself was forced to put a wt response where there was none.
`thermo_to_growth` now gives wt the hill_geno reference curve
(`_pin_wt_reference`; an explicit wt override still wins). Every earlier
hill_geno simulation had a perturbed wt, 12% of seeds a non-normal
category. Rerun: new directories, same pilot calibration (wt is one
spike, a negligible share of the tube totals):

```bash
python make_sim_config.py --out_dir sim_realistic_s1b --seed 1 --calibrate_from pilot_s1
```

```bash
python make_sim_config.py --out_dir sim_poisson_s1b --seed 1 --noise poisson --calibrate_from pilot_s1
```

Corrected arms (2026-10-06/07, commit 8399fb6b; `sim_realistic_s1b`,
`sim_poisson_s1b`): wt carries the reference curve (0.99 to 0.01, log K
-4.1, n 2). **The chain is clean**: simulate, process, configure, staged
MAP (every stage converged on the exact loss), arrowhead Laplace (9 and 11
held shared directions), extract, predict and summarize, with no hand
step. **The offset diagnostic is clean** (smallest BH q 0.63 realistic,
0.31 Poisson), as it should be with no selection-onset transient in the
simulator. The offsets' SD (0.31, 0.34) is about twice the held 0.17: the
simulator's tube noise is a per-tube growth-rate shift
(`tube_noise_sigma` 0.002 per minute, so ~0.4 ln units over a selection),
larger than the real fit's offsets (SD 0.20) and not a level.

Recovery, the baseline for pipeline plan step 8 (realistic / Poisson;
genotypes 124k / 186k with growth data):

| | realistic | Poisson |
|---|---|---|
| X: Pearson r, RMSE, mean error | 0.34, 0.94, +0.06 | 0.76, 0.37, -0.21 |
| X: 95% coverage, median width | 0.66, 0.39 | 0.29, 0.15 |
| log K: r, 95% coverage | 0.28, 0.80 | 0.45, 0.82 |
| log n: r, 95% coverage | 0.43, 0.94 | 0.66, 0.94 |
| dk_geno: r, bias, 95% coverage | 0.69, -0.007, 0.66 | 0.79, -0.011, 0.23 |
| growth k, bias per minute | +0.006 | +0.022 |
| growth m, kanR+kan / pheS+4CP (truth -0.0138 / +0.0141) | -0.0192 / +0.0187 | -0.0168 / +0.0164 |
| k and m 95% coverage | 0 | 0 |

X coverage by genotype reads (95%; realistic / Poisson): <=100 reads
1.00 / 0.93, 1e2-1e3 0.90 / 0.62, 1e3-1e4 0.81 / 0.24, 1e4-1e5 0.61 /
0.10, >1e5 0.20 / 0.04. Coverage falls with depth, and under Poisson
noise the X error is a near-constant -0.22 at every depth: a shared error,
not per-genotype noise. The shared parameters are off and carry no
interval. k slid up with dk_geno down (the k/dk_geno slide,
"Per-condition growth priors" in CLAUDE.md; the loose priors, k SD 0.01,
allow it), |m| came out 20-40% too large, and the arrowhead Laplace held
exactly those directions (k, m, dk_geno_hyper_shift) at the MAP, so k and
m have zero-width intervals and every deep genotype inherits their error.
The relative-fit study's run 4 (small library) covered X 0.97 / 0.94 with
the same route; at full size the shared error dominates. These go to step
8: anchoring k against dk_geno (tighter or measured k priors, base-growth
or wt monoculture data), the held Schur directions, and a simulator tube
noise that matches the real offsets.

**What limits the correlation is depth, not k and m.** The realistic arm's
overall X r of 0.34 is set by shallow genotypes. Reads below are summed
over all 118 tubes (every condition, IPTG, time and replicate, both
libraries); 1,000 reads is about 8 per tube. By genotype-concentration
point, realistic arm:

| total reads | r | RMSE | RMSE after the best affine map | SD of truth |
|---|---|---|---|---|
| <=100 | 0.11 | 2.64 | 0.43 | 0.43 |
| 1e2-1e3 | 0.30 | 1.26 | 0.41 | 0.43 |
| 1e3-1e4 | 0.73 | 0.33 | 0.30 | 0.43 |
| 1e4-1e5 | 0.96 | 0.14 | 0.13 | 0.43 |
| >1e5 | 0.99 | 0.11 | 0.06 | 0.43 |

Deep genotypes rank well; their error is a shared rescaling (slope 1.2,
|m| too large), which the k/m work fixes and r does not see. Below 1e3
reads the realistic noise leaves no information on X (best affine error =
the truth's spread), and the MAP points are wild (RMSE 1.3-2.6) because
the X population SDs are not held (only log n is; `hill_relative`'s
`theta_{X_low,X_delta,log_hill_K}_hyper_scale_fixed` would shrink them).
Their intervals cover (0.996), so they are honest but uninformative.
Above 1e3 reads the realistic arm's r is 0.87 (Spearman 0.82). Report
recovery by depth, not one all-genotype r.

**The simulated depth distribution does not match the real screen.**
Genotypes with growth data by total reads (doubles / singles):

| total reads | realistic sim | Poisson sim | real (recipe) |
|---|---|---|---|
| 1-100 | 10,948 / 25 | 17,870 / 17 | 43,713 / 0 |
| 1e2-1e3 | 18,616 / 36 | 30,618 / 33 | 95,717 / 0 |
| 1e3-1e4 | 30,460 / 90 | 48,356 / 82 | 60,542 / 7 |
| 1e4-1e5 | 39,771 / 159 | 59,435 / 145 | 15,634 / 40 |
| >1e5 | 23,630 / 608 | 28,474 / 644 | 1,387 / 903 |
| all (incl. wt, one triple) | 124,345 | 185,676 | 217,945 |

Median total reads per double: 11,136 / 8,348 / 480 (94 / 71 / 4 per
tube); per single 441,080 / 590,799 / 867,829. The total reads match, but
the simulation concentrates them: many doubles deep, many below the
10-read minimum and dropped (124k of 218k survive under realistic noise),
where the real reads spread thinly over nearly every double (64% of real
doubles, 139,430, have fewer than 1,000 reads). The likely cause is the
simulated pool's unevenness (`lib_assembly_skew_sigma` 1.25 and the
design's `transform_sizes` bottleneck). So:

- The simulation is not yet a stand-in for this screen. Calibrate its
  library composition to the real depth distribution (this table is the
  target) before quoting its recovery; every other result depends on it.
- Most real doubles sit where the realistic arm had no information on X.
  If the real counts are as overdispersed as that arm, those curves are
  not usable one genotype at a time (the noise model is calibrated to the
  real overdispersion, planning/studies/noise-anatomy/, so this is a
  warning, not a measurement). Pooling is the lever there: held X
  population SDs, and mutation-level structure (`hill_mut`) that shares
  information across doubles with the same single.
