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
cp <repo>/planning/studies/full-size-sim/{make_sim_config.py,run_full_size.srun,growth_priors_loose.csv} <repo>/planning/studies/staged-map/score_map.py .
```

First a pilot at the first-guess `cfu0`, only to calibrate it (no raw
files, no growth table):

```bash
python make_sim_config.py --out_dir pilot_s1 --seed 1
```

```bash
cd pilot_s1 && tfs-simulate simulate_config.yaml --out_prefix tfs_sim --seed 1 --no_write_raw --no_write_growth
```

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

Pending.
