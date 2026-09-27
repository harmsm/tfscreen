---
title: Roadmap for analysis improvements alongside the congression plan
status: active
filed: 2026-09-26
area: tfmodel
revisit_when: >-
  Active now, on a branch parallel to main. Update the step list as steps
  finish, and re-check the constraints each time a congression step lands on
  main.
related:
  - planning/congression-physics-plan.md
  - planning/growth-only-relative-fit.md
  - planning/count-likelihood.md
  - planning/per-sample-level-offset.md
  - planning/empirical-mixture-refit.md
  - planning/estimate-spike-fraction-from-growth-curvature.md
  - planning/estimate-dk-alpha-by-varying-lambda.md
  - planning/plasmid-segregation-low-copy.md
  - planning/combination-specific-expression-effects.md
  - planning/absolute-abundance-measurements.md
  - src/tfscreen/process_raw/counts_to_lncfu.py
  - src/tfscreen/tfmodel/generative/components/sample_offset/normal.py
---

## Why

The congression plan (`planning/congression-physics-plan.md`) is close to
done on `main` (step 4 is implemented and its cross-rule grid is set up; step
5, the dk rule, is left). Its remaining work is mostly waiting for
simulations. This plan orders the other analysis changes, to be built in
parallel on a second branch, so that they fit together and fit with what
congression still changes.

The user's priorities (2026-09-26):

1. **Growth-only, wt-relative fit**, then measure the growth-binding map
   empirically (`planning/growth-only-relative-fit.md`).
2. **Low-count genotypes handled rigorously** instead of by pseudocount
   (`planning/count-likelihood.md`, with `planning/per-sample-level-offset.md`
   as its first half).

A third item joined on 2026-09-26: **model the total population in each
tube**, from OD600 through a lab-specific calibration, instead of ingesting a
per-tube CFU estimate. Once the fit works from reads, the total is the only
source of the absolute scale, so it has to be modeled at the same level.

## Reads and totals: where each number comes from

The experiment (`docs/source/process-raw.rst`, "One tube per time-point"):
after the `presplit` sample, a replicate's culture is split into one tube per
condition x IPTG concentration x timepoint; a timepoint pulls a whole row of
tubes. Each pulled tube gets an OD600 reading, is spun down and frozen, and
is later extracted and sequenced. OD600 is read for three bioreplicates;
two are sequenced.

For tube s, with true total population `N_s` and genotype populations
`n_{g,s}`:

- **Reads measure frequencies.** `reads_{g,s} ~ depth_s * n_{g,s} / N_s`,
  plus a small per-tube composition offset `u_s` (PCR and genotype-calling
  efficiency: the `__unknown__` fraction differs between tubes; the
  denominator review measured it at about 0.026 ln units).
- **OD600 measures the total.** `N_s = cal(OD_s) * volume`, where `cal` is
  the lab's OD-to-CFU calibration.
- **Tube growth noise cancels from frequencies.** A tube that grew a little
  faster or slower than its neighbors (a shared rate shift `eps_s`) scales
  every `n_{g,s}` and `N_s` by the same `exp(eps_s)`, so `n_{g,s} / N_s` does
  not see it. It shows up only in the OD600 value.

So the genotype model can predict smooth `n_{g,s}` (no tube term), the
population is a smooth curve `P_s = ln N_s` fitted across tubes, and
`ln_cfu_{g,s} = ln(freq_{g,s}) + P_s` is free of tube noise. This is why the
earlier practice (every tube assigned the pooled CFU of its condition and
timepoint across the three bioreplicates) worked so much better than per-tube
CFU: it removed the tube noise along with the OD noise. The replacement is a
smooth population curve per replicate x condition x concentration, with a
shared shape and per-replicate levels, observed through every OD600 reading,
including the bioreplicate that was not sequenced.

The curve and the genotype model describe the same cells:
`P_s = logsumexp_g ln n_{g,s}`. That coupling is not enforced directly (it
would need every genotype in every step, which breaks mini-batching). It is
enforced by the data: fitting every genotype's frequency makes
`sum_g exp(ln n_{g,s} - P_s) = 1` up to the `__unknown__` share.

What stays per tube and genotype is only what does not cancel: counting
noise, PCR jackpotting, and genotype-by-tube differences (a tube whose
selection was slightly stronger treats genotypes differently).

**Where the total is known and where it is not** (user, 2026-09-26):

- **Start: known.** The culture grows to a presplit OD600 and is diluted by
  the same factor into every tube, so every tube of a replicate starts at
  `cal(OD_presplit) * volume / dilution`, one value shared by all of them.
  Units: `ln_cfu = ln(cfu/mL * 5 mL)`, the calibration giving cfu/mL.
- **Visible tubes: measured.** OD600 on every pulled tube, same reader, plate
  and 200 uL as the calibration.
- **Blind tubes: not measured.** Under the strongest selection (particularly
  kan) the population barely grows and OD600 stays below the detection
  threshold at the timepoints we have; the next screen wants earlier
  timepoints, also below it. Censoring gives only an upper bound. A blind
  tube is constrained by its curve's known start and any later visible
  points, and through the reads by other columns of the same condition
  (`k_c`, `m_c` shared; each genotype's X smooth across columns). When every
  column of a condition is blind at a timepoint, `k_c` trades freely with
  the curves. Measurements that close this are filed separately:
  `planning/absolute-abundance-measurements.md` (a few for the current
  data, notably wt monoculture growth rates at the gauge concentrations; a
  cell spike-in counting standard for the next screen).

## Disposition of every idea in `planning/`

| Idea | Disposition | Where |
|---|---|---|
| `growth-only-relative-fit` | **Track G**, priority 1 | steps 0a, 1, 5, 8, 9 |
| `count-likelihood` | **Track N**, priority 2 | steps 0b, 2, 4, 7 |
| `per-sample-level-offset` | Track N; becomes the population model plus the composition offset `u_s` | step 6 |
| OD-to-CFU at the experimental level (new, 2026-09-26) | Track N | steps 0c, 2, 3, 4, 6 |
| `absolute-abundance-measurements` (new, 2026-09-26) | Bench work, user's choice; the model side is in steps 4, 6, 7 | now (1-3), next screen (4-5) |
| censoring floor rows (congression 3.5 open item) | **Superseded by step 7**; not built (D1) | - |
| lowering the default `binding_weight` (congression open item) | Stays on main; this plan asks it to default to 1 (C7) | main |
| setting lambda from the measurement (congression open item) | Stays on main; independent of this plan | main |
| `empirical-mixture-refit` | Deferred until main's step 5 and this plan's step 5 | after step 9 |
| `estimate-spike-fraction-from-growth-curvature` | Park: future runs tag spikes by codon | - |
| `estimate-dk-alpha-by-varying-lambda` | Experiment; needs main's step 5 | - |
| `plasmid-segregation-low-copy` | Independent; simulator-only; pick up when the low-copy design is planned | any time |
| `combination-specific-expression-effects` | Park: needs bench data | - |

## Compatibility constraints

These hold across all steps. Each one names the collision it prevents.

**C1. Congression rule vs relative X.** Step 4 on main made `homodimer` the
default theta rule. It takes `logit(theta)`, so it needs absolute theta in
(0, 1). The relative growth variable X is real-valued and defined only up to
an affine map, so the partition-function rules do not apply to it (the idea
file's invariance argument holds only for `max`: `max f(X) = f(max X)` for
increasing f). Rule: a relative-X theta component runs with `single` or with
`mixture` under `congression_theta_rule: max`, and the orchestrator refuses
`homodimer`/`heterodimer` with it (the same pattern as the refusal of a
learned activity). The dk rules (dilution now, soft-min after main's step 5)
act on dk in rate units and are unaffected. X must be oriented to increase
with repression for `max` to mean "tightest binder wins". The `max` branch of
`mixture.cell_classes` must not clip X to [THETA_EPS, 1 - THETA_EPS]; check
this in step 5. Once step 9 learns the map f, the partition-function rules
act on `f^-1(X)` and are available again.

**C2. Observers become selectable.** Track N replaces the growth and presplit
likelihoods. Make the observation model an orchestrator setting written to
the config and carried as static data (`growth_likelihood: lncfu | counts`),
the same pattern as `congression_theta_rule`. Do not use a YAML `components:`
key: observers are not registry components, and every `components:` key
becomes an orchestrator kwarg. `tfs-prefit-calibration`, `tfs-sample-prior`
and prediction then follow the setting with no change of their own.

**C3. Tube noise lives in the OD600 likelihood, not on genotypes.** A tube is
a timepoint (one sequenced sample, independent of the other timepoints). Its
shared growth deviation cancels from frequencies (see "Reads and totals"), so
no per-tube latent is added to every genotype for it. It appears as scatter
of OD600 about the population curve, with variance `(sigma_tube * t)^2` plus
the OD reading noise, over `t_pre + t_sel` (pre-growth and selection both
happen in the tube, D7). The only per-tube latent on genotypes is the
composition offset `u_s`, with a time-independent prior. Today's
`sample_offset/normal` (a per-tube rate offset on every genotype) becomes
that `u_s` (D2). The simulator's tube noise is already per timepoint (its
`condition_selector` includes `t_sel`).

**C4. The absolute scale, and with it `k_c` and `m_c`, comes from OD600.**
From reads alone, any per-tube constant can be added to every genotype, which
absorbs `k_condition * t` entirely: reads fix relative growth (dk_geno and
`m * theta` differences), not absolute rates. The population curves fix the
absolute rates. In Track G this matters twice: `k_c` is wt's absolute growth
at 1 mM IPTG, and `m_c` is wt's growth difference between 0 and 1 mM, which
are different tubes (different grid columns), so it rests on the columns'
population curves. Genotypes whose X does not depend on IPTG (broken
repressors) are internal standards across columns and should be reported as a
check. Steps 5 and 6 must report `k_c` and `m_c` recovery, not only theta
coverage.

**C5. Everything stays mini-batch safe.** The population curve parameters
and `u_s` are global sites (per replicate x condition x concentration, per
tube), like the condition parameters. The count likelihood is per genotype
and scaled by `scale_vector`. Index hopping into tube s comes from the same
genotype in the lane's other tubes, which are in the same batch row.
Sequencing errors that turn one genotype into another would couple
genotypes; leave them in `__unknown__`. The OD600 likelihood does not involve
genotypes and is never scaled. The new theta component follows the registry
rules (library-sized plates, `batch_idx` slicing, `return_population`,
`{site}_loc(s)`/`{site}_scale(s)`), and `test_batch_safety.py`,
`test_component_population.py` and `test_initialization.py` must pass
registry-wide.

**C6. Process the raw data once.** Before any Track N model work, the
processed files carry: counts, per-tube depth (including `__unknown__`), and
a per-tube OD600 table covering every bioreplicate, sequenced or not, with
the reader's detection flag. The real data is reprocessed once (step 2) and
every later step reads the same files. `ln_cfu`/`ln_cfu_var` stay, for plots
and for the `lncfu` likelihood.

**C7. Binding at weight 1 when it returns.** Track G removes binding from the
fit; step 9 brings it back as an explicit measurement of the map at weight 1
with realistic SDs. The binding-weight change on main should therefore make
1 the default, not a smaller ratio.

**C8. Downstream tools on the X scale.** Logit epistasis, `in_regime` and
anything that assumes theta in (0, 1) do not apply to X. `tfs-predict-theta`
must label its output as X for a relative fit (a column or metadata flag, so
nothing downstream mistakes one for the other); `tfs-extract-epistasis`
allows only `--scale add` on it until f is known. `tfs-summarize-calibration`
needs the simulated truth on the X scale: with growth linear in theta and
the gauge of D4 (`X_wt(c_lo) = 1`, `X_wt(c_hi) = 0`), true
`X_g(c) = (theta_g(c) - theta_wt(c_hi)) / (theta_wt(c_lo) - theta_wt(c_hi))`.

**C9. Shared files, small diffs.** Main will still touch
`transformation/mixture.py`, `simulate/cell_rules.py`,
`simulate/selection_experiment.py` (cell rules), `model_orchestrator.py` and
`configure_model_cli.py` (a `congression_dk_rule` setting). This branch stays
out of `mixture.py` and `cell_rules.py` except for C1's check, keeps its
orchestrator and CLI changes additive, and syncs with main after every
congression step lands.

**C10. The method is generic; the calibration is the lab's.** An OD-to-CFU
calibration depends on the plate reader, plate, volume and strain, so no
lab's calibration data or constants go into this repository. The repository
holds the method: a CLI that fits a calibration from a documented input
format and writes a calibration file, the functions that read and apply it,
and an example built from synthetic data. The existing lab calibration
(`tfscreen-notebooks/od600-to-cfu/`, outside the repo) is the first user and
the test that the port reproduces it. A lab that measures totals another
way (plating, flow) supplies per-tube totals with their own error model;
that is a scientific alternative, not a compatibility path (D12).

**C11. The calibration's own uncertainty is shared by every tube.** The
current `od600_to_cfu` adds the calibration-curve error (`J C J^T`,
approximated by a quadratic in OD) to each tube's CFU variance as if it were
independent per tube. It is one error, shared by every tube in every
experiment calibrated with that curve: it cannot average out across tubes,
and it mostly does not matter for growth rates (a multiplicative error
cancels from slopes; only curvature error does not). The calibration file
therefore keeps the full parameter covariance, and the fit treats the
calibration parameters as global quantities (D10), never as per-tube noise.

**C12. Count noise scales with the mean, and rare genotypes start from few
cells.** Study 0b found per-tube variance 5-18 times Poisson at 100-3,000
reads, the signature of a bottleneck upstream of the reads, not a
constant-CV floor. The count likelihood's dispersion must allow
`var = phi * mu` (NB1 / quasi-Poisson), with at most a small NB2 floor. A
typical double has a few founder cells per tube at the split, drawn anew for
every tube, so its starting abundance differs between tubes; the shared
`ln_cfu0` is a mean, and founder noise is part of the per-tube dispersion.
The simulator must produce both (step 4), or its calibration grids
understate real noise.

## Steps

Keep this list current. Two tracks, G (growth-only) and N (noise, counts and
totals), interleaved in this order. G comes first by priority; step 8 (the
map) waits for Track N because a real map can look like "no map" when the
posteriors are overconfident (bulk theta 95% coverage was 0.66 in the
congression calibration grid, 0.78-0.87 after convergence fixes).
Breaking changes are allowed throughout (D12); each modeling step ends with
a simulation checked by `tfs-summarize-calibration`.

**Implementation order (D14, 2026-09-26).** While the refined data are
gathered, build what does not hang on it: 1 -> 4 -> 7 (with supplied
totals) -> 5 -> 3, then 2's OD600 part, 6, 7b, 8, 9 as data arrive. Step 7
comes before step 6: until the population model exists, each tube's total
is the supplied `sample_ln_cfu` (as today) and the tube offset is
`ln depth_s - sample_ln_cfu_s + u_s`; step 6 later replaces the supplied
totals. The count and depth columns of step 2 come with step 7. The
congression plan's step 6 (re-validate on the count likelihood) waits on
step 7.

0. **Studies on real data** (no package code; run in parallel; all can start
   now).
   Data: the 2026-07-23 snapshot, extracted by
   `planning/studies/step0-data/` (113 sequenced tubes, 215,402 genotypes).
   **The first results are provisional** (user, 2026-09-26): that
   snapshot's FASTQ processing assigned every unknown read to wt and used a
   very stringent Q cutoff, and its binding data were five genotypes at the
   original protein concentration. A new processing batch is running, and
   new binding data exist (10 genotypes, two lower repressor
   concentrations whose curves match the in vivo concentration range).
   Re-run 0a and 0b on those before acting on them.
   - [x] **0a. Growth-binding map, model-free** (G's step 0). Per genotype with
     binding data: slope of `ln_cfu` on `t_sel` per replicate x condition x
     concentration; subtract wt; plot against binding theta (Hill fits if
     concentrations differ), spiked and bulk separately. Decides the base
     case (one collapsing curve: linear or monotone; genotype-specific scatter:
     no map). Also answers the idea file's open question on whether growth
     and binding concentrations coincide. Study:
     `planning/studies/growth-binding-map/`.
     Result (2026-09-26): concentrations coincide (eight IPTG points). Only
     five genotypes (all spiked, all near wt in vitro) have binding data.
     wt-relative, total-free slopes have no leverage. Absolute slopes (with
     the smoothed totals) reject one linear map at s = 1 in kan (reduced
     chi2 6.7; genotype-specific slopes p = 2e-5): growth rises between
     0.001 and 0.01 mM IPTG where in vitro theta is still 1. An in vivo
     concentration scale s = 30-100 brings reduced chi2 to about 3 (kan m
     about -0.025 per min). 4CP growth does not track binding. **The base
     case is not supported**; the map needs s (step 8), and the direct test
     is monoculture growth of these genotypes across the titration
     (`planning/absolute-abundance-measurements.md`).
   - [x] **0b. Read noise anatomy.** From the real counts: fraction of rows
     below 20 and below 5 reads, by genotype class and condition; the
     per-tube composition offset `u_s` (shared residual of frequencies of
     well-measured genotypes, wt and spikes, within a tube); a look for
     block structure in residual level or dispersion along the likely
     processing order (the extraction batches were not recorded, D8);
     replicate
     spread of high-count spike ratios against the binomial expectation,
     which sizes the count dispersion. Sets priors for steps 6 and 7 and the
     simulator parameters of step 4. Study: `planning/studies/noise-anatomy/`.
     Result (2026-09-26): doubles (97% of rows) have a median of 3-5 reads
     per tube (23-33% zero, 72-85% under 20); singles about 5,300; spikes
     29,000-56,000. Composition offset `u_s` SD 0.018. Per-tube count
     variance is 5-18 times Poisson at 100-3,000 reads (pooled: variance
     = 0.0055 + 8.4 x Poisson): a bottleneck upstream of the reads
     (template molecules and/or founder cells), so the dispersion scales
     with the mean (C12). Today's `ln_cfu_var` understates the noise 5-10
     fold. No extraction-batch analysis was possible from the recorded
     order. Low counts are mostly low abundance, not crashes: about 88% of
     double rows under 5 reads in selective tubes belong to genotypes
     already under 5 reads without the drug; only 0.1% of doubles with >= 20
     reads without the drug fall below 5 in half their selective tubes.
   - [ ] **0c. OD600 anatomy.** (First pass done; waiting on data.) All OD600 readings for the three
     bioreplicates, through the lab calibration: the shape of `ln N(t)` per
     condition x concentration (pre-growth and selection, lag, any
     saturation), how much of the between-bioreplicate difference is a level
     (shared split density) and how much a slope; scatter about a smooth
     curve against the 2% reading noise (the excess is tube growth noise);
     readings below the detection threshold (0.096) or above the calibrated
     range (0.59), listed by condition, concentration and timepoint (which
     tubes are blind decides which bench experiments are worth running);
     whether kan and 4CP tubes differ in ways that suggest the
     calibration does not transfer to stressed cells (D11). Decides the
     curve form (D9). Study: `planning/studies/od600-population/`.
     First pass (2026-09-26, the two sequenced replicates only): replicate
     2 reads +0.24 ln units above replicate 1 (a level; per-condition
     levels add nothing), but replicate-specific slopes are also
     significant (p = 7e-5). Tube scatter 0.17 ln units against 0.03
     reading noise, not growing with time. No tube below threshold (lowest
     1.5x, kan+ 0 mM, which barely grows). Per-condition population slopes
     have SE 0.003-0.007 per min, comparable to `m`: these tubes alone do
     not pin `k_c`. Still needed: the third bioreplicate's OD600, the
     repeated OD600 runs behind the smoothed totals, and the presplit
     OD600 and dilution.
1. [x] **Binding-optional plumbing** (G).
   The list in the idea file: `binding_df` becomes `--binding_df` in
   `tfs-configure-model` (at least one of growth/binding required; no phantom
   `data.binding`); the orchestrator gates binding tensors, `BindingData`, the
   auto `binding_weight`, `theta_binding_noise` and `observe_binding` on
   `binding_df is None`; `model.py` gates the binding path statically;
   `analysis/prediction.py` and `summarize_fit_cli.py` cope (default
   trajectory subset: wt, spikes, and a fixed-seed sample); fail fast on
   `binding_weight` or non-`zero` binding noise without binding;
   `tfs-prefit-calibration` refuses a growth-only config with a clear
   message. Update positional callers (`examples/simulate-and-analyze/run.sh`,
   the grid templates) and the tests named in the idea file; a growth-only
   smoke test on `growth-smoke.csv` + `library-smoke.yaml`. Check that a
   joint fit is unchanged (fixed-seed MAP loss against main): not for
   compatibility, but to show the plumbing changed nothing it should not.
   Done 2026-09-26. Joint model log density at a fixed seed identical to
   before (`single` and `mixture`). Also fixed: absent prior groups
   (`priors.binding` here, `priors.growth` for binding-only) reloaded from
   the priors CSV as NaN; they are now left out and stay `None`. Tests:
   `tests/tfscreen/tfmodel/test_growth_only.py`, a growth-only smoke test
   in `tests/smoke-tests/test_configure_run_smoke.py`.
2. **Counts and OD600 through processing** (N; breaking).
   - `counts_to_lncfu` writes `counts` and per-tube depth (with
     `__unknown__`) next to today's columns. The same for
     `tfs-process-presplit`.
   - A per-tube OD600 input (all bioreplicates, sequenced or not; a
     `sequenced` flag or the absence of reads marks the third) and the
     calibration file of step 3. Per-tube CFU columns are no longer an input;
     totals come from OD600 (or another lab's total measurement, C10).
   - An interim smoothed total: `tfs-process-counts` can compute
     `sample_ln_cfu` from a population curve fitted to all bioreplicates' OD
     (the form from 0c), in place of per-tube or pooled CFU. It generalizes
     the pooled practice and gives step 5 a sound total on real data before
     step 6 exists. Its per-tube SDs are correlated (one curve), which the
     `lncfu` likelihood ignores; step 6 fixes that.
   - The orchestrator requires counts, depth and the OD table. Old
     processed files are not supported; reprocess the real data (C6) and
     regenerate simulations.
3. **Generic OD600 calibration** (N; new CLI; C10, C11).
   `tfs-calibrate-od600`: port the notebook's method (technical-replicate
   reading noise, detection threshold, errors-in-variables polynomial of CFU
   on OD, counting plus pipetting error on the plate counts) with a
   documented CSV input format and a configurable polynomial degree. It
   writes a calibration YAML with coefficients, the **full** parameter
   covariance, reading noise, detection threshold, calibrated OD range and
   the volume convention (per mL; the model's `ln_cfu` is per 5 mL sample,
   so the tube volume is explicit). Shared read/apply functions in
   `process_raw/`. Test: reproduces the lab notebook's constants and CFU
   estimates from its spreadsheets (run locally; the data stay out of the
   repo), and a synthetic example in `examples/`.
4. [x] **Simulator realism** (N; simulator only).
   The simulator has index hopping (`prob_index_hop`) and per-tube growth
   noise (per timepoint, matching the design) but writes
   `sample_cfu_std = 0.0` and no OD. Add: OD600 per tube through an inverse
   calibration (from a calibration YAML) with reading noise, the detection
   threshold and the calibrated range; bioreplicates that get OD only (no
   reads); founder sampling per tube (each tube seeded with a Poisson
   draw of cells per genotype at the split, not `total_cfu0 * freq`; 0b);
   a template bottleneck before sequencing (reads drawn from a limited
   number of template molecules; 0b); PCR jackpotting (Gamma-Poisson counts with a configured
   dispersion); a per-tube composition offset; optional monoculture
   growth-rate measurements and a spike-in counting standard, to size how
   much each would help before it is run. Each source can be switched off,
   so its effect on the fit can be measured (a scientific switch, D12).
   Values from 0b and 0c. Needed to validate steps 6 and 7, and step 5
   under realistic noise.
   Done 2026-09-26: `founder_sampling`, `demographic_growth`,
   `pcr_template_molecules`, `pcr_amplification_cv`, an `od600` block
   (`simulate/od600.py`; OD-only replicates, `tfs_sim_od600.csv`,
   `sample_cfu_from_od600`), plus `shared_transformation`: every replicate
   used to redraw the library assembly and transformation, unlike the real
   protocol's single glycerol stock. With all options off the simulator's
   output is byte-identical to before. Checked on the example config: the
   template bottleneck adds `(reads / templates)(1 + cv^2)` to the
   variance/Poisson ratio, as designed; the example config's 260k reads
   per tube against 2M templates gives only 1.2-1.8x, so matching the real
   5-18x needs templates on the order of reads per tube / 10 (about 1-2M
   at the real ~18M reads per tube). The example config's tubes also reach
   OD600 2.9, above a realistic calibrated range; tune `cfu0` and timing
   before using its OD600. Deferred: monoculture growth-rate outputs and a
   spike-in counting standard (waiting on the bench decisions).
5. [ ] **Relative-X fit** (G; new component, opt-in). (Implemented;
   validation grid pending.)
   - A relative Hill theta component: hill_geno's curve with real-valued
     baselines, population priors centered on wt, registry rules (C5). Gauge
     (D4): wt's curve is pinned to `X = 1` at `c_lo` and `X = 0` at `c_hi`
     (per titrant), so wt keeps its K and n free and its baselines follow
     from them. `c_lo`/`c_hi` are an orchestrator setting written to the
     config, defaulting to the lowest and highest measured concentration
     (0 and 1 mM IPTG in the current data).
   - One X per genotype x titrant concentration, shared by every condition
     (D5); each condition's `k_c` and `m_c` (sign included) map it to growth.
     Report the check: per genotype, kan and 4CP growth should be affine in
     each other. Genotypes that break it are the first evidence against the
     design hypothesis.
   - A growth-only configuration forces `activity: fixed`,
     `theta_rescale: passthrough`, `condition_growth: linear`, and refuses
     `hill_mut`/thermo theta components and partition-function congression
     rules (C1).
   - Outputs labelled as X; downstream restrictions (C8).
   - `tfs-summarize-calibration` computes the X-scale truth (C8).
   - Validation on one simulation: joint vs relative fit, the relative fit
     compared with X-scale truth; genotypes with simulated binding reported
     separately (the calibration tool already stratifies on the simulated
     binding file, so this is a held-out comparison); `k_c`/`m_c` recovery
     and the IPTG-independent genotypes as internal standards (C4). Then
     under step 4's noise. Real data runs use step 2's smoothed total.
   Implemented 2026-09-26: `theta: hill_relative`
   (`generative/components/theta/hill_relative.py`; baselines `X_low` and
   `X_delta` real-valued and hierarchical, priors centered on wt's 1 and
   -1; wt's baselines computed from its K and n by `gauge_baselines`, with
   the occupancy span floored at 1e-3). The gauge is the orchestrator
   setting `theta_gauge_conc` (`--theta_gauge_conc`; default the measured
   min and max, written to the config). The orchestrator refuses binding
   data, learned activity, `logit` rescaling, non-linear growth, theta
   noise, and non-`max` rules under `mixture` (C1; `mixture`'s `max`
   branch does not clip, checked). C8: `THETA_SCALE = "X"`;
   `tfs-predict-theta` writes `theta_scale = X`; epistasis tools allow only
   `add` (no `in_regime`); `tfs-summarize-fit` maps theta and
   `growth_k`/`growth_m` truth onto the gauge (with simulated activity);
   `tfs-summarize-calibration` labels those rows `theta_regime = X`.
   Prediction on a genotype subset keeps wt for the gauge. Registry-wide
   batch-safety, batch-order and guide-naming tests pass for it. Not built:
   the kan-vs-4CP consistency check (X is shared by construction, so a fit
   cannot test it alone; needs a per-condition-family X variant or a
   model-free comparison, to be decided with the real-data run). Validation
   grid: `planning/studies/relative-fit/` (24 runs: noise x likelihood x
   joint/relative x 3 seeds; not yet run).
6. **Population model in the fit** (N; replaces the per-tube offset).
   - Population curves: `P = ln N` per replicate x condition x concentration
     over `t_pre + t_sel`, starting at the replicate's split density (from
     the presplit OD600 and the dilution, with calibration and pipetting
     error; one value per replicate), with a shape shared across replicates
     and per-replicate deviations (form from 0c, D9). Global sites, a few
     per curve.
   - OD600 likelihood: every reading of every bioreplicate, through the
     calibration (global, D10), with reading noise plus tube growth noise
     `(sigma_tube * t)^2` (C3); readings below the threshold are censored,
     readings above the calibrated range are refused or flagged.
   - Optional monoculture growth observer, for bench experiment 1 of
     `planning/absolute-abundance-measurements.md`: a measured growth rate
     of a named genotype in a named condition and concentration observes
     the clean-class rate `k_c + dk_g + m_c X_g(c)`. Unlike `base_growth`
     (`k_ref + dk_geno` in one reference condition), it pins `k_c`/`m_c`
     where OD600 is blind.
   - The `lncfu` likelihood observes frequencies against the model:
     `ln freq_{g,s} ~ ln n_{g,s} - P_s + u_s`, counting variance only. The
     per-row `ln_cfu` stays an output, now built from `P`.
   - `sample_offset` becomes the composition offset `u_s`: per tube, constant
     prior (0b), no time scaling (D2). Exposed in `tfs-configure-model`.
   - Validation: the calibration grid with step 4's noise on, against the
     step 2 interim total: theta coverage, `k_c`/`m_c` recovery (C4), in the
     joint and the relative fit.
7. [ ] **Count likelihood** (N). (Implemented; validation grid pending.)
   `growth_likelihood: counts` (C2): growth and presplit observers with
   `reads_{g,s} ~ NegBin(depth_s * exp(ln n_{g,s} - P_s + u_s), dispersion)`,
   `P` from step 6 (until then, the supplied `sample_ln_cfu`, D14), `u`
   from step 6's `sample_offset`, and a learned dispersion with both a term
   proportional to the mean and a quadratic term,
   `var = mu (1 + phi) + mu^2 / r`, so the refined data only set priors
   (C12; one per experiment; no
   per-batch term, D8), optional index hopping (`E[reads] = N (p_g + h q_g)`),
   and, for future screens, a spike-in counting standard (a non-growing
   pseudo-genotype of known abundance per tube, which fixes `P_s`).
   No pseudocount in the fit. This supersedes censoring (D1). Validation: the
   calibration grid against step 6's `lncfu` model, including the
   true-lambda-0 arm where the mixture currently finds lambda ~0.12, a
   leading suspect for which is the floor. Low-count genotypes are the
   stratum to watch.
   Implemented 2026-09-26 (with supplied totals, D14):
   `generative/observe/growth_counts.py` (`reads ~ NegBin(mu, mu (1 + phi)
   + mu^2 inv_r)`, `log mu = ln reads + ln_cfu_pred - ln total`), a
   float32-accurate log-pmf (Loader's algorithm; numpyro's negative
   binomial lost several nats per observation at 1e4-1e6 reads),
   `sample_offset: level` (one constant-prior offset per tube, D2), and
   `sample_reads` from `counts_to_lncfu`; the orchestrator derives tube
   reads from `adjusted_counts / frequency` for files processed before, so
   the new processing batch needs no rerun. Refuses `growth_noise` other
   than `zero`. Presplit stays `ln_cfu` (its data have no counts yet);
   index hopping is not modeled yet. Validation grid:
   `planning/studies/count-likelihood/` (24 runs). First run (2026-09-27):
   11 of 12 `counts` runs failed, on two bugs now fixed (the pre-fit's
   Hessian was taken at `exp` of every positive site's MAP, overflowing on
   `phi`; the concentration underflowed for near-extinct genotypes, giving
   NaN gradients). Rerun needed, `lncfu` arms included (the Hessian bug
   also loosened their `k_scale`). The one finished `counts` mixture run
   found lambda 0.027 at a true 0 (the `lncfu` mixture found ~0.12).
   - [ ] **7b. Growth regime: test, choose, purge** (N). The
     `growth_transition` components (`instant`, `memory`, `baranyi`,
     `baranyi_k`, `baranyi_tau`, `two_pop`) were unconstrained by the data
     under the old noise model (user, 2026-09-26). Re-test them once steps
     6-7 are in: on simulations with known transitions (does the fit recover
     a lag it was given, and stay at `instant` when there is none?), then on
     real data. Keep `instant` (the null) and at most one physically
     motivated model, most likely `two_pop` (cells switch from pre-selection
     to selection growth at a first-order rate `k_trans`, so each genotype's
     trajectory is a two-exponential, the per-genotype counterpart of D9's
     population form); fix its fallback for `g_pre - g_sel - k_trans <= 0`
     if it is kept. **Purge the rest** from the fit
     (`generative/components/growth_transition/`, `_baranyi.py`), the
     simulator (`simulate/growth/transition_linkage.py`,
     `TRANSITION_REGISTRY`), examples, grid `auto` enumeration, tests, docs
     and CLAUDE.md (D12, D13).
8. **The growth-binding map** (G's stage 2; study first).
   On real data, relative-X fits (step 5) with the step 7 likelihood; errors in
   both variables, genotype as the unit; the nested family identity -> affine
   f -> monotone f, with a global concentration scale s; spiked and bulk
   separately (bulk on a compressed spiked curve = congression). A study in
   `planning/studies/`; a CLI only if it earns one. On simulation first with a
   known nonlinear f and scale s, to check recovery.
9. **Joint fit with the learned map** (G; scope set by step 8).
   A learnable `theta_rescale` (f) and concentration scale s; binding back at
   weight 1 (C7); absolute theta and the partition-function congression rules
   return through `f^-1` (C1). If step 8 finds no map, this step becomes a
   report of the disagreement, not a model.

After step 9: revisit `planning/empirical-mixture-refit.md` (Stage 1 could
then use the count likelihood and needs no binding-based prefit).

## Decisions

User, 2026-09-26.

- **D1. No censoring.** Step 7 replaces it (and the masked 3.5 grid showed
  the floor does not drive the mixture's behavior). If main needs a floor fix
  before step 7 lands, dropping rows below a read threshold is available now.
- **D2. `sample_offset` becomes the composition offset `u_s`.** Revised
  2026-09-26 after the OD discussion: tube growth noise moves to the OD600
  likelihood (C3), so the per-tube latent on genotypes keeps only the
  time-independent composition part (PCR, calling). No new component.
- **D3. One `u_s` per tube, independent across timepoints.** No random walk.
- **D4. Gauge: pin wt at two reference concentrations**, `X_wt = 1` at 0 and
  `X_wt = 0` at 1 mM IPTG, whether or not those are asymptotes. It uses the
  full accessible dynamic range of growth.
- **D5. X is shared between kan and 4CP** (the core design hypothesis), with
  a consistency check (step 5).
- **D6. One long-lived branch.**
- **D7. Tube growth noise scales with `t_pre + t_sel`.** Protocol: transform,
  grow to a set OD, take `presplit`, split into one tube per condition x
  concentration x timepoint (a grid: rows timepoints, columns IPTG), pull a
  whole row per timepoint. Pre-growth and selection both happen in the
  tube. Everything before the split is shared by a replicate's tubes and
  lands in `ln_cfu0`, which `presplit` measures.
- **D8. No per-row term.** A row's tubes are read, spun down and frozen
  immediately (stop-time spread: seconds); all samples are then thawed and
  extracted in batches of about 20 (three for 60 samples) and amplified in
  one PCR block. Extraction batch is the only shared step; yield does not
  change genotype frequencies, but it could change PCR template input and so
  count dispersion. The batch assignment was not recorded (user,
  2026-09-26): batches of 20 generally followed processing order, but not
  reliably. So the model has **no batch term**: an assignment that is only
  probably right would put guesses into the likelihood. If 0b sees block
  structure along the likely order, it is reported as unexplained
  dispersion and absorbed by the shared dispersion, not modeled. Future
  screens should record each sample's extraction batch (a `sample_df`
  column), so the question can be answered then.

- **D10. Calibration parameters are latent** (user, 2026-09-26; may be
  revised): the calibration file's mean and covariance are a
  multivariate-normal prior on the polynomial coefficients (three for a
  quadratic), so their shared uncertainty is carried once (C11). Fixed at
  the best fit stays available as a check.
- **D12. Breaking changes are fine** (user, 2026-09-26). No compatibility
  with old files, configs or checkpoints is kept for its own sake. A
  switch or an old path survives only for a scientific reason: to measure
  whether something matters (the `lncfu` vs `counts` likelihood, the noise
  sources in the simulator, `instant` as the null growth transition).
- **D13. One framing for growth regimes** (user, 2026-09-26). Once a
  physically motivated model is chosen (7b; D9 for the population curve),
  models outside that framing are removed, not kept as options.
- **D14. Build order while data are gathered** (user, 2026-09-26): steps
  1, 4, 7 (supplied totals), 5, 3; see "Steps".

## Open

- **D9. Population curve form.** Per replicate x condition x concentration,
  a known start (presplit), a shared shape plus per-replicate deviations
  (the user's observation: slopes agree across bioreplicates, levels do
  not). Under selection the total can fall before it rises (susceptible
  cells stall or die while resistant genotypes take over), so a monotone
  form is wrong there. Candidates: a sum of two exponentials,
  `log(a e^{r1 t} + b e^{r2 t})` (physically motivated: two
  subpopulations); piecewise linear over pre-growth and selection; a
  low-order polynomial or spline. 0c decides. Deriving the curve from the
  genotype model (`logsumexp_g ln n_g`) is exact in principle but not
  closable: `__unknown__` reads (about two thirds of kan reads) are cells
  of unmodeled genotypes that still count toward OD600.
  Decide together with 7b, but keep the two questions apart: the
  population curve can bend with no per-genotype lag at all, because
  selection changes the library's composition (fast growers take over the
  total). Curvature in the totals is not evidence for `two_pop` or a
  Baranyi lag; only the per-genotype trajectories in the reads are. If
  genotypes do follow `two_pop`, the population is a sum of their
  two-exponentials, and a two-exponential population curve is a
  reduction of that, not the same model.
- **D11. Does the calibration transfer to selected cells?** It was measured
  on unstressed Keio cells in LB. Kanamycin- or 4CP-stressed cells can change
  size and shape, and dead cells still scatter light (and still carry
  plasmid DNA into the reads). 0c looks for signs of it; a bench check
  (OD vs plate counts under selection) would settle it.
- **Where the extra count noise comes from** (C12). A technical replicate
  (re-amplify and re-sequence the same extracted DNA for a few tubes)
  separates the template/PCR part from founder sampling and biology. Cheap
  if DNA remains.

Answered (user, 2026-09-26): OD600 is read with the calibration's setup
(same reader, plate, 200 uL); the presplit is read and then diluted
identically into every tube (above); `ln_cfu` is `ln(cfu/mL * 5 mL)`; each
row of a bioreplicate is read as a block within 5 minutes of its freeze.
