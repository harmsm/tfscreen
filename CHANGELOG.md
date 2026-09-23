# Changelog

All notable changes to `tfscreen` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

Entries cite files and functions rather than commit hashes, so they stay valid
across rebases and amended commits.

## Provenance

The contemporaneous log begins on 2026-09-10, at package version 0.4.3; this
file was assembled on 2026-09-22 around the reconstruction made then. Entries
fall into two kinds:

- **Contemporaneous.** Written at the time of the change, as part of the
  change. Every entry from 0.4.4 onward is contemporaneous *except* blocks
  explicitly headed "Reconstructed".
- **Reconstructed.** Backfilled on 2026-09-10 from `git diff` between release
  tags, commit messages, and the GitHub
  [release notes](https://github.com/harmslab/tfscreen/releases). Every
  version section from 0.4.3 back to 0.1.0 is reconstructed and opens with a
  marker saying so. Those sections describe the net change between tags; a
  feature added and removed within one release is not listed. Commit messages
  from this period are terse, so bullets were written from the diffs and kept
  only where the diff or code confirmed them; they may still be incomplete.
  Where a GitHub release note exists, it is quoted verbatim as the author's
  own contemporaneous summary, separate from the reconstructed bullets.

## [Unreleased]

### Added

- **Guide selection for SVI (`tfs-fit-model --guide_type`).** With
  `analysis_method=svi`, the variational family can now be any numpyro
  autoguide as well as the component guide (still the default):
  `delta`, `auto_normal`, `auto_diagonal_normal`, `auto_multivariate_normal`
  or `auto_low_rank_multivariate_normal`. Numpyro class names such as
  `AutoNormal` are accepted, case-insensitively. `--guide_rank` sets the rank
  of `auto_low_rank_multivariate_normal`, and `--guide_init_scale` sets an
  autoguide's initial scale. An option the chosen guide does not take is
  refused before any fitting starts, as is any guide option with
  `analysis_method` `map` or `nuts`.
  - `RunInference.setup_svi` takes `guide_type`, `guide_kwargs` and
    `init_values` (constrained site values an autoguide's location starts
    from, via `init_to_value`). The registry is `AUTOGUIDES`/`GUIDE_TYPES` in
    `inference/run_inference.py`, with `resolve_guide_type` and
    `check_guide_kwargs`.
  - An autoguide starts from the pre-MAP solution when `pre_map_num_epoch > 0`,
    and from the configured guesses otherwise.
  - Checkpoints record `guide_type` and `guide_kwargs`.
    `restore_svi_from_checkpoint` rebuilds that guide; a checkpoint written
    before this change is treated as a component-guide checkpoint.
    `tfs-fit-model` refuses to resume a checkpoint with a different guide.
  - `tfs-sample-posterior` sends `delta` checkpoints to the Laplace
    approximation and every other guide to guide sampling. It routes by the
    recorded `guide_type`, because `AutoNormal` parameters share AutoDelta's
    `{site}_auto_loc` names. Older checkpoints still fall back to that name
    check.
  - `get_posteriors` drops autoguide auxiliary sites (names starting with
    `_`, such as `AutoContinuous`'s `_auto_latent`) before the forward pass.

- **`tfs-summarize-calibration`** (`analysis/calibration_grid.py`) pools
  posterior calibration across a `tfs-setup-sim-grid` grid. It reads each
  run's `combo.json` and the `tfs-summarize-fit` outputs in its `summary/`
  directory, and reports coverage at several interval levels, calibration
  error and bias, PIT uniformity (KS), interval width, and median RMSE and
  Pearson r against the simulated truth. Rows are per run x quantity x
  stratum: whether the genotype has binding data, whether it is encoded by a
  spiked sequence, and, for theta, whether the true value is resolvable or
  saturated. Arms (runs sharing every grid variable except `--replicate_keys`)
  are averaged, and `--baseline key=value ...` pairs each run with the
  baseline run fit to the same simulated data. Unfinished runs are listed in
  `{out_prefix}_run_status.csv`. No thresholds or grades. New console script:
  reinstall to register it.

### Fixed

- **Per-genotype latents scrambled across genotypes under autoguides.**
  Several components sampled their per-genotype latents in a plate sized
  `data.batch_size`, so every numpyro autoguide -- including the `AutoDelta`
  behind MAP, pre-MAP and `tfs-prefit-calibration` -- held one parameter per
  batch *position* rather than per genotype. The full-batch index is
  binding-first and reshuffled every step, so this happened even without
  mini-batching: binding genotypes kept their positions, but every bulk
  genotype's latents were overwritten by whichever genotype landed in that
  position. The component SVI guide was not affected, since it holds
  library-sized parameters and slices them itself.
  - `activity/hierarchical_geno`, `activity/horseshoe_geno`,
    `dk_geno/hierarchical_geno`, `ln_cfu0/hierarchical`,
    `ln_cfu0/hierarchical_factored`, `theta/categorical_geno` and
    `noise/logit_normal` now sample these latents in a library-sized
    `{name}_genotype_plate` and slice them to the batch with
    `data.batch_idx`. The latent priors and guides that were wrapped in
    `scale_vector` no longer are; only the likelihood is scaled.
  - The theta-noise components (`zero`, `beta`, `logit_normal`) take an
    optional `data` argument, which `generative/model.py` now passes.
  - `noise/beta` is a documented exception: its `{name}_dist` latent is the
    noisy theta itself and remains batch-shaped.
  - `RunInference.get_map_posteriors` and `get_laplace_posteriors` now raise
    on a MAP checkpoint whose genotype-indexed latents are not library-sized,
    instead of clipping the indices and reusing one genotype's value for
    another. **MAP checkpoints written before this fix must be refit.**
  - `draw_prior` no longer returns plate sites as if they were latents.
- **New `tfmodel/inference/batch_safety.py`** detects batch-dependent latents
  by tracing a model at two batch sizes (abstractly, under `jax.eval_shape`).
  `tests/tfscreen/tfmodel/inference/test_batch_safety.py` runs it across the
  component registry. `orchestrator_latent_dimension` gives the size of the
  flattened latent vector an `AutoContinuous` guide works in.
- **`RunInference.run_optimization` refuses unsafe autoguide fits.** When the
  guide is any numpyro `AutoGuide` (including the `AutoDelta` behind MAP,
  pre-MAP and `tfs-prefit-calibration`), it first runs the batch-safety check
  and raises a `ValueError` naming the offending sites -- for example, a MAP
  fit with `theta_growth_noise=beta`. The component guide is not checked. For
  an `AutoMultivariateNormal` guide it also reports the latent dimension and
  the memory the dense covariance needs, and warns above 4 GB.

- **`tfs-summarize-fit` could pick the wrong config or losses file in a grid
  run directory.** It matched `*_config.yaml` and `*_losses.txt`, which there
  also match the simulate config and the prefit/pre-MAP losses; the right file
  won only by alphabetical order. It now takes the config that has `data` and
  `components` sections, and ignores `*_prefit_losses.txt` and
  `*_premap_losses.txt`.

## [0.4.4] - 2026-09-22

A processing and bookkeeping release: two fixes in the read-counting and
`ln_cfu` paths, a library description for `tfs-configure-model`, and a
general cross-run comparison tool. The inference model itself is unchanged --
this is the groundwork for 0.5.0, which changes how congression and spiked
genotypes are handled.

### Added

- **Library description for `tfs-configure-model` (`--library_config`).** The
  model now learns which genotypes were spiked into the library from the
  library YAML itself -- the same file handed to `tfs-process-fastq` -- instead
  of a hand-written list of genotype names. It is required whenever
  `--growth_df` is given and refused for a binding-only model.
  - `tfscreen.genetics.library_design`: `library_composition_table()` turns a
    library config into one row per genotype (`is_wt`, `in_spiked_origin`,
    `pool_fraction`, `bulk_fraction`, `origins`), with
    `read_library_composition` / `write_library_composition` for the snapshot.
    The module also carries `expected_library_composition`,
    `scale_library_design` and `estimate_library_mixture` (the last estimates a
    realized `library_mixture` from an observed abundance table, e.g. a
    pre-split sample).
  - `tfs-configure-model` writes `{out_prefix}_library.csv` and records it in
    the config as `library_file`, alongside a `library` block holding the
    source path and the declared `library_mixture`. `tfs-fit-model` reads the
    snapshot CSV, never the YAML, so the table is where a design number can be
    overridden -- the same pattern as the priors and guesses CSVs.
  - `tfs-configure-model` now **fails** if any genotype in `growth_df`,
    `presplit_df` or `base_growth_df` is absent from the library
    (`__unknown__` excepted). A residue-numbering or wt-sequence difference
    between the library description and the reads it was called against
    mismatches most or all genotypes and is caught on the first run.
  - `ModelOrchestrator` gains `library_file` and exposes the table as
    `orchestrator.library_df`.
  - `examples/process_raw/library_config.yaml` gains `library_mixture`.

### Changed

- **Breaking: `tfs-configure-model --spiked` is removed**, replaced by
  `--library_config` (above). **Action:** pass the experiment's library YAML
  instead of listing spiked genotypes. Configs already written with
  `spiked_genotypes` still load, and `ModelOrchestrator(spiked_genotypes=...)`
  still works, but the two inputs are mutually exclusive.
- **Breaking: `tfs-fit-genotypes` and `tfs-build-empirical --spiked_file`
  become `--library_config`.** The congression-free genotypes are read from
  the library description rather than a text file of names.
- A spiked genotype named by the library but absent from the growth data is
  reported and ignored, rather than raising. The list is derived from the
  library design, and real spikes can drop out of a dataset. A hand-supplied
  `spiked_genotypes` list still raises, since it is hand-written.
- `counts_to_lncfu` works in log space. It now reads
  `sample_ln_cfu`/`sample_ln_cfu_std`, inferring them via `get_scaled_cfu`
  from `sample_ln_cfu_var`, or from `sample_cfu` plus
  `sample_cfu_std`/`sample_cfu_var` when the log-space columns are absent
  (log-space wins). Genotype `ln_cfu`/`ln_cfu_var` are computed directly in
  log space and `cfu`/`cfu_var` derived from them. The new helper
  `get_sample_ln_cfu` is called early by `tfs-process-counts` and
  `tfs-process-presplit`, so a bad sample sheet fails before any counts are
  read; `get_scaled_cfu` gains a `prefix` argument. Existing sample sheets
  with `sample_cfu`/`sample_cfu_std` keep working and give identical results;
  outputs gain `sample_ln_cfu`/`sample_ln_cfu_std`.

### Known issues

- **`pool_fraction` and `bulk_fraction` are recorded but not yet used by the
  model.** The fit still treats "spiked" as a binary (congression-free plus a
  separate `ln_cfu0` prior class), although wt and the spiked single mutants
  are in fact mostly bulk cells. `bulk_fraction` is the per-genotype weight an
  observable-level congression mixture will need.
- **Nothing cross-checks `library_mixture`.** No upstream artifact ever reads
  it, so a stale or wrong mixture cannot be detected -- only recorded. The
  genotype-set check covers the genetics keys only. Use one `run_config.yaml`
  per experiment for `tfs-process-fastq`, `tfs-simulate` and
  `tfs-configure-model`.

### Reconstructed: changes between v0.4.3 and 2026-09-10

> *Reconstructed on 2026-09-10 from `git diff v0.4.3..HEAD` and commit
> messages. These entries were not written when the changes were made. They
> cover commits from before the contemporaneous log began that are not
> described above.*

#### Added

- **Pre-fit model accounting in `tfs-configure-model`.**
  `tfmodel/model_stats.py` (`count_model_dimensions`, `format_model_stats`,
  `write_model_stats`) traces the numpyro model under `jax.eval_shape` and
  counts latent parameters (by plate, support and component) against
  observations (from `good_mask`). It also records per-genotype coverage and
  the anchors that pin the k/dk_geno/k_ref slide.
  - `tfs-configure-model` prints a summary and writes
    `{out_prefix}_model_stats.csv` and `{out_prefix}_model_stats.json`.
    `--skip_model_stats` disables this.
- `tfs-process-fastq` logs at startup whether numba JIT is active.

#### Changed

- **Breaking:** `tfs-compare-feature` is replaced by `tfs-compare-runs`
  (`analysis/compare_runs.py`, `analysis/scripts/compare_runs_cli.py`). The
  new tool is estimate-agnostic: it accepts predicted features
  (theta/growth/epistasis) and `tfs-extract-params` parameter tables.
  - Keys are configurable: an auto-detected match key, plus `--match_by`,
    `--index_by` and `--group_by`. The run count (`n_present`/`n_runs`) is
    reported.
  - It reports only raw statistics (`rms_sd`, `overdispersion` with p and a
    BH q). The graded A–D `tier` column, the `overdispersed` flag, the
    crosstabs output, and `--sd_tier_edges`/`--overdispersion_threshold` are
    gone.
  - Output is sorted in canonical genotype order.
  - Estimates are passed as two or more CSV paths or as a single manifest
    file.
  - The aggregate mixture CSV is written by default (`--no_aggregate`
    suppresses it) and resolved settings go to `{out_prefix}_metadata.json`.
- **Faster `tfs-process-fastq`.** The expected-library reconciliation pass
  used to walk `pybktree` in Python. It now flattens the BK-tree to CSR
  arrays and searches it in a single numba kernel, which is about 4.4× faster
  on that pass with identical output. The `FastqToCounts` object is shipped to
  each pool worker once, via an initializer, instead of on every submit.

#### Fixed

- **`__unknown__` reads were counted as wildtype.** `standardize_genotypes`
  (`genetics/genotype_sorting.py`) parsed the reserved `__unknown__` label as
  a self-to-self mutation and collapsed it to `wt`. This produced a phantom
  second `wt` row carrying all unattributable reads.
  - Reserved labels (`UNKNOWN_GENOTYPE`, `RESERVED_GENOTYPES`) now pass
    through unchanged. Mutation tokens without an integer site number now
    raise instead of being treated as synonymous.
  - `counts_to_lncfu` keeps the `__unknown__` bucket in the per-sample
    read-depth denominator, drops it from genotype rows, and excludes it from
    the pseudocount term. The binomial frequency variance now uses that same
    denominator.
  - **Action:** regenerate `ln_cfu` tables produced from counts files that
    contain an `__unknown__` row.
- `tfs-process-fastq --num_workers` was parsed with the wrong type because it
  defaults to `None`. It is now forced to `int`.

## [0.4.3] - 2026-07-23

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.4.2..v0.4.3`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> This release improves the epistasis and cat response code. One of the major theoretical predictions motivating this work is that epistasis in dG versus log(effector) should exhibit peak-like behavior if the ensemble is redistributed. This implements that, allowing for epistasis estimates using logit(theta) (proportional to dG), improved/numerically stabilized peak calling, and joint estimates of epistasis from the full posterior rather than marginal theta distributions.
>
> - joint estimates of epistasis
> - improved cat response code
> - ability to predict genotype growth on a small subset of genotypes
> - added ability to pool runs with different seeds to generate aggregate posteriors
> - general bug fixes

**Reconstructed changes:**

### Added
- **Joint epistasis from the posterior (`tfs-predict-epistasis`).** New CLI (`tfmodel/scripts/predict_epistasis_cli.py`, backed by `extract_theta_epistasis` in `tfmodel/analysis/extraction.py`). It draws θ for every genotype from the same posterior sample, computes mutant-cycle epistasis within each draw, and reports quantiles across draws. The uncertainty therefore includes the posterior covariance between the four cycle corners, which the marginal-θ route drops. Output columns are `genotype`, `titrant_name`, `titrant_conc`, bare `q<level>` columns, and a trailing `in_regime` flag. Flags: `--scale` (`logit` default, `add`, `mult`), `--scale_constant`, `--regime_eps` (default 0.01), `--regime_ci` (default 0.95), plus the usual `--genotypes_file`/`--titrant_names_file`/`--titrant_concs_file`/`--only_files`. Only genotypes seen in training are supported. A MAP `.pkl` gives a single point estimate.
- **`in_regime` flag.** Set to 1 only when all four cycle corners have their θ posterior (central `regime_ci` interval) inside `[regime_eps, 1 - regime_eps]`. Rows with `in_regime == 0` depend on the model's functional form, not directly on the data.
- **Logit-scale epistasis.** `tfscreen.analysis.extract_epistasis` accepts `scale="logit"`: additive epistasis of `logit(Y)`, which is proportional to ΔG for an occupancy. It also gains `scale_constant` (multiplies the transform, e.g. `-RT` to report an interaction free energy; rejected for `mult`) and `logit_eps` (clamp before the logit; warns on inputs outside [0, 1]). The per-scale definitions now live in one helper, `_epistasis_from_corners`, which the posterior path also uses.
- **`tfs-extract-epistasis` CLI** (`analysis/scripts/extract_epistasis_cli.py`). Computes epistasis from any long-form observable table. Flags: `--y_obs`, `--y_std`, `--group_by`, `--scale add|mult|logit`, `--scale_constant`, `--keep_extra`, `--logit_eps`. When no cycles are found, it prints a diagnostic that names the column(s) that probably need to go in `--group_by`.
- **`tfs-compare-feature` CLI** (`analysis/compare_feature.py`, `analysis/scripts/compare_feature_cli.py`). Grades per-genotype agreement of any quantile-summarized feature (θ, growth, epistasis) across N runs, such as different seeds or k-fold dropouts. It scores two axes: reproducibility (`rms_sd` of `q0.5` across runs, graded into tiers A–D by `--sd_tier_edges`) and self-consistency (an `overdispersion` statistic, with a flag set by `--overdispersion_threshold`). It has a symmetric mean mode and a reference mode (`--reference`), plus `--min_coverage`. It writes `{out_prefix}.csv` and `{out_prefix}_crosstabs.txt`. The input is a manifest file that lists the estimate CSVs.
- **Pooled aggregate posteriors across runs.** `tfs-compare-feature --write_aggregate` (`aggregate_feature`) writes `{out_prefix}_aggregate.csv`. Each row pools the N runs as an equal-weight mixture of their per-run marginal posteriors, rebuilt from the `q<level>` ladder. The pooled error covers both per-run width and run-to-run spread, and it does not shrink with N.
- **`tfs-fit-genotypes` CLI** (`tfmodel/scripts/fit_genotypes_cli.py`). Runs the per-genotype MLE fit of the growth model as a standalone step, with optional congression de-attenuation (`--congression_lambda`). Other flags: `--spiked_file`, `--intercept_cols`, `--dk_geno_prior_sd`, `--min_obs`, `--num_workers`, `--save_theta_history`. It writes `<prefix>_params.csv`, `<prefix>_theta.csv`, and, with de-attenuation, `<prefix>_params_deattenuated.csv` and optionally `<prefix>_theta_history.csv`.
- **Genotype-subset growth prediction.** `tfs-predict-growth --subset_genotypes` (with `--subset_seed`) predicts one memory-sized block of genotypes instead of the whole library, for a quick input/output ln_cfu check. The block always keeps the binding, spiked, and file-specified genotypes and fills the rest with a random sample. If the device runs out of memory, the random part is halved and the prediction retried.
- **Log-concentration curve models** in `mle/curve_models`: `linear_log`, `bell_peak_log`, `bell_dip_log`. Each is a shape in `log10(x)` that takes raw concentration and applies the transform internally (`_to_log10_x`, which floors `x <= 0` to `min(x>0)/100`). New curated model lists `DEFAULT_MODELS` and `SHAPE_MODELS`.
- **Cat-response magnitude assessment** (`analysis/cat_response/cat_assess.py`). Tests each curve against zero on the observed data: a χ² test on `y_obs/y_std` gives `nonzero_p`, then Benjamini–Hochberg across curves gives `nonzero_q`, and together they drive a boolean `fittable`. Per-point `z`/`sig_nonzero` are also reported. The model-based omnibus test (`omnibus_W/df/p/q`) is reported but does not gate anything. An equivalence rollup (`all_equiv_zero`) uses a ROPE (`--rope_cutoff`, or auto `rope_multiplier * median(y_std)`).
- **Cat-response model-selection modes** (`cat_fit.py`, `--select_by`). `shape` is the default: it gates flat vs curvy on the flat fit's weighted residual autocorrelation (`--curvy_cutoff`), then picks the curvy model with the best weighted R². `aicc` picks the lowest-AICc model. `adequacy` keeps the AICc pick but escalates (never demotes) when a runs test flags clustered residuals (`--adequacy_alpha`). Per-model diagnostics are reported for every model: `runs_p`, `autocorr`/`autocorr_p`, and weighted-χ² `gof_p`. New rollup columns: `shape`, `shape_status`, `aicc_best_model`.
- **New cat-response outputs:** `{out_prefix}_assessment.csv` (per-point record with `y_obs`/`y_std`, `y_model`/`y_model_std`, `z`, `sig_nonzero`, `fittable`) and one `{out_prefix}_{model}.csv` parameter table per model. `--write_all_predictions` writes the predicted curves of every model instead of only the best one.
- `tfscreen.util.resolve_obs_columns`: shared default of `y_obs=q0.5` and `y_std=(q0.841 - q0.159)/2`, used by `tfs-cat-response` and `tfs-extract-epistasis`. `tfscreen.util.parallel.resolve_workers` handles joblib-style worker counts (`-1` = `cpu_count - 1`).
- `predict_with_error(..., full_cov=True)` also returns the full prediction covariance. `xfill` gains `min_value` to clamp the padded lower bound (e.g. at 0 for concentrations).
- `simulate/toy_thermo/`: a self-contained four-state thermodynamic toy TF model for building intuition about observables and logit epistasis (basis curves, measurement window, plotting helpers), with the notebook `examples/toy_thermo/toy_thermo.ipynb`. Also new: `examples/simulate-empirical/simulate_config.yaml` and `docs/manuscript/methods.md`.

### Changed
- **Breaking: `tfs-cat-response` interface rewritten.** The CLI moved to `tfscreen.analysis.scripts.cat_response_cli` (the old `analysis/cat_response/scripts/` module was removed). It is now generic over `data_file x_obs [--y_obs] [--y_std] [--group_by ...]`, and groups are `genotype` plus the `--group_by` columns. It replaces the θ-specific `theta_file` + `--theta_col`/`--sigma_col`, which grouped on `genotype`/`titrant_name` with `titrant_conc` as x. `--workers` became `--num_workers` (default `-1`). Without `--models`, it fits `SHAPE_MODELS` in the default shape mode and `DEFAULT_MODELS` in the other modes; the old default was every model in `MODEL_LIBRARY`. **Action:** update scripts that call `tfs-cat-response`, e.g. `tfs-cat-response theta.csv titrant_conc --group_by titrant_name`.
- **Breaking: cat-response prediction columns renamed.** `y`/`y_std` became `y_model`/`y_model_std` in `{out_prefix}_predictions.csv`. `plot.cat_fits` now expects the new names. By default, predictions hold only the best model. `cat_fit` now returns `(flat_output, pred_df, assess_df)`, and `cat_response` returns `(results_df, predictions_df, assessment_df, rope_cutoff)`.
- **Breaking: `condition_selector` renamed to `group_by`** in `extract_epistasis` and `mutant_cycle_pivot`. `plot.heatmap.epistasis_heatmap` is updated to match.
- `biphasic_dip` bounds on `baseline`/`amplitude` are now unbounded. The old `>= 0` bounds made it unfittable on signed data such as logit epistasis.
- The per-genotype MLE fit and congression de-attenuation moved from `simulate/empirical/` to `tfmodel/genotype_fit/` (`fit.py`, `congression.py`). `simulate.empirical.fit_phenotypes` and `.congression` remain as re-export shims. New helpers: `fits_to_results_df` and `predict_theta`.
- `tfs-build-empirical` keeps `<prefix>_stage1_fits.csv` as the raw (pre-de-attenuation) fit. When `--congression_lambda` is given, it now also writes the de-attenuated fits that feed Stage 2 to `<prefix>_stage1p5_fits.csv`.
- The auto-sized `genotype_batch_size` overhead multiplier (`tfmodel/analysis/batch_sizing.py`) changed from 8.0 to 6.5.
- Cat-response fitting is parallelized over chunks of groups.

### Fixed
- **Parameter covariance on ill-conditioned fits** (`mle/fitters/_util.get_cov`). The covariance is now built from the SVD of the Jacobian, as `scipy.optimize.curve_fit` does, instead of inverting `JᵀJ`. Before, fits that were ill-conditioned but still identifiable could return a spurious all-NaN covariance. Truly rank-deficient Jacobians still return NaN.
- **`linear` curve-model guess swapped slope and intercept.** `guess_linear` built its design matrix as `[1, x]`, so the guess came back as `[b, m]` while the model expects `[m, b]`. It now builds `[x, 1]`.
- **Crash in empirical Stage 2** (`simulate/empirical/population.py`). A numerically singular per-genotype covariance, e.g. from an unidentified Stage-1 parameter, raised `LinAlgError: Singular matrix` and could fail the whole batched inversion. Inversion and the marginal log-likelihood now use a floored eigendecomposition, which always returns finite values.

## [0.4.2] - 2026-07-12

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.4.1..v0.4.2`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> - Added dk_geno inputs to calibration
> - Added pinning to fit (can pin growth parameters in SVI for example)
> - Improved simulation architecture
> - Added empirical simulations (fit experimental data and use to as an empirical generator for simulated data)
> - Fixed multiple bugs

**Reconstructed changes:**

### Added
- **Empirical phenotype simulation (`simulate/empirical/`).** A new pipeline fits real screen data and uses the fit as the phenotype generator for simulated libraries. `fit_phenotypes.py` does per-genotype MLE of the growth model with the calibration frozen (Stage 1, parallelizable via `num_workers`). `population.py` runs a measurement-error EM that deconvolves estimation noise into a `PopulationModel` (Stage 2). `resample.py` draws per-genotype phenotypes and builds `theta_gc_override`/`theta_params_override`/`dk_geno_override` (Stage 3).
- **`tfs-build-empirical` CLI** (`simulate/scripts/build_empirical_cli.py`). A one-command orchestrator that runs `configure_model`, then `run_prefit_calibration`, then Stages 1–2, and writes `<prefix>_phenotype_model.json` plus a diagnostic `<prefix>_stage1_fits.csv`. Flags include `--binding_file`, `--spiked_file`, `--base_growth_file`, `--thermo_data`, `--calibration_file` (skips configure+prefit and reuses an existing calibration), `--congression_lambda`, `--dk_geno_prior_sd`, `--min_obs`, `--drop_railed` and `--num_workers`.
- **Optional congression de-attenuation** (`simulate/empirical/congression.py`, enabled by `--congression_lambda`). Corrects bulk genotypes' Stage-1 theta curves for co-transformation with a fixed-point iteration on the inference's own congression operator, then refits Hill. Spiked genotypes are left unchanged.
- **`phenotype_source: empirical` simulate mode.** `library_prediction` resamples every genotype from the model named in an `empirical: {phenotype_model: ...}` block. This forces `theta_component=hill_geno` (with a warning if the config asks for something else) and fixed activity.
- **`dk_geno` component `pinned`** (`generative/components/dk_geno/pinned.py`, registered as `dk_geno: pinned`). Sets `dk_geno` deterministically from caller-supplied per-genotype values and defaults every other genotype to 0. wt is always 0. Values come from a pins CSV passed through `ModelOrchestrator(dk_geno_pins_file=...)`.
- **dk_geno inputs to calibration.** When a `base_growth_df` is configured, `tfs-prefit-calibration` turns its measured rates into per-genotype `dk_geno` pins (`_dk_geno_pins_from_base_growth`), so the calibration MAP uses real `dk_geno` values instead of assuming 0.
- **`tfs-prefit-calibration --pin_m`.** Hard-clamps the `linear` growth slope `m` to its per-condition calibration loc. The new `ModelPriors.m_pinned` makes `m` a deterministic site and drops its guide parameters, so the growth likelihood cannot pull `m` away during the SVI fit.
- **Per-condition growth priors.** The `linear`, `power` and `saturation` `condition_growth` components now accept either a scalar or a per-`condition_rep` array for each prior loc/scale, and declare `get_scale_bounds()`. The priors CSV (`configuration_io.py`) supports indexed per-condition rows that are name-joined to the model's `condition_rep` order. `tfs-prefit-calibration` writes per-condition MAP locs and floored scalar scales into those rows. New floor/ceiling flags: `--k_scale_floor`, `--m_scale_floor`, `--k_scale_ceiling`, `--m_scale_ceiling`; `--hessian_chunk_size` is also new.
- **Base growth-rate calibration data (`base_growth_df`).** Inference now takes direct reference-condition growth-rate measurements that anchor a global `k_ref` latent through the new `observe_base_growth` observer (`generative/observe/base_growth.py`). Added `tfs-configure-model --base_growth_df`; `k_ref` is included in the extracted parameters.
- **`observe_presplit` observer** (`generative/observe/presplit.py`). The presplit likelihood now lives in the observer layer alongside `observe_base_growth`, and each is wired in only when its data is supplied.
- **`tfs-process-presplit` CLI** (`process_raw/scripts/process_presplit_cli.py`). Converts per-sample count CSVs into a presplit `ln_cfu` table for `tfs-configure-model`. Count I/O shared with `tfs-process-counts` moved into `process_raw/_counts_io.py`.
- **`tfs-report-cfu0` CLI** (`simulate/scripts/report_cfu0_cli.py`). Prints the mean `ln_cfu0` and the number of surviving genotypes by class (`wt`/`spiked`/`single`/`double`) across `--num_replicates` simulated transformations. Use it to tune `transform_sizes`/`library_mixture`.
- **Finer control of simulated binding data.** A new `binding_data` schema has two sub-blocks, `spiked_binding` (clean controls drawn from `spiked_seqs`) and `library_binding` (in-library, congression-affected controls). Each takes `choose_by: stratified | random | <params file>` and an optional or required `num`, validated fail-fast in `_validate_binding_config`. `library_binding` genotypes are selected after the growth simulation from the genotypes that survived (`simulate/library_binding_data.py`) and written to `tfs_sim_library_binding.csv`. `sample_theta` gained `select_mode="random"`.
- **New simulation outputs.**
  - `base_growth_data` YAML block (`simulate/base_growth_data.py`): writes the `base_growth` CSV and a single-row `k_ref` echo.
  - `presplit_data` generation moved to `simulate/presplit_data.py`.
  - `tfs_sim_growth_parameters.csv`: per-condition ground truth for `condition_growth` (`simulate/growth_parameters_output.py`).
  - Transformation-lambda ground-truth file (`simulate/transformation_lam_output.py`).
- **`tfs-summarize-fit` ground-truth comparison** for condition-growth parameters, `k_ref` and the transformation Poisson lambda.
- **Genotype batching in `tfs-predict-growth`** (`--genotype_batch_size`). When the flag is omitted, the batch size is estimated automatically from available device memory (`tfmodel/analysis/batch_sizing.py`).
- `tfs-setup-grid` rewrites `presplit_df`/`base_growth_df` paths and accepts `growth_noise` as a grid axis.

### Changed
- **Breaking:** the library-genetics config keys `sub_libraries` and `library_combos` are renamed to `tiles` and `tile_combos` (`LibraryManager`, simulate/process_raw configs). **Action:** rename these keys in existing run configs.
- **Breaking:** the `library` column is now required in `growth_df` and `presplit_df`. It was previously filled with `"default"` when missing, and it is now part of the prediction merge keys. **Action:** add a `library` column to hand-built input tables.
- **Breaking:** `tfs-configure-model`'s default `--transformation_model` changed from `empirical` to `single`. The `empirical` and `logit_norm` models now require `--transformation_lambda MEAN STD` (the measured congression lambda), which anchors the transformation prior. Passing it together with `single` is an error. **Action:** add `--transformation_lambda` to any configure call that uses `empirical`/`logit_norm`.
- **Breaking:** simulated `binding_data` no longer takes a top-level `genotypes` list for choosing binding genotypes. Use `spiked_binding`/`library_binding` instead (see `examples/simulate/simulate_config.yaml`).
- Growth-component priors are now built inside each component instead of in `ModelOrchestrator`, and observer variable naming and prior handling were standardized.
- Binding genotype-params CSVs now ignore extra columns rather than rejecting them, so a `*_stage1_fits.csv` can be used directly as a `choose_by` file.
- `tfs-predict-growth`/`predict()` report parameter sites plated on `condition_rep` at one row per `condition_rep` instead of forcing them onto the growth-row grid.
- The ln_cfu0 components (`hierarchical`, `hierarchical_factored`) and the posterior-sampling path in `run_inference.py` were reworked to reduce memory use when sampling posteriors.

### Removed
- `tfscreen.mle.parse_patsy`.
- `src/tfscreen/simulate/scripts/example_config.yaml` (use `examples/simulate/simulate_config.yaml`).
- `tfs-summarize-grid` no longer merges `*_calib_stats.json` into its rows.

### Fixed
- **Tile mixing in simulation:** each transformant pool's weight is now normalized by its number of transformants. Previously `transform_sizes` leaked into the `library_mixture` ratio in `_sim_transform_and_mix`.
- **Congression correction under batching (`empirical` transformation):** the background theta CDF is now built from the full genotype population, not just the genotypes in the current minibatch or prediction subset (`update_thetas(..., population_theta=...)`).
- **Thermodynamic theta models under batching:** the `O2_C4_*`/`O2_C12_*` `thermo.py` models now index genotypes through `data.batch_idx`, so minibatches no longer pick up the wrong genotypes' theta.
- **Growth prediction batching:** each batch now includes the binding genotypes, so the binding tensor is never empty; their duplicate rows are then stripped from the output.
- Out-of-memory failures in the new posterior-sampling path.
- Presplit data simulated by `tfs-simulate` now carries and groups by the `library` column.
- The prefit calibration model no longer receives the production `transformation_lambda` after its transformation is forced to `single`.
- `base_growth_data` is now a recognized simulate config key.
- `tfs-fit-model`'s pre-MAP stage now honors `convergence_tolerance`, `convergence_window` and `patience`.

## [0.4.1] - 2026-06-22

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.4.0..v0.4.1`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> Release contains bug-fixes, improved tools for evaluating fit results (`tfs-summarize-fit`) and improved documentation and examples.

**Reconstructed changes:**

### Added
- **Expanded `tfs-summarize-fit`.** It now writes a `summary/` directory containing `{out_prefix}_fit_summary.json` (loss, parameter count, and theta/growth prediction statistics), training and test theta correlation CSVs and a PDF (`_theta_corr.pdf`), a growth correlation plot (`_growth_corr.pdf`), a training-loss curve (`_losses.pdf`), per-genotype theta-fit plots and per-genotype growth-trajectory plots. When a `*_sim_parameters.csv` ground-truth file is present in the run directory, it annotates each `*_params_*.csv` (direct, `_params_d_*` per-mutation and `_params_epi_*` epistasis files) with a `ref` column and plots ref against median. `--ref_theta_file` supplies ground-truth theta explicitly; otherwise a `*_sim_genotype_theta.csv` in the run directory is used.
- **Posterior calibration diagnostics** in the new `tfmodel/analysis/error_calibration.py`:
  - PIT from samples or from stored quantiles (`pit_from_samples`, `pit_from_quantiles`)
  - `calibration_curve` and `pit_uniformity_test`
  - PIT-histogram and calibration-curve plots
  - `calibration_summary` for single runs
  - SBC rank statistics (`compute_sbc_ranks`, `summarize_sbc`)

  `tfs-summarize-fit` uses `calibration_summary` when ground truth is available.
- **Pre-split observations.** An optional `presplit_df` holds the sequencing aliquot taken at t = −t_pre, with columns `replicate, condition_pre, genotype, ln_cfu, ln_cfu_std`. It enters the model as a `presplit_obs` Normal likelihood directly on `ln_cfu0` (`PreSplitData` in `data_class.py`). Pass it with `tfs-configure-model --presplit_df`; the path is recorded under `data: presplit` in the config. Genotypes absent from the growth data are dropped with a message.
- **Simulated pre-split data.** `tfs-simulate` writes a presplit CSV when a `presplit_data` block (optional `noise`) is present in the simulate config. The CSV includes an `ln_cfu_0_true` ground-truth column.
- **Perturbation-based theta simulation for `hill_geno` and `hill_mut`.** These components gain `simulate()`, a `SimPriors` class and `get_sim_hyperparameters()`, configured by the new `theta_sim_priors` simulate-config block:
  - a wild-type reference curve
  - per-genotype or per-mutation perturbation widths
  - `hill_geno` mixture fractions for stuck-bound, never-binds and inverted phenotypes
  - `hill_mut` pairwise epistasis drawn from a regularized horseshoe; `epi_tau_scale: 0.0` disables epistasis
- **Measured Hill parameters for binding genotypes.** The new `simulate/binding_params.py` reads a per-genotype CSV (`genotype, theta_low, theta_high, log_hill_K, hill_n`) set by `binding_data.genotype_params_file`. The measured curves are injected into the growth simulation (`build_theta_gc_override_hill_geno`, `build_theta_gc_override_hill_mut`). For `hill_mut`, singles are converted to per-mutation deltas and the curves are assembled for the whole library. This is only supported for Hill-based theta components.
- **Stratified binding-genotype sampling.** Binding genotypes can be chosen by greedy-maximin selection over a pool of prior-predictive theta curves (`sample_theta_stratified` in `simulate/sample_theta.py`), so their curves are diverse and span the binding concentrations. Pool size is set by `binding_stratify_pool_size` (default 500).
- **`dk_geno_zero`** simulate-config key pins every genotype's `dk_geno` to 0. When it is set, the `dk_geno_hyper_*` keys become optional.
- **Unknown-key validation.** Simulate and tfmodel YAML configs now fail on unrecognized top-level keys (`check_unknown_keys`, `SIMULATE_KNOWN_KEYS`, `TFMODEL_KNOWN_KEYS`).
- **Pre-fit scale bounds.** `tfs-prefit-calibration` gains `--k_scale_floor`, `--m_scale_floor`, `--k_scale_ceiling` and `--m_scale_ceiling`. The calibrated k and m prior scales are now derived from the Hessian and clamped to these bounds (`_build_hessian_scale_updates`).
- **Trajectory plotting module.** `tfscreen.plot.geno_trajectory` now holds `plot_geno_trajectory` (plots from a prediction DataFrame), `predict_geno_trajectory_df` and `predict_and_plot_geno_trajectory`. Trajectory plotting was previously embedded in the pre-fit calibration script.
- **Library spec shorthand.** `LibraryManager` now strips internal whitespace from `wt_seq`, `degen_sites` and `sub_libraries`, and accepts `.` in `degen_sites` and `spiked_seqs` to mean "wild-type base at this position".
- **Documentation and examples.**
  - New `docs/source/quickstart.rst`.
  - New `docs/source/summarize-fit.rst`, a guide to every `tfs-summarize-fit` output.
  - `analysis.rst` expanded with the pre-fit output files, the parameter CSVs and the quantile-column convention.
  - New `examples/simulate-and-analyze/` (`simulate_config.yaml`, `hill_params.csv`, `run.sh`, `README.md`) running the full simulate → configure → prefit → fit → sample → extract → predict → summarize pipeline.
  - `examples/simulate/simulate_config.yaml` now documents the thermo and Hill theta components, `theta_sim_priors`, `dk_geno_zero` and `presplit_data`.

### Changed
- **Breaking: quantile columns renamed.** `tfs-extract-params`, `tfs-predict-theta` and `tfs-predict-growth` now write 17 quantile columns named `q<level>` (`q0.001` … `q0.5` … `q0.999`, including `q0.159` and `q0.841` for ±1σ). The previous named columns (`min`, `lower_95`, `lower_std`, `lower_quartile`, `median`, `upper_quartile`, `upper_std`, `upper_95`, `max`) are gone. MAP-checkpoint input now yields a single `q0.5` column instead of `point_est`. `tfs-cat-response` now reads `q0.5` by default and derives sigma from `(q0.841 − q0.159)/2`. **Action:** update any downstream scripts that read the old column names.
- **Breaking: posterior sampling removed from `tfs-fit-model`.** The `num_posterior_samples`, `sampling_batch_size` and `always_get_posterior` options are gone. Run `tfs-sample-posterior` after fitting. `tfs-sample-posterior` now restores the SVI state directly from the checkpoint (`RunInference.restore_svi_from_checkpoint`) instead of re-entering the optimizer with zero epochs. `run_optimization` also writes a checkpoint when a run stops before converging.
- **Breaking: simulate-config seed keys unified.** `random_seed` and `theta_rng_seed` are replaced by a single `seed` key, and `tfs-simulate` gains `--seed`. Because of the new unknown-key validation, a config that still uses the old keys now errors. **Action:** rename these keys in existing simulate configs.
- **Breaking: `final_cfu_pct_err` removed** from the simulate config. Simulated `sample_cfu_std` is now 0.
- **Default output prefixes.** `tfs-predict-theta` now defaults to `tfs_pred_theta` (was `tfs_theta_pred`) and `tfs-predict-growth` to `tfs_pred_growth` (was `tfs_growth_pred`). The default `tfs-simulate` output prefix is now `tfs_sim_`.
- **Selection vs. control priors on the slope `m` (`linear` growth).** Each condition's name is checked for `+` (selection) or `-` (control). Control conditions get a tight `m_scale_minus` (0.001) prior on `m` and selection conditions get `m_scale_plus` (0.01). This keeps the optimizer from moving selection-phase theta signal into control conditions, a cause of theta undershoot. `ModelOrchestrator` passes `condition_labels` to any component whose `get_priors` accepts them. A condition name containing both `+` and `-`, or neither, raises a `ValueError`.
- **Growth prior defaults.**
  - `linear` and `power`: `k_loc` 0.025 → 0.020.
  - `linear`: `m_scale` 0.05 → 0.01.
  - `saturation`: `min_loc` 0.025 → 0.020 and `max_loc` 0.025 → 0.030; the initial guesses change to match.
- **Per-genotype curves in the pre-fit calibration.** The `_simple` theta component used by the pre-fit now accepts per-genotype `(T, C, G)` theta values, so each calibration genotype keeps its own observed binding curve instead of a population average. `_simple` is now registered in `model_registry["theta"]` (calibration-only).
- **Binding/growth mismatch is no longer fatal.** `binding_df` genotype/titrant pairs that are missing from `growth_df` are now dropped with a per-titrant message; previously this raised an error.
- **Clearer stale-guesses error.** A guesses CSV whose parameter sizes no longer match the model now raises a `ValueError` telling you to regenerate it.
- **Module moves and renames.** Console-script names are unchanged.
  - `run_prefit_calibration_cli.py` → `prefit_calibration_cli.py`
  - `prior_predictive_cli.py` → `sample_prior_cli.py`
  - `simulate/scripts/run_simulation_cli.py` → `simulate_cli.py`
  - `analysis/cat_response/cat_response_cli.py` → `analysis/cat_response/scripts/cat_response_cli.py`
  - `mle/stats_test_suite.py` → `analysis/stats_test_suite.py`, now exported from `tfscreen.analysis`
- **Internal renames.** Code that took a `gm` model object now uses `orchestrator` throughout (e.g. `write_configuration(orchestrator=...)`, `predict(orchestrator=...)`). The test tree now mirrors `tfmodel/generative/components/`.

### Removed
- `analysis/cat_response/fit_response_cli.py`, which was not registered as a console script; use `tfs-cat-response`.
- `tfmodel/analysis/sbc.py`; its SBC functions now live in `tfmodel/analysis/error_calibration.py`.

### Fixed
- **Genotypes scrambled in growth predictions.** `predict` used the static binding-first `batch_idx`, which misaligned posterior values (stored in canonical genotype order) with per-genotype masks and offsets. Prediction now builds a canonical-order batch with `get_batch(data, arange(num_genotype))`. `_setup_batching` also fills the non-binding positions of the static index with valid genotype indices instead of zeros.
- **MAP predictions corrupted by a double transform.** MAP (`AutoDelta`) parameters are already stored in constrained space, but `predict` applied the support bijection again, corrupting positive-constrained sites such as `ln_cfu0_hyper_scale`. The extra transform is removed.
- **Misplaced values for non-tail-aligned prediction sites.** Sites whose dimensions are not tail-aligned with the tensor layout (e.g. `ln_cfu0`, shape `(rep, condition_pre, genotype)`) were assigned to the wrong rows. The new `_align_site_to_tm_dims` maps them correctly.
- **Pinned hyperpriors ignored at prediction.** `predict` now uses the original orchestrator's priors, so calibrated hyperpriors stay pinned; previously a rebuilt orchestrator with default priors was used.
- **Empty `condition_rep` groups.** The `linear` growth component and the `memory` growth-transition component no longer fail extraction when the `condition_rep` map group is absent.
- **`theta_sim_priors` rejected by the simulator.** The key was missing from the recognized simulate-config keys and is now accepted.
- **Integer time columns.** `t_pre` and `t_sel` are cast to float on load, avoiding dtype problems when merging with float prediction grids.

## [0.4.0] - 2026-05-31

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff 99b77e0..v0.4.0` (i.e. changes after the untagged 0.3.1 bump), and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> Reorganized code around three core functions: running the hierarchical model, simulating an experiment, and processing raw experimental data. Standardized command line interface, yaml inputs, and model naming conventions. Lots of bug fixes throughout.

**Reconstructed changes:**

### Added
- **Thermodynamic theta models.** Lac-dimer and MWC-dimer partition-function theta components, with and without an unfolded state, each in three parameterizations: per-mutation lnK (`PK`), structure-based neural-net prior (`PnnC`), and ddG prior (`PddG`). They are registered under standardized names such as `thermo.O2_C4_K3_U0_a.PK` and `thermo.O2_C12_K5_U1_a.PddG`. Supporting pieces are `scripts/generate_struct_ensemble.py` and a sparse `mut_geno_matrix`.
- **New model components:**
  - `activity`: `horseshoe_mut` (horseshoe epistasis prior).
  - `ln_cfu0`: `hierarchical_factored`.
  - `theta_rescale` (new category): `passthrough`, `logit`.
  - `theta_growth_noise`: `logit_normal`.
  - `growth_noise` (new category): `zero`, `normal_kt`.
  - `sample_offset` (new category): `zero`, `normal`.
  - `growth_transition`: `baranyi_k`, `baranyi_tau`, `two_pop`.
- **Two-stage calibration.** `tfs-prefit-calibration` pre-fits the growth linking function by MAP and pins the hyperpriors (empirical Bayes) for the production fit.
- **Empirical initial guesses** for `ln_cfu0`, `dk_geno`, the growth components and the Hill components. `ln_cfu0` also accepts spiked-genotype priors and more than one prior class.
- **Inference options:**
  - NUTS is available from the command line (`tfs-fit-model --analysis_method nuts`, with `--nuts_*` options).
  - MAP fits can produce Hessian-based Laplace posteriors (`tfs-sample-posterior`, chunked through `hessian_chunk_size`).
  - Models can be fit on binding data alone (`binding_only`).
  - Checks for exploding NaNs were added, along with the `tfs-diagnose-nan` script.
- **Prior predictive and simulation-based calibration.** Adds `tfmodel/analysis/prior_predictive.py` (`tfs-sample-prior`) and `tfmodel/analysis/sbc.py` (`tfs-summarize-sbc`).
- **Prediction and extraction:**
  - `tfs-predict-growth` and `tfs-predict-theta` take over from `tfs-predict`. `tfs-predict-theta` supports categorical theta models and predicts unmeasured genotypes (`tfmodel/analysis/predict_unmeasured.py`).
  - `tfs-extract-params` reads parameters directly from a checkpoint `.pkl` without sampling, and uses a point-estimate/median output contract.
  - Individual posterior samples can be pulled during prediction and extraction.
- **Grid and summary tooling:**
  - `tfs-setup-grid` and `tfs-summarize-grid` handle grids of model configs; `tfs-setup-sim-grid` handles grids of simulations. Both rely on the shared helpers in `util/grid_utils.py` (Jinja2 templates, which adds `jinja2` as a new dependency).
  - `tfs-summarize-fit` summarizes a fit, including a fit-summary plot.
  - `tfs-subset-genotypes` subsets genotype data.
  - `tfs-cat-response` is registered as a CLI.
- **Simulation:**
  - The `tfs-simulate` CLI (`simulate/scripts/run_simulation_cli.py`).
  - Prior-predictive theta sampling (`simulate/sample_theta.py`).
  - A `SimData` container (`simulate/sim_data_class.py`).
  - Simulation activity, `dk_geno`, growth-transition and thermodynamic theta are now shared with or synchronized to the `tfmodel` components.
- `genetics/count_mutation_backgrounds.py` (mutation background counter) and `plot/plot_theta_fits.py`.
- `examples/` gained reference configs and templates: `simulate/`, `tfmodel/` and `process_raw/`.
- New docs pages (`grid.rst`, `process-raw.rst`, `ligandmpnn-features.rst`) and a GitHub Actions test workflow (`.github/workflows/tests.yml`).

### Changed
- **Breaking: package reorganized** around three core functions: the hierarchical model, simulation and raw-data processing.
  - `analysis/hierarchical/growth_model/` → `tfmodel/`:
    - `model.py`, `registry.py`, `components/` and `observe/` → `tfmodel/generative/`.
    - `model_class.py` → `tfmodel/model_orchestrator.py`.
    - `configuration_io.py` and `data_class.py` → `tfmodel/`.
    - `prediction.py` and `extraction.py` → `tfmodel/analysis/`.
    - Scripts → `tfmodel/scripts/*_cli.py`.
  - `analysis/hierarchical/{run_inference,posteriors}.py` → `tfmodel/inference/`, plus a new `checkpoint_io.py`.
  - `analysis/hierarchical/{tensor_manager,populate_dataclass}.py` and `growth_model/batch.py` → `tfmodel/tensors/`.
  - `fitting/` → `mle/`; `models/generic/` → `mle/curve_models/`.
  - `models/growth_linkage.py` and `models/transition_linkage.py` → `simulate/growth/`.
  - `data.py` → `genetics/data.py`.
  - `process_raw/{process_fastq,process_counts}.py` → `process_raw/scripts/*_cli.py`.
- **Breaking: CLI renamed and standardized.** Every entry point now lives in a `<name>_cli.py` file and is built with `generalized_main`.
  - `tfs-configure-growth-analysis` → `tfs-configure-model`.
  - `tfs-growth-analysis` → `tfs-fit-model`.
  - `tfs-predict` → `tfs-predict-growth` / `tfs-predict-theta`.
  - `tfs-summarize-posteriors` → `tfs-summarize-fit`, `tfs-extract-params` and `tfs-sample-posterior`.
- **Breaking: component names standardized** on a `_geno`/`_mut`/`thermo.*` scheme:
  - `activity`: `hierarchical` → `hierarchical_geno`, `horseshoe` → `horseshoe_geno`.
  - `dk_geno`: `hierarchical` → `hierarchical_geno`.
  - `theta`: `hill` → `hill_geno`, `categorical` → `categorical_geno`.
- **Breaking: YAML inputs standardized.**
  - `read_yaml` no longer converts whole-number floats to ints; only quoted scientific-notation strings are converted.
  - `read_yaml` `override_keys` rejects unknown keys.
  - `read_yaml` validates that the `growth` and `condition_blocks` sections agree.
- **Breaking:** `sample_df` uses `sample_cfu` instead of OD600, and `process_raw/od600_to_cfu.py` is removed.
- `tfs-configure-model` uses a default batch size of 1024.
- Epistasis is represented with a sparse matrix, and epistasis prediction runs in batches to avoid running out of memory.
- `read_dataframe` converts `genotype` columns to categoricals automatically.
- Hierarchical-model parameter names are standardized.
- The convergence criteria were updated.
- All positive constraints get a small numerical floor.
- Requires Python >= 3.11 and `numpyro>=0.19.0`. `patsy` was dropped from the dependencies.

### Removed
- **Breaking:** the independent (non-hierarchical) analysis (`analysis/independent/`) and the MLE calibration package (`calibration/`).
- **Breaking:** the `models/` package, which held the old growth regressions (`ols`, `wls`, `gls`, `glm`, `gee`, `nls`, `kf`, `ukf`), `lac_model/` and `eee_model.py`. The rest of it moved (see above).
- **Breaking:** these components:
  - `condition_growth`: `linear_independent`, `linear_fixed`.
  - `dk_geno`: `hierarchical_mut`.
- `fitting/fitters/ols_2D.py`, `wls_2D.py`, `simulate/setup_observable.py` and `util/design.py`.
- The `notebooks/` directory.

### Fixed
- Genotype labels in the binding data were being swapped during fits.
- `tfs-prefit-calibration` was upweighting the binding data.
- Posterior writing overran its buffer.
- Hessian computation ran out of memory on GPUs.
- MAP "posteriors" were wrong, and posterior generation required a seed.
- Indexing bugs, including growth data with discontinuous indexes (`get_scaled_cfu`) and indexing in `run_inference`.
- The `epistasis` setting was dropped from the config.
- Guide bug in the `growth_transition` components; NaN in the memory-model term (now clamped).
- Batching bugs in the `_mut` components; collapse in the `horseshoe_mut` activity prior.
- Guesses for some MWC-dimer models under `binding_only`.
- Out-of-memory errors in the lac-dimer mutation and NN-prior theta models.

## [0.3.1] - 2026-03-31 (untagged)

> *Reconstructed on 2026-09-10 from `git diff v0.3.0..99b77e0`, the untagged commit that bumped `__version__` to 0.3.1 ("version bump; ready for pyro port"), and commit messages. These entries were not written at release time and may be incomplete.*
>
> *This version was never tagged or released on GitHub; it is recorded because `__version__` reported 0.3.1 between 2026-03-31 and 2026-05-31.*

**Reconstructed changes:**

### Added
- **Mutation-level and epistasis inference:**
  - `theta`: `hill_mut`.
  - `activity`: `hierarchical_mut`.
  - `dk_geno`: `hierarchical_mut`.
- **Growth-transition components** (`instant`, `memory`, `baranyi`) under a new `growth_transition` category.
- **Nonlinear condition-growth models:** `power` and `saturation`.
- **New `transformation` components:** `empirical` (empirical transformation correction) and `logit_norm`.
- **New CLIs:**
  - `tfs-configure-growth-analysis` (config writing and reading through `configuration_io.py`).
  - `tfs-predict`, which predicts `ln_cfu` from parameter samples (`prediction.py`).
  - Parameter extraction split out into `extraction.py`.
- A MAP optimization can run before SVI, and checkpoints are protected from being overwritten by accident.
- Growth parameters can be shared across replicates.
- Base documentation.
- Growth-linkage modules (`models/growth_linkage.py`, `transition_linkage.py`, `occupancy_growth_model.py`), plus `genetics/build_mut_geno_matrix.py`.

### Changed
- **Breaking: CLIs replaced.**
  - `tfs-hier-theta` → `tfs-growth-analysis`.
  - `tfs-summarize-posteriors` moved to `growth_model/scripts/`.
  - `tfs-indep-theta` was removed from the entry points.
- **Breaking: components reorganized** into per-category subpackages (`components/<category>/<name>.py`), with these registry renames:
  - `condition_growth`: `hierarchical` / `independent` / `fixed` → `linear` / `linear_independent` / `linear_fixed`.
  - Noise: `none` → `zero`.
  - `transformation`: `congression` replaced by `empirical` / `logit_norm`.
- **Breaking:** the `condition` plate and mapper are renamed to `condition_rep`.
- The categorical theta component was modernized.
- The MLE calibration routine and FASTQ parsing were improved.

### Removed
- **Breaking:** `.npz` posterior files; posteriors are `.h5` only.

### Fixed
- Overflow in long inference runs.
- `.h5` posterior file sync on clusters.
- Indexing bugs in prediction.

## [0.3.0] - 2026-01-02

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.2.2..v0.3.0`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> This release includes updates to the posterior sampling and summarization code. It also switches to a more consistent run interface, switching from number of iterations to number of epochs for SVI and MAP inference.

**Reconstructed changes:**

### Added
- **Posterior summarization CLI** `tfs-summarize-posteriors` (`analysis/hierarchical/summarize_posteriors.py`): reads a posterior file plus the run's `{out_root}_config.yaml` and writes per-parameter CSVs, `{out_root}_growth_pred.csv`, and (for the `hill` theta model) `{out_root}_theta_curves.csv`. `tfs-hier-theta` now calls it automatically after posterior sampling.
- New `GrowthModel` methods `extract_growth_predictions` and `extract_theta_curves`, plus `write_config`/`load_config` for a YAML record of the run's data and model settings.
- `tfs-hier-theta` flag `--config_file`: re-runs from a saved config. `growth_df`, `binding_df`, and `seed` are now optional (`seed` is required unless resuming from a checkpoint).
- `tfs-hier-theta` flag `--transformation_model` (`congression` (default) | `single`) exposes the transformation component on the command line.
- Epoch-based convergence controls `--patience` (consecutive passing checks required) and `--convergence_check_interval` (in epochs).
- End-to-end smoke tests (`tests/smoke-tests/`) and unit tests for batch consistency, posterior mapping, and posterior extraction.

### Changed
- **Breaking:** **Epoch-based run interface for SVI and MAP.** `--num_steps` was replaced by `--max_num_epochs`. `--convergence_window` and `--checkpoint_interval` are now counted in epochs (one epoch is `ceil(num_genotypes / batch_size)` iterations), and the defaults changed (`convergence_tolerance` 1e-6 → 0.01, `convergence_window` 10000 → 10, `checkpoint_interval` 10000 → 10). The learning-rate decay now spans the full run.
- **Breaking:** Posterior samples are now written to HDF5 (`{out_root}_posterior.h5`, via `h5py`) instead of `.npz`. The summarization code still reads `.npz`.
- **Breaking:** `RunInference.setup_map` was removed. MAP now uses `setup_svi(guide_type="delta")` (`AutoDelta`), and SVI uses `guide_type="component"`.
- Default theta noise models are now `none` for both `--theta_growth_noise_model` and `--theta_binding_noise_model` (previously `beta`).
- Posterior prediction now uses better batching and has a progress bar. The congression correction reuses its grid integration across genotypes (`n_grid` 512 → 256), and the default `lam_scale` prior changed from 1.0 to 0.5.
- `pyproject.toml` now lists the inference stack as dependencies (`jax`, `jaxlib`, `numpyro`, `optax`, `flax`, `dill`, `h5py`, `scipy`).

### Fixed
- Fixed genotype indexing errors in posterior sampling, including when reading from a checkpoint, and added guards so indexing is consistent in batched and full-batch modes.
- Fixed an out-of-memory error in `run_inference`.
- Fixed an overflow error in `model_class`.
- `fitting.fitters.least_squares` now warns and returns NaN when SVD fails to converge instead of raising. `ols_2D` and `stats_test_suite` now handle zero-variance and constant inputs.

## [0.2.2] - 2025-12-18

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.2.1..v0.2.2`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> - Added correction to inferred theta values accounting for the fact cells can pick up more than one plasmid
> - Refactored the util submodule from flat to hierarchical
> - Tweaks to increase speed of inference
> - Updated command line interface to analyze_theta to interface with the model changes
> - Lots of docstring clean up
> - Added a bunch of unit tests
> - Various bug fixes

**Reconstructed changes:**

### Added
- **Multi-plasmid (congression) correction of theta.** A new `transformation` category was added to the hierarchical model registry, with components `congression` (`transformation_congression.py`) and `single` (`transformation_single.py`). `congression` corrects inferred theta for cells carrying more than one plasmid by integrating over a logit-normal background theta distribution with co-transformation rate `lam`. `GrowthModel` uses `transformation="congression"` by default.
- `tfs-hier-theta --spiked <genotype> [...]` (`spiked_genotypes` in `GrowthModel`): lists genotypes that are excluded from the congression correction. Unknown genotypes raise an error.
- `--adam_final_step_size` for `tfs-hier-theta`. The Adam step size now decays exponentially from `--adam_step_size` to this value.
- A human-readable `{out_root}_losses.txt` loss log.
- A large set of unit tests, covering `util/` and the hierarchical model among others.

### Changed
- **Breaking:** **`tfscreen.util` restructured from flat modules into subpackages:**
  - `util/cli/`: `generalized_main`
  - `util/dataframe/`: `add_group_columns`, `check_columns`, `chunk_by_group`, `df_to_arrays`, `expand_on_conditions`, `get_group_mean_std`, `get_scaled_cfu`
  - `util/io/`: `read_dataframe`, `read_yaml`
  - `util/numerical/`: `array_search`, `broadcast_args`, `xfill`, `zero_truncated_poisson`, plus `padding.py` (was `vstack_padded.py`) and `transform.py` (was `numerical.py`)
  - `util/validation/`: `check` (was `check.py`)
  - Top-level re-exports from `tfscreen.util` are preserved. Direct imports of old module paths, such as `tfscreen.util.generalized_main`, must be updated.
- **Faster inference.** Optimization steps now run in `jax.lax.scan` blocks of `checkpoint_interval` steps with vmapped batch extraction, replacing the per-step Python loop.
- New `tfs-hier-theta` defaults: `--adam_step_size` 1e-6 → 1e-3, `--elbo_num_particles` 10 → 2.
- `analysis_method=posterior` now disables batching.
- Docstrings were cleaned up throughout the hierarchical model.

### Removed
- **Breaking:** `tfscreen.analysis.growth_to_theta` and `tfscreen.fitting.fitters.map` (`run_map`) were removed.

### Fixed
- Fixed posterior-sampling indexing: latents are now sampled against a full-genotype batch, and batch outputs are concatenated on the last axis. `not_binding_idx` is now a pytree array.
- Fixed `fitting.fit_manager` scale-parameter masking, error-message propagation, and robustness in the `stats_test_suite` Breusch-Pagan test. Other small fixes found by the new tests in `fitting/`, `models/growth/`, `calibration/io.py`, `cat_response`, and `plot/cat_fits.py`.

## [0.2.1] - 2025-12-15

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.2.0..v0.2.1`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> + Hierarchical modeling analysis now complete.
> + Cleaned up analysis and demo notebooks.

**Reconstructed changes:**

### Added
- **Hand-written variational guides for every hierarchical model component.** Each module in `analysis/hierarchical/growth_model/components/` (`growth_independent`, `growth_hierarchical`, `growth_fixed`, `ln_cfu0`, `dk_geno_fixed`, `dk_geno_hierarchical`, `activity_fixed`, `activity_hierarchical`, `activity_horseshoe`, `theta_cat`, `theta_hill`, `no_noise`, `beta_noise`) and both observers (`observe/binding.py`, `observe/growth.py`) now define a `guide()` function alongside `define_model()`. Hyperparameter scales use LogNormal guides with positive-constrained `pyro.param` sites. The growth observer's guide learns the Student-t degrees of freedom `nu`, and `beta_noise` learns `kappa` with an amortized Beta guide.
- **Component registry** (`growth_model/registry.py`): a `model_registry` dict that maps each category (`condition_growth`, `ln_cfu0`, `dk_geno`, `activity`, `theta`, `theta_growth_noise`, `theta_binding_noise`) to its component modules, plus `observe_binding`/`observe_growth`.
- **GPU-resident mini-batching with likelihood rescaling.** `GrowthModel` takes a `batch_size` argument. Binding genotypes are always included in each batch, and the remaining slots are filled with random non-binding genotypes. A per-genotype `scale_vector` (applied via `pyro.handlers.scale`) reweights the sampled genotypes so the batch stands in for the full library. `GrowthData`/`BindingData` gained `batch_size`, `batch_idx`, `scale_vector` and `geno_theta_idx` fields. `RunInference` moves the data to the device once with `jax.device_put` and slices each batch there with a jitted `get_batch`.
- `GrowthModel.jax_model_guide`, `GrowthModel.get_batch` and `GrowthModel.get_random_idx`. `RunInference` now requires the model to provide `jax_model_guide`.
- Validation in `_read_binding_df`: it raises `ValueError` if `binding_df` contains genotype/titrant_name pairs that are not in `growth_df`.
- Unit tests for `run_inference` and `analyze_theta` (`tests/tfscreen/analysis/hierarchical/test_run_inference.py`, `test_analyze_theta.py`). Hierarchical-model tests were updated to match the new structure.
- New notebooks: `notebooks/monoculture-calibration/` (growth calibration from control experiments), `notebooks/development-tools/` (`distributions`, `run-nuts_single`, `check-for-int-in-svi`), `notebooks/create-test-data/`, `notebooks/tune-ensemble-parameters/` (lac model behavior, with AlphaFold/experimental structures), and several `notebooks/dev/` notebooks. Test and coverage badges were added under `docs/badges/`.

### Changed
- **Hierarchical model restructured around the registry.** `jax_model` (`growth_model/model.py`) no longer dispatches through integer flags and `if/elif` chains. It receives the selected component modules as keyword arguments plus an `is_guide` flag, and `GrowthModel` binds these with `functools.partial` to build both the model and the guide from the same function.
- **SVI and MAP use the component guide instead of numpyro autoguides.** `RunInference.setup_map`/`setup_svi` now pass `model.jax_model_guide` to `SVI`, replacing `AutoDelta` and `AutoLowRankMultivariateNormal`. As a result, the `guide_rank` argument was removed from `setup_svi`, `_run_svi` and `analyze_theta`.
- `batch_size` moved from `RunInference.run_optimization`/`_run_map`/`_run_svi` to the `GrowthModel` constructor. `analyze_theta` passes it through.
- **Growth tensor layout.** The combined `treatment` axis was split into separate `condition_pre`, `condition_sel`, `titrant_name` and `titrant_conc` axes, giving a 7-D tensor with genotype last. `condition_sel` is re-indexed within each `condition_pre` (`condition_sel_reduced`). The separate growth-theta TensorManager and the `map_ln_cfu0`/`map_genotype`/`map_theta`/`map_theta_group` tensors are gone, and `titrant_conc`/`log_titrant_conc` are now 1-D arrays taken from the tensor axis labels. The growth observer's plates were updated to match, and `ln_cfu0` is sampled over replicate × condition_pre × genotype plates.
- The wild-type location changed from `wt_index`/`not_wt_mask`/`num_not_wt` to a `wt_indexes` array.
- Posterior sampling (`RunInference.get_posteriors`) slices the genotype-indexed latent samples to each forward-prediction batch before running `Predictive`, instead of running the forward pass on unsliced latents.
- `analyze_theta` defaults: `activity_model` changed from `"fixed"` to `"horseshoe"`, and the default `convergence_tolerance` in `_run_svi` changed from `1e-7` to `1e-4`.
- `populate_dataclass` converts numpy arrays to JAX arrays, and `TensorManager` builds `good_mask` with an explicit `bool` dtype.

### Removed
- `ControlClass` (from `growth_model/data_class.py`), together with the `MODEL_COMPONENT_NAMES` table and the integer component constants in `growth_model/model.py`. The registry replaces them.
- `sample_batch` and `deterministic_batch` in `growth_model/batch.py`. Both are replaced by `get_batch(full_data, idx)`.

### Fixed
- `simulate/selection_experiment.py`: the growth-rate noise standard deviation is now `abs(mean * growth_rate_noise)`, so a negative mean growth rate no longer produces an invalid (negative) noise scale.
- `calibration/plot.py`: fixed a broken import of `get_indiv_growth` by importing it from `tfscreen.analysis.independent.get_indiv_growth`.

## [0.2.0] - 2025-11-26

> *Reconstructed on 2026-09-10 from the GitHub release notes, `git diff v0.1.0..v0.2.0`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> Code now does:
>
> - Simulation of experiments
> - Processing raw reads (fastq through ln_cfu/condition)
> - Independent modeling (extract that for each genotype independently)
> - Hierarchical modeling (Bayesian analysis of theta together)
> - Epistasis analysis
> - Epistasis vs. [titrant] categorization analysis
> - Various helpful plotting utilities.
>
> Additionally, the API is now (roughly) stable. I antiquate no major changes to the simulation, raw-processing, or independent modeling code base. I expect to build out the hierarchical modeling codebase and clean up the utilities functions. None of these will be architectural changes, however, just minor refactors.

**Reconstructed changes:**

### Added
- **Hierarchical Bayesian theta inference** (`tfscreen.analysis.hierarchical`): a JAX/NumPyro/Flax joint growth + binding model (`growth_model/model.py`) that infers per-genotype operator occupancy (theta) together with condition growth, `ln_cfu0`, and pleiotropic `dk_geno` effects. Components in `growth_model/components/` are selectable per axis: condition growth (`hierarchical`, `independent`, `fixed`), `dk_geno` (`hierarchical`, `fixed`), activity (`fixed`, `hierarchical`, `horseshoe`), theta (`hill`, categorical `theta_cat`), and theta noise (`beta`, none). Separate observation layers for growth and binding data live in `growth_model/observe/`.
- **`RunInference`** (`analysis/hierarchical/run_inference.py`): SVI (low-rank autoguide, `ClippedAdam`) and MAP optimization with mini-batching, convergence detection over a loss window, checkpoint write/restore, loss logging, posterior sampling, and prediction. `TensorManager` (`tensor_manager.py`) and `populate_dataclass.py` pack ragged per-genotype data into JAX tensors.
- **`tfs-hier-theta`** CLI: wraps `analysis.hierarchical.analyze_theta` (growth + binding DataFrames, component choices, `analysis_method` `svi`/`map`, checkpoint restart).
- **Independent per-genotype analysis** (`tfscreen.analysis.independent`) with the **`tfs-indep-theta`** CLI: per-genotype growth-rate estimation (`get_indiv_growth`), pre-growth modeling (`model_pre_growth`), and direct counts-to-theta fitting (`cfu_to_theta`); plus `analysis/growth_to_theta.py`.
- **Raw read processing** (`tfscreen.process_raw`): paired FASTQ to genotype counts (`FastqToCounts`, `process_fastq`, multiprocess chunked; uses `pyfastx` and `pybktree`), counts to `ln_cfu` (`counts_to_lncfu`, `process_counts`), and OD600-to-CFU/mL conversion (`od600_to_cfu`). CLIs **`tfs-process-fastq`** and **`tfs-process-counts`**.
- **Genetics utilities** (`tfscreen.genetics`): `LibraryManager` (library definition from wt sequence, degenerate sites, tiles), genotype standardization/sorting (`standardize_genotypes`, `argsort_genotypes`, `set_categorical_genotype`), mutant-cycle construction (`build_cycles`), and `combine_mutation_effects`.
- **Epistasis analysis**: `analysis/extract_epistasis.py` (`extract_epistasis`, `mutant_cycle_pivot`).
- **Categorical response fitting** (`analysis/cat_response/`): `cat_response`/`cat_fit` fit a library of empirical curve shapes (flat, linear, polynomial, Hill repressor/inducer, bell, biphasic peak/dip) defined in `models/generic/` (`MODEL_LIBRARY`).
- **Plotting subpackage** (`tfscreen.plot`): heatmaps (amino acid vs residue, amino acid vs titrant, epistasis), corner plots, Hill and categorical fit plots, error/uncertainty-calibration plots, and x-y correlation plots.
- **General fitting framework** (`tfscreen.fitting`): `FitManager`, patsy formula parsing (`parse_patsy`), `predict_with_error`, and fitters in `fitting/fitters/` (least squares, MAP with configurable priors, matrix NLS/WLS, vectorized 2D OLS/WLS).
- **Shared utilities** in `tfscreen.util`, including `generalized_main` (builds a CLI from a function signature), `read_yaml`, DataFrame/column validation, and `zero_truncated_poisson`.

### Changed
- **Simulation rewritten** around `library_prediction` (thermodynamics to per-genotype theta and growth, `thermo_to_growth.py`) and `selection_experiment` (transformation, mixing, index hopping, growth, and sequencing); the simulation uses `LibraryManager` to generate libraries. The old step modules (`generate_libraries`, `initialize_population`, `transform_and_mix`, `simulate_growth`, `sequence_samples`, `load_simulation_config`, `cell_growth_moves`) were removed.
- `tfscreen.analyze` renamed to `tfscreen.analysis`; phenotype models moved from `simulate/generate_phenotypes/` to `tfscreen.models` (`lac_model/`, `eee_model.py`), and time-series growth estimators (OLS/WLS/GLS/GEE/GLM/NLS/Kalman/UKF) moved from `fitting/` to `models/growth/`.
- Calibration rebuilt (`calibration/`): separate wt growth-rate, wt theta, background, and k-vs-theta steps (`get_wt_k`, `get_wt_theta`, `get_background`, `get_k_vs_theta`) plus calibration plotting.
- Dependencies: `eee` and `dataprob` dropped; `numba`, `patsy`, `pybktree`, `pyfastx`, `corner`, and `statsmodels` added.

## [0.1.0] - 2025-09-02

> *Reconstructed on 2026-09-10 from the GitHub release notes, the source tree at tag `v0.1.0`, and commit messages. These entries were not written at release time and may be incomplete.*

**Release notes (contemporaneous, by the author):**

> This is a first draft of the code that allows simulation and analysis of a transcription factor high-throughput screen. Expect large changes to the API.

**Reconstructed changes:**

### Added
- Initial release of `tfscreen`, a library for simulating and analyzing high-throughput screens of transcription factor libraries. No command-line entry points; use is through the Python API and example notebooks.
- **Simulation** (`tfscreen.simulate`, `run_simulation` driven by a YAML config): library generation, population initialization, transformation and mixing, growth under selection conditions, and sequencing of samples.
- **Phenotype models** (`simulate/generate_phenotypes/`): lac repressor thermodynamic models (`lac_model/`: MWC dimer, microscopic dimer, linkage dimer and dimer-tetramer) and an `eee`-based ensemble model for mapping mutational effects to operator occupancy.
- **Analysis** (`tfscreen.analyze`): growth-rate estimation from time series (`estimate_growth_rates`) and per-genotype theta estimation (`estimate_theta`).
- **Growth calibration** (`tfscreen.calibration`): fitting of condition growth rates versus theta, with calibration read/write.
- **Fitting routines** (`tfscreen.fitting`): OLS, WLS, GLS, GEE, GLM, NLS, matrix NLS/WLS, and Kalman / unscented Kalman filter estimators.
- Walkthrough notebook (`notebooks/tfscreen-walkthrough/`).

[Unreleased]: https://github.com/harmslab/tfscreen/compare/v0.4.4...HEAD
[0.4.4]: https://github.com/harmslab/tfscreen/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/harmslab/tfscreen/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/harmslab/tfscreen/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/harmslab/tfscreen/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/harmslab/tfscreen/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/harmslab/tfscreen/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/harmslab/tfscreen/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/harmslab/tfscreen/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/harmslab/tfscreen/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/harmslab/tfscreen/releases/tag/v0.1.0
