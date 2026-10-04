# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription factor (TF) libraries. It models bacterial growth in plasmid-based libraries where TFs regulate selection markers (antibiotic resistance or pheS/4CP). The core analysis is a hierarchical Bayesian model (JAX/Numpyro) that infers per-genotype TF operator occupancy from growth data.

## Commands

### Testing

`NUMBA_DISABLE_JIT=1` is set automatically by `tests/conftest.py` — no prefix needed on the command line.

```bash
# Run all unit tests
~/miniconda3/bin/pytest tests/tfscreen

# Run a single test file
~/miniconda3/bin/pytest tests/tfscreen/tfmodel/test_model.py

# Run slow tests too
~/miniconda3/bin/pytest tests/tfscreen --runslow

# Run smoke tests
~/miniconda3/bin/pytest tests/smoke-tests --runslow

# Run with coverage
~/miniconda3/bin/coverage run --branch -m pytest tests/tfscreen --runslow
```

### Linting
```bash
# Fatal errors only
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full lint (non-fatal)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127
```

### CLI Entry Points
```
tfs-process-fastq          # FASTQ → read counts (library_config f1 f2 --out_dir; library YAML first)
tfs-process-counts         # per-tube counts → growth table (tube table + counts dir; tube totals from --od600_file/--od600_calibration_file/--tube_volume_mL or supplied sample_cfu; every library in one call; --presplit writes the presplit table)
tfs-calibrate-od600        # Fit an OD600-to-CFU/mL calibration from a dilution series of replicate readings + plate counts (process_raw/od600.py)
tfs-configure-model        # Generate YAML config template (+ library composition snapshot); --set_priors name=value ... sets scalar priors (held SDs, sigma_fixed); --growth_priors / --growth_priors_wt_rates set per-condition linear k/m priors where the pre-fit does not run (priors_edit.py); a library spike with no growth data is an error unless --allow_missing_spikes
tfs-prefit-calibration     # Pre-fit linking function via MAP
tfs-fit-model              # Main hierarchical Bayesian inference (--guide_type component|delta|auto_normal|auto_diagonal_normal|auto_multivariate_normal|auto_low_rank_multivariate_normal; --guide_rank, --guide_init_scale; convergence: --convergence_window_steps, --patience, --convergence_z, --loss_rtol, --param_tolerance, --adam_step_size -> --adam_final_step_size by --adam_step_size_cut; --adam_clip_norm opts back into elementwise gradient clipping, off by default; --init_from <params.npz> starts at an earlier MAP's {site}_auto_loc values, e.g. a level-offset fit from a zero-offset fit, since guesses cannot name sites outside the configured guesses; --stage_offsets auto|on|off, --staged_step_size: a fresh MAP with level tube offsets runs the staged MAP, inference/staged_map.py)
tfs-fit-genotypes          # Per-genotype MLE fits of the growth model (no congression correction)
tfs-sample-posterior       # Draw posterior samples from fitted model (MAP: --laplace auto|full|arrowhead|blocks|point; auto = full up to --laplace_max_params 20000 MAP parameters, arrowhead above; arrowhead writes {out_prefix}_held_directions.csv; blocks holds the shared parameters at the MAP; point = the MAP point, no Hessian; --skip_growth_observations: leave out growth_pred/growth_obs, most of the file on a large library)
tfs-sample-prior           # Draw prior predictive samples
tfs-extract-params         # Extract parameters from checkpoint
tfs-predict-growth         # Predict growth from fitted model
tfs-predict-theta          # Predict operator occupancy
tfs-predict-epistasis      # Joint second-order epistasis from the theta posterior (per-draw ep, then quantiles; captures cross-genotype posterior covariance the marginal tfs-extract-epistasis path drops)
tfs-cat-response           # Fit categorical response curves
tfs-extract-epistasis      # Calculate second-order epistasis from a long-form observable table (--scale add|mult|logit; --scale_constant rescales the transform before epistasis, e.g. -RT to put logit onto a free-energy scale)
tfs-compare-runs           # Cross-run agreement statistics for any quantile-summarized estimate -- predicted features (theta/growth/epistasis) or fitted parameters (log_hill_K/dk_geno/growth_k/k_ref) -- across N runs (seeds / k-fold dropouts). Reports raw numbers (rms_sd, overdispersion + p/q, n_present/n_runs); no thresholds, no grades. --index_by picks the entity, --group_by breaks it out further, --match_by overrides key detection
tfs-simulate               # Simulate a full experiment (config_file --out_prefix tfs_sim → {out_prefix}_{name}.csv)
tfs-build-empirical # Fit real data → empirical phenotype-generating distribution (Stages 1-2)
tfs-setup-sim-grid         # Set up grid of simulation runs
tfs-setup-grid             # Set up grid of model configs
tfs-summarize-grid         # Summarize grid results
tfs-summarize-fit          # Summarize a fitted model
tfs-summarize-calibration  # Pool posterior calibration (coverage, PIT, width, RMSE) across a tfs-setup-sim-grid grid, by arm and stratum; --baseline pairs runs fit to the same simulated data
```

Removed in 0.5.0: `tfs-process-presplit` (now `tfs-process-counts --presplit`), `tfs-diagnose-nan`, `tfs-subset-genotypes`, `tfs-report-cfu0`, `tfs-summarize-sbc` (the library function `error_calibration.summarize_sbc` remains). `docs/source/cli.rst` is the generated reference of every command's flags.

## Architecture

### Data Flow
```
FASTQ files
    → tfs-process-fastq → counts CSV
    → tfs-process-counts → ln_cfu DataFrame (CSV)
    → tfs-fit-model (with config YAML) → checkpoint .pkl (MAP/SVI)
    → tfs-sample-posterior → posterior samples .h5
    → tfs-predict-growth → growth predictions CSV
    → tfs-predict-theta  → theta predictions CSV
    → tfs-extract-params → per-parameter CSVs
    → tfs-summarize-fit  → summary plots/CSVs
```

### Source Layout (`src/tfscreen/`)

| Module | Responsibility |
|--------|---------------|
| `tfmodel/` | Core hierarchical Bayesian inference engine (the heart of the package) |
| `tfmodel/generative/` | Numpyro model definition, component registry, pluggable components |
| `tfmodel/inference/` | JAX/Numpyro sampling, MAP estimation, checkpoint I/O, posteriors |
| `tfmodel/tensors/` | Ragged-tensor management, JAX array population |
| `tfmodel/analysis/` | Prediction, parameter extraction, error calibration, prior predictive |
| `process_raw/` | FASTQ parsing, count normalization, ln_cfu calculation |
| `simulate/` | Full experiment simulation from thermodynamics to read counts |
| `simulate/growth/` | Growth/growth-transition linkage models for simulation |
| `analysis/` | Downstream statistical analysis of inference outputs (cat_response, extract_epistasis, compare_runs) |
| `mle/` | General-purpose MLE regression (FitManager, least squares, WLS, NLS) |
| `mle/curve_models/` | Empirical curve-fitting functions and MODEL_LIBRARY used by cat_response |
| `mle/fitters/` | Low-level fitter implementations (least_squares, matrix_nls, matrix_wls) |
| `genetics/` | Genotype library management, mutation effect combination |
| `plot/` | Visualization (heatmaps, corner plots, error plots) |
| `util/` | Shared IO, DataFrame ops, numerical helpers, validation, CLI |

### Core Analysis: `tfmodel/`

The hierarchical Bayesian inference engine. Key files:

- **`generative/model.py`** — Numpyro probabilistic model definition. The generative model is:
  `ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre·A·θ)·t_pre + (k_sel + dk_geno + m_sel·A·θ)·t_sel`
  where θ = operator occupancy, A = per-genotype TF activity, dk_geno = pleiotropic growth effect.
  With `transformation: mixture` each genotype's cells are split into classes (clean, plus K congressed classes built from fixed co-resident sets) with cell-level θ/A/dk_geno (θ by `congression_theta_rule`, default the homodimer partition function over equal plasmid shares; dk_geno by `congression_dk_rule`, default dilution); every class is grown through the same pipeline on a leading class axis and the classes are mixed as `ln_cfu = ln_cfu0 + logsumexp_c(log w_c + G_c)`. The transformation component's `cell_classes(focal, population, params, data)` builds the classes (`single`: one class); `NEEDS_POPULATION` makes `jax_model` supply library-ordered θ/activity/dk_geno (`return_population=True`, or `GrowthData.external_*_population` in prediction).

- **`model_orchestrator.py`** — `ModelOrchestrator`: top-level class orchestrating data loading, inference, and prediction.

- **`inference/run_inference.py`** — `RunInference`: coordinates JAX/Numpyro sampling (SVI or NUTS) and MAP estimation (optax). **Guide selection:** `setup_svi(guide_type=..., guide_kwargs=..., init_values=...)` takes any name in `GUIDE_TYPES` — `component` (the guide assembled from the model components) or a numpyro autoguide from the `AUTOGUIDES` registry (snake_case of the class name, `delta` = `AutoDelta`; CamelCase class names accepted via `resolve_guide_type`). `tfs-fit-model --guide_type` exposes it for `analysis_method=svi`; an autoguide's location starts at the pre-MAP point (`init_to_value`), and `init_params` (component-guide param names) is not passed to it. Checkpoints record `guide_type`/`guide_kwargs`; `restore_svi_from_checkpoint` rebuilds that guide (legacy checkpoints → `component`) and `tfs-sample-posterior` routes `delta` → Laplace, everything else → guide sampling. `run_optimization` refuses any autoguide (AutoDelta included) when `inference/batch_safety.py` finds batch-dependent latents. Because per-genotype latents are library-sized, the `AutoContinuous` guides (incl. `auto_multivariate_normal`) are valid under mini-batching; the dense MVN just costs O(D²) memory (reported, warned above `_DENSE_GUIDE_WARN_GB`).

- **`inference/convergence.py`** — when to cut the step size and when to stop; see **Convergence and step size** below.

- **`tensors/tensor_manager.py`** — `TensorManager`: maps ragged per-genotype observations into JAX-compatible tensors.

- **`generative/registry.py`** — `model_registry` dict mapping component names to module implementations. This is where all swappable components are registered.

- **`generative/components/`** — Pluggable model components selected via the YAML config's `components:` section (registry key noted in parens where it differs from the directory name):
  - `activity/`: `fixed`, `hierarchical_geno`, `hierarchical_mut`, `horseshoe_geno`, `horseshoe_mut`
  - `growth/` (registry key `condition_growth`): `linear`, `power`, `saturation`. Their per-condition prior loc/scale fields (in `linear`, `m_scale_plus`/`m_scale_minus` too) accept a scalar (broadcast to all conditions) **or** a per-condition array; the pre-fit calibration writes per-condition arrays to pin the baselines (see **Per-condition growth priors** below). Each declares `get_scale_bounds()`.
  - `growth_transition/`: `instant`, `memory`, `baranyi`, `baranyi_k`, `baranyi_tau`, `two_pop`
  - `transformation/`: `single`, `mixture` (`empirical`/`logit_norm` were removed; `ModelOrchestrator` refuses them by name with a message, `_RETIRED_TRANSFORMATIONS`). The old E[max] θ operator (`_congression.py`) is gone, along with Stage 1.5 of the empirical pipeline that used it
  - `theta/`: `categorical_geno`, `hill_geno`, `hill_relative` (wt-relative X, see **Relative-X fit** below), `hill_mut`; `hill_geno`/`hill_relative` take `theta_log_hill_n_hyper_scale_fixed` (> 0 holds the log(n) population SD; learned, a MAP on real data ran it to about 19, and n near 0.01 hid each genotype's response below the lowest nonzero concentration, in the 0-titrant tubes); `hill_relative` also takes `theta_{X_low,X_delta,log_hill_K}_hyper_scale_fixed` (the dev-data MAP ran them to 3.6, 4.3 and 11 against about 0.45, 0.45 and 0.7 among well-measured genotypes, so it shrank nothing; `planning/dev-data/real_fit/make_init_hs.py` rescales a MAP's non-centered offsets for a warm start at the held values); thermodynamic partition-function variants under `theta/thermo/` (lac dimer and MWC dimer, with/without unfolded state, PK/PnnC/PddG parameterizations)
  - `theta_rescale/`: `passthrough`, `logit`
  - `dk_geno/`: `fixed`, `hierarchical_geno`, `pinned`
  - `noise/` (theta observation noise; registry keys `theta_growth_noise`: `zero`/`beta`/`logit_normal`, `theta_binding_noise`: `zero`/`beta`)
  - `growth_noise/`: `zero`, `normal_kt`
  - `ln_cfu0/`: `hierarchical` (one starting abundance per replicate x pre-condition x genotype; use this), `hierarchical_factored` (one genotype baseline per replicate shared by every pre-condition, plus a per-tube offset; tube offsets non-centered, `tube_offset = tube_scale * tube_offset_z`, because the centered form had an unbounded density as `tube_scale` went to 0). The orchestrator refuses `hierarchical_factored` when a replicate's pre-conditions come from different libraries (`_check_factored_ln_cfu0`): kanR and pheS are transformed and grown up separately (user, 2026-09-28), so their starting abundances differ per genotype. On simulations the fit pushed that difference into a confident per-genotype theta error: 95% coverage 0.09 above 1000 reads, against 0.80 with `hierarchical` (`planning/studies/svi-overconfidence/`). Every study grid used the factored model until then.
  - `sample_offset/`: `zero`, `normal` (per-tube growth-rate offset, scaled by elapsed time), `level` (per-tube ln_cfu offset, constant prior; `sigma_fixed` > 0 holds its SD instead of learning it: learned, it grew to a free level per tube on the first real-data fit and absorbed the population's growth)

- **`generative/observe/`** — *not* under `components/`, and not swappable via YAML. Holds the four observation-likelihood layers (`binding`, `growth`, `presplit`, `base_growth`), registered under flat `model_registry` keys `observe_binding`/`observe_growth`/`observe_presplit`/`observe_base_growth`. `ModelOrchestrator` wires in `observe_binding` whenever binding data were supplied and (unless `binding_only`) the growth observer chosen by `growth_likelihood` (`lncfu` → `observe_growth`, Student-t on `ln_cfu`; `counts` → `observe_growth_counts`, see **Count likelihood** below), plus `observe_presplit`/`observe_base_growth` only when the corresponding data (`presplit_df`/`base_growth_df`) was supplied — these are parallel, independently-gated observers, not alternative choices for one axis. **Growth-only models** (`binding_df=None`, `tfs-configure-model` without `--binding_df`) have `data.binding = None`, `priors.binding = None`, no `theta_binding_noise` component and no binding sites in the model (`jax_model` gates them on `data.binding is not None`, static structure); the orchestrator refuses a `binding_weight` or a non-`zero` `theta_binding_noise` without binding, and `tfs-prefit-calibration` refuses a growth-only config. Absent prior groups are left out of the priors CSV (`configuration_io._extract_scalars` / `_update_dataclass` skip `None`), so they reload as `None`, not NaN.

### Count likelihood (`growth_likelihood: counts`)

Roadmap step 7. **The `tfs-configure-model` default** (with `sample_offset: level`), since the count-likelihood study (2026-09-27: theta RMSE 35-60% lower than `lncfu` at similar coverage). `ModelOrchestrator`'s own defaults stay `lncfu`/`zero` on purpose: configs are read back through it, and one written before step 7 has no `growth_likelihood` key, so changing the orchestrator default would silently turn old `lncfu` configs into count models (tests with synthetic `ln_cfu` tables also rely on it). `observe/growth_counts.py` observes reads directly: `reads ~ NegBin(mu, var = mu (1 + phi) + mu^2 inv_r)` with `log mu = ln_sample_reads + ln_cfu_pred - sample_ln_cfu` (predicted frequency times the tube's reads), learned `growth_phi`/`growth_inv_r` (LogNormal priors, medians 5 and 0.01), no pseudocount. The log-pmf is `CountNegativeBinomial`, Loader's algorithm (R's `dnbinom_mu`: `bd0` deviance + `stirlerr`): numpyro's own negative binomial loses several nats per observation to float32 cancellation at 1e4-1e6 reads; keep any rewrite of it tested against scipy at high depth (`test_observe_growth_counts.py`), and guard `jnp.where` branches so the unused one cannot overflow (NaN gradients). The concentration has a floor (`c + e^-30`, `_LOG_C_FLOOR`): without it `mu / phi` underflows for genotypes predicted near extinction (all-zero genotypes drift there) and the gradient/Hessian go NaN; `test_finite_value_gradient_and_hessian` covers log mu down to -200. Data: `GrowthData.counts`/`ln_sample_reads`/`sample_ln_cfu` (same layout as `ln_cfu`, None for `lncfu`; sliced by `tensors/batch.py`) and the static `GrowthData.growth_likelihood`, built by `model_orchestrator._add_count_columns` (tube reads from `sample_reads`, else `adjusted_counts / frequency`, about 1% high from pseudocounts, the same for every genotype in a tube; tube totals from `sample_ln_cfu` or `sample_cfu`). The orchestrator refuses `growth_noise` other than `zero` with counts. Pair it with `sample_offset: level` (one constant-prior ln_cfu offset per tube), since an error in a tube's supplied total otherwise moves every genotype in it. Presplit is still observed as `ln_cfu`. Validation grid: `planning/studies/count-likelihood/`.

### OD600 calibration (`tfs-calibrate-od600`)

Roadmap step 3. `process_raw/od600.py` holds the method (C10: the lab's data never enter the repo): `reading_noise`/`detection_threshold` from repeated readings of a dilution series (largest relative SD; midway between the two most dilute means), `plate_counts_to_cfu` (`colonies * dilution / plated_volume_mL`, Poisson counting + one pipetting error per step), `fit_polynomial` (weighted least squares, covariance scaled by the reduced chi-square as in `curve_fit`), `calibrate`, `write_calibration`/`read_calibration`/`check_calibration` (`kind: od600_to_cfu_per_mL`, coefficients, full covariance, `reading_rel_sd`, `detection_threshold`, `calibrated_od600_range`; the lab notebook's `A_CFU`... format is refused with a pointer), `od600_to_cfu_per_mL`, `cfu_per_mL_error_components` (curve SD `sqrt(J C J^T)`, shared by every tube (C11), kept apart from the per-reading SD) and `cfu_per_mL_to_od600` (bisection). It reproduces the lab notebook's coefficients to 2e-9 given the notebook's CFU/mL, but the notebook never divided by the plated volume (0.1 mL), so its constants are 10x low; the tool divides.

`tube_totals_from_od600(od600_df, cal, tube_volume_mL)` turns one reading per tube into `sample_cfu` (CFU/mL times the culture volume: totals are cells per tube), `sample_cfu_std`, `sample_cfu_curve_std`, `sample_cfu_reading_std` and `od600_in_calibrated_range`; a reading below the detection threshold is an error. `tfs-process-counts` calls it (`add_tube_totals`) when given `--od600_file` (or an `od600` column) with `--od600_calibration_file` and `--tube_volume_mL`, and refuses OD600 plus supplied totals. This is roadmap/pipeline step 1 (P3: the OD600 table stays a first-class input so the population model can replace the totals step). Synthetic inputs in `examples/process_raw/` (`make_example_data.py`).

### Relative-X fit (`theta: hill_relative`)

Roadmap step 5. Growth alone fixes an occupancy-like variable only up to an affine map (`k`/`m` absorb it), so `theta/hill_relative.py` fits a wt-relative growth variable X: hill_geno's curve with real-valued baselines (`X_low`, `X_delta`), wt gauged to `X = 1` at `c_lo` and `X = 0` at `c_hi` (`gauge_baselines`, from wt's own K and n; `data.growth.theta_gauge_log_conc`). `theta_gauge_conc` is an orchestrator setting (`tfs-configure-model --theta_gauge_conc`; default the measured min/max, resolved and written to the config; refused for other theta components). `model_orchestrator._check_relative_theta` refuses binding data, activity other than `fixed`, `theta_rescale` other than `passthrough`, non-`linear` condition growth, theta noise other than `zero` and, with `mixture`, any `congression_theta_rule` but `max` (the only rule invariant to an increasing map; `mixture`'s `max` branch does not clip). The module's `THETA_SCALE = "X"` is how downstream code tells: `tfs-predict-theta` adds `theta_scale = X`, `extract_theta_epistasis`/`tfs-extract-epistasis` allow only `scale='add'` (no `in_regime`), `tfs-summarize-fit` maps simulated truth to the gauge (`hill_relative.x_scale_truth`/`x_scale_growth_truth`, activity from `*_sim_parameters.csv`), `calibration_grid` labels those rows `theta_regime = X`. `prediction.copy_orchestrator` always keeps wt in a genotype subset (its K/n set the gauge) and `predict` drops it again. `tfs-prefit-calibration` refuses it (growth-only). **Which inference to use is open.** Growth alone constrains mostly `m·(X_g - X_wt)`, so the posterior is a curved ridge along `m·X` that only wt's absolute growth change (via the tube totals) pins. Runs 2 and 3 of `planning/studies/relative-fit/` (2026-09-28) found the mean-field component guide sliding along it (smaller |m|, stretched X) while the low-rank guide (`--guide_type auto_low_rank_multivariate_normal`) recovered X, and MAP + Laplace blowing up in 5 of 6 runs. Both comparisons ran with bugs fixed on 2026-09-29: every component-guide hyperparameter scale frozen at its 1e-4 bound (2026-09-27 16:48 to 2026-09-29; the low-rank autoguide was unaffected) and the flat 1e-3 Laplace eigenvalue floor (the blowups); the learned log(hill_n) spread was also free then. The svi-overconfidence study's relative arm, in the same window on another library, found the two guides about equal (X 95% coverage 0.68 vs 0.74, RMSE 0.071 vs 0.072). Every guide collapses k and m. The post-fix rerun (`planning/studies/relative-fit/grid_v4.yaml`, run 4, 2026-10-01, 30 runs) confirmed the guide verdict: the component guide is biased even with the fixes (X RMSE 0.29-0.30, 95% coverage 0.06 Poisson / 0.38 realistic), and the low-rank guide recovers X (RMSE 0.07 / 0.12, coverage 0.81 / 0.76); rank 20 matches numpyro's default rank. MAP + floored Laplace now works and covers X and k/m best, but with the widest intervals and the noisiest point. Two-stage is the best calibrated under Poisson noise (0.91) and no better than low-rank under realistic noise (0.78). Every guide collapses k and m. So: the low-rank guide, rank 20, expecting X intervals to undercover somewhat and k/m intervals to be too narrow. On a full library the low-rank guide slid along the ridge (dev data, 3.3e5 nats worse than the MAP), and the MAP plus the arrowhead Laplace (below) covered best on run 4 (X 0.97 / 0.94, k/m about the full Laplace's), so that is the full-library route. Validation grid: `planning/studies/relative-fit/`.

### Adding a New Model Component

1. Create `src/tfscreen/tfmodel/generative/components/<category>/myname.py`
2. Implement the required interface (follow an existing component as reference)
3. Register it in `tfmodel/generative/registry.py` under the appropriate category key
4. Add a test in `tests/tfscreen/tfmodel/components/<category>/`
5. For a new `condition_growth` component: make its per-condition prior loc/scale fields accept scalar-or-array (broadcast-then-index by the condition plate) and implement `get_scale_bounds()` so the pre-fit can pin its per-condition baselines — see **Per-condition growth priors** above.
6. Per-genotype latents must be **mini-batch safe**: sample them in a library-sized plate (`pyro.plate(f"{name}_genotype_plate", data.num_genotype)`), then slice with `[..., data.batch_idx]` whenever the value is library-sized (a smaller value is an already-sliced substitution from the posterior forward pass). Never size a latent's plate with `data.batch_size`, and do not wrap latent priors/guides in `scale_vector` (only the likelihood is scaled). A batch-sized latent silently aliases genotypes onto batch positions under every numpyro autoguide, including `AutoDelta` (MAP, pre-MAP, `tfs-prefit-calibration`) — even at full batch, because the full-batch index is binding-first and reshuffled each step. Slice **unconditionally**: never `if data.batch_size < data.num_genotype`, and never index a library-ordered array by the batch-relative `geno_theta_idx` alone (translate through `data.batch_idx[data.geno_theta_idx]`, the `hill_geno` pattern). At full batch the index is a permutation, not the identity, so skipping the slice pairs batch positions with the wrong genotypes while every shape stays right. `inference/batch_safety.py` checks both failure modes: `find_batch_dependent_latents` traces at two batch sizes (shape; `RunInference.run_optimization` runs it before fitting with any autoguide and refuses an unsafe model), and `find_batch_order_mismatches` traces two orderings of the full batch with the latents held fixed and checks that `growth_pred`/`theta_growth_pred`/`binding_pred`/`theta_binding_pred` permute with the index (order). `tests/tfscreen/tfmodel/inference/test_batch_safety.py` enforces both registry-wide. Known exception: `noise/beta`'s `{name}_dist` (its latent is the noisy θ itself).

7. A new `dk_geno` or `activity` component must accept `return_population=False` in `define_model` and `guide` and, when True, return `(tensor, population)`: the library-ordered per-genotype values from the same draw, or `None` if the latents arrived batch-sized. Build both with `generative/components/_population.py::per_genotype` (default path = the old slice-then-compute, so no extra cost when not requested). The congression mixture looks co-resident genotypes up in these by library index; `tests/tfscreen/tfmodel/generative/components/test_component_population.py` checks every registered variant.

8. Horseshoe priors use `components/_horseshoe.py`: its `HalfCauchy` (log density via `hypot`, finite far in the tail) and `regularized_scale(lam, tau, c2)` (the slab-regularized `sqrt(c2 lam^2 / (c2 + tau^2 lam^2))` without squaring `lam`). Never write `lam ** 2` there: a mean-field guide can widen a local scale until a draw near 1e20 squares to inf, and the whole fit goes NaN (count-likelihood grid v2, run 0007, hill_mut epistasis on low-read doubles).

9. Guide sites should follow the `{site}_loc(s)` / `{site}_scale(s)` naming convention (a Normal or LogNormal guide whose location/scale params are named after the site). The MAP warm-up hands its solution to the component guide through it (`inference/initialization.py`); `tests/tfscreen/tfmodel/inference/test_initialization.py::test_component_guides_follow_loc_scale_convention` fails for a new variant that does not, unless it is added to `_KNOWN_UNMAPPED` with a reason.

(This applies to `components/` categories. `generative/observe/` is not a `<category>/<variant>` registry entry — see above.)

### Convergence and step size

`RunInference.run_optimization` is the one optimization loop (SVI, MAP, the SVI pre-MAP warm-up, `tfs-prefit-calibration`). It runs in tumbling **windows** of `convergence_window_steps` optimizer steps (default 2000; raised to ≥ `MIN_WINDOW_EPOCHS` = 10 epochs so mini-batched genotypes are each visited several times; split into `PARAM_BLOCKS` = 8 blocks). `inference/convergence.py` holds the decisions as host-side numpy, testable on synthetic traces (`tests/tfscreen/tfmodel/inference/test_convergence.py`, including a recorded real-data and a simulated trace in `loss_traces/`).

- **Loss test** (`loss_trend`): OLS line through the medians of 20 blocks of the window's per-step losses → drop per window ± SE. *Improving* iff `drop > z·SE` and `drop > loss_rtol·|loss|`. Judged against the loss's own noise: never against the start-up transient (the old rule's `|Δ|/(loss_start − loss_best)` stopped runs ~500× above their final loss and never stopped after a resume) nor against `|loss|` (the rule before that, which never fired on a noisy ELBO). A rising or oscillating loss is *not* improving.
- **Loss skew** (`loss_skew`, reported by `loss_trend`): `(mean − median) / (1.4826·MAD)` of the window's losses about the trend line. Medians hide rare huge penalties that dominate the mean (the objective); benign ELBO noise, skewed or heavy-tailed, stays < ~1 (all recorded clean traces ≤ 0.4), rare penalties give 10–40. At the floor a window with skew > `MAX_LOSS_SKEW` (3) is not a plateau; skew never forces a cut (a smaller step size does not remove penalties). Blind once penalized steps are the majority (median and MAD follow them) — then the loss level shows it instead.
- **Parameter test** (`param_drift`): each tracked param's block means (accumulated on the device inside the `lax.scan` carry — no per-step host transfer) → per-element trend over the window, in **normalizer** units (`RunInference._param_normalizer_specs`): paired guide scale for a location (`_loc`/`_locs`/`_auto_loc` ↔ `_scale`/`_scales`/`_auto_scale`; the posterior SD), derived SD for AutoContinuous `auto_loc`, prior SD for an AutoDelta location on a real site, and 1 for positive/bounded params (log/logit units, so relative). Excess = `|drift| − z·SE − floor`, where `floor = MIN_STEP_COHERENCE (0.01) · step_size · window` is the optimizer's resolution (Adam moves ~step size per step whatever the gradient; without the floor, jitter of a location whose guide scale collapsed below the step size reads as hundreds of SDs of drift). The posterior SD is floored at `PRIOR_SD_FLOOR` (0.01) × the site's prior SD in the location's unconstrained units (`initialization.site_unconstrained_prior_sds`: exact SD for real sites, robust MC `IQR/1.349` of prior draws otherwise; `RunInference._posterior_sd_floors` maps component-guide locations via `{site}_loc(s)`, AutoNormal via `{site}_auto_loc`, and AutoContinuous `auto_loc` via numpyro's `_ravel_dict` latent order): the mean-field component guide collapses hierarchical scales' guide SDs to ~4e-4 (hill_mut `theta_sigma_d_*_scale`, `theta_epi_tau_scale`), and in those units a negligible creep never settles. Summarized per array (max if ≤ 100 elements, else 99th percentile); *moving* iff any summary > `param_tolerance` (0.05). `scale_tril`/`cov_factor` are not tracked.
- **Decisions** (`ConvergenceMonitor.end_window`): above the floor step size, `patience` consecutive windows without loss improvement → cut (`step_size_cut`, default 0.1), unless the stalls pooled are a descent: before a cut or a stop, `pooled_loss_trend` fits one line through the block medians of all `patience` stalled windows (same `z`/`loss_rtol`), and a significant drop resets the stall count (`pooled_loss_t` in the convergence CSV; the stalled windows' medians are kept in checkpoints). A single window misses a slow, noisy descent: on the count-likelihood v2 grid, 20 of 23 first cuts came during a descent that pooled to t = 4-14, one (run 0012) into a wrong optimum (`loss_traces/sim_svi_counts_noisy_descent_run0012.npy`). **Parameters and skew never block a cut**: some directions never settle (a horseshoe local scale's MAP drifts to 0 forever at ~no loss change) and holding the step size high for them only wastes time. At `final_step_size`, `patience` consecutive windows with no loss improvement, no parameter movement and no loss skew → converged; a still-moving parameter or a skewed loss keeps the run going and is named in the log, so a slide, degenerate direction or hidden penalty is reported, never called converged. `max_num_epochs` is only a cap. A hierarchical MAP generally has no finite optimum (hyper-scale funnel), so MAP/pre-MAP runs often end at the cap — the honest outcome.
- **Optimizer** (`run_inference.adam_optimizer`): plain numpyro `Adam` by default; `adam_clip_norm` (every CLI and `setup_svi`) opts into `ClippedAdam`. **Do not re-enable clipping by default.** `ClippedAdam` clips each gradient *element* to ±`clip_norm`; these losses have gradients of 1e3–1e6 per element, so every element is clipped every step and Adam follows each draw's gradient *sign* — its fixed point is where the per-draw signs balance, not where the expected gradient is zero. With a heavy-tailed ELBO (step-like binding curve M42C/K84N, `hill_n` 24.6, measured with SD 0.001; a guide draw crossing it costs ~2.4e5 nats) that held 5–90% of draws in violation on every congression-calibration run (2026-09-25; the "blow-ups" were block medians flipping once the rate passed 50%, and step-size cuts cannot move a sign-balance point). Unclipped, the same fits drop to ~0 violations and converge (`planning/studies/congression-calibration/diagnosis/`). The two optimizers share their state, so old checkpoints resume unclipped.
- **Step-size cuts** swap `svi.optim` for a new optimizer of the same kind (clipped or not, read off the current `svi.optim`) at the smaller constant step (the optimizer state is step-size independent) and re-trace the scan (a fresh closure, so no stale compiled step size). `setup_svi` records `_step_size`/`_adam_clip_norm` (the latter written to checkpoints); a step-size *schedule* is still accepted but refuses cuts. The old `optax.exponential_decay` over `max_num_epochs` is gone (it collapsed the pre-MAP's step size within its 1000 epochs and tied resumes to `max_num_epochs`).
- **Runaway loss** (`RUNAWAY_LOSS_FACTOR` = 1e3): a window whose loss falls below -1000 times the magnitude of the run's first block median ends the run as `diverged` (not converged; `monitor.diverged`, kept in checkpoints with `reference_loss`). It means the objective is unbounded: a hierarchical density singularity (the old centered `hierarchical_factored` tube offset), or float32 error in a dense guide's own log density (`auto_multivariate_normal` on the small SVI-overconfidence library, where `log q` of its own draws reached -4e22 through an ill-conditioned `scale_tril`).
- **Records**: `{out_prefix}_convergence.csv` (one row per window: loss (median), drop ± SE, t, `loss_mean`/`loss_skew`/`loss_skewed`, worst param and its excess/drift, step size, plateau count, decision; a resume rewrites an older file under the current columns, `_migrate_convergence_columns`); `{out_prefix}_losses.txt` is `epoch,loss,step,step_size` (loss = block median). Checkpoints carry `step_size` and `convergence` (the monitor's `state_dict`); a resume continues at that step size and plateau count (a checkpoint without them resumes at `adam_step_size`). A run started with `svi_state=None` resets `_current_step` to 0 (the pre-MAP no longer shifts SVI's epoch numbering).

**Starting points** (`inference/initialization.py`). Configured guesses are keyed mostly by *site* name and partly by component-guide param name; AutoDelta and the component guide both ignored site-keyed guesses (AutoDelta started at prior medians; the MAP → component-guide hand-off matched no names, so the pre-MAP was discarded). Now `RunInference.site_values` turns any mix of guesses / `{site}_auto_loc` MAP params / guide locations into constrained site values; MAP (`_run_map`, prefit, pre-MAP) starts there through `setup_svi(init_values=...)` (fallback `init_to_median` for AutoDelta); `RunInference.component_guide_start` maps site values onto the component guide via the `{site}_loc(s)`/`{site}_scale(s)` convention (LogNormal locations take the log; each match verified against the traced distribution) and caps every mapped scale at `guide_init_scale` (`fit_model_cli.DEFAULT_GUIDE_INIT_SCALE`, 1e-4; also the autoguides' `init_scale` for a fresh fit). A scale param with a lower-bounded constraint (the hyperparameter scales' `greater_than(1e-4)`) starts at least `guide_init_scale` above the bound: starting on it is `-inf` unconstrained, and the param never moves (every hyperparameter scale stayed frozen at 1e-4 from 2026-09-27 to 2026-09-29). The scale is one number in every site's unconstrained units, so it must be small: at the old 0.1 (numpyro's autoguide default) an SD of 0.1 on a growth rate per minute is 17 ln units over a selection, SVI started ~1000x above the pre-MAP's loss and re-descended into other optima (mirror mode, halved m, the library shifted against wt; relative-fit grid, 2026-09-27). SVI widens the scales itself (about e-fold per 1000 steps at step size 1e-3). `init_param_jitter` defaults to 0 for the same reason (it is multiplicative, so 0.1 moves an ln_cfu0 location near 15 by ~1.5 ln units). `test_default_guide_start_stays_near_its_point` checks the start on a count model; a fit's first logged SVI loss should sit near its pre-MAP loss. `tests/tfscreen/tfmodel/inference/test_initialization.py` checks the convention registry-wide; exceptions: horseshoe_geno (`activity_*_offset_*` param names) and prior-guided noise latents. A new component guide should follow the convention. `read_configuration` keeps only guess rows whose names are in the orchestrator's `init_params`; it silently drops any others, such as a bare `condition_growth_k` or `sample_offset_offset` row. To start at a point the guesses cannot name, use `tfs-fit-model --init_from <params.npz>`: an earlier MAP's `{site}_auto_loc` arrays, which `site_values` ranks first. The dev-data level-offset MAP is the case: Adam moves every parameter about one step size per step, so in the first window it carried the tube offsets and k/m (k prior SD 0.002 per minute) by whole units into a ±2.8 offset mode. The step-size cuts then froze it there. Starting from the no-offset MAP plus small per-tube offsets was 1.2e5 nats better (`planning/dev-data/real_fit`).

**Staged MAP** (`inference/staged_map.py`, pipeline plan step 3). `tfs-fit-model` builds that start itself: a fresh MAP (no `--checkpoint_file`, no `--init_from`) of a model with `sample_offset: level` runs (1) the MAP with `sample_offset_offset` held at 0 (and a learned `sample_offset_sigma` held at `sigma_prior_scale`: with every offset at 0 its MAP is 0 and the density unbounded), files `{out_prefix}_stage1_*`; (2) the offsets alone, every other site held at stage 1's MAP (`{out_prefix}_stage2_*`; the tubes do not couple given the rest, so this is `small_offsets.py`); (3) the joint MAP from stage 1 + stage 2's offsets at `--staged_step_size` 1e-4, writing `{out_prefix}_*`. A stage is an ordinary `_run_map` on `HeldModel(orchestrator, held)`, whose `jax_model` is `numpyro.handlers.condition` on the held constrained values (as jax arrays: the model indexes per-genotype values with traced batch indices); the AutoDelta then has no parameters for them. Fit a `HeldModel` with autoguides only (its `jax_model_guide` is the wrapped model's, read by `site_values` for names). A stage whose `_params.npz` exists is reused on rerun; one with only a checkpoint resumes. `--stage_offsets auto` (default) stages only that case; SVI's pre-MAP is not staged. Validation on the dev data: `planning/studies/staged-map/`.

**MAP values: constrained or not.** AutoDelta's `svi.get_params(state)` (what `run_optimization` returns and `write_params` saves) is *constrained*; `svi.optim.get_params(state.optim_state)` (what `tfs-sample-posterior`, `tfs-extract-params` and `checkpoint_io` read from checkpoints) is *unconstrained*. `compute_hessian_sigmas` takes the former, `get_laplace_posteriors`/`get_map_posteriors`/`_map_params_to_constrained` the latter. Mixing them up silently evaluates positive sites at `exp(value)` (the pre-fit did until 2026-09-27).

**Block Laplace.** `get_laplace_posteriors(block_genotypes=True)` (`tfs-sample-posterior --laplace blocks`) holds the shared parameters at the MAP and gives each genotype the Laplace of its own parameters. Given the shared parameters the genotypes do not couple (checked: every cross-genotype Hessian entry was 0 on a relative-fit run, wt included, since the X gauge fixes only wt's own baselines), so `block_laplace_factors` gets every B x B block from B Hessian-vector products whose probe is 1 on one slot of every genotype, over genotype chunks, and refuses a model whose blocks leak. Each block is floored like the full Laplace. Both Laplace paths take the Hessian on `_unscaled_batch`, the full batch with the mini-batch scale removed: `scale_vector` carries `num_genotype / batch_size` even at full batch. On relative-fit run 4 the block Laplace covered X 0.93 / 0.84 (Poisson / realistic) and k/m 0. The MAP point's X is poor at low depth under realistic noise (RMSE 1.3 at <= 100 reads, against 0.22 for the low-rank guide's point; the MAP's hyperscales run away, so nothing shrinks it), but the intervals are wide enough to cover there.

**Arrowhead Laplace.** `block_shared=True` (`--laplace arrowhead`; what `--laplace auto` picks above 20,000 MAP parameters) keeps the shared parameters' uncertainty: `_genotype_hessian_parts` adds one probe per shared element (its response on a chunk's genotype rows is their coupling; on the shared rows it is summed over chunks, less `(chunks - 1)` times the chunk-invariant part, measured as `H(a) + H(b) - H(a + b)` on two one-genotype batches since every chunk's potential carries the priors and any unsliced binding data). `arrowhead_laplace_factors` then draws the shared parameters from the Schur complement and each genotype from its conditional, the full Laplace (`test_arrowhead_laplace_matches_full_laplace`). Two departures for MAPs short of an optimum, both needed: a floored direction of a genotype block carries no coupling (a short dev-data MAP floored 167 of 308 blocks, and the kept couplings subtracted 6e10 from m's curvature of 1e10), and a negative direction of the Schur complement is held at the MAP, named in the log and recorded on `RunInference.held_shared_directions` (`tfs-sample-posterior` writes it to `{out_prefix}_held_directions.csv`) (floored at the prior, the k/dk_geno slide gave k its prior SD and X widths of 1-3 on run 4). On run 4 it covered X 0.97 / 0.94 and k/m 0.92 / 1.0 (Poisson) and 0.50 / 0.42 (realistic), about the full Laplace's.

**Laplace floor.** `get_laplace_posteriors` floors each Hessian eigenvalue at the prior's curvature along its eigenvector (`laplace_eigenvalue_floor`; prior SDs from `site_unconstrained_prior_sds` with the MAP substituted; `LAPLACE_MIN_EIGENVALUE` = 1e-3 where no prior SD is usable), so no direction of the Laplace is wider than the prior. The old flat 1e-3 floor turned a MAP's negative eigenvalues (a hierarchical MAP stopped at its epoch cap always has a few) into variances of 1000 that leaked into whatever they touched: on the two-stage grid the draws of the growth slope m spread 50x (seed 8).

### Per-condition growth priors

The `condition_growth` components (`linear`/`power`/`saturation`) carry a per-condition **additive baseline** (`k` for linear/power, `min` for saturation) that is only jointly identified with the shared per-genotype `dk_geno`: the growth likelihood `g = k_condition + dk_geno + A·m·θ` is invariant to `k += C, dk_geno −= C`. With only weak/rare anchors (wt's pinned `dk_geno=0`, `base_growth`) the whole system slides by a global constant `C`, inflating all condition baselines and `k_ref` and making genotypes with constrained `dk_geno` (notably wt) badly mis-fit their `ln_cfu`. The fix is to pin each condition baseline with a per-condition prior — a prior on the 4 baseline params acts at full strength, unlike the abundance-diluted genotype anchors.

Mechanism, spanning three files:

- **Components** (`generative/components/growth/*.py`): `ModelPriors` loc/scale fields accept a scalar (broadcast, the default/back-compat) or a length-`num_condition_rep` array; `define_model`/`guide` broadcast-then-index by the condition plate. Each component declares `get_scale_bounds() → {suffix: {floor, ceiling, scale_field}}`, giving the pre-fit per-parameter scale floors (tight for the baseline term, looser for e.g. `power`'s log-exponent `n`).
- **CSV (de)serialization** (`configuration_io.py`): the priors CSV supports per-condition **indexed rows** (`flat_index` + `condition_rep`/`replicate` label columns) alongside scalar rows. On load, indexed rows are **name-joined** to the model's `map_condition_rep` order (`_read_priors_flat`/`_assemble_condition_array`) and **fail fast** on an unknown or missing condition. A fresh `tfs-configure-model` still writes scalar rows (2-column CSV, legacy path).
- **Pre-fit** (`prefit_calibration_cli.py`): after the MAP calibration, `_build_csv_updates` writes each `condition_growth` site's per-condition MAP **loc** array into the priors CSV (the baseline pin), and `_build_hessian_scale_updates` writes a tight **scalar** scale (floored via `get_scale_bounds()`). `_apply_priors_updates` expands a scalar prior row into per-condition indexed rows tagged with `condition_rep`. `growth_transition` sites keep warm-start-only behavior.

Net flow: `configure` (scalar) → `prefit` (per-condition `k_loc` indexed rows + tight scalar `k_scale`) → `fit` (loads per-condition priors that hold the baselines, closing the slide). This is the primary mechanism; the `base_growth_data`/`k_ref` anchor (below) is complementary but insufficient alone because `k_ref` is itself free to slide.

**Hard clamp on `m` (`tfs-prefit-calibration --pin_m`).** A soft Normal prior on the slope `m` — however tight — is only a KL penalty in SVI, and the growth likelihood over many observations can override it (empirically m walked ~5σ off even a 0.0005-scale pin). `linear`'s `ModelPriors.m_pinned` (bool, static) makes `m` a `deterministic` site clamped to its per-condition `m_loc` instead of a sampled site (guide drops the `m` variational params); `--pin_m` sets `condition_growth.m_pinned=1` in the priors CSV. `m` is safe to clamp because its calibration MAP loc is unbiased (dk_geno is uncorrelated with θ). **`k` is not clamped this way in production fits**: it carries real per-experiment tube-noise variance (`tube_noise_sigma`; mirrored by `k_scale_floor`) and sits in the additive k/dk_geno slide, so it keeps a floored soft prior — pin its loc, not its scale. `ModelPriors.k_pinned` exists for conditional fits at a fixed draw of (k, m): the two-stage fit (`planning/studies/svi-overconfidence/two_stage.py`) refits everything else by SVI at Laplace draws of k and m and pools the conditional posteriors, which carries the k/m uncertainty the guides collapse into each genotype's interval.

### Key Abstractions

**`TensorManager`** (`tfmodel/tensors/tensor_manager.py`): Handles ragged tensors. Genotypes have different numbers of observations; this class pads and indexes into JAX-compatible arrays.

**`DataClass` / `PriorsClass`** (`tfmodel/data_class.py`): Flax pytree dataclasses holding structured experimental data and prior specifications for JAX compilation.

### Configuration (YAML)

See **YAML Standards** below for the full conventions. The tfmodel config (`tfs_configure_config.yaml`) is generated by `tfs-configure-model` and drives `tfs-fit-model`. It contains `data`, `components`, `priors_file`, `guesses_file` and — for a growth model — `library_file` plus a `library` provenance block. Do not hand-edit it.

### Library description (`--library_config`)

`tfs-configure-model` takes `--library_config`, the library YAML that describes the screened library — **the same file handed to `tfs-process-fastq`**. It is required whenever `growth_df` is given (a binding-only model has no congression correction and no ln_cfu0 latents, so it refuses one). It replaced the old `--spiked` list, which could not express the central fact that the spiked controls are *also* encoded in the bulk sub-libraries: wt and each spiked single mutant arrive both as a monoclonal spike and as bulk library members, and they cannot be told apart by sequence.

Keys read (via `LibraryManager`): `reading_frame`, `first_amplicon_residue`, `wt_seq`, `degen_sites`, `tiles`, `tile_combos`, `spiked_seqs`, plus `library_mixture`. Everything else (a full simulate config, say) is ignored.

Flow: `genetics.library_composition_table(library_config)` (`genetics/library_design.py`) →  one row per genotype with `is_wt`, `in_spiked_origin`, `pool_fraction` and `bulk_fraction` → written to `{out_prefix}_library.csv`. The **snapshot CSV is what `tfs-fit-model` reads**, never the YAML; this mirrors the priors/guesses CSV pattern and makes the table the place to override the design's numbers. `ModelOrchestrator(library_file=...)` derives today's `spiked_genotypes` from `in_spiked_origin` and exposes the full table as `orchestrator.library_df`. `spiked_genotypes` is still accepted directly (and is what `tfs-prefit-calibration` uses when it clears both keys for its calibration-subset model), but the two are mutually exclusive, and a config carrying `spiked_genotypes` is the legacy path.

**Two checks, two different guarantees.** The *genetics* keys are attestable: they determine the genotype names, so `tfs-configure-model` fails fast if any genotype in `growth_df`/`presplit_df`/`base_growth_df` is absent from the library (`check_genotypes_in_library`; `__unknown__` excepted). A residue-numbering shift mismatches essentially every mutant and is caught immediately. `library_mixture` is **declared-only**: no upstream artifact ever saw it, so nothing can cross-check it. It is recorded verbatim in the config's `library` block, and that is the whole defense — the values should be the best available estimate of the realized pool, not just the intended design.

A library-derived spiked genotype that has no growth data is an **error** at configure time (`configure_model_cli.check_spikes_in_data`; the real-data fit's processing config named three spikes, one with the wrong codon, and nothing caught it), unless `--allow_missing_spikes`, which reports it (spikes can drop out of a real dataset). The orchestrator itself still only reports it.

`pool_fraction`/`bulk_fraction` are computed and snapshotted. `bulk_fraction` is carried into the model as `GrowthData.bulk_fraction` (library-sized float per genotype, indexed by `batch_idx`; built by `ModelOrchestrator._build_bulk_fraction`: the table's value, `__unknown__` → 1.0, any other missing genotype raises; legacy `spiked_genotypes` → 0 spiked / 1 otherwise) and is the `f_g` of the `mixture` transformation: a genotype's congressed fraction is `f_g (1 - P(M = 1))`, with `M` ~ zero-truncated Poisson(λ) plasmids per transformant (a cell's abundance is shared among its plasmids, so abundance by co-resident count n is `P(M = n + 1)`; λ is the simulator's `transformation_poisson_lambda`). Purity (`bulk_fraction`) and the `ln_cfu0` prior class (`ln_cfu0_spiked_mask`, from `in_spiked_origin`) are separate fields; the old binary `congression_mask` is gone. The orchestrator also draws each genotype's fixed co-resident plasmid sets for the mixture — `GrowthData.coresident_idx`/`coresident_n`, via `_draw_coresident_sets`, from the bulk share `pool_fraction * bulk_fraction` of genotypes with growth data (legacy: uniform over non-spiked); `congression_sets` (default `[12, 3, 1]`) and `congression_seed` are orchestrator settings written to the config, so draws are reproducible. `congression_theta_rule` (default `homodimer`; also `heterodimer`, `max`; `tfs-configure-model --congression_theta_rule`) is a third such setting, carried into the model as the static `GrowthData.congression_theta_rule`; the partition-function rules require `activity='fixed'` (the default activity component; the orchestrator refuses a mixture with these rules and a learned activity). `congression_dk_rule` (default `dilution`; also `softmin`, which needs `congression_dk_alpha` > 0, and `min`; `tfs-configure-model --congression_dk_rule`/`--congression_dk_alpha`) sets a congressed cell's dk_geno the same way (static `GrowthData.congression_dk_rule`/`congression_dk_alpha`).

### Pre-fit model accounting (`tfmodel/model_stats.py`)

`tfs-configure-model` runs a parameter/observation census after writing the config, printing a summary to stdout and writing `{out_prefix}_model_stats.csv` (one row per latent sample site) + `{out_prefix}_model_stats.json` (headline counts, coverage quantiles, anchors, warnings). `--skip_model_stats` disables it. The API entry point is `count_model_dimensions(orchestrator) -> ModelStats` (plus `format_model_stats` / `write_model_stats`), so the same census runs from a notebook or any other caller holding a `ModelOrchestrator`.

**Nothing is hard-coded per component.** The census *traces the Numpyro model*, so a newly registered component is counted automatically — there is no parameter table to keep in sync. Each `pyro.sample` site yields its scalar count (traced shape), its **plate membership** (what it is indexed by, hence what it scales with), its **support** (positive ⇒ scale/shrinkage latent), and its **component** (longest-prefix match on the site name against the registry key each component receives as its `name`).

- **Cost.** The trace runs under `jax.eval_shape` — abstract evaluation, no FLOPs and no allocation — so it is free even on a full-size library. If a component does something abstract evaluation can't handle, it silently falls back to a concrete forward pass; `summary["trace_mode"]` records which ran, and a test asserts the default stays `"abstract"`.
- **Full batch, always.** The trace uses `get_batch(data, arange(num_genotype))`. Tracing a mini-batch would report every per-genotype parameter count as the *batch size* rather than the library size.
- **Observation counts come from `good_mask`**, not tensor shapes, so padding in the ragged tensors is never counted as data. `_observation_counts` reads the masks directly (the four observers' masks reduce to plain `good_mask` sums under a full batch); `test_observation_counts_match_applied_masks` pins that shortcut against the masks a *concrete* trace actually applies, so it fails if an observer changes its masking.

**Site classification** (`_site_scope`): entity plates (`pair` > `mutation` > `genotype`, checked in that order) set `scope`/`entity` and `scales_with_library`; a site indexed by an entity *and* by a measurement axis (`titrant_conc`/`time`, size > 1) is `per_datum` — a random effect that grows with the data, scope-labelled `per_genotype_datum`. Sites with no entity plate fall back to the most specific design axis (`per_condition`, `per_replicate`, `per_sample`, …) or `global`. `level` is `hyperparameter` when a site sits above its component's entity-level latents (components with no entity latents at all, e.g. `condition_growth`, are all `individual`).

**Effective parameters are reported as a bracket, never a single number.** `p_eff_lower` counts everything that is *not* an entity-indexed latent (hypers, condition params, globals); `p_eff_upper` counts every latent. Partial pooling makes each entity-level latent cost strictly less than one dof, so the truth is inside. `per_datum` latents are excluded from **both** bounds and from `params_per_genotype` — counting them against a genotype's observation budget would double-count the data on both sides. `n_obs/n_param` is reported only as the two ends of that bracket. A real `p_eff` (WAIC/PSIS-LOO) needs a fitted model and would belong in `tfs-summarize-fit`.

**Per-genotype coverage** (`ModelStats.per_genotype`, kept in memory for plotting; only its quantiles go to JSON) records per genotype: growth/binding/presplit/base_growth observation counts, and the number of **distinct** titrant concentrations / replicates / timepoints / conditions covered. The distinct-level counts matter independently of the raw count — a genotype measured at 2 concentrations cannot support a 4-parameter Hill curve no matter how many timepoints back it, which is what `n_genotype_theta_under_determined` flags. `anchors` counts what pins the k/dk_geno/k_ref slide (binding genotypes split spiked vs in-library, base_growth, presplit, pinned dk_geno).

## Simulation Internals (`simulate/`)

### Overview

The simulation pipeline generates ground-truth phenotypes for a synthetic TF screen experiment. The top-level entry point is `library_prediction` (`simulate/library_prediction.py`), which returns five dataframes:

| Return value | Content |
|---|---|
| `library_df` | One row per genotype in the library |
| `phenotype_df` | Long-form growth predictions (one row per genotype × condition) |
| `genotype_theta_df` | Long-form theta predictions (one row per genotype × titrant_conc) |
| `parameters_df` | One row per genotype; per-genotype Hill/theta params + dk_geno + activity |
| `binding_theta_df` | Theta at binding concentrations for calibration genotypes (`None` if not configured) |

The core calculation happens in `thermo_to_growth` (`simulate/thermo_to_growth.py`):

1. **Prior-predictive theta sampling** — `sample_theta_prior` draws a `theta_gc` matrix of shape `(G, C)` where G = number of library genotypes and C = number of unique titrant concentrations. The genotype order here matches the library (`sim_data`) order.
2. **theta_gc_override injection** — specific rows of `theta_gc` can be replaced before any further computation (see below).
3. **Growth rate calculation** — theta is mapped to growth rates via the configured `growth_params`.
4. **parameters_df assembly** — per-genotype Hill parameters extracted from the `theta_param` pytree, then patched with `theta_params_override` (see below).

### Binding data and calibration genotypes

The optional `binding_data` YAML block configures calibration genotypes for which measured binding curves are available. Top-level keys `titrant_name`, `titrant_conc`, `noise` describe the (shared) binding assay. Two sub-blocks select which genotypes are measured, each with a `choose_by` (`stratified` | `random` | *params-file path*) and, for non-file modes, a `num`:

**`spiked_binding`** — clean, monoclonal (congression-free) controls; **pool = `spiked_seqs`**.
- `choose_by: <file>` — the named genotypes get the file's Hill params as their true phenotype (see the measured-params bullets below). `num` is forbidden with a file.
- `choose_by: stratified|random` — assign diverse (greedy-maximin) / random prior-predictive theta curves to `num` spiked genotypes (default all); `wt` keeps its natural reference. Their theta at *growth* concentrations is injected via `theta_gc_override` so growth matches binding.

**`library_binding`** — in-library controls drawn from the **bulk** (congression-affected growth); **pool = library genotypes NOT in `spiked_seqs`**. Omit to disable.
- These are regime-matched to the bulk: on the fit side they are simply extra `binding_df` rows and, because they are not spiked, their growth gets the bulk `bulk_fraction` automatically (binding itself is never congressed). No fit-side change.
- **Pre/post-sim split** (`simulate/library_binding_data.py::generate_library_binding_df`): the `file` path injects the genotypes' phenotype *pre-sim* (in `library_prediction`, via the same override machinery), but the binding *measurement* — and, for `stratified`/`random`, the *selection* — happen **post-growth-sim** so selection is restricted to genotypes that actually survived with growth data (guaranteeing `num` usable anchors). `file`-specified genotypes that don't survive are **warned** and dropped. `simulate_cli.py` writes the selected set to `tfs_sim_library_binding.csv`.

**Noise** (`binding_data.noise`): Gaussian on `theta_obs`, not clipped to [0, 1] (the fit's binding likelihood is an unclipped Normal; clipped anchors biased the absolute theta scale). `binding_data.clip_theta_obs: true` reproduces simulations made before 2026-09-29.

**Validation** (`library_prediction._validate_binding_config`, fail-fast): `spiked_binding` file genotypes must be ⊆ `spiked_seqs`; `library_binding` file genotypes must be disjoint from `spiked_seqs`; `num` + a file is an error; a file requires a Hill theta component; `spiked_binding.num` ∈ `[1, num_spiked]`; `library_binding` stratified/random requires `num`.

**Measured Hill parameters** (a `choose_by` *params file*, either block):
- Reads a CSV with columns `genotype, theta_low, theta_high, log_hill_K, hill_n`.
- Only supported for Hill-based theta components (`hill_geno`, `hill_mut`).
- **`theta_low` and `theta_high` are clamped to `[1e-4, 1-1e-4]` at read time** with a `UserWarning`. This is critical: a value of e.g. `1.000004` (a common float-rounding artefact) maps to `logit ≈ +16`, making all per-mutation deltas ~13–15 σ under the `HalfNormal(1)` prior on delta scales and preventing inference from recovering reasonable theta values.
- For `hill_mut`: `build_theta_gc_override_hill_mut` assembles theta for **all library genotypes** (not just the measured ones) by additively combining per-mutation logit-space deltas. The WT reference is taken from `SimPriors` defaults. Multi-mutant genotypes not directly measured in the CSV are assembled from single-mutant deltas; directly-measured multi-mutants use their CSV values directly.
- Measured genotypes override any earlier simulated-path values in `theta_gc_override`.

### Base growth-rate calibration data

The optional `base_growth_data` YAML block generates a simulated `base_growth_df`: direct reference-condition growth-rate "measurements" for a subset of genotypes, mirroring the inference-side `base_growth_df` input (see `tfmodel/model_orchestrator.py::_read_base_growth_df` and `generative/model.py`'s `base_growth_obs` block, which anchor `condition_growth`'s k/m against dk_geno's hierarchical hyperparameters to resolve an identifiability confound).

- Generation lives in `simulate/base_growth_data.py::generate_base_growth_df`, called from `simulate/scripts/simulate_cli.py` (not `library_prediction.py`) since it only needs `parameters_df`, not the theta/growth machinery.
- Config: `k_ref` (required, the reference wt growth rate) plus optional `genotypes` (default `["wt"]`), `rates` (per-genotype true-rate override), and `noise`.
- Every requested genotype must already exist in `parameters_df` — `dk_geno` is looked up from the value already assigned during the normal per-library draw (`_assign_dk_geno` in `thermo_to_growth.py`), never redrawn. This mirrors the inference-side requirement that `base_growth_df` genotypes already exist in `growth_df`.
- `rate_true = k_ref + dk_geno[genotype]` unless overridden via `rates`; the observed `rate` adds Gaussian noise (sigma = `noise`), and `rate_std` is reported as that same flat `noise` value for every row (matching the `binding_data.noise` → `theta_std` convention).
- **`noise` must be strictly positive if the CSV is fed to `tfs-configure-model --base_growth_df`.** `rate_std` is used directly as a Normal likelihood scale — both when `_read_base_growth_df` inverse-variance-combines multiple rows per genotype, and in the `base_growth_obs` likelihood itself. `rate_std == 0.0` (the default when `noise` is omitted) causes a division-by-zero (`1/rate_std**2`) that produces a NaN `k_ref` prior location, surfacing as a numpyro "invalid loc parameter" crash the first time the model is traced (`tfs-prefit-calibration` or `tfs-fit-model`) — not at `tfs-simulate` or `tfs-configure-model` time, which makes it confusing to diagnose. `_read_base_growth_df` raises a clear `ValueError` naming the offending genotypes if any `rate_std <= 0` is present.
- `simulate/base_growth_data.py::generate_k_ref_df` writes a single-row `tfs_sim_k_ref.csv` (`parameter="k_ref"`, `ref=<configured value>`) alongside `base_growth_df`, purely as an echo of the configured `k_ref` — the ground-truth counterpart to the fit's single global `*_params_k_ref.csv` (see `tfmodel/analysis/extraction.py`'s `k_ref` block). It is kept out of `tfs_sim_parameters.csv`/`tfs_sim_growth_parameters.csv` because it is neither genotype- nor condition-indexed.

### Growth parameter ground-truth output (`tfs_sim_growth_parameters.csv`)

`tfs-simulate` writes `tfs_sim_growth_parameters.csv`, the per-condition ground-truth counterpart to the fit's `condition_growth` component outputs (`*_params_growth_k.csv`, `*_params_growth_m.csv`, `*_params_growth_n.csv`, `*_params_growth_min.csv`, `*_params_growth_max.csv` — see `generative/components/growth/*.py`'s `get_extract_specs`). It is always written (the top-level `growth` config block is required, unlike the optional `*_data` blocks).

- Generation lives in `simulate/growth_parameters_output.py::generate_growth_parameters_df`, called from `simulate/scripts/simulate_cli.py` right after `library_prediction`, since it only needs `cf['growth']`.
- One row per condition, keyed by `condition_rep` — the same raw condition string used as `growth`'s dict keys (`{"kanR+kan": {...}}`) *is* the fit side's `condition_rep` value (see `model_orchestrator._build_growth_tm`, which pools `condition_pre`/`condition_sel` directly into a column literally named `condition_rep`), so no name-mapping step is needed to join this file against a fit's extracted params.
- Each growth model uses different YAML parameter names than the fit's extract names, and the module hand-maintains the mapping (`b,m` → `growth_k,growth_m` for `linear`; `b,a,n` → `growth_k,growth_m,growth_n` for `power`; `kmin,kmax` → `growth_min,growth_max` for `saturation`) by comparing `simulate/growth/growth_linkage.py`'s numpy formulas against the corresponding JAX formulas in `generative/components/growth/*.py`. If a new `condition_growth` component is added, this mapping must be extended too, or its parameters won't get ground-truth comparison in `tfs-summarize-fit`.
- On the `tfmodel` side, `tfmodel/scripts/summarize_fit_cli.py::_summarize_condition_growth_params` and `_summarize_k_ref` join these two files against the fit's extracted params (on `condition_rep`, and trivially for the single-row `k_ref`, respectively) to annotate them with a `ref` column, mirroring `_summarize_params`'s genotype-keyed comparison but for growth's condition-keyed/global-scalar parameters.

### Sampling noise (roadmap step 4)

Optional simulate-config keys, all off by default (a config without them simulates byte-for-byte as before): `founder_sampling` (`_sim_growth`: each tube gets a Poisson number of cells per transformant clone, drawn per tube), `demographic_growth` (given founders, Gamma(n0, e^kt) growth or Binomial(n0, e^kt) death; needs `founder_sampling`), `shared_transformation` (one library assembly and transformation shared by all replicates through the `shared_state` dict `simulate_cli` passes to `selection_experiment`; without it each replicate redraws both, unlike the real protocol's single glycerol stock), `pcr_template_molecules` / `pcr_amplification_cv` (`_sim_sequencing`: reads drawn from a multinomial set of template molecules, each amplified by a Gamma factor; count variance grows by about `(reads/templates)(1 + cv^2)`). `selection_experiment(..., sequence=False)` simulates totals (and OD600) without reads, for OD-only replicates. Real counts are 5-18x Poisson at 100-3,000 reads (`planning/studies/noise-anatomy/`); matching that needs templates on the order of reads per tube / 10.

### theta_gc_override and theta_params_override

These two dicts are the mechanism by which binding data is "pinned" into the growth simulation.

**`theta_gc_override`** (`dict[str, np.ndarray]`):
- Keys are genotype strings; values are 1-D arrays of theta at the growth titrant concentrations (sorted-unique order from `sample_df`).
- Applied in `thermo_to_growth` *after* prior-predictive sampling but *before* noise and all downstream computations.
- Overwrites the corresponding row of `theta_gc` in-place using `geno_to_sim_idx` (which maps genotype → its index in the original library list).

**`theta_params_override`** (`dict[str, dict[str, float]]`):
- Keys are genotype strings; values are dicts with Hill parameter keys (`theta_low`, `theta_high`, `log_hill_K`, `hill_n`).
- Applied in `thermo_to_growth` *after* `parameters_df` is assembled from the `theta_param` pytree, overwriting the relevant columns.
- Purpose: the `theta_param` from `sample_theta_prior` reflects the prior-predictive draw, not the override. Without this patch, `tfs_sim_parameters.csv` would show the pre-override values, making the saved parameters inconsistent with what was actually simulated.
- Only columns already present in `parameters_df` are updated; unknown keys are silently skipped.
- Genotype keys not found in `parameters_df` are silently skipped.

### Key files

| File | Role |
|---|---|
| `simulate/library_prediction.py` | Top-level orchestrator; assembles override dicts and calls `thermo_to_growth` |
| `simulate/thermo_to_growth.py` | Prior-predictive sampling, theta injection, growth calculation, parameters_df assembly |
| `simulate/binding_params.py` | CSV reading (with theta clipping), per-component override builders, binding theta assemblers |
| `simulate/binding_data.py` | Noise-injects the pre-sim `binding_theta_df` (spiked binding genotypes) into an observed binding CSV; `generate_binding_df` |
| `simulate/library_binding_data.py` | Post-sim generator for `binding_data.library_binding` (in-library, congression-affected binding genotypes; survivor-restricted selection); `generate_library_binding_df` |
| `simulate/base_growth_data.py` | Generates simulated direct growth-rate calibration data (`base_growth_data` YAML block) and the single-row `k_ref` ground-truth echo |
| `simulate/growth_parameters_output.py` | Generates per-condition `condition_growth` ground truth (`tfs_sim_growth_parameters.csv`) from the `growth` YAML block |
| `simulate/presplit_data.py` | Generates simulated pre-split (t = -t_pre) data (`presplit_data` YAML block; `generate_presplit_df`) |
| `simulate/od600.py` | Simulated OD600 per tube (`od600` YAML block): inverts a `tfs-calibrate-od600` calibration (read/apply/invert live in `process_raw/od600.py`) to get each tube's true OD600, adds reading noise, flags detectable/in-calibrated-range, and (`sample_cfu_from_od600`) runs it forward to the lab's estimate. `simulate_cli` writes `tfs_sim_od600.csv` for every replicate, including `num_od_only_replicates` extra unsequenced ones |
| `simulate/sample_theta.py` | `sample_theta_prior` (prior-predictive) and `sample_theta_stratified` (greedy maximin) |
| `simulate/sim_data_class.py` | `SimData` container and `build_sim_data` factory |
| `simulate/build_sample_dataframes.py` | Constructs sample/timepoint DataFrames from simulation config |
| `simulate/selection_experiment.py` | Models the selection experiment (transformation, growth, sequencing). A co-transformed cell splits its abundance among its plasmids (`_plasmid_shares`) and has one growth rate built from cell-level physics (`_cell_kt`); `_check_cell_components` fails fast if the phenotype's `k_pre`/`k_sel` can't be rebuilt from its theta/activity/dk_geno |
| `simulate/cell_rules.py` | Pluggable rules combining a co-transformed cell's plasmids: `THETA_RULES` (`homodimer`, the default: `logit θ_cell = log Σ x_g exp(l_g)` over equal shares; `heterodimer`: `2 log Σ x_g exp(l_g/2)`; both require activity 1 and raise otherwise; `max`: highest-θ plasmid sets θ and activity) and `DK_RULES`, the soft-min family `dk_cell = -(1/alpha) log sum_g x_g exp(-alpha dk_g)` (`dilution`, the default: share-weighted mean, alpha -> 0; `softmin`: finite `congression_dk_alpha`; `min`: the worst variant, alpha -> inf), chosen by `congression_theta_rule`/`congression_dk_rule`. The fit's mixture uses the same rules (`mixture.cell_classes`, `THETA_EPS` shared). Growth is then `thermo_to_growth.growth_rate_one_condition`, the same formula used per genotype. Never combine whole growth rates across plasmids |
| `simulate/scripts/simulate_cli.py` | `tfs-simulate` CLI entry point; runs `library_prediction` + `selection_experiment` across replicates and writes all output CSVs, delegating the optional `binding_data`/`presplit_data`/`base_growth_data` blocks to their respective generator modules above |
| `simulate/scripts/report_cfu0_cli.py` | `tfs-report-cfu0` CLI entry point; reuses `library_prediction` + `selection_experiment` (same pattern as `simulate_cli.py`) across `num_replicates` to report mean `ln_cfu_0` and surviving-genotype counts by class (`wt`/`spiked`/`single`/`double`), for tuning `transform_sizes`/`library_mixture`/`cfu0` against observed real-library values |
| `simulate/run_simulation.py` | Simpler, non-CLI orchestrator (`run_simulation`); does not support `binding_data`/`presplit_data`/`base_growth_data` |

### Empirical phenotype pipeline (`simulate/empirical/`)

An alternative to prior-predictive phenotype sampling: instead of drawing theta from made-up priors, fit **real** screen data to an empirical phenotype-generating distribution and resample from it, so the simulated library's phenotype *distribution* matches reality while ground truth stays known. Targets `hill_geno` + linear growth; asserts `A ≡ 1` (repressor: blocks or not, leaky binding absorbed into `theta_low`). Three stages:

**Module location note.** Stage 1 (per-genotype MLE fit) lives in `tfmodel/genotype_fit/fit.py` — a general per-genotype inference engine for the growth model, exposed standalone as **`tfs-fit-genotypes`** (`tfmodel/scripts/fit_genotypes_cli.py`). `simulate/empirical/fit_phenotypes.py` is a back-compat re-export shim. `tfs-fit-genotypes` takes `growth_df` + `growth_calibration_file` (positional; a prefit priors CSV or wide `condition_rep,growth_k,growth_m`) and writes `<prefix>_params.csv` and `<prefix>_theta.csv` (predicted θ, column `theta_raw`, vs `[genotype,titrant_name,titrant_conc]`). Reusable helpers `fits_to_results_df(fits)` (rebuild the params table from any fits dict) and `predict_theta(fits, growth_df)` back both this CLI and `tfs-build-empirical`.

**No congression correction (Stage 1.5 retired, 2026-09-24).** The Stage 1 fits of bulk genotypes carry the congression bias (small in the measured regime, about 0.05 ln units), and Stage 2 builds the distribution from them as-is. The old Stage 1.5 de-attenuated θ with the θ-only `E[max]` operator the main fit dropped in step 3.3c (Poisson rather than zero-truncated co-resident weights, binary spiked/bulk purity, no dk dilution); `--congression_lambda` was removed from `tfs-fit-genotypes` and `tfs-build-empirical`. A proper replacement (an iterated per-genotype refit under the same observable-level mixture as the fit) waits on a review of the whole empirical pipeline (`planning/empirical-mixture-refit.md`).

- **Stage 1** (`fit_phenotypes.py`): per-genotype MLE of the growth model on a real `ln_cfu` DataFrame (derives `ln_cfu_std` from the processed `ln_cfu_var` via `get_scaled_cfu`, like `model_orchestrator`), calibration (`growth_k`/`growth_m` per `condition_rep`) frozen. Fits `(dk_geno, theta_low, theta_high, log_hill_K, hill_n)` in transformed coords (logit theta bounds, log n) so `run_least_squares` returns covariance in the space Stage 2 needs. Per-genotype independent (no cross-genotype coupling); weak `dk_geno` Tikhonov prior; ±16 logit clamp. Returns `GenotypeFit(estimate, covariance)` per genotype. The fits are embarrassingly parallel — `num_workers` (`-1` = `cpu_count-1`) runs them over a `ProcessPoolExecutor`; the parallel path strips the genotype `Categorical` first (it carries all categories on every group → O(N²) pickling). Worker startup pays a ~2s JAX import, so parallelism is a wash for tiny/fast runs and near-linear for large real libraries. `_hill_theta` is numerically identical to `hill_geno.run_model` and `binding_params._hill_theta` (verified — same `_ZERO_CONC_SENTINEL`).
- **Stage 2** (`population.py`): measurement-error EM (`z_i~N(mu,Σ)`, `y_i~N(z_i,S_i)`) that **deconvolves estimation noise** (`Cov(y)=Σ+mean(S_i)`), so it recovers a narrower population than a naive KDE on the point estimates. `fit_population()` → `PopulationModel` (single MV-Normal in transformed space; `wt_ref` field holds wt's actual Stage-1 fit; `.sample(n)` → natural-space DataFrame; `.save()`/`.load()`).
- **Stage 3** (`resample.py`): `resample_phenotypes` draws one i.i.d. phenotype per genotype (wt pinned to `dk_geno=0` + `wt_ref` Hill); `make_empirical_overrides` reuses `build_theta_gc_override_hill_geno` → `(theta_gc_override, theta_params_override, dk_geno_override)`. `thermo_to_growth` gained a `dk_geno_override` param (the one growth-path change; `theta_params_override` only patches θ columns). `build_empirical_binding_theta` rebuilds spiked `binding_theta_df` from resampled params (selection from the `binding_data` config; θ from resampled Hill, not `sample_theta_stratified`).

Integration: `library_prediction` gains a `phenotype_source: empirical` branch (needs an `empirical: {phenotype_model: <path>}` block pointing at the single self-contained `<out_prefix>_phenotype_model.json`; `_resolve_phenotype_model_path` accepts that path with/without `.json` or the bare `<out_prefix>`, and the fit prints the absolute path — use it so no file-copying is needed) that resamples all genotypes, injects the overrides, forces `theta_component=hill_geno` (warns + ignores any other value; the resampled phenotypes are per-genotype Hill curves and the discarded prior draw must match the `parameters_df` schema — this also avoids running/overflowing an e.g. `hill_mut` draw), drops the ignored `theta_priors`/`theta_sim_priors`, and forces `activity=fixed/1`. `phenotype_source`/`empirical` are in `selection_experiment.SIMULATE_KNOWN_KEYS`. Resampled ground truth flows into `parameters_df`/`genotype_theta_df` automatically (no new return value); `library_binding` regenerates for free (reads `parameters_df`). `tfs-build-empirical` (`simulate/scripts/build_empirical_cli.py`) is a **one-command orchestrator**: given the experimental inputs (`growth_df` positional; `--binding_df` and `--seed` required; optional `--library_config`/`--base_growth_df`/`--thermo_data`/`--num_workers`) it internally calls `configure_model` (linear + hill_geno defaults — no model choices exposed, since here they'd only be wrong) then `run_prefit_calibration` (MAP-calibrates per-condition k/m), then Stages 1-2, saving the deliverable `<prefix>_phenotype_model.json` (one self-contained, human-readable file = the generating distribution; `PopulationModel.save`/`.load`) + the diagnostic `<prefix>_stage1_fits.csv`, plus the `<prefix>_configure_*`/`<prefix>_prefit_*` intermediates. The configure/prefit imports are lazy (the heavy JAX stack loads only on this path). The MAP prefit is the slow step, so `--growth_calibration_file` skips configure+prefit and reuses a calibration for fast Stage-1/2 iteration — either a prefit **priors CSV** (read via `fit_phenotypes.read_calibration`, which pivots the `growth.condition_growth.k_loc`/`growth.condition_growth.m_loc` per-`condition_rep` rows — prefit writes these via `_csv_row_name` = `growth.{component}.{field}` — to wide k/m) or a wide `(condition_rep, growth_k, growth_m)` CSV. This **complements** the fully-synthetic prior path (the *accuracy* benchmark) as a *realism* benchmark.

## Categorical response assessment (`analysis/cat_response/`)

`tfs-cat-response` fits a family of empirical shapes (`MODEL_LIBRARY`) to each group's `y_obs`-vs-`x_obs` curve and answers two **orthogonal** questions. Do not conflate them:

**Model x-scale (concentration-parameterized vs log-conc).** `MODEL_LIBRARY` models are **not** interchangeable on x-scale. The Hill family (`repressor`/`inducer`/`hill_*`) and `biphasic_*` are parameterized in **raw concentration** and take `log(x)` internally (`_hill` does `np.log(x)`), so they are already sigmoids/peaks *in log-concentration* and **must be handed raw x** — feeding them `log10(x)` (negative) both double-logs and NaNs them. The geometric models (`bell_peak`/`bell_dip` Gaussian-in-x, `linear`) are shapes in raw x; their `*_log` counterparts (`bell_peak_log`/`bell_dip_log` = Gaussian in `log10(x)` with a free real `center`, and `linear_log` = line in `log10(x)`) are the log-concentration versions. The `*_log` models own their transform via `models._to_log10_x` (x stays **raw concentration** in the data — there is no `--log_x` flag and no separate log column): `x <= 0` (the no-titrant point) is floored to `min(x[x>0])/100` before the log, computed per-call (identical across groups for a shared titration grid). `flat` is scale-invariant. When adding a new model, decide which camp it's in — never blanket-transform x. `DEFAULT_MODELS` (in `curve_models/__init__.py`) is the curated set fit when `--models`/`models_to_run` is omitted (was: all of `MODEL_LIBRARY`): `flat, linear_log, repressor, inducer, bell_peak_log, bell_dip_log` — one parameterization per qualitative response. All other models (raw-x `bell_*`/`linear`, 4-param `hill_*`, `biphasic_*`) stay registered and reachable via `--models`.

- **Shape** (which model): selection is controlled by `select_by` in `cat_fit.py` (three modes; **default `"shape"`**). **`"aicc"`**: `best_model` = lowest-AICc model (small-sample-corrected AIC on the **weighted** residuals `chi2 = sum(((y-yfit)/y_std)**2)`, `aic = 2k + chi2`; `aicc=inf` when `n-k-1 <= 0`, params still reported). Robust default — the weighted χ² correctly weights the few informative points, which the sign-based runs test does **not**. **`"adequacy"`** (`select_by_adequacy`): **escalate-only** refinement — keep the AICc pick unless its residuals are systematically clustered (one-sided lower-tail Wald-Wolfowitz runs test, `runs_p < adequacy_alpha`), then move to the lowest-AICc adequate model that is **no simpler** (`k >=` the AICc pick's `k`). It **never demotes**, so it cannot collapse a confident curved fit to `flat` — the failure mode of the earlier (removed) "simplest-adequate" rule, which on noisy heteroscedastic (logit) data let the diluted runs test override AICc and demote real curves to `flat`. **`"shape"`** (`select_by_shape`): liberal, prior-aligned classifier for *exploration* (AICc is too conservative — its small-n penalty buries a well-fit curve, e.g. an R²=0.96 dip called `flat`). Two steps, **no AICc parsimony**: (1) **flat-vs-curvy** gate on structure in the *flat* fit's residuals — curvy iff `autocorr_p|flat < curvy_cutoff` (weighted Durbin-Watson lag-1 autocorrelation p, `residual_autocorr`; magnitude/`y_std`-aware, so unlike the runs test it isn't washed out by many near-baseline points); (2) among curvy-shape models (step/peak/dip/biphasic; `linear` excluded as unphysical) pick the best **weighted R²**, preferring the simpler within `r2_margin` (0.02). `curvy_cutoff` (default 0.1) is the sweepable knob — run a set and visually inspect. When `models_to_run` is None, shape mode defaults to `SHAPE_MODELS` (physical vocabulary: `flat, inducer, repressor, bell_peak_log, bell_dip_log, biphasic_peak, biphasic_dip` — no `linear_log`; adds biphasic) instead of `DEFAULT_MODELS`. Note `biphasic_dip`'s `baseline`/`amplitude` bounds are **unbounded** (`curve_models/__init__.py`); a prior `>= 0` bound assumed a non-negative observable and gave a large-negative R² on signed data (logit epistasis), so it could never be selected.

  Per-model diagnostics are always reported and (except the shape gate) **do not gate selection**: `runs_p|*` (sign-based; needs `n >= _MIN_RUNS_N` (4), power only at `n >~ 8`), `autocorr|*`/`autocorr_p|*` (weighted DW lag-1, the shape gate's signal), weighted-χ² `gof_p|*` (`goodness_of_fit_p`). `shape` = qualitative form of `best_model` (`flat`/`linear`/`step`/`peak`/`dip`/`biphasic`, via `_SHAPE_BY_MODEL`; `_CURVY_SHAPES` = the non-flat/non-linear ones); `shape_status` = runs-test diagnostic on the **selected** model (`adequate`/`misfit`/`unassessable`/`none`, via `_shape_status`); `aicc_best_model` records the AICc pick (differs from `best_model` only when adequacy escalates or shape reclassifies). This form axis is **orthogonal to** the magnitude/`fittable` axis below — the intended exploratory hierarchy is their cross: `fittable` (× `all_equiv_zero`) gives *flat-real / can't-tell (indeterminate) / confident_zero*, and `shape` gives *flat vs which kind of curvy*.
- **Magnitude** (distinguishable from zero): a post-hoc pass, `cat_assess.py`, grading each curve against zero **on the observed data, not the fitted curve** (`assess_best_model` takes `y_obs`/`y_std`). The driver is a model-free **portmanteau** `nonzero_chi2 = sum((y_obs/y_std)**2) ~ χ²(n)` (`_nonzero_chi2`) → `nonzero_p` → **Benjamini-Hochberg** across curves → `nonzero_q`. This replaced a model-based **omnibus** `W = yhat @ pinv(J·Cov·Jᵀ) @ yhat` as the gate because that test reads the *fitted* curve's covariance, which is wildly overconfident when a flexible model is fit to noisy data (it called curves whose observed error bars all overlap zero "real"). The omnibus (`omnibus_W/df/p/q`) is **still computed and reported** but **gates nothing**; `y_model`/`y_model_std` are still emitted for plotting. Per-point `sig_nonzero`/`z` also use observed `y_obs/y_std`. `alpha` has two roles: the per-point `sig_nonzero`/equiv CI level **and** the `nonzero_q` threshold that calls `real` (it *is* a calling threshold, not just a stat level).

The equivalence rollup `all_equiv_zero` (every observed point's CI `|y_obs| + z·y_std ⊂ [-rope_cutoff, rope_cutoff]`, via `classify_equiv`) separates "confidently flat" from "too noisy to tell"; `rope_cutoff` defaults to `rope_multiplier * median(observed y_std)` (`compute_rope`, a **detectability** threshold computed globally after all fits) — which scales with the noise and so **rarely lets a whole CI fit inside**, meaning `confident_zero` seldom fires under the auto value; pass an explicit `--rope_cutoff` (a biological region) to make it fire. The per-point `equiv_zero` flag is computed internally for this rollup but **not written** to the assessment CSV. The magnitude call is a **bool `fittable`** (`_fittable`): `True` iff `nonzero_q < alpha` (distinguishable from zero, worth interpreting the shape). The old 3-way is recoverable as `fittable` × `all_equiv_zero`: `fittable=True` = "real"; `fittable=False & all_equiv_zero=True` = "confidently flat at zero"; `fittable=False & all_equiv_zero=False` = "can't tell". This magnitude axis is **orthogonal to** the `shape` axis above; the exploratory read is their cross-tab (filter `fittable`, then look at `shape`).

Outputs: rollups (`best_model`/`aicc_best_model`/`shape`/`shape_status`/`best_model_runs_p`/`best_model_autocorr_p`/`best_model_gof_p`, data-based `nonzero_p/q` (drives `fittable`), reported-only model `omnibus_p/q`, `n_nonzero`, `all_equiv_zero`, `fittable` (bool)) land in `{prefix}.csv`; `{prefix}_assessment.csv` is the self-contained per-point record — `model` (best model name), `fittable` (bool, right after `model`; carried on every point for filtering — model name and fitted values left intact), `x`, observed `y_obs`/`y_std`, fitted `y_model`/`y_model_std` (curve value + propagated fit error, **not** the observed error), then `z` (= y_obs/y_std) and `sig_nonzero` (per-observed-point, not the model). `equiv_zero`/`direction` were dropped (the ROPE `equiv_zero` was ~always False; `direction` = `sign(y_obs)`). `{prefix}_predictions.csv` holds only each group's **best** model (columns `model,x,y_model,y_model_std,is_best_model`; `best_only=True` threaded `cat_response→cat_fit` so the all-model curve is never built) unless `--write_all_predictions`. Note `y_model`/`y_model_std` are the **model prediction** at each observed x, distinct from `y_obs`/`y_std` (the experimental point + its input error). `cat_fit` returns a 3-tuple `(flat_output, pred_df, assess_df)`; `cat_response` a 4-tuple `(results_df, predictions_df, assessment_df, rope_cutoff)`.

## Cross-run comparison (`analysis/compare_runs.py`)

`tfs-compare-runs` measures how much N independent estimates of the same
quantity disagree, and whether that disagreement is explained by the uncertainty
each run reports. It is estimate-agnostic: anything long-form with a point
estimate works — `tfs-predict-theta`/`-growth`/`-epistasis` output *and*
`tfs-extract-params` parameter files.

**No thresholds, no grades.** Every quantity is a raw number; filter downstream
(`df["overdispersion"] > 2`, `df["rms_sd"] < 0.05`) so the cutline is recorded in
the analysis that used it. There is no `tier` column, no `overdispersed` flag,
and no crosstab output — an earlier version had all three, and the arbitrary,
unrecorded `--sd_tier_edges` cutlines were the reason they were removed.
`{out_prefix}_metadata.json` records every resolved setting.

**Three key sets** (all printed at runtime and recorded in the metadata):

- **match key** — what makes a row "the same row" across runs. Auto-detected as
  every column shared by all runs that is not a value column (`q<level>`,
  `y_obs`, `y_std`) and not incidental (`in_training_data`, `in_regime`,
  `Unnamed: *`, …). `--match_by` overrides. Fails fast if it is not unique
  within a run — a non-unique key silently becomes a many-to-many join.
- **`index_by`** — the entity being scored. Auto: `genotype`, else `parameter`,
  else the sole match-key column. The last rule is what makes
  `*_params_growth_k.csv` (keyed `replicate` + `condition_rep`) work — but it is
  ambiguous there, so that file needs `--index_by condition_rep`.
- **report key** = `index_by + group_by`; one output row per distinct value.

`match key - report key` is the *residual*: the axes each output row pools over.
`--group_by` is a statistical **zoom**, not a free refinement — at the finest
grouping only `n_runs - 1` dof remain per row. `n_rows`/`n_eff` report the
pooling depth. `dynamic_range` is the target's range over the residual axes and
is NaN when there is no residual axis; it pools across *all* residual axes at
once, so put a stratifier in `--group_by` to get per-stratum ranges.

**Row order.** Both output tables sort by the report key (compare) / match key
(aggregate) in **canonical genotype order** via
`genetics.set_categorical_genotype` — `wt`, then singles, then doubles, by site —
and the `genotype` column comes back as an ordered Categorical. Plain
lexicographic order would put `A100V` before `A2V`. With no `genotype` column
there is no canonical order for the entity (`parameter`, `condition_rep`, …), so
compare falls back to `rms_sd` ascending and aggregate to the match key.

**Statistical notes.** `rms_sd` is in native units and is not comparable across
parameters; `overdispersion` (χ²/dof, with p and a BH q) is the unit-free axis.
The default sigma is the symmetric quantile half-width `(q0.841 - q0.159)/2`,
which stays a fine robust scale but biases Axis 2 for skewed posteriors
(`theta_low`/`theta_high` at 0/1, `hill_n` at its bound); `rms_sd` is unaffected.
`--y_obs`/`--y_std` accept explicit columns (via the shared
`resolve_obs_columns`) for tables with no quantile ladder; Axis 2 is NaN when no
uncertainty is available.

**Aggregate.** `{out_prefix}_aggregate.csv` is written **by default**
(`--no_aggregate` suppresses it; it is the slow step on a big library). It mixes
the per-run posteriors as an equal-weight mixture keyed on the match key, so it
needs ≥2 shared `q<level>` columns and is skipped with a warning otherwise. It
is a row-level combination, so `--group_by` does not affect it, and a reference
run is never mixed in.

**Weights (not yet wired up).** Every reduction is written in weighted form
against a `_w` column pinned at 1.0, so uniform weights reproduce the unweighted
formulas exactly (`ddof=1` variance, arithmetic means) and enabling weights later
changes no existing number. `_mixture_quantiles` already accepts `weights`. When
wiring them up: on the **run** axis use *design* weights (fold size, run
validity), never inverse-variance — precision weighting across seed/dropout runs
down-weights runs that honestly report wide posteriors and tilts toward the
overconfident ones, exactly what Axis 2 exists to detect.

**Input forms.** The `estimates` positional takes `nargs="+"`: two or more
arguments are direct CSV paths (`rep1.csv rep2.csv`); a single argument is a
manifest file (one path per line) *unless* it ends in `.csv`.

## Planning (`planning/`)

Ideas, active plans and studies live in `planning/` (called `future/` before
2026-09-23); `planning/README.md` has the conventions. Ideas and plans are one
Markdown file each with a YAML header whose `status` (`idea`/`active`/
`promoted`/`dropped`/`done`) tells them apart. An active plan keeps its step
list current. Studies live in `planning/studies/<slug>/` (a `README.md` with
question, decision fed, how to run, inputs, commit and results, plus the
script); once a plan cites a study it is frozen. `dev/` is untracked scratch;
anything a plan cites moves into `planning/studies/`. The active plans are
`planning/congression-physics-plan.md`, `planning/analysis-roadmap.md`
(steps 2, 6, 7b, 8 and 9 left; outcome so far in
`planning/analysis-roadmap-summary.md`) and `planning/experiment-pipeline.md`
(one experiment file from raw data to posterior; steps 0, 1, 2 and 4 done in
0.5.0, the staged MAP, orchestrator and simulator raw formats left).

## YAML Standards

All YAML files in this codebase follow these conventions. Apply them when creating or modifying any YAML file.

### YAML types

There are two categories of YAML in tfscreen:

**Flat config files** drive a single CLI run (`tfs-simulate`, `tfs-fit-model`, etc.).  They have a flat list of top-level keys; nested dicts appear only where a parameter is itself structured (e.g. `observable_calc_kwargs`, `growth`).  See `examples/simulate_config.yaml` for the canonical simulate/process_raw reference.

**Grid YAML files** drive `tfs-setup-grid` or `tfs-setup-sim-grid`.  They describe a Cartesian product of parameter variants and produce one run subdirectory per combination.  See `examples/grid.yaml` (tfmodel) and `examples/simulate_grid.yaml` (simulate).

### Flat config conventions

- All keys in `snake_case`.
- Sections separated by `# --- Section name ---` comment header lines.
- File paths relative to the config file's own location.
- **Int vs float**: write count-like values as integers (`25_000_000`, `100_000`) and rate/fraction values as floats (`0.01`, `1.5e-7`). PyYAML preserves this distinction; `read_yaml` does not coerce between them. Underscores in integer literals are valid YAML and improve readability.
- Optional top-level blocks (`growth_transition`, `binding_data`) are omitted, not null, when not needed.

### Grid YAML structure

Both grid CLIs share the same top-level skeleton:

```yaml
# (simulate grids only)
base_config: path/to/simulate_config.yaml

run_name: "{{ var1 }}__{{ var2 }}"   # Jinja2 template; optional
output_file: run.sh                   # Jinja2 template file; optional

# Phase blocks (configure_model for tfs-setup-grid; simulate for tfs-setup-sim-grid)
simulate:          # or: configure_model:
  - name: <block_name>
    variants:
      - key1: value1
      - key1: value2
  - name: <joint_block>
    variants:
      - key1: val_a   # these two keys always move together
        key2: val_x
      - key1: val_b
        key2: val_y

# Variables forwarded to the Jinja2 template only (not to the config or CLI)
template:
  - name: <block_name>
    variants:
      - var: value
```

**Block rules:**
- Each block item needs `name` + either `auto` (tfmodel grids only — enumerates all registered components for an axis) or `variants` (explicit list of dicts).
- The Cartesian product is taken across **all** blocks (`configure_model`/`simulate` + `template`).
- Multi-key variants (multiple keys in one dict) always travel together and are never split.
- `simulate` / `configure_model` variables go to the config; `template` variables go to the Jinja2 template only.  To share a variable, list it in both sections.
- Variable names in blocks: `snake_case`.
- File paths are the grid tool's job, never the run template's, and a grid output directory must stay movable as a unit (on/off a cluster, onto a data partition). Both grid CLIs (output directory `--out_dir`) copy every input file into `<out_dir>/inputs/` once (`grid_utils.InputStager`; a numeric suffix keeps two different files with the same name apart, and a changed file never overwrites an existing copy) and each run refers to it as `../inputs/<name>`. Relative paths resolve against the grid YAML's directory (overrides, `configure_model` variables and template variables) or, for `tfs-setup-sim-grid`, the base config's directory (base config values). A template variable naming an existing file is copied the same way. Setup fails, before writing anything, on a missing or directory input, on a config value outside the known path keys that names an existing file (`grid_utils.check_no_outside_paths`), or on a template that does not render.
- `tfs-setup-sim-grid`: the config file paths are the key paths in `_SIM_PATH_KEYS` (`setup_sim_grid_cli.py`; nested keys such as `binding_data.*.choose_by` and `empirical.phenotype_model` included, `choose_by` keywords excepted). Add any new file-valued simulate key there.
- `tfs-setup-grid`: the inputs are the `configure_model` file arguments in `_PATH_KEYS` (`setup_grid_cli.py`: `binding_df`, `growth_df`, `presplit_df`, `base_growth_df`, `thermo_data`, `library_config`). `tfs-configure-model` reads the originals; the written config is then pointed at the copies wherever it recorded them (`data.*`, `components.thermo_data`/`presplit_df`/`base_growth_df`, `library.source`), and setup fails on any other path in it that names a file outside the grid. The priors/guesses/library CSVs are per-run outputs next to the config and stay as they are. Data paths are read relative to the working directory, so a run is launched from its own directory. Add any new file-valued `configure_model` argument to `_PATH_KEYS`.

### Shared grid utilities

Both grid CLIs import from `tfscreen.util.grid_utils` for run-name generation, Jinja2 environment setup and input staging (`InputStager`, `check_no_outside_paths`, `stage_template_vars`, `render_run_template`).  Add new reusable grid helpers there rather than duplicating them.

### `examples/` directory

| File | Purpose |
|------|---------|
| `simulate/simulate_config.yaml` | Canonical well-commented simulate config reference |
| `simulate/simulate_grid.yaml` | Example simulate grid for `tfs-setup-sim-grid` |
| `od600/` | Synthetic `tfs-calibrate-od600` inputs (`replicates.csv`, `plate_counts.csv`, from `make_example_data.py`) and the calibration made from them (`od600_calibration.yaml`, used by the simulate example's `od600` block); not any lab's real calibration |
| `simulate/run.sh` | Jinja2 shell template rendered into each simulate grid run subdir |
| `simulate-and-analyze/simulate_config.yaml` | Combined simulate + analyze workflow config |
| `simulate-empirical/simulate_config.yaml` | Simulate config that resamples phenotypes from a `tfs-build-empirical` model (`phenotype_source: empirical`) |
| `simulate-and-analyze/hill_params.csv` | Example Hill parameter CSV for binding data input |
| `simulate-and-analyze/run.sh` | Jinja2 shell template for simulate-and-analyze runs |
| `tfmodel/grid.yaml` | Example tfmodel grid for `tfs-setup-grid` |
| `tfmodel/run.srun` | Jinja2 SLURM template rendered into each tfmodel grid run subdir |
| `process_raw/library_config.yaml` | Minimal library config for `tfs-process-fastq` and `tfs-configure-model --library_config` (genetics keys + `library_mixture`) |
| `process_raw/` (`make_example_data.py`, `tube_table.csv`, `od600.csv`, `counts/`) | Synthetic tube table, OD600 table and per-tube count files in the formats `tfs-process-counts` reads; run with `../od600/od600_calibration.yaml` |

### `process_raw` YAML

`tfs-process-fastq` takes the library YAML as its first positional (`library_config`), passed to `LibraryManager`. It reads only the library genetics keys: `reading_frame`, `wt_seq`, `degen_sites`, `tiles`, `expected_5p`, `expected_3p`, `tile_combos`, `spiked_seqs`. All other keys are ignored.

**Required practice**: maintain **one** library YAML per experiment and pass that same file to `tfs-simulate`, `tfs-process-fastq` and `tfs-configure-model --library_config`. Nothing in the pipeline checks that the three saw the same file: the genotype-set check in `tfs-configure-model` catches a genetics mismatch, but `library_mixture` (which only `tfs-configure-model` and `tfs-simulate` read) can silently drift. Use `examples/process_raw/library_config.yaml` only when you need a minimal standalone library config (e.g. for processing real data without a matching simulation); note it carries `library_mixture` as well as the genetics keys.

### Terminology

- **Condition**: unique growth setting (marker + selection) — same avg growth rate for same genotype
- **Tube = sequenced sample = timepoint**: after `presplit` the culture is split into one tube per condition × titrant × planned timepoint (a grid: rows = timepoints, columns = IPTG); pre-growth and selection both happen in the tubes; a timepoint is taken by pulling every tube in its row and reading its OD600, spinning it down and freezing it; sequencing comes later. Total CFU per tube comes from OD600 through a lab-specific calibration (not in this repo). There are no aliquots from a shared tube (stopping the shaker to pipette disturbed the trajectories more than tube-to-tube noise does). So anything tube-level (growth environment, OD600 reading, PCR) is independent across timepoints, not a trajectory-wide effect. Rationale: `docs/source/process-raw.rst`, "One tube per time-point"
- **Sample** (in `sample_df`/`sample_ln_cfu`): one sequenced tube, i.e. one replicate × condition × titrant × time. Older text sometimes says "sample" for a replicate × condition; read it per this entry
- **theta (θ)**: operator occupancy — fraction of operators bound by TF
- **A**: per-genotype TF activity (multiplied by theta to scale occupancy)
- **dk_geno**: pleiotropic growth effect of mutation independent of TF activity

### Testing Notes

- Always set `NUMBA_DISABLE_JIT=1` when running tests — Numba JIT causes test failures
- Slow tests (marked `@pytest.mark.slow`) are skipped by default; use `--runslow` to include them
- Smoke tests live in `tests/smoke-tests/` and test end-to-end pipelines
- **Write or update unit tests for any new code added in a session.** Tests mirror the source layout under `tests/tfscreen/`; a new module at `src/tfscreen/foo/bar.py` gets tests at `tests/tfscreen/foo/test_bar.py`.

## CLI Standards

All `tfs-*` entry points follow these conventions. Apply them when writing or modifying any CLI script.

### File naming

Each entry point lives in `<name_of_script>_cli.py`. The registered console script is `tfs-<name-of-script>` (hyphens in entry point, underscores in filename). Example: `predict_theta_cli.py` → `tfs-predict-theta`.

### Argument layout — use `generalized_main`, no manual argparse

All scripts use `generalized_main` from `tfscreen.util.cli.generalized_main`. The function signature is the CLI spec:

- Parameters **without** a default → positional (required) arguments
- Parameters **with** a default (including `None`) → `--flag` arguments

- A bool that defaults to False is a `--name` switch; one that defaults to True is turned off with `--no_name` (`--name` is accepted and does nothing). Never invert a flag's meaning.
- A `None` default reads a string unless `manual_arg_types` gives a type (`int`, `float`, or a list type with `manual_arg_nargs`).
- The numpydoc docstring is the `--help`: the text before `Parameters` is the description, each parameter's entry is its help, and defaults are appended unless the entry already says "default". Document every parameter.
- `generalized_main` returns None (console scripts run `sys.exit(main())`; a returned value would exit 1), prints the run's provenance (`util/provenance.py`: version, commit, dirty flag, command) and writes `{out_prefix}_provenance.json` when the function takes `out_prefix`. Configs (`provenance:` block) and fit checkpoints embed `get_provenance()`.
- After changing a CLI signature or docstring, run `python docs/scripts/make_cli_reference.py`; `tests/tfscreen/util/cli/test_cli_reference.py` fails while `docs/source/cli.rst` is stale.

Positional argument order (use only what the script needs):
1. `config_file` — path to YAML config (`tfs-process-fastq`: the library YAML)
2. `param_file` — a posterior `.h5` or a MAP checkpoint `.pkl` (`tfs-sample-posterior` takes `checkpoint_file`)
3. `data_file` — path to a long-form observable CSV (e.g. `tfs-cat-response`
   takes `data_file x_obs` with `--y_obs`; `tfs-extract-epistasis` takes `data_file` with `--y_obs`)

Shared names: `--seed` for every seed, `--num_workers` (-1 = all CPUs but one), data inputs named as `tfs-configure-model` names them (`growth_df`, `binding_df`, `presplit_df`, `base_growth_df`), the library YAML is always `library_config`.

### `--library_config`

`tfs-configure-model` and `tfs-build-empirical` take the library YAML as `--library_config`. It is an exception to the positional-if-required rule: `tfs-configure-model` requires it only when `growth_df` is given, and `growth_df` is itself a flag, so a positional would be wrong for the binding-only call. Validate it in the function body instead. The same holds for the data: `tfs-configure-model` takes `--binding_df` and `--growth_df` as flags and requires at least one (both: joint model; binding only: binding-only model; growth only: growth-only model).

### Output flag

Always `--out_prefix` (never `--out_root`, `--out`, `--output_file` or `--output_prefix`). The function parameter must also be named `out_prefix`, and outputs are `{out_prefix}.ext` or `{out_prefix}_<name>.ext`. A command whose output is a directory of files takes `--out_dir` instead (`tfs-process-fastq`, `tfs-setup-grid`, `tfs-setup-sim-grid`).

### File-backed list arguments

When a list of genotypes, titrant names, or concentrations is needed, the `_cli` wrapper takes file-path strings (one value per line, `#` comments allowed). Use `manual_arg_types` in `generalized_main` to override the `NoneType` inferred from `default=None`. The shared helper `read_lines(path)` lives in `tfscreen.util.cli`.

### `in_training_data` column

`tfs-predict-growth` and `tfs-predict-theta` output a boolean column `in_training_data` (1/0) at the `(genotype, titrant_name, titrant_conc)` tuple level.

### `in_regime` column (`tfs-predict-epistasis`)

`tfs-predict-epistasis` appends a trailing `in_regime` (int 0/1) after the `q<level>` columns. It is `1` only when **all four corners** of the mutant cycle (wt, both singles, double) have their θ posterior — the central `regime_ci` interval (default 95%) — inside the resolvable band `[regime_eps, 1 - regime_eps]` (default `eps=0.01`, whose logit band is ±~4.6). Outside that band `logit(θ)` saturates and the linear-in-θ growth likelihood constrains it weakly, so the epistasis there leans on the θ-model's functional form and cross-genotype posterior covariance — `in_regime == 0` rows are **model-conditional** (a tight CI can still be flagged 0: the perfectly-correlated-but-saturated case). It is the posterior-mass analogue of the toy model's `measurement_window` join (`simulate/toy_thermo/basis.py`); it does **not** separately test whether the growth signal exceeds the growth noise (that is the heavier `m·A/σ_growth` identifiability check). Computed in `extract_theta_epistasis` (`tfmodel/analysis/extraction.py`) on the raw θ samples, so it is independent of `scale`/`scale_constant`. A MAP checkpoint has one draw, so the interval collapses to a point-value band check.

### Quantile-output column convention

Any CLI that emits a posterior/quantile summary of an estimate writes those quantiles as **bare `q<level>` columns** — `q0.5` (median), `q0.025`, `q0.975`, etc. — with **no feature-name prefix** on the column. This holds for `tfs-predict-theta`, `tfs-predict-growth`, and `tfs-predict-epistasis`; the feature a file describes is conveyed by the file, not by the column name. Downstream tools rely on this: `resolve_obs_columns` (used by `tfs-extract-epistasis`) defaults `y_obs` to `q0.5` and `y_std` to `(q0.841 - q0.159)/2`, and `tfs-compare-runs` reads the whole `q<level>` ladder. Point estimate + std outputs (e.g. the marginal `tfs-extract-epistasis`'s `ep_obs`/`ep_std`) are a different, non-quantile output shape and keep their descriptive names.

### Registered entry points

All scripts under `tfmodel/scripts/` and `analysis/scripts/` follow the `_cli.py` naming convention and are registered in `pyproject.toml`.
