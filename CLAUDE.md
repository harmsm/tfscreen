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
tfs-process-fastq          # FASTQ → read counts
tfs-process-counts         # counts → ln_cfu DataFrames
tfs-process-presplit       # Process pre-split count files
tfs-configure-model        # Generate YAML config template
tfs-prefit-calibration     # Pre-fit linking function via MAP
tfs-fit-model              # Main hierarchical Bayesian inference
tfs-sample-posterior       # Draw posterior samples from fitted model
tfs-sample-prior           # Draw prior predictive samples
tfs-extract-params         # Extract parameters from checkpoint
tfs-predict-growth         # Predict growth from fitted model
tfs-predict-theta          # Predict operator occupancy
tfs-cat-response           # Fit categorical response curves
tfs-diagnose-nan           # Diagnose NaN issues in inference
tfs-simulate               # Simulate a full experiment
tfs-setup-sim-grid         # Set up grid of simulation runs
tfs-setup-grid             # Set up grid of model configs
tfs-summarize-grid         # Summarize grid results
tfs-summarize-fit          # Summarize a fitted model
tfs-summarize-sbc          # Summarize simulation-based calibration runs
tfs-subset-genotypes       # Subset genotype data
```

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
| `analysis/` | Downstream statistical analysis of inference outputs (cat_response, extract_epistasis) |
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

- **`model_orchestrator.py`** — `ModelOrchestrator`: top-level class orchestrating data loading, inference, and prediction.

- **`inference/run_inference.py`** — `RunInference`: coordinates JAX/Numpyro sampling (SVI or NUTS) and MAP estimation (optax).

- **`tensors/tensor_manager.py`** — `TensorManager`: maps ragged per-genotype observations into JAX-compatible tensors.

- **`generative/registry.py`** — `model_registry` dict mapping component names to module implementations. This is where all swappable components are registered.

- **`generative/components/`** — Pluggable model components selected via YAML config:
  - `activity/`: `fixed`, `hierarchical_geno`, `hierarchical_mut`, `horseshoe_geno`, `horseshoe_mut`
  - `growth/`: `linear`, `power`, `saturation`
  - `growth_transition/`: `instant`, `memory`, `baranyi`, `baranyi_k`, `baranyi_tau`, `two_pop`
  - `transformation/`: `empirical`, `logit_norm`, `single`
  - `theta/`: `categorical_geno`, `hill_geno`, `hill_mut`; thermodynamic partition-function variants under `theta/thermo/` (lac dimer and MWC dimer, with/without unfolded state, PK/PnnC/PddG parameterizations)
  - `theta_rescale/`: `passthrough`, `logit`
  - `dk_geno/`: `fixed`, `hierarchical_geno`
  - `noise/`: `zero`, `beta`, `logit_normal` (theta observation noise)
  - `growth_noise/`: `zero`, `normal_kt`
  - `ln_cfu0/`: `hierarchical`, `hierarchical_factored`
  - `sample_offset/`: `zero`, `normal`
  - `observe/`: `binding`, `growth` (observation likelihood layers)

### Adding a New Model Component

1. Create `src/tfscreen/tfmodel/generative/components/<category>/myname.py`
2. Implement the required interface (follow an existing component as reference)
3. Register it in `tfmodel/generative/registry.py` under the appropriate category key
4. Add a test in `tests/tfscreen/tfmodel/components/<category>/`

### Key Abstractions

**`TensorManager`** (`tfmodel/tensors/tensor_manager.py`): Handles ragged tensors. Genotypes have different numbers of observations; this class pads and indexes into JAX-compatible arrays.

**`DataClass` / `PriorsClass`** (`tfmodel/data_class.py`): Flax pytree dataclasses holding structured experimental data and prior specifications for JAX compilation.

### Configuration (YAML)

See **YAML Standards** below for the full conventions. The tfmodel config (`tfs_configure_config.yaml`) is generated by `tfs-configure-model` and drives `tfs-fit-model`. It contains `data`, `components`, `priors_file`, and `guesses_file` sections. Do not hand-edit it.

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

The optional `binding_data` YAML block configures calibration genotypes for which measured binding curves are available. It has two sub-paths that can coexist:

**Path 1: Simulated genotypes** (`binding_data.genotypes` list)
- `wt` gets its natural unperturbed prior-predictive reference curve.
- Non-wt entries are drawn via a stratified (greedy maximin) algorithm across binding concentrations, ensuring diverse coverage.
- The selected theta values at *growth* concentrations are injected via `theta_gc_override` so that the simulated growth data is consistent with the binding data.

**Path 2: Measured Hill parameters** (`binding_data.genotype_params_file`)
- Reads a CSV with columns `genotype, theta_low, theta_high, log_hill_K, hill_n`.
- Only supported for Hill-based theta components (`hill_geno`, `hill_mut`).
- **`theta_low` and `theta_high` are clamped to `[1e-4, 1-1e-4]` at read time** with a `UserWarning`. This is critical: a value of e.g. `1.000004` (a common float-rounding artefact) maps to `logit ≈ +16`, making all per-mutation deltas ~13–15 σ under the `HalfNormal(1)` prior on delta scales and preventing inference from recovering reasonable theta values.
- For `hill_mut`: the function `build_theta_gc_override_hill_mut` assembles theta for **all library genotypes** (not just the measured ones) by additively combining per-mutation logit-space deltas. The WT reference is taken from `SimPriors` defaults. Multi-mutant genotypes not directly measured in the CSV are assembled from single-mutant deltas; directly-measured multi-mutants use their CSV values directly.
- Measured genotypes override any earlier simulated-path values in `theta_gc_override`.

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
| `simulate/sample_theta.py` | `sample_theta_prior` (prior-predictive) and `sample_theta_stratified` (greedy maximin) |
| `simulate/sim_data_class.py` | `SimData` container and `build_sim_data` factory |
| `simulate/build_sample_dataframes.py` | Constructs sample/timepoint DataFrames from simulation config |
| `simulate/selection_experiment.py` | Models the selection experiment (growth + sequencing) |
| `simulate/run_simulation.py` | End-to-end simulation runner called by `tfs-simulate` |

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

### Shared grid utilities

Both grid CLIs import from `tfscreen.util.grid_utils` for run-name generation, Jinja2 environment setup, and config-path rewriting.  Add new reusable grid helpers there rather than duplicating them.

### `examples/` directory

| File | Purpose |
|------|---------|
| `simulate/simulate_config.yaml` | Canonical well-commented simulate config reference |
| `simulate/simulate_grid.yaml` | Example simulate grid for `tfs-setup-sim-grid` |
| `simulate/run.sh` | Jinja2 shell template rendered into each simulate grid run subdir |
| `simulate-and-analyze/simulate_config.yaml` | Combined simulate + analyze workflow config |
| `simulate-and-analyze/hill_params.csv` | Example Hill parameter CSV for binding data input |
| `simulate-and-analyze/run.sh` | Jinja2 shell template for simulate-and-analyze runs |
| `tfmodel/grid.yaml` | Example tfmodel grid for `tfs-setup-grid` |
| `tfmodel/run.srun` | Jinja2 SLURM template rendered into each tfmodel grid run subdir |
| `process_raw/library_config.yaml` | Minimal library genetics config for `tfs-process-fastq` |

### `process_raw` YAML

`tfs-process-fastq` accepts a `run_config` argument that is passed to `LibraryManager`. It reads only the library genetics keys: `reading_frame`, `wt_seq`, `degen_sites`, `tiles`, `expected_5p`, `expected_3p`, `tile_combos`, `spiked_seqs`. All other keys are ignored.

**Recommended practice**: maintain one `run_config.yaml` per experiment and pass it to both `tfs-simulate` and `tfs-process-fastq` — no need for a separate file. Use `examples/process_raw/library_config.yaml` only when you need a minimal standalone library config (e.g. for processing real data without a matching simulation).

### Terminology

- **Condition**: unique growth setting (marker + selection) — same avg growth rate for same genotype
- **Sample**: one experimental tube (replicate + condition)
- **Timepoint**: one aliquot (replicate + condition + time)
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

Positional argument order (use only what the script needs):
1. `config_file` — path to YAML config
2. `posterior_file` — path to posteriors `.h5`/`.npz`
3. `theta_file` — path to theta CSV (for `tfs-cat-response`)

### Output flag

Always `--out_prefix` (never `--out_root`, `--out`, or `--output_file`). The function parameter must also be named `out_prefix`.

### File-backed list arguments

When a list of genotypes, titrant names, or concentrations is needed, the `_cli` wrapper takes file-path strings (one value per line, `#` comments allowed). Use `manual_arg_types` in `generalized_main` to override the `NoneType` inferred from `default=None`. The shared helper `_read_lines(path)` lives in `tfscreen.util.cli`.

### `in_training_data` column

`tfs-predict-growth` and `tfs-predict-theta` output a boolean column `in_training_data` (1/0) at the `(genotype, titrant_name, titrant_conc)` tuple level.

### Registered entry points

All scripts under `tfmodel/scripts/` and `analysis/cat_response/` follow the `_cli.py` naming convention and are registered in `pyproject.toml`.
