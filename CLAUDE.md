# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription factor (TF) libraries. It models bacterial growth in plasmid-based libraries where TFs regulate selection markers (antibiotic resistance or pheS/4CP). The core analysis is a hierarchical Bayesian model (JAX/Numpyro) that infers per-genotype TF activity from growth data.

## Commands

### Testing
```bash
# Run all unit tests (NUMBA_DISABLE_JIT=1 is required)
NUMBA_DISABLE_JIT=1 pytest tests/tfscreen

# Run a single test file
NUMBA_DISABLE_JIT=1 pytest tests/tfscreen/analysis/hierarchical/growth_model/test_model.py

# Run slow tests too
NUMBA_DISABLE_JIT=1 pytest tests/tfscreen --runslow

# Run smoke tests
NUMBA_DISABLE_JIT=1 pytest tests/smoke-tests --runslow

# Run with coverage
NUMBA_DISABLE_JIT=1 coverage run --branch -m pytest tests/tfscreen --runslow
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
tfs-growth-analysis        # Main hierarchical Bayesian inference
tfs-configure-growth-analysis  # Generate YAML config template
tfs-summarize-posteriors   # Summarize posterior samples
tfs-predict                # Make predictions from fitted model
```

## Architecture

### Data Flow
```
FASTQ files
    → tfs-process-fastq → counts CSV
    → tfs-process-counts → ln_cfu DataFrame
    → tfs-growth-analysis (with run_config.yaml) → posteriors (HDF5 + pkl)
    → tfs-summarize-posteriors → summary CSV
    → tfs-predict → predictions CSV
```

### Source Layout (`src/tfscreen/`)

| Module | Responsibility |
|--------|---------------|
| `process_raw/` | FASTQ parsing, count normalization, ln_cfu calculation |
| `fitting/` | General-purpose regression (FitManager, least squares, WLS, NLS) |
| `models/` | Mathematical growth models and thermodynamic models (lac, EEE) |
| `genetics/` | Genotype library management, mutation effect combination |
| `calibration/` | Wildtype parameter extraction and calibration pipeline |
| `simulate/` | Full experiment simulation from thermodynamics to read counts |
| `analysis/` | Statistical inference (hierarchical Bayesian, independent, cat_response) |
| `util/` | Shared IO, DataFrame ops, numerical helpers, validation, CLI |
| `plot/` | Visualization (heatmaps, corner plots, error plots) |

### Core Analysis: `analysis/hierarchical/growth_model/`

The hierarchical Bayesian inference engine. Key files:

- **`model.py`** — Numpyro probabilistic model definition. The generative model is:
  `ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre·A·θ)·t_pre + (k_sel + dk_geno + m_sel·A·θ)·t_sel`
  where θ = operator occupancy, A = per-genotype TF activity, dk_geno = pleiotropic growth effect.

- **`model_class.py`** — `GrowthModel`: top-level class orchestrating data loading, inference, and prediction.

- **`run_inference.py`** — `RunInference`: coordinates JAX/Numpyro sampling (NUTS) and MAP estimation (optax).

- **`tensor_manager.py`** — `TensorManager`: maps ragged per-genotype observations into JAX-compatible tensors.

- **`registry.py`** — `model_registry` dict mapping component names to module implementations. This is where all swappable components are registered.

- **`components/`** — Pluggable model components selected via YAML config:
  - `activity/`: `fixed`, `hierarchical`, `horseshoe`
  - `growth/`: `linear`, `linear_independent`, `linear_fixed`, `power`, `saturation`
  - `growth_transition/`: `instant`, `memory`, `baranyi`
  - `transformation/`: `empirical`, `logit_norm`, `single`
  - `theta/`: `categorical`, `hill`
  - `dk_geno/`: `fixed`, `hierarchical`
  - `noise/`: `zero`, `beta`

### Adding a New Model Component

1. Create `src/tfscreen/analysis/hierarchical/growth_model/components/<category>/myname.py`
2. Implement the required interface (follow an existing component as reference)
3. Register it in `registry.py` under the appropriate category key
4. Add a test in `tests/tfscreen/analysis/hierarchical/growth_model/components/<category>/`

### Key Abstractions

**`FitManager`** (`fitting/fit_manager.py`): General regression wrapper. Set parameter transformations, bounds, fixed vs. free parameters, and a model function via `set_model_func()`.

**`TensorManager`** (`analysis/hierarchical/tensor_manager.py`): Handles ragged tensors. Genotypes have different numbers of observations; this class pads and indexes into JAX-compatible arrays.

**`DataClass` / `PriorsClass`** (`analysis/hierarchical/growth_model/data_class.py`): Flax pytree dataclasses holding structured experimental data and prior specifications for JAX compilation.

### Configuration (YAML)

The `run_config.yaml` drives `tfs-growth-analysis`. Key sections:
- `library`: defines the genetic library (WT sequence, degenerate sites, sub-libraries)
- `observable_calculator`: thermodynamic model (`lac`, `eee`, `linkage_dimer`) and parameters
- `condition_blocks`: growth conditions (marker, titrant concentrations, selection timepoints)
- Model component selections (which `activity`, `growth`, `transformation`, etc. modules to use)
- `calibration_file`: path to wildtype calibration JSON

### Terminology

- **Condition**: unique growth setting (marker + selection + IPTG concentration) — same avg growth rate for same genotype
- **Sample**: one experimental tube (replicate + condition)
- **Timepoint**: one aliquot (replicate + condition + time)
- **theta (θ)**: operator occupancy — fraction of operators bound by TF
- **A**: per-genotype TF activity (the primary output of inference)
- **dk_geno**: pleiotropic growth effect of mutation independent of TF activity

### Testing Notes

- Always set `NUMBA_DISABLE_JIT=1` when running tests — Numba JIT causes test failures
- Slow tests (marked `@pytest.mark.slow`) are skipped by default; use `--runslow` to include them
- Smoke tests live in `tests/smoke-tests/` and test end-to-end pipelines
