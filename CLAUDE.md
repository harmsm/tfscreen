# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tfscreen` is a Python library for simulating and analyzing high-throughput screens of transcription factor (TF) libraries. It models bacterial growth in plasmid-based libraries where TFs regulate selection markers (antibiotic resistance or pheS/4CP). The core analysis is a hierarchical Bayesian model (Pyro/PyTorch) that infers per-genotype TF activity from growth data.

This repository is the **PyTorch/Pyro port** of the original JAX/NumPyro codebase, maintained on branch `pyro-port`. The original lives in the `tfscreen/` git worktree (branch `main`). The motivation for the port is to enable native LigandMPNN integration as a `theta/ligandmpnn.py` component — both use PyTorch, eliminating JAX↔PyTorch tensor bridging.

## Commands

### Testing
```bash
# Run all unit tests
pytest tests/tfscreen

# Run a single test file
pytest tests/tfscreen/analysis/hierarchical/growth_model/test_model.py

# Run slow tests too (includes smoke tests)
pytest tests/tfscreen --runslow
pytest tests/smoke-tests --runslow

# Run NUTS equivalence tests (very slow, requires reference fixtures)
pytest tests/smoke-tests/test_nuts_equivalence.py --runnuts

# Run with coverage
coverage run --branch -m pytest tests/tfscreen --runslow
```

No `NUMBA_DISABLE_JIT=1` prefix needed — Numba is not used in this PyTorch version.

### Linting
```bash
# Fatal errors only
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full lint (non-fatal)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127
```

### CLI Entry Points
```
tfs-process-fastq              # FASTQ → read counts
tfs-process-counts             # counts → ln_cfu DataFrames
tfs-growth-analysis            # Main hierarchical Bayesian inference
tfs-configure-growth-analysis  # Generate YAML config template
tfs-summarize-posteriors       # Summarize posterior samples
tfs-predict                    # Make predictions from fitted model
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

- **`model.py`** — Pyro probabilistic model definition. The generative model is:
  `ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre·A·θ)·t_pre + (k_sel + dk_geno + m_sel·A·θ)·t_sel`
  where θ = operator occupancy, A = per-genotype TF activity, dk_geno = pleiotropic growth effect.

- **`model_class.py`** — `ModelClass`: top-level class orchestrating data loading, inference, and prediction. Exposes `pyro_model` and `pyro_model_guide` attributes.

- **`run_inference.py`** — `RunInference`: coordinates Pyro SVI (MAP via `AutoDelta`) and MCMC (NUTS). The optimization loop is an explicit Python `for` loop — there is no `jax.lax.scan` equivalent. Also exposes `run_nuts()` for diagnostic MCMC runs.

- **`tensor_manager.py`** — `TensorManager`: maps ragged per-genotype observations into PyTorch-compatible tensors.

- **`registry.py`** — `model_registry` dict mapping component names to module implementations.

- **`components/`** — Pluggable model components selected via YAML config:
  - `activity/`: `fixed`, `hierarchical`, `horseshoe`, `hierarchical_mut`
  - `growth/`: `linear`, `linear_independent`, `linear_fixed`, `power`, `saturation`
  - `growth_transition/`: `instant`, `memory`, `baranyi`
  - `transformation/`: `empirical`, `logit_norm`, `single`
  - `theta/`: `categorical`, `hill`, `hill_mut`
  - `dk_geno/`: `fixed`, `hierarchical`, `hierarchical_mut`
  - `noise/`: `zero`, `beta`

### Adding a New Model Component

1. Create `src/tfscreen/analysis/hierarchical/growth_model/components/<category>/myname.py`
2. Implement the required interface (follow an existing component as reference)
3. Register it in `registry.py` under the appropriate category key
4. Add a test in `tests/tfscreen/analysis/hierarchical/growth_model/components/<category>/`

### Key Abstractions

**`FitManager`** (`fitting/fit_manager.py`): General regression wrapper. Set parameter transformations, bounds, fixed vs. free parameters, and a model function via `set_model_func()`.

**`TensorManager`** (`analysis/hierarchical/tensor_manager.py`): Handles ragged tensors. Genotypes have different numbers of observations; this class pads and indexes into PyTorch-compatible arrays.

**`DataClass` / `PriorsClass`** (`analysis/hierarchical/growth_model/data_class.py`): Standard Python `@dataclass(frozen=True)` instances holding structured experimental data and prior specifications.

### Key Differences from the JAX/NumPyro Version

- **No `dim=` in `pyro.plate()`** — Pyro infers plate dimensions from tensor shapes; all `dim=-N` arguments have been removed.
- **`pyro_model` / `pyro_model_guide`** — renamed from `jax_model` / `jax_model_guide`.
- **SVI loop** — explicit Python `for` loop; no `jax.lax.scan`. Slightly slower per-step for small models but functionally equivalent.
- **NUTS** — available via `RunInference.run_nuts()` for diagnostics. Pyro's NUTS runs the tree-traversal in Python (vs. XLA-compiled in NumPyro), so it is significantly slower; use `jit_compile=True` to partially mitigate.
- **`pyro.poutine`** — replaces `numpyro.handlers` for trace/condition/seed handlers.
- **Posteriors** — stored identically (HDF5); the downstream extraction and prediction code is framework-agnostic.

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

- `--runslow` is required for smoke tests and the NUTS equivalence test
- The NUTS equivalence test (`tests/smoke-tests/test_nuts_equivalence.py`) requires a pre-generated reference JSON at `tests/fixtures/nuts_gold_standard_reference.json`; generate it once with `NUMBA_DISABLE_JIT=1 python tests/fixtures/generate_nuts_reference.py` in the NumPyro environment, then copy to `tests/fixtures/` here
- The `@pytest.mark.requires_torch` marker skips tests if PyTorch/Pyro are not installed (used to allow the two worktrees to share test infrastructure)
