# simulate-and-analyze

A complete example: simulate a small TF-library screen, then fit the
hierarchical Bayesian model to the simulated data and compare the fit with
the known truth.

## Contents

| File | Purpose |
|------|---------|
| `simulate_config.yaml` | Simulation settings (library genetics, conditions, growth, binding and presplit data); also the library description for `tfs-configure-model --library_config` |
| `hill_params.csv` | Hill parameters for wt and three spiked single mutants, used as their true θ curves and binding data |
| `run.sh` | The pipeline: simulate, configure, pre-fit, fit, sample the posterior, extract, predict, summarize |

## Running

Run from this directory, since the config names `hill_params.csv` by a
relative path:

```bash
bash run.sh simulate_config.yaml out 1
```

The arguments are the simulate config, the output directory (created if
needed) and the random seed (default 1).

The fit is the slow step: expect tens of minutes on a laptop CPU and a few
minutes on a GPU. On a cluster, comment out the `XLA_FLAGS` line in `run.sh`
and uncomment `module load cuda/...`; the `#SBATCH` lines let you submit it
with `sbatch`.

## Documentation

The [quickstart](https://tfscreen.readthedocs.io/en/latest/quickstart.html)
walks through every step and every output file. The
[tfs-summarize-fit guide](https://tfscreen.readthedocs.io/en/latest/summarize-fit.html)
explains the diagnostic outputs in `out/summary/`.
