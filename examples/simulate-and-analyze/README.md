# simulate-and-analyze

A complete end-to-end example: simulate a synthetic TF-library screen and fit
the hierarchical Bayesian model to the simulated data.

## Contents

| File | Purpose |
|------|---------|
| `simulate_config.yaml` | Simulation parameters (library genetics, conditions, binding data) |
| `hill_params.csv` | Per-genotype Hill binding parameters used to generate ground-truth θ |
| `run.sh` | Pipeline script (simulate → configure → calibrate → fit → predict → summarise) |

## Running

```bash
bash run.sh simulate_config.yaml out/ 1
```

The three arguments are the config file, the output directory (created
automatically), and the random seed.  Seed `1` reproduces the documented
example outputs.

Running time is roughly 20–60 minutes on a laptop CPU or a few minutes on a
GPU.  On a cluster, comment out the `XLA_FLAGS` line in `run.sh` and
uncomment `module load cuda/...` instead.

## Documentation

See the [quickstart](https://tfscreen.readthedocs.io/en/latest/quickstart.html)
for a guided walkthrough and the
[tfs-summarize-fit reference](https://tfscreen.readthedocs.io/en/latest/summarize-fit.html)
for a detailed guide to interpreting the diagnostic outputs in `out/summary/`.
