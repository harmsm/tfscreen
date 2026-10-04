# Staged level-offset MAP

**Question.** Does `tfs-fit-model`'s staged MAP (`--stage_offsets auto`)
reach the hand-chained real-data fit from one command? A level-offset MAP
started cold settled with the tube offsets near ±2.8. The real fit avoided
that by hand: a no-offset MAP, then each tube's best offset with everything
else held (`small_offsets.py`), then the joint MAP from there at step size
1e-4 (`make_warm_start.py`, `make_init_npz.py`, `check_warm.py`, six runs;
`../real-data-fit/README.md`).

**Decision it feeds.** Pipeline plan step 3
(`planning/experiment-pipeline.md`): whether the staged MAP replaces the
hand chain and those scripts retire. If it does, the shortened recipe for the
real fit is `run_staged.srun`.

## Design

The staged MAP (`tfmodel/inference/staged_map.py`) runs three MAPs on the
same model. Stage 1 holds the tube offsets at 0. Stage 2 holds every other
site at stage 1's MAP and fits the offsets alone. Stage 3 is the joint MAP
from stage 1's point plus stage 2's offsets, at `--staged_step_size` 1e-4.

Two arms on the dev data, both with `rel_off_n05`'s model: the growth-only
relative fit, level offsets with SD held at 0.17, log(n) population SD held
at 0.5, loose k/m priors (`growth_priors_loose.csv`, the old
`set_growth_priors.py loose`). Every setting is a `tfs-configure-model`
flag; nothing is edited by hand.

| arm | fit |
|---|---|
| `staged_off` | `--stage_offsets off`: the cold MAP. Expected to reproduce the ±2.8 offset mode. |
| `staged_auto` | `--stage_offsets auto`: the three stages. |

Both then run the arrowhead Laplace, extract parameters and score the count
likelihood (`score_counts.py`).

**Pass criteria** (the plan's done criterion for the real fit):
`staged_auto`'s count log-likelihood within a few thousand nats of
`rel_off_n05`'s, and every spike's log K and n median inside
`rel_off_n05_alap`'s 95% intervals. `staged_off` should fail them; if it
does not, the trap was not reproduced and the comparison says nothing about
staging.

A local check on the 483-genotype simulate-and-analyze library, cold against
staged, found no trap there (Results), as the relative-fit study's MAPs
suggested.

## How to run

On the cluster, in `planning/dev-data/real_fit/` (gitignored, not pulled),
with this repository at the commit that has `--stage_offsets`:

```bash
cp <repo>/planning/studies/staged-map/{run_staged.srun,growth_priors_loose.csv,compare.py} .
```

```bash
grep -c stage_offsets run_staged.srun
```

```bash
mkdir staged_auto && cd staged_auto && sbatch --export=ALL,STAGE=auto ../run_staged.srun
```

```bash
mkdir staged_off && cd staged_off && sbatch --export=ALL,STAGE=off ../run_staged.srun
```

When both are done, from `real_fit/`:

```bash
python compare.py rel_off_n05_alap staged_auto staged_off
```

## Inputs

`real_fit/inputs/growth.csv.gz` and `real_fit/inputs/library_config.yaml`
(the nine-spike config), as for `rel_off_n05`; `score_counts.py` from
`../real-data-fit/analysis/` (symlinked into `real_fit/`).

## Commit

The commit that adds `--stage_offsets`. Every run writes its provenance
(`*_provenance.json`, and the config's `provenance:` block).

## Results

Local check (2026-10-03, uncommitted working tree after 0.5.0; CPU). The
simulate-and-analyze library (483 genotypes, `examples/simulate-and-analyze/
simulate_config.yaml`, seed 7), configured as in `run_staged.srun` without
the growth-priors table, MAP with default settings, `--stage_offsets off`
against `auto`:

| arm | final loss | tube offsets (nonzero) | wall time |
|---|---|---|---|
| off (cold) | 161,083 | SD 0.32, -1.18 to +1.02 | 7.8 min |
| auto (staged) | 160,617 | SD 0.37, -1.15 to +1.07 | 16.8 min |

Stage 1 ran to the 100,000-epoch cap (loss 213,522 with no offsets), stage
2 converged in 28,000 steps (197,602; offsets SD 0.28), stage 3 ran to the
cap. The cold MAP did not fall into the offset trap on this library, so this
only shows that staging costs no fit (466 nats better) at about twice the
time. Every run hit the epoch cap. Whether staging avoids the trap is the
cluster arm's question.

Cluster runs: pending.
