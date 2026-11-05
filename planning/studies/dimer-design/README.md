# dimer-design: the NSF background design on the full dimer model

Plan step 4 of `planning/dimer-thermo-model.md` (4c-4e), 2026-11-04.

## Question

The NSF proposal's design analysis
(`~/work/Documents/writing/grants/2026-11-20_nsf-grant/analyses/2026-11-04_results-narrative.md`)
chose five separated double backgrounds and their TF levels by Fisher
information on a reduced seven-primitive ensemble at wt, and predicted
per-parameter standard errors of 0.09-0.19 kcal/mol. Its handoff asks for
the same calculation on the full topology, over the library. Does the
design resolve the per-mutation constants when the truth is
`thermo.dimer.PddG` and the fit is the real pipeline? Does it beat a panel
of random backgrounds on the same read budget, and how does the
structural prior cope with a table that over-predicts?

## Decision fed

Whether the proposal's design claim survives the full model, in time for
the proposal (due 2026-11-20). These are pilots, as in dimer-utility's
A3/A4: undercoverage is acceptable short of a collapse.

## Design

Truth: `thermo.dimer.PddG` with monomers and the unfolded monomer on and
the unfolded dimer off, from the 2026-11-03 state table (20 states; the
10 the model reads) in kcal/mol at 310.15 K, plus 0.5 kcal/mol noise per
mutation and state. wt is the model default except the monomer terms,
set from the narrative's assumptions: folding -4 kcal/mol per monomer
(ln K_m_u = -6.49) and dimerization -15 kcal/mol (ln K_d_h = 24.34). Our
wt is calibrated at 10 µM TF and the narrative's at 0.1 µM; free monomer
goes as sqrt(Kd/[TF]), so 100-fold TF is about 2.8 kcal/mol of
dimerization, and the -12/-15/-18 dial brackets the narrative's case.

The unfolded dimer is off because, with it on, a dimer state's ΔΔG_H also
drives Δln K_u: the folding background L114T/I289D then unfolds
completely (θ ≈ 0 at every TF level, checked 2026-11-04). The narrative's
ensemble has no U2. Without it the dimer sink buffers that background,
as the narrative describes, and its library spreads over θ (interquartile
0.40-0.99 at TF mid, apo).

Library: the A-series smoke design subsampled to 50 + 50 singles and 25
doubles, the same members in every background (about 134 genotypes per
background). Each background is its own strain, so its own library per
marker; it carries the reference's design without spikes. A background's
own mutations add without a pair term (dimer-utility A4).

Arms, one read budget (the N arm's 218 tubes at A2's reads per tube):

| arm | backgrounds | tubes |
|---|---|---|
| W | reference only, A2's 118-tube design at TF mid | 118 |
| N | W plus the narrative's five doubles, a 20-tube block each at its TF level | 218 |
| R | W plus 15 random single-mutant backgrounds at TF mid (the count control) | 418 |

The narrative's five: L114T+I289D (folding, hi), R51P+C281D (interface,
hi), S61P+D149M (conformational and effector, mid), S97A+G187E
(conformational and effector, lo), R51S+D149M (conformational, mid). R51P,
R51S, S61P and S97A lie inside the amplicon (27-98), so those backgrounds
carry an edited `wt_seq`; R51 and S97 are also library sites, and
`backgrounds.background_mutations` now takes the majority so a library
mutation there (R51X) does not drop R51P from the additive set. The random
15 are single mutations outside the amplicon, one per residue, with every
state within 2 kcal/mol (a proxy for leaving a readable curve), seed
20261104. TF levels: mid 10 µM held; lo 3.3 µM and hi 30 µM learned (ln
SD 0.5). One operator.

Runs (8 simulations x 3 fits = 24):

| run | arm | table slope c | dimerization (kcal/mol) |
|---|---|---|---|
| W_c1, N_c1, R_c1 | W, N, R | 1 | -15 |
| W_c03, N_c03, R_c03 | W, N, R | 0.3 | -15 |
| N_c1_d12, N_c1_d18 | N | 1 | -12, -18 |

c is `ddG_truth_scale`: truth = c x table + noise, while the fit's prior
stays centered on the table. 0.3 is the middle of the narrative's 3-20
fold compression.

Fits (`run_fit.sh`), every one with the truth's variant, the pair term,
the contexts file and the Oii scales held at 1 kT: `PK` (every scale held
at 1 kT), `PddG_s1` (σ_s held at 1 kT) and `PddG_dial` (σ_s held at the
truth noise, 0.811 kT). Joint MAP does not estimate these scales (A2:
σ_s collapsed to 0), so a learned-σ_s arm would need dimer-utility's EM
loop, several MAPs each; the two held values stand in for it. Staged MAP
capped at 50,000 epochs a stage, then the arrowhead Laplace.

## How to run

`make_configs.py` writes the run configs (`runs/`, not tracked) and the
arm designs and contexts (one per arm, naming only the TF levels
its design uses) in `inputs/`, which also holds the base config,
the A2 design, the OD600 calibration and the state table. `submit.sh`
runs it once before it submits anything. On the cluster, from this directory:

```bash
bash submit.sh
```

submits every run (`bash submit.sh N_c1` for one): a simulate job, then
per arm a configure-and-MAP job and a Laplace-to-summary job after it, 6 h
each on `gpu`. The MAP checkpoints every 5000 epochs, so a job that times
out resumes when resubmitted. Locally: `bash run_fit.sh N_c1 PK` after
`python make_configs.py` and simulating with `tfs-simulate
tfs_sim_config.yaml --out_prefix sim/tfs_sim --no_write_raw` in
`runs/N_c1`.

## Inputs and commit

`inputs/2026-11-03_laci-state-ddg.csv` is the grant analysis's state
table. Smoke-tested on the laptop (2026-11-04, N_c1 with a 200-epoch MAP
and the point posterior, PK and PddG_s1): configure, MAP, extract, predict
and summarize run; N_c1 has 805 genotype x background combinations and
39,432 MAP parameters.

## Results

Not run yet. `score.py` (to come) will score the per-mutation constants
by arm against the narrative's predicted standard errors, and θ as in
dimer-utility.
