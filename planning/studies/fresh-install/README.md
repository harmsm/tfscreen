# Fresh install

**Question.** Does a new lab member's install of tfscreen work: a clean
clone, a new virtual environment, `pip install -e ".[cuda12]"` with the
release's constraints, on a cluster GPU node?

**Decision fed.** `planning/release-0.5.0.md`, items 11 and 13: whether
0.5.0's install instructions and dependency floors are right before
Natalie's dry run.

**How to run.** On the cluster, after pulling the commit that adds this
study and pushing it to GitHub (the job clones from GitHub, not from the
cluster checkout):

```bash
sbatch --export=ALL,OUT=/gpfs/projects/harmslab/harms/studies/fresh-install/run1 /gpfs/home/harms/tfscreen/planning/studies/fresh-install/fresh_install.srun
```

`OUT` must not hold `tfscreen/` or `venv/` already. Set `PY` to a Python
3.11 or newer if `python3` on the node is older, and `REF` to test a tag.

**What it checks, in order:** the clone at `REF`; the environment
(`installed.txt` is its `pip freeze`); that jax sees the GPU (the job
fails here if pip installed CPU-only jax); the unit tests
(`unit_tests.txt`); the quickstart example on the GPU
(`quickstart/run.log`, `quickstart/out/`).

**Inputs.** None beyond the repository.

**Pass.** jax on the GPU, the unit tests as on CI, and the quickstart
running to its summary.

## Results

### Local fresh install (2026-10-20)

A new venv on the maintainer's laptop (macOS, Python 3.13.11),
`pip install -e ".[test]"` with no constraints, took the newest releases
(pandas 3.0.6, numpy 2.5.4, jax 0.12.0, scipy 1.18.1, numba 0.68.0).
The full unit suite passed (6,127 passed, 18 skipped, `pytest -n 4`,
8.7 min). Its `pip freeze` became `constraints/tfscreen-0.5.0.txt`.

A dry-run resolve of that set with `jax[cuda12]` for Linux x86-64
(`pip install --dry-run --platform manylinux... --only-binary=:all:`)
gives 83 packages for Python 3.12, with jax, jaxlib, jax-cuda12-plugin
and jax-cuda12-pjrt all at 0.12.0 (pybktree left out of the check: it
ships only a source package, which pip does not cross-resolve). For
Python 3.11 it fails: contourpy 1.4.0, at least, has dropped 3.11. The
constraints file and the install docs say Python 3.12 or newer; without
`-c`, a 3.11 install takes older releases, as CI's 3.11 jobs do.

### Cluster fresh install

Not run yet (Claude's ssh access needs the user's two-factor login).
