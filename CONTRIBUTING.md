# Contributing to tfscreen

Thanks for helping. This file covers how to propose a change and what a pull
request needs before it can merge.

## Before you start

For anything larger than a bug fix, open an issue first and describe what you
want to change and why. The model is a coupled system, and a change that looks
local often touches the priors, the guides, the simulator and the CLIs at once.
A short discussion up front saves rework.

Ideas, active plans and the studies behind them live in `planning/`. Check
there before starting work on the model; your idea may already have a plan or
a recorded reason it was set aside. See `planning/README.md`.

## Workflow

We develop on forks and merge into `main` through pull requests. Nobody pushes
directly to `main`.

1. Fork `harmslab/tfscreen` on GitHub and clone your fork.
2. Add the main repository as `upstream`:

   ```bash
   git remote add upstream git@github.com:harmslab/tfscreen.git
   ```

3. Create a branch for each change, named for what it does
   (`fix-presplit-offset`, `add-baranyi-guide`). Keep one logical change per
   branch.
4. Before opening the pull request, bring your branch up to date with
   `upstream/main` and rerun the tests.
5. Open a pull request into `harmslab/tfscreen:main`. The pull request
   template asks what changed, why, and how you verified it, and gives a
   checklist of everything below. Work through it before asking for review.

`main` is a protected branch. GitHub enforces the following:

- Every change goes through a pull request. Direct pushes and force pushes to
  `main` are blocked.
- A pull request needs one approving review.
- All CI checks must pass: the test matrix and the coverage job.
- The branch must be up to date with `main` before it can merge. If `main`
  moves while your pull request is open, merge or rebase onto it and let CI
  run again.
- Pushing new commits dismisses earlier approvals, so a reviewer has to
  approve the final version. Finish your changes before asking for a last
  review.

## Development setup

tfscreen needs Python 3.11 or later. Install it in editable mode with the test
dependencies:

```bash
pip install -e ".[test]"
```

## Tests

Every pull request needs full unit test coverage of the code it adds or
changes. In practice that means:

- Every new function, branch and error path has a test that exercises it.
  We measure branch coverage, not just line coverage.
- A bug fix comes with a test that fails before the fix and passes after it.
- Tests mirror the source layout: `src/tfscreen/foo/bar.py` is tested in
  `tests/tfscreen/foo/test_bar.py`.
- Tests that take more than a few seconds get `@pytest.mark.slow`. They are
  skipped by default and run with `--runslow`.
- End-to-end pipeline checks belong in `tests/smoke-tests/`.

`tests/conftest.py` sets `NUMBA_DISABLE_JIT=1`, so no prefix is needed. Run the
unit tests:

```bash
pytest tests/tfscreen
```

Add `-n auto` to run them in parallel (`pytest-xdist`, in the `test` extra).

Check coverage, including slow tests. Coverage settings (branch coverage,
source) live in `pyproject.toml`, so the same commands work locally and in CI:

```bash
pytest -n auto --runslow --cov --cov-report= tests/tfscreen
```

```bash
coverage report
```

To see what CI will say about the lines your branch changes:

```bash
coverage xml
```

```bash
diff-cover coverage.xml --compare-branch=upstream/main
```

Some tests check rules across the whole component registry, for example that
every per-genotype latent is mini-batch safe and that guides follow the
`{site}_loc` / `{site}_scale` naming convention. If one of these fails for a
new component, fix the component rather than the test.

CI runs the unit tests on Linux and macOS for Python 3.11, 3.12 and 3.13, plus
a lint check for fatal errors. A separate coverage job runs the tests with
`--runslow` and fails if either check misses:

- Every line a pull request adds or changes must be covered by a test
  (`diff-cover --fail-under=100`). diff-cover counts lines, not branches, so
  check the branch columns of `coverage report` for your files yourself.
- Total coverage of the package must not drop below the floor set in
  `.github/workflows/tests.yml`. Raise the floor when coverage improves.

Run the same lint locally:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Code conventions

`CLAUDE.md` is the detailed reference for the codebase. It is written for AI
coding tools but serves human contributors equally well. The parts you are
most likely to need:

- **New model components:** the checklist under "Adding a new model
  component", including registration and mini-batch safety.
- **CLI scripts:** the "CLI standards" section. Every `tfs-*` script uses
  `generalized_main`, lives in `<name>_cli.py`, writes with `--out_prefix`
  and is registered in `pyproject.toml`. Its docstring is its `--help`, one
  entry per parameter. After changing any CLI signature or docstring,
  regenerate the reference page with `python docs/scripts/make_cli_reference.py`;
  `tests/tfscreen/util/cli/test_cli_reference.py` fails until you do.
- **YAML files:** the "YAML standards" section.
- **Quantile outputs:** bare `q<level>` column names, no feature prefix.

### Docstrings

Docstrings follow the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) convention.
Every public function, class and method gets a full docstring: a one-line
summary, an extended description where needed, then `Parameters`, `Returns`
and `Raises` sections as they apply. Give each parameter's type and shape,
for example `numpy.ndarray of shape (num_genotype,)`, and say what units it
is in when that is not obvious.

```python
def detect_match_keys(dfs, value_columns=(), match_by=None):
    """
    Determine the columns that make a row "the same row" across runs.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        All tables that must share the key.
    value_columns : iterable of str, optional
        Columns holding estimates, excluded from the key.
    match_by : list of str, optional
        Explicit key columns. Overrides auto-detection.

    Returns
    -------
    list of str
        The match key columns.

    Raises
    ------
    ValueError
        If the key is not unique within a run.
    """
```

Private helpers (leading underscore) may use a one-line docstring when the
behavior is simple.

### Validation

Configuration errors should fail fast with a clear message. Do not silently
drop or coerce a value the user supplied.

### Keeping `CLAUDE.md` current

If your change alters how the code works, update `CLAUDE.md` in the same pull
request so it stays accurate.

## Changelog

Record every user-visible change in `CHANGELOG.md` under `[Unreleased]`, in
the same pull request as the change. The file follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Cite files and
functions rather than commit hashes.

## Commits

Write commit messages with an imperative subject line of 72 characters or
fewer ("Add soft-min congression dk rule", not "Added..."). Use the body to say
why the change was made and what it affects. Use American spelling.

Do not commit data, checkpoints, notebooks of exploratory work or other large
files. `dev/` is untracked scratch space.

## License

By contributing, you agree that your contributions are licensed under the
project's license (see `LICENSE`).
