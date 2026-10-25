## What and why

<!-- What does this change do, and why is it needed? Link the issue or
planning/ file it addresses. -->

## How it was verified

<!-- Tests added, commands run, simulations or fits checked. -->

## Checklist

See [CONTRIBUTING.md](https://github.com/harmslab/tfscreen/blob/main/CONTRIBUTING.md) for details on each item.

- [ ] Branch is up to date with `upstream/main`
- [ ] New and changed code has full unit test coverage, including branches and error paths
- [ ] Bug fixes include a test that fails without the fix
- [ ] Slow tests are marked `@pytest.mark.slow`
- [ ] `pytest tests/tfscreen --runslow` passes locally
- [ ] `diff-cover coverage.xml --compare-branch=upstream/main` reports 100%
- [ ] `flake8 . --count --select=E9,F63,F7,F82` reports no errors
- [ ] Public functions, classes and methods have numpydoc docstrings
- [ ] New components, CLIs and YAML keys follow the conventions in `CLAUDE.md`
- [ ] `CLAUDE.md` is updated if the change alters how the code works
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` for any user-visible change
- [ ] No data, checkpoints or other large files are committed
