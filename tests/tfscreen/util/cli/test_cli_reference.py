"""docs/source/cli.rst must match what the commands' parsers print."""
import importlib.util
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
SCRIPT = os.path.join(ROOT, "docs", "scripts", "make_cli_reference.py")


def _module():
    spec = importlib.util.spec_from_file_location("make_cli_reference", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_entry_point_builds_a_parser():
    mod = _module()
    for name, target in mod.entry_points():
        text = mod.help_text(name, target)
        assert text.startswith(f"usage: {name}"), name


def test_cli_reference_is_current():
    mod = _module()
    with open(mod.OUT) as fh:
        current = fh.read()
    assert current == mod.render(), (
        "docs/source/cli.rst is stale; run "
        "python docs/scripts/make_cli_reference.py")
