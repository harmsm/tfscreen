import json

from tfscreen.util import provenance


def test_git_state_has_commit_in_checkout():
    state = provenance.git_state()
    assert set(state) == {"commit", "dirty"}
    # The test suite runs from a git checkout
    assert state["commit"] is None or len(state["commit"]) == 40


def test_git_state_without_git(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *a: None)
    assert provenance.git_state() == {"commit": None, "dirty": None}


def test_get_provenance_fields():
    provenance.set_command(["tfs-x", "a b", "--y"])
    prov = provenance.get_provenance()
    for key in ("tfscreen_version", "commit", "dirty", "command",
                "python", "host", "cwd", "time"):
        assert key in prov
    assert prov["command"] == "tfs-x 'a b' --y"


def test_format_provenance():
    prov = {"tfscreen_version": "9.9.9", "commit": "a" * 40,
            "dirty": True, "command": "tfs-x"}
    line = provenance.format_provenance(prov)
    assert line == "tfscreen 9.9.9, commit aaaaaaaaaa (dirty): tfs-x"

    prov.update(commit=None, command=None)
    assert provenance.format_provenance(prov) == "tfscreen 9.9.9, no git commit"


def test_write_provenance(tmp_path):
    path = provenance.write_provenance(str(tmp_path / "d" / "p.json"),
                                       {"tfscreen_version": "1"})
    assert json.load(open(path)) == {"tfscreen_version": "1"}
