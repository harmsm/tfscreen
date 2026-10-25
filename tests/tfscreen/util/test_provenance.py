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


def test_git_returns_none_when_git_cannot_run(monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert provenance._git("rev-parse", "HEAD") is None

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("git", 10)
    monkeypatch.setattr(provenance.subprocess, "run", timeout)
    assert provenance._git("rev-parse", "HEAD") is None


def test_git_returns_none_on_nonzero_exit(monkeypatch):
    import subprocess

    def fail(*a, **k):
        return subprocess.CompletedProcess(a, 128, stdout="", stderr="no repo")
    monkeypatch.setattr(provenance.subprocess, "run", fail)
    assert provenance._git("rev-parse", "HEAD") is None


def test_git_returns_stripped_stdout(monkeypatch):
    import subprocess

    def ok(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="abc123\n", stderr="")
    monkeypatch.setattr(provenance.subprocess, "run", ok)
    assert provenance._git("rev-parse", "HEAD") == "abc123"


def test_write_provenance_defaults_to_current(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_GIT_STATE",
                        {"commit": "b" * 40, "dirty": False})
    monkeypatch.setattr(provenance, "_COMMAND", None)
    provenance.set_command(["tfs-y"])
    path = provenance.write_provenance(str(tmp_path / "p.json"))
    prov = json.load(open(path))
    assert prov["commit"] == "b" * 40
    assert prov["dirty"] is False
    assert prov["command"] == "tfs-y"


def test_print_provenance_defaults_to_current_and_stderr(capsys, monkeypatch):
    monkeypatch.setattr(provenance, "_GIT_STATE",
                        {"commit": "c" * 40, "dirty": True})
    monkeypatch.setattr(provenance, "_COMMAND", None)
    provenance.set_command(["tfs-z", "--a"])
    provenance.print_provenance()
    err = capsys.readouterr().err
    assert "commit cccccccccc (dirty): tfs-z --a" in err
