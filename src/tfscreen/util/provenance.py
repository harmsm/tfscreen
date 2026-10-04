"""
Provenance of a run: the tfscreen version, the git commit it ran from and
the command line.

Every ``tfs-*`` command prints this when it starts and writes it to
``{out_prefix}_provenance.json`` (see ``generalized_main``). Writers that keep
their own records (configs, checkpoints) embed ``get_provenance()``.
"""

import datetime
import json
import os
import platform
import shlex
import subprocess
import sys

from tfscreen.__version__ import __version__

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set by generalized_main for the run in progress; None outside a CLI run.
_COMMAND = None

# git_state() of this process, looked up once (checkpoints ask often).
_GIT_STATE = None


def _git(*args):
    """Run git in the package directory; None if git or a repository is missing."""
    try:
        out = subprocess.run(["git", "-C", _PACKAGE_DIR, *args],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state():
    """
    The commit the package was imported from.

    Returns
    -------
    dict
        ``commit`` (full hash, or None when the package is not inside a git
        checkout, as for an installed wheel) and ``dirty`` (True when tracked
        files differ from that commit, None when unknown).
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def set_command(argv):
    """Record the command line of the run in progress (called by generalized_main)."""
    global _COMMAND
    _COMMAND = " ".join(shlex.quote(a) for a in argv)


def get_provenance():
    """
    Provenance of the current process.

    Returns
    -------
    dict
        ``tfscreen_version``, ``commit``, ``dirty``, ``command`` (None outside
        a ``tfs-*`` run), ``python``, ``host``, ``cwd`` and ``time`` (local,
        ISO 8601).
    """
    global _GIT_STATE
    if _GIT_STATE is None:
        _GIT_STATE = git_state()
    prov = {"tfscreen_version": __version__}
    prov.update(_GIT_STATE)
    prov.update({
        "command": _COMMAND,
        "python": platform.python_version(),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    return prov


def format_provenance(prov):
    """One line for a log: version, commit (with a dirty flag) and command."""
    commit = prov.get("commit")
    if commit is None:
        where = "no git commit"
    else:
        where = f"commit {commit[:10]}" + (" (dirty)" if prov.get("dirty") else "")
    line = f"tfscreen {prov['tfscreen_version']}, {where}"
    if prov.get("command"):
        line += f": {prov['command']}"
    return line


def write_provenance(path, prov=None):
    """Write provenance (default: ``get_provenance()``) to a JSON file."""
    if prov is None:
        prov = get_provenance()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    return path


def print_provenance(prov=None, stream=None):
    """Print the one-line provenance summary (to stderr by default)."""
    if prov is None:
        prov = get_provenance()
    if stream is None:
        stream = sys.stderr
    print(format_provenance(prov), file=stream, flush=True)
