"""Tests for path-rewriting helpers (now in grid_utils, re-exported from setup_grid_cli)."""

import os

import yaml

from tfscreen.util.grid_utils import (
    relativize_config_paths as _relativize_config_paths,
    relativize_node as _relativize_node,
    relativize_template_vars as _relativize_template_vars,
)


# ---------------------------------------------------------------------------
# _relativize_node
# ---------------------------------------------------------------------------

def test_relativize_node_absolute_existing(tmp_path):
    """Absolute path to an existing file is rewritten relative to subdir."""
    f = tmp_path / "data.csv"
    f.write_text("x")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    result = _relativize_node(str(f), str(subdir))
    assert result == os.path.relpath(str(f), str(subdir))


def test_relativize_node_absolute_missing(tmp_path):
    """Absolute path to a non-existent file is left unchanged."""
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    missing = str(tmp_path / "ghost.csv")
    assert _relativize_node(missing, str(subdir)) == missing


def test_relativize_node_relative_string(tmp_path):
    """Relative (non-absolute) strings are never touched."""
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    assert _relativize_node("data/foo.csv", str(subdir)) == "data/foo.csv"


def test_relativize_node_non_string(tmp_path):
    """Non-string values are returned unchanged."""
    subdir = str(tmp_path / "run_0001")
    os.makedirs(subdir)
    assert _relativize_node(42, subdir) == 42
    assert _relativize_node(None, subdir) is None
    assert _relativize_node(True, subdir) is True


def test_relativize_node_dict(tmp_path):
    """Dict values with absolute paths are recursively rewritten."""
    f = tmp_path / "binding.csv"
    f.write_text("x")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    node = {"binding_df": str(f), "n_samples": 100}
    result = _relativize_node(node, str(subdir))
    assert result["binding_df"] == os.path.relpath(str(f), str(subdir))
    assert result["n_samples"] == 100


def test_relativize_node_list(tmp_path):
    """List values with absolute paths are recursively rewritten."""
    f = tmp_path / "file.csv"
    f.write_text("x")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    result = _relativize_node([str(f), "unchanged"], str(subdir))
    assert result[0] == os.path.relpath(str(f), str(subdir))
    assert result[1] == "unchanged"


def test_relativize_node_nested(tmp_path):
    """Nested dicts/lists are handled recursively."""
    f = tmp_path / "deep.csv"
    f.write_text("x")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    node = {"outer": {"inner": str(f)}}
    result = _relativize_node(node, str(subdir))
    assert result["outer"]["inner"] == os.path.relpath(str(f), str(subdir))


# ---------------------------------------------------------------------------
# _relativize_config_paths
# ---------------------------------------------------------------------------

def test_relativize_config_paths_data_section(tmp_path):
    """Absolute paths in the data section are rewritten to relative."""
    f = tmp_path / "binding.csv"
    f.write_text("x")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    cfg = {"data": {"binding_df": str(f), "n_samples": 100}}
    yaml_path = subdir / "config.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    _relativize_config_paths(str(yaml_path), str(subdir))

    with open(yaml_path) as fh:
        result = yaml.safe_load(fh)
    assert not os.path.isabs(result["data"]["binding_df"])
    assert result["data"]["n_samples"] == 100


def test_relativize_config_paths_top_level(tmp_path):
    """Absolute paths outside the data section are also rewritten."""
    cal = tmp_path / "calibration.json"
    cal.write_text("{}")
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    cfg = {"calibration_file": str(cal), "n_samples": 200}
    yaml_path = subdir / "config.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    _relativize_config_paths(str(yaml_path), str(subdir))

    with open(yaml_path) as fh:
        result = yaml.safe_load(fh)
    assert not os.path.isabs(result["calibration_file"])
    assert result["n_samples"] == 200


def test_relativize_config_paths_no_change(tmp_path):
    """Config with no absolute paths is not rewritten."""
    subdir = tmp_path / "run_0001"
    subdir.mkdir()
    cfg = {"data": {"binding_df": "../../data/binding.csv"}}
    yaml_path = subdir / "config.yaml"
    original = yaml.dump(cfg)
    yaml_path.write_text(original)
    mtime_before = yaml_path.stat().st_mtime

    _relativize_config_paths(str(yaml_path), str(subdir))

    assert yaml_path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# _relativize_template_vars
# ---------------------------------------------------------------------------

def test_relativize_template_vars_existing_file(tmp_path):
    """A relative path in a template var that resolves to an existing file is rewritten."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    geno = data_dir / "genotypes.csv"
    geno.write_text("x")

    grid_yaml_dir = str(tmp_path)
    subdir = str(tmp_path / "grid" / "run_0001")
    os.makedirs(subdir)

    tmpl_vars = {"genotypes_file": "data/genotypes.csv", "seed": 42}
    result = _relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir)

    assert not os.path.isabs(result["genotypes_file"])
    # Path from run subdir should navigate up to tmp_path/data/genotypes.csv
    resolved = os.path.normpath(os.path.join(subdir, result["genotypes_file"]))
    assert resolved == str(geno)
    assert result["seed"] == 42


def test_relativize_template_vars_nonexistent(tmp_path):
    """A string that doesn't point to an existing path is left unchanged."""
    grid_yaml_dir = str(tmp_path)
    subdir = str(tmp_path / "run_0001")
    os.makedirs(subdir)

    tmpl_vars = {"genotypes_file": "data/missing.csv", "label": "control"}
    result = _relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir)
    assert result == tmpl_vars


def test_relativize_template_vars_absolute_path(tmp_path):
    """An absolute path in a template var is also rewritten if the file exists."""
    f = tmp_path / "abs.csv"
    f.write_text("x")
    grid_yaml_dir = str(tmp_path)
    subdir = str(tmp_path / "run_0001")
    os.makedirs(subdir)

    tmpl_vars = {"genotypes_file": str(f)}
    result = _relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir)
    assert not os.path.isabs(result["genotypes_file"])
    resolved = os.path.normpath(os.path.join(subdir, result["genotypes_file"]))
    assert resolved == str(f)


def test_relativize_template_vars_non_string_passthrough(tmp_path):
    """Non-string template variable values are returned unchanged."""
    grid_yaml_dir = str(tmp_path)
    subdir = str(tmp_path / "run_0001")
    os.makedirs(subdir)

    tmpl_vars = {"seed": 0, "flag": True, "ratio": 1.5, "none_val": None}
    result = _relativize_template_vars(tmpl_vars, grid_yaml_dir, subdir)
    assert result == tmpl_vars
