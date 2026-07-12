"""Tests for path-rewriting helpers (now in grid_utils, re-exported from setup_grid_cli)."""

import os

import yaml

import inspect

from tfscreen.util.grid_utils import (
    relativize_config_paths as _relativize_config_paths,
    relativize_node as _relativize_node,
    relativize_template_vars as _relativize_template_vars,
)
from tfscreen.tfmodel.scripts.setup_grid_cli import (
    _cm_kwargs,
    _resolve_cm_paths,
    _COMPONENT_AXES,
    _PATH_KEYS,
)
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model


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


# ---------------------------------------------------------------------------
# _cm_kwargs — component-axis → configure_model parameter translation
# ---------------------------------------------------------------------------

def test_cm_kwargs_adds_model_suffix_to_axes():
    """Registry-axis keys gain a ``_model`` suffix; other keys pass through."""
    out = _cm_kwargs({
        "condition_growth": "linear",
        "growth_noise": "normal_kt",   # regression: was missing from _COMPONENT_AXES
        "epistasis": True,             # non-axis: forwarded unchanged
        "binding_df": "b.csv",         # non-axis: forwarded unchanged
    })
    assert out == {
        "condition_growth_model": "linear",
        "growth_noise_model": "normal_kt",
        "epistasis": True,
        "binding_df": "b.csv",
    }


# ---------------------------------------------------------------------------
# _resolve_cm_paths — relative path resolution for file-path arguments
# ---------------------------------------------------------------------------

def test_resolve_cm_paths_resolves_all_df_args(tmp_path):
    """presplit_df/base_growth_df (regression) and the other df args resolve to abs."""
    base = str(tmp_path)
    cm_vars = {
        "binding_df": "b.csv",
        "growth_df": "g.csv",
        "presplit_df": "p.csv",
        "base_growth_df": "bg.csv",
        "theta_model": "hill_mut",     # not a path: untouched
    }
    out = _resolve_cm_paths(cm_vars, base)
    for key in ("binding_df", "growth_df", "presplit_df", "base_growth_df"):
        assert out[key] == os.path.normpath(os.path.join(base, cm_vars[key]))
        assert os.path.isabs(out[key])
    assert out["theta_model"] == "hill_mut"


def test_resolve_cm_paths_leaves_absolute_unchanged(tmp_path):
    """An already-absolute path argument is not rewritten."""
    abs_p = os.path.join(str(tmp_path), "b.csv")
    out = _resolve_cm_paths({"base_growth_df": abs_p}, "/some/other/dir")
    assert out["base_growth_df"] == abs_p


# ---------------------------------------------------------------------------
# Drift guards against the configure_model signature
# ---------------------------------------------------------------------------

def test_component_axes_cover_all_model_params():
    """Every ``*_model`` param of configure_model must have a _COMPONENT_AXES entry.

    Guards against adding a new swappable component to configure_model without
    teaching tfs-setup-grid how to forward it (the ``growth_noise`` drift).
    """
    params = inspect.signature(configure_model).parameters
    model_axes = {p[:-len("_model")] for p in params if p.endswith("_model")}
    missing = model_axes - _COMPONENT_AXES
    assert not missing, f"_COMPONENT_AXES missing configure_model axes: {sorted(missing)}"


def test_path_keys_cover_all_df_params():
    """Every ``*_df`` path param of configure_model must be in _PATH_KEYS.

    Guards against adding a new data-file argument (e.g. presplit_df,
    base_growth_df) without wiring its relative-path resolution.
    """
    params = inspect.signature(configure_model).parameters
    df_params = {p for p in params if p.endswith("_df")}
    missing = df_params - _PATH_KEYS
    assert not missing, f"_PATH_KEYS missing configure_model df args: {sorted(missing)}"
