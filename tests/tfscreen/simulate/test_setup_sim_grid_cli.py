"""Tests for tfs-setup-sim-grid (setup_sim_grid_cli)."""

import json
import os

import pytest
import yaml

from tfscreen.simulate.scripts.setup_sim_grid_cli import (
    _expand_block,
    _resolve_paths,
    setup_sim_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_config(tmp_path):
    """Write a minimal valid simulate config and return its path."""
    cfg = {
        "reading_frame": 0,
        "observable_calculator": "lac",
        "tube_noise_sigma": 0.01,
        "seed": None,
    }
    p = tmp_path / "simulate_config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


@pytest.fixture()
def minimal_grid_yaml(tmp_path, base_config):
    """Write a minimal grid YAML with one simulate variant."""
    content = f"""\
base_config: {base_config}

simulate:
  - name: noise
    variants:
      - tube_noise_sigma: 0.05
"""
    p = tmp_path / "grid.yaml"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# _expand_block
# ---------------------------------------------------------------------------

def test_expand_block_variants():
    block = {"name": "noise", "variants": [{"x": 1}, {"x": 2}]}
    assert _expand_block(block) == [{"x": 1}, {"x": 2}]


def test_expand_block_no_variants_raises():
    with pytest.raises(ValueError, match="variants"):
        _expand_block({"name": "bad"})


def test_expand_block_auto_raises():
    with pytest.raises(ValueError, match="auto.*not supported"):
        _expand_block({"name": "x", "auto": "condition_growth"})


# ---------------------------------------------------------------------------
# _resolve_paths
# ---------------------------------------------------------------------------

def test_resolve_paths_relative(tmp_path):
    f = tmp_path / "struct.h5"
    f.write_text("x")
    result = _resolve_paths({"thermo_data": "struct.h5"}, str(tmp_path))
    assert os.path.isabs(result["thermo_data"])
    assert result["thermo_data"] == str(f)


def test_resolve_paths_absolute_unchanged(tmp_path):
    f = tmp_path / "struct.h5"
    f.write_text("x")
    abs_path = str(f)
    result = _resolve_paths({"thermo_data": abs_path}, "/some/other/dir")
    assert result["thermo_data"] == abs_path


def test_resolve_paths_nested_choose_by_file(tmp_path):
    """A choose_by params file inside binding_data is resolved too."""
    cfg = {"binding_data": {"spiked_binding": {"choose_by": "hill.csv"},
                            "library_binding": {"choose_by": "stratified",
                                                "num": 20}}}
    result = _resolve_paths(cfg, str(tmp_path))
    assert result["binding_data"]["spiked_binding"]["choose_by"] == \
        os.path.join(str(tmp_path), "hill.csv")
    # Builtin keywords are not paths.
    assert result["binding_data"]["library_binding"]["choose_by"] == "stratified"


def test_resolve_paths_does_not_mutate_input(tmp_path):
    """Nested resolution works on a deep copy (the base config is shared)."""
    cfg = {"binding_data": {"spiked_binding": {"choose_by": "hill.csv"}}}
    _resolve_paths(cfg, str(tmp_path))
    assert cfg["binding_data"]["spiked_binding"]["choose_by"] == "hill.csv"


def test_setup_sim_grid_copies_input_files_into_runs(tmp_path):
    """Each run directory gets its own copy of the base config's binding
    params file, and its config names the local copy -- even when the base
    config lives outside the grid directory."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "hill.csv").write_text("genotype\nwt\n")
    base = base_dir / "simulate_config.yaml"
    base.write_text(yaml.dump({
        "seed": None,
        "binding_data": {"spiked_binding": {"choose_by": "hill.csv"}},
    }))
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    grid = grid_dir / "grid.yaml"
    grid.write_text(
        f"base_config: {base}\n"
        "simulate:\n"
        "  - name: seed\n"
        "    variants:\n"
        "      - seed: 1\n"
        "      - seed: 2\n"
    )

    out_prefix = str(tmp_path / "out")
    setup_sim_grid(str(grid), out_prefix=out_prefix)

    run_dirs = sorted(d for d in os.listdir(out_prefix)
                      if os.path.isdir(os.path.join(out_prefix, d)))
    assert len(run_dirs) == 2
    for d in run_dirs:
        run_dir = os.path.join(out_prefix, d)
        with open(os.path.join(run_dir, "tfs_sim_config.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["binding_data"]["spiked_binding"]["choose_by"] == "hill.csv"
        with open(os.path.join(run_dir, "hill.csv")) as fh:
            assert fh.read() == "genotype\nwt\n"


def test_setup_sim_grid_missing_input_file_fails_at_setup(tmp_path):
    """A referenced input file that does not exist fails before any run is made."""
    base = tmp_path / "simulate_config.yaml"
    base.write_text(yaml.dump({
        "binding_data": {"spiked_binding": {"choose_by": "missing.csv"}},
    }))
    grid = tmp_path / "grid.yaml"
    grid.write_text(f"base_config: {base}\n"
                    "simulate:\n  - name: seed\n    variants:\n      - seed: 1\n")
    with pytest.raises(FileNotFoundError, match="spiked_binding.choose_by"):
        setup_sim_grid(str(grid), out_prefix=str(tmp_path / "out"))


def test_setup_sim_grid_keyword_choose_by_untouched(tmp_path):
    base = tmp_path / "simulate_config.yaml"
    base.write_text(yaml.dump({
        "binding_data": {"library_binding": {"choose_by": "stratified", "num": 5}},
    }))
    grid = tmp_path / "grid.yaml"
    grid.write_text(f"base_config: {base}\n"
                    "simulate:\n  - name: seed\n    variants:\n      - seed: 1\n")
    out = str(tmp_path / "out")
    runs = setup_sim_grid(str(grid), out_prefix=out)
    with open(os.path.join(out, runs[0]["run"], "tfs_sim_config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["binding_data"]["library_binding"]["choose_by"] == "stratified"


def test_resolve_paths_unknown_key_unchanged(tmp_path):
    result = _resolve_paths({"tube_noise_sigma": 0.01}, str(tmp_path))
    assert result["tube_noise_sigma"] == 0.01


# ---------------------------------------------------------------------------
# setup_sim_grid — basic output structure
# ---------------------------------------------------------------------------

def test_setup_sim_grid_creates_subdirs(tmp_path, minimal_grid_yaml):
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(minimal_grid_yaml, out_prefix=out)
    assert len(runs) == 1
    run_dir = os.path.join(out, runs[0]["run"])
    assert os.path.isdir(run_dir)


def test_setup_sim_grid_writes_config(tmp_path, minimal_grid_yaml):
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(minimal_grid_yaml, out_prefix=out)
    cfg_path = os.path.join(out, runs[0]["run"], "tfs_sim_config.yaml")
    assert os.path.isfile(cfg_path)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["tube_noise_sigma"] == 0.05


def test_setup_sim_grid_writes_combo_json(tmp_path, minimal_grid_yaml):
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(minimal_grid_yaml, out_prefix=out)
    combo_path = os.path.join(out, runs[0]["run"], "combo.json")
    assert os.path.isfile(combo_path)
    with open(combo_path) as fh:
        combo = json.load(fh)
    assert "simulate" in combo
    assert "template" in combo


def test_setup_sim_grid_writes_summary(tmp_path, minimal_grid_yaml):
    out = str(tmp_path / "grid_out")
    setup_sim_grid(minimal_grid_yaml, out_prefix=out)
    summary_path = os.path.join(out, "grid_summary.json")
    assert os.path.isfile(summary_path)
    with open(summary_path) as fh:
        summary = json.load(fh)
    assert "runs" in summary
    assert len(summary["runs"]) == 1


# ---------------------------------------------------------------------------
# setup_sim_grid — Cartesian product
# ---------------------------------------------------------------------------

def test_setup_sim_grid_cartesian_product(tmp_path, base_config):
    grid = f"""\
base_config: {base_config}

simulate:
  - name: noise
    variants:
      - tube_noise_sigma: 0.01
      - tube_noise_sigma: 0.05

  - name: seed
    variants:
      - seed: 0
      - seed: 42
"""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(grid)

    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(grid_path), out_prefix=out)
    assert len(runs) == 4


# ---------------------------------------------------------------------------
# setup_sim_grid — Jinja2 template rendering
# ---------------------------------------------------------------------------

def test_setup_sim_grid_renders_template(tmp_path, base_config):
    template = "NUM_REPLICATES={{ num_replicates }}\n"
    tmpl_path = tmp_path / "run.sh"
    tmpl_path.write_text(template)

    grid = f"""\
base_config: {base_config}
output_file: run.sh

simulate:
  - name: noise
    variants:
      - tube_noise_sigma: 0.05

template:
  - name: reps
    variants:
      - num_replicates: 5
"""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(grid)

    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(grid_path), out_prefix=out)
    rendered = open(os.path.join(out, runs[0]["run"], "run.sh")).read()
    assert "NUM_REPLICATES=5" in rendered


# ---------------------------------------------------------------------------
# setup_sim_grid — error handling
# ---------------------------------------------------------------------------

def test_setup_sim_grid_missing_grid_yaml(tmp_path):
    with pytest.raises(FileNotFoundError, match="Grid YAML not found"):
        setup_sim_grid(str(tmp_path / "nonexistent.yaml"))


def test_setup_sim_grid_missing_base_config(tmp_path):
    grid = """\
base_config: /does/not/exist.yaml

simulate:
  - name: x
    variants:
      - k: v
"""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(grid)
    with pytest.raises(FileNotFoundError, match="Base config not found"):
        setup_sim_grid(str(grid_path))


def test_setup_sim_grid_no_base_config_key(tmp_path):
    grid = """\
simulate:
  - name: x
    variants:
      - k: v
"""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(grid)
    with pytest.raises(ValueError, match="base_config"):
        setup_sim_grid(str(grid_path))


def test_setup_sim_grid_override_applied(tmp_path, base_config):
    grid = f"""\
base_config: {base_config}

simulate:
  - name: calc
    variants:
      - observable_calculator: eee
"""
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(grid)
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(grid_path), out_prefix=out)

    cfg_path = os.path.join(out, runs[0]["run"], "tfs_sim_config.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["observable_calculator"] == "eee"
    # Base config values not overridden are preserved
    assert cfg["reading_frame"] == 0
