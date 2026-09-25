"""Tests for tfs-setup-sim-grid (setup_sim_grid_cli)."""

import json
import os
import shutil

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


# ---------------------------------------------------------------------------
# Nested file paths (binding_data.*.choose_by, empirical.phenotype_model)
# ---------------------------------------------------------------------------

def _read_run_config(out, run):
    with open(os.path.join(out, run["run"], "tfs_sim_config.yaml")) as fh:
        return yaml.safe_load(fh)


def test_resolve_paths_nested_choose_by(tmp_path):
    (tmp_path / "hill_params.csv").write_text("x")
    cfg = {"binding_data": {"spiked_binding": {"choose_by": "hill_params.csv"},
                            "library_binding": {"choose_by": "stratified",
                                                "num": 5}}}
    result = _resolve_paths(cfg, str(tmp_path))
    assert (result["binding_data"]["spiked_binding"]["choose_by"]
            == str(tmp_path / "hill_params.csv"))
    assert result["binding_data"]["library_binding"]["choose_by"] == "stratified"
    # The input is not modified
    assert cfg["binding_data"]["spiked_binding"]["choose_by"] == "hill_params.csv"


def test_resolve_paths_nested_phenotype_model(tmp_path):
    result = _resolve_paths({"empirical": {"phenotype_model": "m.json"}},
                            str(tmp_path))
    assert result["empirical"]["phenotype_model"] == str(tmp_path / "m.json")


def test_resolve_paths_missing_nested_parent(tmp_path):
    cfg = {"binding_data": None, "empirical": "not-a-dict"}
    assert _resolve_paths(cfg, str(tmp_path)) == cfg


def _write_study(tmp_path, base, grid_body, grid_dir=None):
    """Write simulate_config.yaml (base) and grid.yaml; return the grid path."""
    study = tmp_path / "study"
    study.mkdir(exist_ok=True)
    (study / "simulate_config.yaml").write_text(yaml.dump(base))
    grid_dir = grid_dir or study
    base_ref = os.path.relpath(study / "simulate_config.yaml", grid_dir)
    grid_path = grid_dir / "grid.yaml"
    grid_path.write_text(f"base_config: {base_ref}\n" + grid_body)
    return str(grid_path)


_SEED_BLOCK = """\
simulate:
  - name: seed
    variants:
      - seed: 1
      - seed: 2
"""


def test_setup_sim_grid_base_config_choose_by_file(tmp_path):
    """A choose_by params file is copied into inputs/ and referenced from there."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "hill_params.csv").write_text("genotype\nwt\n")
    base = {"seed": 1,
            "binding_data": {"spiked_binding": {"choose_by": "hill_params.csv"},
                             "library_binding": {"choose_by": "random",
                                                 "num": 3}}}
    grid_path = _write_study(tmp_path, base, _SEED_BLOCK)
    out = str(tmp_path / "elsewhere" / "grid_out")
    runs = setup_sim_grid(grid_path, out_prefix=out)
    assert len(runs) == 2
    assert os.listdir(os.path.join(out, "inputs")) == ["hill_params.csv"]
    for run in runs:
        cfg = _read_run_config(out, run)
        assert (cfg["binding_data"]["spiked_binding"]["choose_by"]
                == os.path.join("..", "inputs", "hill_params.csv"))
        # Keyword left alone
        assert cfg["binding_data"]["library_binding"]["choose_by"] == "random"


def test_setup_sim_grid_is_movable(tmp_path):
    """The grid works after it is moved and the original inputs are deleted."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "hill_params.csv").write_text("genotype\nwt\n")
    (study / "struct.h5").write_text("h5")
    (study / "run.sh").write_text("cat {{ extra }}\n")
    (study / "extra.txt").write_text("extra")
    base = {"thermo_data": "struct.h5",
            "binding_data": {"spiked_binding": {"choose_by": "hill_params.csv"}}}
    grid_path = _write_study(tmp_path, base, """\
output_file: run.sh
simulate:
  - name: seed
    variants:
      - seed: 1
template:
  - name: extra
    variants:
      - extra: extra.txt
""")
    out = tmp_path / "grid_out"
    runs = setup_sim_grid(grid_path, out_prefix=str(out))

    moved = tmp_path / "far" / "away" / "grid_out"
    moved.parent.mkdir(parents=True)
    shutil.move(str(out), str(moved))
    shutil.rmtree(study)

    run_dir = moved / runs[0]["run"]
    cfg = _read_run_config(str(moved), runs[0])
    assert (run_dir / cfg["thermo_data"]).read_text() == "h5"
    assert ((run_dir / cfg["binding_data"]["spiked_binding"]["choose_by"])
            .read_text() == "genotype\nwt\n")
    rendered = (run_dir / "run.sh").read_text()
    assert rendered == f"cat {os.path.join('..', 'inputs', 'extra.txt')}"
    assert (run_dir / ".." / "inputs" / "extra.txt").read_text() == "extra"


def test_setup_sim_grid_same_name_different_files(tmp_path):
    """Two different files with one name get distinct copies."""
    study = tmp_path / "study"
    for sub in ("a", "b"):
        (study / sub).mkdir(parents=True)
        (study / sub / "params.csv").write_text(f"genotype\n{sub}\n")
    grid_path = _write_study(tmp_path, {"seed": 1}, """\
simulate:
  - name: binding
    variants:
      - binding_data:
          spiked_binding:
            choose_by: a/params.csv
      - binding_data:
          spiked_binding:
            choose_by: b/params.csv
      - binding_data:
          spiked_binding:
            choose_by: a/params.csv
""")
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(grid_path, out_prefix=out)
    got = [_read_run_config(out, r)["binding_data"]["spiked_binding"]["choose_by"]
           for r in runs]
    assert got == [os.path.join("..", "inputs", "params.csv"),
                   os.path.join("..", "inputs", "params_2.csv"),
                   os.path.join("..", "inputs", "params.csv")]
    assert sorted(os.listdir(os.path.join(out, "inputs"))) == [
        "params.csv", "params_2.csv"]
    assert open(os.path.join(out, "inputs", "params_2.csv")).read() == "genotype\nb\n"


def test_setup_sim_grid_rerun_changed_input_not_overwritten(tmp_path):
    """Re-running into the same grid never changes a file older runs use."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "hill_params.csv").write_text("v1")
    base = {"binding_data": {"spiked_binding": {"choose_by": "hill_params.csv"}}}
    grid_path = _write_study(tmp_path, base, _SEED_BLOCK)
    out = str(tmp_path / "grid_out")
    setup_sim_grid(grid_path, out_prefix=out)
    runs = setup_sim_grid(grid_path, out_prefix=out)  # unchanged: reused
    assert os.listdir(os.path.join(out, "inputs")) == ["hill_params.csv"]

    (study / "hill_params.csv").write_text("v2")
    runs = setup_sim_grid(grid_path, out_prefix=out)
    cfg = _read_run_config(out, runs[0])
    assert (cfg["binding_data"]["spiked_binding"]["choose_by"]
            == os.path.join("..", "inputs", "hill_params_2.csv"))
    assert open(os.path.join(out, "inputs", "hill_params.csv")).read() == "v1"


def test_setup_sim_grid_phenotype_model_prefix(tmp_path):
    """A bare phenotype_model prefix copies the JSON the simulator would load."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "emp_phenotype_model.json").write_text("{}")
    base = {"empirical": {"phenotype_model": "emp"}}
    grid_path = _write_study(tmp_path, base, _SEED_BLOCK)
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(grid_path, out_prefix=out)
    cfg = _read_run_config(out, runs[0])
    assert (cfg["empirical"]["phenotype_model"]
            == os.path.join("..", "inputs", "emp_phenotype_model.json"))


def test_setup_sim_grid_missing_input_file_raises(tmp_path):
    base = {"binding_data": {"spiked_binding": {"choose_by": "nope.csv"}}}
    grid_path = _write_study(tmp_path, base, _SEED_BLOCK)
    out = tmp_path / "grid_out"
    with pytest.raises(FileNotFoundError, match="binding_data.spiked_binding.choose_by"):
        setup_sim_grid(grid_path, out_prefix=str(out))
    assert not out.exists()


def test_setup_sim_grid_directory_input_raises(tmp_path):
    study = tmp_path / "study"
    (study / "structs").mkdir(parents=True)
    grid_path = _write_study(tmp_path, {"thermo_data": "structs"}, _SEED_BLOCK)
    with pytest.raises(ValueError, match="directory"):
        setup_sim_grid(grid_path, out_prefix=str(tmp_path / "grid_out"))


def test_setup_sim_grid_unknown_path_key_raises(tmp_path):
    """A file named by a key setup does not know about is a hard error."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "mystery.csv").write_text("x")
    base = {"presplit_data": {"source": "mystery.csv"}}
    grid_path = _write_study(tmp_path, base, _SEED_BLOCK)
    out = tmp_path / "grid_out"
    with pytest.raises(ValueError, match="presplit_data.source.*_SIM_PATH_KEYS"):
        setup_sim_grid(grid_path, out_prefix=str(out))
    assert not out.exists()


def test_setup_sim_grid_unknown_absolute_path_raises(tmp_path):
    grid_path = _write_study(tmp_path, {"some_dir": str(tmp_path)}, _SEED_BLOCK)
    with pytest.raises(ValueError, match="some_dir"):
        setup_sim_grid(grid_path, out_prefix=str(tmp_path / "grid_out"))


def test_setup_sim_grid_template_directory_raises(tmp_path):
    study = tmp_path / "study"
    (study / "data").mkdir(parents=True)
    (study / "run.sh").write_text("{{ d }}")
    grid_path = _write_study(tmp_path, {"seed": 1}, """\
output_file: run.sh
template:
  - name: d
    variants:
      - d: data
""")
    with pytest.raises(ValueError, match="Template variable 'd'.*directory"):
        setup_sim_grid(grid_path, out_prefix=str(tmp_path / "grid_out"))


def test_setup_sim_grid_keyword_choose_by_unchanged(tmp_path):
    """A choose_by keyword is kept even when a file of that name exists."""
    (tmp_path / "stratified").write_text("x")
    base = {"binding_data": {"spiked_binding": {"choose_by": "stratified"}}}
    (tmp_path / "simulate_config.yaml").write_text(yaml.dump(base))
    (tmp_path / "grid.yaml").write_text("""\
base_config: simulate_config.yaml
simulate:
  - name: seed
    variants:
      - seed: 1
""")
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(tmp_path / "grid.yaml"), out_prefix=out)
    cfg = _read_run_config(out, runs[0])
    assert cfg["binding_data"]["spiked_binding"]["choose_by"] == "stratified"


def test_setup_sim_grid_override_choose_by_from_grid_dir(tmp_path):
    """A binding_data override resolves its choose_by against the grid YAML."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "simulate_config.yaml").write_text(yaml.dump(
        {"binding_data": {"spiked_binding": {"choose_by": "stratified"}}}))
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    (grid_dir / "measured.csv").write_text("genotype\nwt\n")
    (grid_dir / "grid.yaml").write_text("""\
base_config: ../base/simulate_config.yaml
simulate:
  - name: binding
    variants:
      - binding_data:
          spiked_binding:
            choose_by: measured.csv
""")
    out = str(tmp_path / "grid_out")
    runs = setup_sim_grid(str(grid_dir / "grid.yaml"), out_prefix=out)
    cfg = _read_run_config(out, runs[0])
    assert (cfg["binding_data"]["spiked_binding"]["choose_by"]
            == os.path.join("..", "inputs", "measured.csv"))
    assert open(os.path.join(out, "inputs", "measured.csv")).read() == "genotype\nwt\n"
    run_dir = os.path.join(out, runs[0]["run"])
    # combo.json keeps the override as written
    with open(os.path.join(run_dir, "combo.json")) as fh:
        combo = json.load(fh)
    assert (combo["simulate"]["binding_data"]["spiked_binding"]["choose_by"]
            == "measured.csv")
