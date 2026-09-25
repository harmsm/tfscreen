"""Tests for tfs-setup-grid: path helpers and self-contained, movable grids."""

import inspect
import os
import shutil

import pandas as pd
import pytest
import yaml

from tfscreen.util.grid_utils import INPUTS_DIRNAME
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.scripts.setup_grid_cli import (
    setup_grid,
    _cm_kwargs,
    _resolve_cm_paths,
    _stage_written_config,
    _COMPONENT_AXES,
    _PATH_KEYS,
)
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model


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
        "library_config": "library.yaml",
        "theta_model": "hill_mut",     # not a path: untouched
    }
    out = _resolve_cm_paths(cm_vars, base)
    for key in ("binding_df", "growth_df", "presplit_df", "base_growth_df",
                "library_config"):
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


# ---------------------------------------------------------------------------
# End to end: self-contained, movable grid directories
# ---------------------------------------------------------------------------

_GENOTYPES = ["wt", "A2L", "A2C"]

_LIBRARY = {
    "reading_frame": 0,
    "first_amplicon_residue": 1,
    "wt_seq":      "atggcaaaaccggaatgc",
    "degen_sites": "...nnt......nnt...",
    "tiles":       "...111......222...",
    "tile_combos": ["single-1", "single-2", "double-1-2"],
    "spiked_seqs": ["..................",
                    "...ctt............"],
    "library_mixture": {"single-1": 10, "single-2": 10,
                        "double-1-2": 100, "spiked": 1},
}

_TEMPLATE = (
    "SEED={{ seed }}\n"
    "GENOTYPES={{ predict_genotypes_file }}\n"
)


def _write_growth(path, ln_cfu=1.0):
    rows = []
    for g in _GENOTYPES:
        for t in (0.0, 10.0):
            rows.append({"library": "lib", "replicate": 1, "time": t,
                         "genotype": g, "ln_cfu": ln_cfu, "ln_cfu_std": 0.1,
                         "condition_pre": "pre-cond",
                         "condition_sel": "sel+cond",
                         "t_pre": 1.0, "t_sel": 10.0,
                         "titrant_name": "iptg", "titrant_conc": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_binding(path):
    pd.DataFrame([{"genotype": g, "titrant_name": "iptg", "titrant_conc": 0.0,
                   "theta_obs": 0.5, "theta_std": 0.05}
                  for g in _GENOTYPES]).to_csv(path, index=False)


@pytest.fixture
def project(tmp_path):
    """Inputs in <tmp>/data, grid YAML + template in <tmp>/grids."""
    data = tmp_path / "data"
    data.mkdir()
    _write_growth(data / "growth.csv")
    _write_binding(data / "binding.csv")
    (data / "library.yaml").write_text(yaml.dump(_LIBRARY))
    (data / "genotypes.txt").write_text("wt\nA2L\n")
    grids = tmp_path / "grids"
    grids.mkdir()
    (grids / "run.sh").write_text(_TEMPLATE)
    return tmp_path


def _data_block(**extra):
    variant = {"binding_df": "../data/binding.csv",
               "growth_df": "../data/growth.csv",
               "library_config": "../data/library.yaml",
               "skip_model_stats": True}
    variant.update(extra)
    return {"name": "data", "variants": [variant]}


def _write_grid(project, cm_blocks, tmpl_blocks=None):
    grid = {"run_name": "{{ condition_growth }}__seed{{ seed }}",
            "output_file": "run.sh",
            "configure_model": cm_blocks,
            "template": tmpl_blocks if tmpl_blocks is not None else [
                {"name": "seed", "variants": [{"seed": 0}, {"seed": 1}]},
                {"name": "predict", "variants": [
                    {"predict_genotypes_file": "../data/genotypes.txt"}]},
            ]}
    path = project / "grids" / "grid.yaml"
    path.write_text(yaml.dump(grid, sort_keys=False))
    return str(path)


def _default_cm_blocks():
    return [_data_block(),
            {"name": "condition_growth",
             "variants": [{"condition_growth": "linear"}]}]


def _config_file_refs(cfg):
    """Every (key, path) in a written config that names a file, relative to cwd."""
    refs = [("data." + k, v) for k, v in cfg["data"].items()]
    for k in ("thermo_data", "presplit_df", "base_growth_df", "dk_geno_pins_file"):
        if cfg["components"].get(k):
            refs.append(("components." + k, cfg["components"][k]))
    if isinstance(cfg.get("library", {}).get("source"), str):
        refs.append(("library.source", cfg["library"]["source"]))
    return refs


def test_grid_survives_move_and_deleting_inputs(project, monkeypatch):
    """Move the grid, delete the original inputs: every run still resolves."""
    grid_yaml = _write_grid(project, _default_cm_blocks())
    out = project / "grid_out"
    runs = setup_grid(grid_yaml, out_prefix=str(out))
    assert len(runs) == 2

    moved = project / "elsewhere" / "deeper" / "grid_moved"
    moved.parent.mkdir(parents=True)
    shutil.move(str(out), str(moved))
    shutil.rmtree(project / "data")
    shutil.rmtree(project / "grids")

    for run in runs:
        run_dir = moved / run["run"]
        cfg_path = run_dir / "tfs_configure_config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())

        # Data/library paths are read relative to the run directory.
        refs = _config_file_refs(cfg)
        assert {k for k, _ in refs} >= {"data.growth", "data.binding",
                                        "library.source"}
        for key, path in refs:
            assert not os.path.isabs(path), key
            assert path.startswith(os.path.join("..", INPUTS_DIRNAME)), key
            assert os.path.isfile(run_dir / path), key

        # Per-run outputs sit next to the config.
        for key in ("priors_file", "guesses_file", "library_file"):
            assert os.path.isfile(run_dir / cfg[key]), key

        # Rendered template paths resolve too.
        rendered = (run_dir / "run.sh").read_text()
        geno_path = rendered.split("GENOTYPES=")[1].strip()
        assert geno_path == os.path.join("..", INPUTS_DIRNAME, "genotypes.txt")
        assert (run_dir / geno_path).read_text() == "wt\nA2L\n"

        # No string in the config points outside the grid.
        assert str(project) not in cfg_path.read_text()

        # And the run actually loads from its own directory.
        monkeypatch.chdir(run_dir)
        orchestrator, _ = read_configuration("tfs_configure_config.yaml")
        assert orchestrator.growth_tm is not None


def test_inputs_copied_once(project):
    """Runs sharing an input share one copy under inputs/."""
    grid_yaml = _write_grid(project, _default_cm_blocks())
    out = project / "grid_out"
    setup_grid(grid_yaml, out_prefix=str(out))
    assert sorted(os.listdir(out / INPUTS_DIRNAME)) == [
        "binding.csv", "genotypes.txt", "growth.csv", "library.yaml"]


def test_same_name_different_files_kept_apart(project):
    """Two different growth.csv files get distinct copies, each run its own."""
    other = project / "data" / "other"
    other.mkdir()
    _write_growth(other / "growth.csv", ln_cfu=2.0)
    blocks = [
        {"name": "data", "variants": [
            _data_block()["variants"][0],
            _data_block(growth_df="../data/other/growth.csv")["variants"][0],
        ]},
        {"name": "condition_growth", "variants": [{"condition_growth": "linear"}]},
    ]
    grid_yaml = _write_grid(project, blocks, tmpl_blocks=[
        {"name": "seed", "variants": [{"seed": 0}]},
        {"name": "predict", "variants": [
            {"predict_genotypes_file": "../data/genotypes.txt"}]}])
    out = project / "grid_out"
    runs = setup_grid(grid_yaml, out_prefix=str(out))
    assert len(runs) == 2

    growth_refs = []
    for run in runs:
        cfg = yaml.safe_load(
            (out / run["run"] / "tfs_configure_config.yaml").read_text())
        growth_refs.append(cfg["data"]["growth"])
        src = os.path.normpath(os.path.join(
            project / "grids", run["configure_model"]["growth_df"]))
        copy = out / run["run"] / cfg["data"]["growth"]
        assert copy.read_text() == open(src).read()
    assert sorted(os.path.basename(p) for p in growth_refs) == [
        "growth.csv", "growth_2.csv"]


def test_changed_input_never_overwrites_existing_copy(project):
    """Re-running setup with a changed input leaves earlier runs' copy intact."""
    grid_yaml = _write_grid(project, _default_cm_blocks())
    out = project / "grid_out"
    setup_grid(grid_yaml, out_prefix=str(out))
    before = (out / INPUTS_DIRNAME / "growth.csv").read_text()

    _write_growth(project / "data" / "growth.csv", ln_cfu=3.0)
    runs = setup_grid(grid_yaml, out_prefix=str(out))

    assert (out / INPUTS_DIRNAME / "growth.csv").read_text() == before
    cfg = yaml.safe_load(
        (out / runs[0]["run"] / "tfs_configure_config.yaml").read_text())
    assert cfg["data"]["growth"] == os.path.join("..", INPUTS_DIRNAME,
                                                 "growth_2.csv")


def test_missing_input_fails_before_writing(project):
    blocks = [_data_block(presplit_df="../data/missing.csv"),
              {"name": "condition_growth",
               "variants": [{"condition_growth": "linear"}]}]
    out = project / "grid_out"
    with pytest.raises(FileNotFoundError, match="presplit_df"):
        setup_grid(_write_grid(project, blocks), out_prefix=str(out))
    assert not out.exists()


def test_directory_input_fails_before_writing(project):
    (project / "data" / "thermo").mkdir()
    blocks = [_data_block(thermo_data="../data/thermo"),
              {"name": "condition_growth",
               "variants": [{"condition_growth": "linear"}]}]
    out = project / "grid_out"
    with pytest.raises(ValueError, match="directory"):
        setup_grid(_write_grid(project, blocks), out_prefix=str(out))
    assert not out.exists()


def test_unknown_file_argument_fails_before_writing(project):
    """A configure_model value outside _PATH_KEYS naming a file is refused."""
    blocks = [_data_block(),
              {"name": "extra",
               "variants": [{"mystery_file": "../data/genotypes.txt"}]}]
    out = project / "grid_out"
    with pytest.raises(ValueError, match="mystery_file.*_PATH_KEYS"):
        setup_grid(_write_grid(project, blocks), out_prefix=str(out))
    assert not out.exists()


def test_template_errors_fail_before_writing(project):
    out = project / "grid_out"
    grid_yaml = _write_grid(project, _default_cm_blocks(), tmpl_blocks=[
        {"name": "seed", "variants": [{"seed": 0}]}])  # predict_genotypes_file undefined
    with pytest.raises(ValueError, match="Undefined template variable"):
        setup_grid(grid_yaml, out_prefix=str(out))
    assert not out.exists()

    grid_yaml = _write_grid(project, _default_cm_blocks(), tmpl_blocks=[
        {"name": "seed", "variants": [{"seed": 0}]},
        {"name": "predict", "variants": [{"predict_genotypes_file": "../data"}]}])
    with pytest.raises(ValueError, match="directory"):
        setup_grid(grid_yaml, out_prefix=str(out))
    assert not out.exists()


def test_skipped_combination_copies_nothing(project):
    """An incompatible combination is skipped and leaves no orphan copies."""
    blocks = [_data_block(),
              {"name": "combo", "variants": [
                  {"condition_growth": "power", "theta_rescale": "logit"}]}]
    out = project / "grid_out"
    runs = setup_grid(_write_grid(project, blocks), out_prefix=str(out))
    assert runs == []
    assert not (out / INPUTS_DIRNAME).exists()


# ---------------------------------------------------------------------------
# _stage_written_config
# ---------------------------------------------------------------------------

def test_stage_written_config_rewrites_every_occurrence(tmp_path):
    src = str(tmp_path / "lib.yaml")
    open(src, "w").write("x")
    cfg = {"data": {"growth": "/g.csv"},
           "components": {"theta": "hill_geno"},
           "library": {"source": src},
           "priors_file": "tfs_configure_priors.csv"}
    out = _stage_written_config(cfg, {src: "../inputs/lib.yaml"}, str(tmp_path))
    assert out["library"]["source"] == "../inputs/lib.yaml"
    assert cfg["library"]["source"] == src  # input not modified


def test_stage_written_config_refuses_unknown_outside_path(tmp_path):
    stray = str(tmp_path / "pins.csv")
    open(stray, "w").write("x")
    cfg = {"components": {"dk_geno_pins_file": stray}}
    with pytest.raises(ValueError, match="dk_geno_pins_file"):
        _stage_written_config(cfg, {}, str(tmp_path / "run"))
