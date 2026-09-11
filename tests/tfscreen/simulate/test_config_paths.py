"""Tests for tfscreen.simulate.config_paths."""

import os

from tfscreen.simulate.config_paths import (
    existing_input_file,
    iter_path_entries,
    resolve_config_paths,
)


def _cfg():
    return {
        "thermo_data": "struct.h5",
        "seed": 1,
        "binding_data": {
            "spiked_binding": {"choose_by": "hill.csv"},
            "library_binding": {"choose_by": "stratified", "num": 20},
        },
        "empirical": {"phenotype_model": "emp"},
    }


def test_iter_path_entries_skips_keywords_and_non_paths():
    names = sorted(kp for _, _, kp in iter_path_entries(_cfg()))
    assert names == ["binding_data.spiked_binding.choose_by",
                     "empirical.phenotype_model", "thermo_data"]


def test_iter_path_entries_tolerates_missing_blocks():
    assert list(iter_path_entries({"seed": 1, "binding_data": {}})) == []


def test_resolve_config_paths_relative_to_base_dir(tmp_path):
    out = resolve_config_paths(_cfg(), str(tmp_path))
    assert out["thermo_data"] == os.path.join(str(tmp_path), "struct.h5")
    assert out["binding_data"]["spiked_binding"]["choose_by"] == \
        os.path.join(str(tmp_path), "hill.csv")
    assert out["binding_data"]["library_binding"]["choose_by"] == "stratified"
    assert out["empirical"]["phenotype_model"] == os.path.join(str(tmp_path), "emp")


def test_resolve_config_paths_keeps_absolute_and_does_not_mutate(tmp_path):
    cfg = _cfg()
    cfg["thermo_data"] = "/abs/struct.h5"
    out = resolve_config_paths(cfg, str(tmp_path))
    assert out["thermo_data"] == "/abs/struct.h5"
    assert cfg["binding_data"]["spiked_binding"]["choose_by"] == "hill.csv"


def test_existing_input_file_phenotype_model_candidates(tmp_path):
    model = tmp_path / "emp_phenotype_model.json"
    model.write_text("{}")
    prefix = str(tmp_path / "emp")
    assert existing_input_file("empirical.phenotype_model", prefix) == str(model)
    # Other keys only accept the literal path.
    assert existing_input_file("thermo_data", prefix) is None
