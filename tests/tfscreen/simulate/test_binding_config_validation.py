"""Fail-fast validation of the binding_data spiked_binding/library_binding blocks."""

import pytest

from tfscreen.simulate.library_prediction import (
    _validate_binding_config,
    _is_file_choice,
)

SPIKED = ["wt", "M42I", "H74A", "K84L"]


def _pfile(tmp_path, genos):
    p = tmp_path / "p.csv"
    lines = ["genotype,theta_low,theta_high,log_hill_K,hill_n"]
    lines += [f"{g},0.05,0.9,-4.0,2.0" for g in genos]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_is_file_choice():
    assert not _is_file_choice("stratified")
    assert not _is_file_choice("random")
    assert _is_file_choice("hill_params.csv")


def test_valid_configs_do_not_raise(tmp_path):
    f = _pfile(tmp_path, ["wt", "M42I"])
    _validate_binding_config({"spiked_binding": {"choose_by": f}}, SPIKED, "hill_mut")
    _validate_binding_config(
        {"spiked_binding": {"choose_by": "stratified", "num": 2},
         "library_binding": {"choose_by": "random", "num": 3}},
        SPIKED, "hill_mut")


def test_spiked_file_genotype_not_spiked_errors(tmp_path):
    f = _pfile(tmp_path, ["wt", "D88A"])   # D88A not in SPIKED
    with pytest.raises(ValueError, match="must be spiked"):
        _validate_binding_config({"spiked_binding": {"choose_by": f}}, SPIKED, "hill_mut")


def test_library_file_genotype_in_spiked_errors(tmp_path):
    f = _pfile(tmp_path, ["M42I"])         # M42I IS spiked
    with pytest.raises(ValueError, match="must NOT be spiked"):
        _validate_binding_config({"library_binding": {"choose_by": f}}, SPIKED, "hill_mut")


def test_num_with_file_errors(tmp_path):
    f = _pfile(tmp_path, ["wt"])
    with pytest.raises(ValueError, match="incompatible with a choose_by file"):
        _validate_binding_config(
            {"spiked_binding": {"choose_by": f, "num": 2}}, SPIKED, "hill_mut")


def test_file_requires_hill_component(tmp_path):
    f = _pfile(tmp_path, ["wt"])
    with pytest.raises(ValueError, match="Hill theta component"):
        _validate_binding_config(
            {"spiked_binding": {"choose_by": f}}, SPIKED, "thermo.O2_C12_K5_U0_a.PK")


def test_spiked_num_out_of_range_errors():
    with pytest.raises(ValueError, match=r"num must be in \[1, 4\]"):
        _validate_binding_config(
            {"spiked_binding": {"choose_by": "stratified", "num": 99}}, SPIKED, "hill_mut")


def test_library_stratified_requires_num():
    with pytest.raises(ValueError, match="requires 'num'"):
        _validate_binding_config(
            {"library_binding": {"choose_by": "stratified"}}, SPIKED, "hill_mut")


def test_missing_choose_by_errors():
    with pytest.raises(ValueError, match="requires a 'choose_by'"):
        _validate_binding_config({"spiked_binding": {"num": 2}}, SPIKED, "hill_mut")
