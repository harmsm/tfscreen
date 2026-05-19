"""
Tests for setup_grid_calibration.py and summarize_grid_calibration.py.
"""
import json
import math
import os
import stat

import numpy as np
import pandas as pd
import pytest
import yaml

from tfscreen.analysis.hierarchical.growth_model.scripts.setup_grid_calibration_cli import (
    _get_axis_values,
    _subdir_name,
    setup_grid_calibration,
    _SEP,
    _RUN_OUT_PREFIX,
    _CONFIG_STEM,
    _INCOMPATIBLE_CG_TR,
    main as setup_main,
)
from tfscreen.analysis.hierarchical.growth_model.scripts.summarize_grid_calibration_cli import (
    _compute_aic,
    _compute_aic_weights,
    _find_stats_json,
    _read_components,
    summarize_grid_calibration,
    main as summarize_main,
)
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _write_base_config(tmp_path, extra=None):
    """Write a minimal base config YAML.

    Uses the real tfscreen YAML structure: component choices live inside a
    ``components:`` dict, and data paths under ``data:``.
    """
    cfg = {
        "components": {
            "condition_growth": "linear",
            "growth_transition": "instant",
            "theta_rescale": "passthrough",
        },
        "data": {
            "growth": "growth.csv",
            "binding": "binding.csv",
        },
    }
    if extra:
        cfg.update(extra)
    cfg_path = tmp_path / "run_config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh)
    return str(cfg_path)


def _write_axis_file(tmp_path, name, values):
    """Write a file-backed axis list and return its path."""
    p = tmp_path / name
    p.write_text("\n".join(values) + "\n")
    return str(p)


def _make_completed_run(grid_dir, cg, gt, tr, stats):
    """Create a fake completed run subdirectory under grid_dir.

    run_config.yaml uses the nested ``components:`` structure that
    write_configuration produces (and that _read_components now reads).
    """
    name = f"{cg}{_SEP}{gt}{_SEP}{tr}"
    subdir = grid_dir / name
    subdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "components": {
            "condition_growth": cg,
            "growth_transition": gt,
            "theta_rescale": tr,
        }
    }
    (subdir / "run_config.yaml").write_text(yaml.dump(cfg))
    (subdir / f"{_RUN_OUT_PREFIX}_calib_stats.json").write_text(json.dumps(stats))
    return subdir


def _minimal_stats(rmse=0.5, n_params=10, n_obs=100):
    return {
        "pct_success": 1.0,
        "rmse": rmse,
        "normalized_rmse": 0.1,
        "pearson_r": 0.95,
        "r_squared": 0.90,
        "mean_error": 0.01,
        "coverage_prob": 0.94,
        "residual_corr": 0.02,
        "residual_corr_p_value": 0.3,
        "bp_p_value": 0.2,
        "n_params": n_params,
        "n_obs": n_obs,
    }


# ---------------------------------------------------------------------------
# Fixture: patch GrowthModel + write_configuration so no real data is needed
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_configure(mocker):
    """Patch GrowthModel and write_configuration to avoid real data loading.

    The fake write_configuration writes stub files so that setup_grid_calibration
    can complete normally.  Returns ``(mock_gm_cls, mock_write_cfg)``.
    """
    mock_gm_cls = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".setup_grid_calibration_cli.GrowthModel"
    )
    mock_gm_cls.return_value = mocker.MagicMock(name="gm_instance")

    def _fake_write(gm, out_prefix, growth_df_path, binding_df_path):
        # Simulate the three files written by the real write_configuration.
        cfg = {
            "data": {"growth": growth_df_path, "binding": binding_df_path},
            "components": {},
            "priors_file": os.path.basename(out_prefix) + "_priors.csv",
            "guesses_file": os.path.basename(out_prefix) + "_guesses.csv",
        }
        with open(f"{out_prefix}_config.yaml", "w") as fh:
            yaml.dump(cfg, fh)
        open(f"{out_prefix}_priors.csv", "w").close()
        open(f"{out_prefix}_guesses.csv", "w").close()

    mock_wc = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.scripts"
        ".setup_grid_calibration_cli.write_configuration",
        side_effect=_fake_write,
    )

    return mock_gm_cls, mock_wc


# ===========================================================================
# Tests for setup_grid_calibration
# ===========================================================================

class TestGetAxisValues:

    def test_returns_all_registered_when_no_file(self):
        vals = _get_axis_values("theta_rescale", None)
        assert set(vals) == set(model_registry["theta_rescale"].keys())

    def test_returns_sorted_when_no_file(self):
        vals = _get_axis_values("theta_rescale", None)
        assert vals == sorted(vals)

    def test_reads_subset_from_file(self, tmp_path):
        f = _write_axis_file(tmp_path, "cg.txt", ["linear", "power"])
        vals = _get_axis_values("condition_growth", f)
        assert vals == ["linear", "power"]

    def test_unknown_value_in_file_raises(self, tmp_path):
        f = _write_axis_file(tmp_path, "cg.txt", ["linear", "not_a_real_component"])
        with pytest.raises(ValueError, match="not_a_real_component"):
            _get_axis_values("condition_growth", f)

    def test_comments_ignored_in_file(self, tmp_path):
        f = _write_axis_file(tmp_path, "tr.txt", ["# comment", "passthrough", "logit"])
        vals = _get_axis_values("theta_rescale", f)
        assert vals == ["passthrough", "logit"]


class TestSubdirName:

    def test_format(self):
        assert _subdir_name("linear", "instant", "logit") == f"linear{_SEP}instant{_SEP}logit"


class TestIncompatibleCgTr:

    def test_power_logit_is_incompatible(self):
        assert ("power", "logit") in _INCOMPATIBLE_CG_TR

    def test_saturation_logit_is_incompatible(self):
        assert ("saturation", "logit") in _INCOMPATIBLE_CG_TR

    def test_linear_logit_is_compatible(self):
        assert ("linear", "logit") not in _INCOMPATIBLE_CG_TR

    def test_power_passthrough_is_compatible(self):
        assert ("power", "passthrough") not in _INCOMPATIBLE_CG_TR

    def test_incompatible_combos_skipped_in_grid(self, tmp_path, patched_configure):
        """Incompatible (cg, tr) combos must not appear in the returned list."""
        cfg = _write_base_config(tmp_path)
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["linear", "power"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["passthrough", "logit"])
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        for cg, gt, tr in combos:
            assert (cg, tr) not in _INCOMPATIBLE_CG_TR

    def test_incompatible_combos_not_created_as_subdirs(self, tmp_path, patched_configure):
        """No subdirectory should be created for an incompatible combination."""
        cfg = _write_base_config(tmp_path)
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["power"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["logit"])
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        assert combos == []
        # Only run_all.sh should be in the grid dir; no combo subdirs.
        entries = os.listdir(out)
        assert "run_all.sh" in entries
        assert not any(os.path.isdir(os.path.join(out, e)) for e in entries)

    def test_compatible_combos_still_created_when_some_incompatible(
            self, tmp_path, patched_configure):
        """Compatible combos in the same grid must still be created."""
        cfg = _write_base_config(tmp_path)
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["linear", "power"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["passthrough", "logit"])
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        # linear×instant×passthrough, linear×instant×logit, power×instant×passthrough = 3
        # power×instant×logit is incompatible → skipped
        assert len(combos) == 3
        assert ("linear", "instant", "logit") in combos
        assert ("power", "instant", "passthrough") in combos
        assert ("power", "instant", "logit") not in combos


class TestSetupGridCalibration:

    def _small_grid(self, tmp_path):
        """Base config + file-backed axes limited to 2×2×2 = 8 combos."""
        cfg = _write_base_config(tmp_path)
        cg_file = _write_axis_file(tmp_path, "cg.txt", ["linear", "power"])
        gt_file = _write_axis_file(tmp_path, "gt.txt", ["instant", "memory"])
        tr_file = _write_axis_file(tmp_path, "tr.txt", ["passthrough", "logit"])
        return cfg, cg_file, gt_file, tr_file

    def test_creates_correct_number_of_subdirs(self, tmp_path, patched_configure):
        # 2 cg × 2 gt × 2 tr = 8 total, minus 2 incompatible (power × logit)
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        subdirs = [d for d in os.listdir(out)
                   if os.path.isdir(os.path.join(out, d))]
        assert len(subdirs) == 6

    def test_returns_combo_list(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        assert len(combos) == 6
        assert all(len(c) == 3 for c in combos)

    def test_subdir_names_reflect_components(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        for cg, gt, tr in combos:
            name = _subdir_name(cg, gt, tr)
            assert os.path.isdir(os.path.join(out, name))

    def test_each_subdir_has_yaml_and_csvs(self, tmp_path, patched_configure):
        """write_configuration produces run_config.yaml, run_priors.csv, run_guesses.csv."""
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        for cg, gt, tr in combos:
            subdir = os.path.join(out, _subdir_name(cg, gt, tr))
            assert os.path.exists(os.path.join(subdir, f"{_CONFIG_STEM}_config.yaml"))
            assert os.path.exists(os.path.join(subdir, f"{_CONFIG_STEM}_priors.csv"))
            assert os.path.exists(os.path.join(subdir, f"{_CONFIG_STEM}_guesses.csv"))

    def test_gm_called_with_correct_components(self, tmp_path, patched_configure):
        """GrowthModel must be called with the right component triple for each combo."""
        mock_gm_cls, _ = patched_configure
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        assert mock_gm_cls.call_count == len(combos)
        called_combos = set()
        for call in mock_gm_cls.call_args_list:
            _, kwargs = call
            called_combos.add((
                kwargs.get("condition_growth"),
                kwargs.get("growth_transition"),
                kwargs.get("theta_rescale"),
            ))
        assert called_combos == set(combos)

    def test_write_configuration_called_per_combo(self, tmp_path, patched_configure):
        """write_configuration must be called once per combination."""
        mock_gm_cls, mock_wc = patched_configure
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        assert mock_wc.call_count == len(combos)

    def test_write_configuration_out_prefix_ends_with_config_stem(
            self, tmp_path, patched_configure):
        """out_prefix passed to write_configuration must end with the config stem."""
        _, mock_wc = patched_configure
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        for call in mock_wc.call_args_list:
            _, kwargs = call
            assert kwargs["out_prefix"].endswith(os.sep + _CONFIG_STEM)

    def test_data_paths_made_absolute(self, tmp_path, patched_configure):
        mock_gm_cls, _ = patched_configure
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        for call in mock_gm_cls.call_args_list:
            args, _ = call
            assert os.path.isabs(args[0]), "growth_path should be absolute"
            assert os.path.isabs(args[1]), "binding_path should be absolute"

    def test_absolute_data_paths_left_unchanged(self, tmp_path, patched_configure):
        """Data paths that are already absolute must not be modified."""
        mock_gm_cls, _ = patched_configure
        cfg = _write_base_config(tmp_path, extra={
            "data": {"growth": "/abs/path/growth.csv",
                     "binding": "/abs/path/binding.csv"},
        })
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["linear"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["passthrough"])
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        args, _ = mock_gm_cls.call_args_list[0]
        assert args[0] == "/abs/path/growth.csv"
        assert args[1] == "/abs/path/binding.csv"

    def test_run_all_sh_is_created_and_executable(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        sh = os.path.join(out, "run_all.sh")
        assert os.path.exists(sh)
        mode = os.stat(sh).st_mode
        assert mode & stat.S_IXUSR, "run_all.sh is not user-executable"

    def test_run_all_sh_contains_all_subdir_names(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f)
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        for cg, gt, tr in combos:
            assert _subdir_name(cg, gt, tr) in sh_text

    def test_run_all_sh_uses_correct_seed(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f,
                               seed=42)
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        assert "--seed 42" in sh_text

    def test_run_all_sh_references_config_yaml(self, tmp_path, patched_configure):
        """run_all.sh must invoke tfs-prefit-calibration with the right YAML name."""
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        assert f"{_CONFIG_STEM}_config.yaml" in sh_text

    def test_prefit_args_appear_in_run_all_sh(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f,
                               prefit_args="--convergence_tolerance 0.001")
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        assert "--convergence_tolerance 0.001" in sh_text

    def test_prefit_args_appear_on_every_line(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out,
                                        condition_growth_file=cg_f,
                                        growth_transition_file=gt_f,
                                        theta_rescale_file=tr_f,
                                        prefit_args="--max_num_epochs 500")
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        prefit_lines = [l for l in sh_text.splitlines()
                        if "tfs-prefit-calibration" in l]
        assert len(prefit_lines) == len(combos)
        for line in prefit_lines:
            assert "--max_num_epochs 500" in line

    def test_no_prefit_args_leaves_command_clean(self, tmp_path, patched_configure):
        """When prefit_args is None the command must not have a trailing space or
        stray argument."""
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        for line in sh_text.splitlines():
            if "tfs-prefit-calibration" in line:
                assert line.rstrip().endswith(f"--out_prefix {_RUN_OUT_PREFIX})")

    def test_prefit_args_multiple_flags(self, tmp_path, patched_configure):
        """Multiple flags in prefit_args must all appear in run_all.sh."""
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f,
                               prefit_args="--convergence_tolerance 0.001 --max_num_epochs 500")
        sh_text = open(os.path.join(out, "run_all.sh")).read()
        assert "--convergence_tolerance 0.001" in sh_text
        assert "--max_num_epochs 500" in sh_text

    def test_defaults_use_all_registered_values(self, tmp_path, patched_configure):
        cfg = _write_base_config(tmp_path)
        out = str(tmp_path / "grid")
        combos = setup_grid_calibration(cfg, out_prefix=out)
        # Total registered minus incompatible pairs (multiplied by all gt values).
        n_gt = len(model_registry["growth_transition"])
        n_incompatible = len(_INCOMPATIBLE_CG_TR) * n_gt
        expected = (
            len(model_registry["condition_growth"])
            * n_gt
            * len(model_registry["theta_rescale"])
            - n_incompatible
        )
        assert len(combos) == expected

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            setup_grid_calibration(str(tmp_path / "does_not_exist.yaml"),
                                   out_prefix=str(tmp_path / "grid"))

    def test_missing_data_key_raises(self, tmp_path):
        cfg_path = tmp_path / "run_config.yaml"
        yaml.dump({"components": {}}, cfg_path.open("w"))
        with pytest.raises((ValueError, KeyError)):
            setup_grid_calibration(str(cfg_path),
                                   out_prefix=str(tmp_path / "grid"))

    def test_out_prefix_directory_is_created(self, tmp_path, patched_configure):
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "brand_new_dir")
        assert not os.path.exists(out)
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        assert os.path.isdir(out)

    def test_does_not_modify_base_config(self, tmp_path, patched_configure):
        cfg_path, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        original = open(cfg_path).read()
        setup_grid_calibration(cfg_path, out_prefix=str(tmp_path / "grid"),
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        assert open(cfg_path).read() == original

    def test_unknown_component_file_raises(self, tmp_path):
        cfg = _write_base_config(tmp_path)
        bad_file = _write_axis_file(tmp_path, "bad.txt", ["not_real"])
        with pytest.raises(ValueError, match="not_real"):
            setup_grid_calibration(cfg, out_prefix=str(tmp_path / "grid"),
                                   condition_growth_file=bad_file)

    def test_existing_out_prefix_is_reused(self, tmp_path, patched_configure):
        """Calling setup on an existing directory should not raise."""
        cfg, cg_f, gt_f, tr_f = self._small_grid(tmp_path)
        out = str(tmp_path / "grid")
        os.makedirs(out)
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        subdirs = [d for d in os.listdir(out)
                   if os.path.isdir(os.path.join(out, d))]
        assert len(subdirs) == 6

    def test_non_grid_components_forwarded_to_gm(self, tmp_path, patched_configure):
        """Non-grid settings in config['components'] must be passed to GrowthModel."""
        mock_gm_cls, _ = patched_configure
        cfg = _write_base_config(tmp_path, extra={
            "components": {
                "condition_growth": "linear",
                "growth_transition": "instant",
                "theta_rescale": "passthrough",
                "activity": "fixed",
                "dk_geno": "hierarchical",
            }
        })
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["linear"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["passthrough"])
        out = str(tmp_path / "grid")
        setup_grid_calibration(cfg, out_prefix=out,
                               condition_growth_file=cg_f,
                               growth_transition_file=gt_f,
                               theta_rescale_file=tr_f)
        _, kwargs = mock_gm_cls.call_args_list[0]
        assert kwargs.get("activity") == "fixed"
        assert kwargs.get("dk_geno") == "hierarchical"

    def test_main_cli_runs(self, tmp_path, mocker, patched_configure):
        """main() should invoke setup_grid_calibration via generalized_main."""
        cfg = _write_base_config(tmp_path)
        cg_f = _write_axis_file(tmp_path, "cg.txt", ["linear"])
        gt_f = _write_axis_file(tmp_path, "gt.txt", ["instant"])
        tr_f = _write_axis_file(tmp_path, "tr.txt", ["passthrough"])
        out = str(tmp_path / "grid")
        argv = [
            cfg,
            "--out_prefix", out,
            "--condition_growth_file", cg_f,
            "--growth_transition_file", gt_f,
            "--theta_rescale_file", tr_f,
        ]
        mocker.patch("sys.argv", ["tfs-setup-grid-calibration"] + argv)
        setup_main()
        assert os.path.isdir(out)


# ===========================================================================
# Tests for summarize_grid_calibration
# ===========================================================================

class TestComputeAic:

    def test_known_values(self):
        n_params, n_obs, rmse = 5, 100, 1.0
        expected = 2 * 5 + 100 * (1 + math.log(2 * math.pi * 1.0 ** 2))
        assert _compute_aic(n_params, n_obs, rmse) == pytest.approx(expected)

    def test_none_input_returns_nan(self):
        assert math.isnan(_compute_aic(None, 100, 0.5))
        assert math.isnan(_compute_aic(5, None, 0.5))
        assert math.isnan(_compute_aic(5, 100, None))

    def test_zero_obs_returns_nan(self):
        assert math.isnan(_compute_aic(5, 0, 0.5))

    def test_zero_rmse_returns_nan(self):
        assert math.isnan(_compute_aic(5, 100, 0.0))

    def test_negative_rmse_returns_nan(self):
        assert math.isnan(_compute_aic(5, 100, -0.1))

    def test_more_params_gives_higher_aic_at_same_rmse(self):
        aic_few = _compute_aic(5, 100, 0.5)
        aic_many = _compute_aic(20, 100, 0.5)
        assert aic_many > aic_few

    def test_lower_rmse_gives_lower_aic_at_same_params(self):
        aic_good = _compute_aic(5, 100, 0.1)
        aic_bad = _compute_aic(5, 100, 1.0)
        assert aic_good < aic_bad


class TestComputeAicWeights:

    def test_weights_sum_to_one(self):
        aics = [10.0, 12.0, 15.0]
        w = _compute_aic_weights(aics)
        assert np.isclose(w.sum(), 1.0)

    def test_best_model_has_highest_weight(self):
        aics = [10.0, 12.0, 20.0]
        w = _compute_aic_weights(aics)
        assert w[0] > w[1] > w[2]

    def test_equal_aics_give_equal_weights(self):
        aics = [10.0, 10.0, 10.0]
        w = _compute_aic_weights(aics)
        assert np.allclose(w, 1.0 / 3)

    def test_nan_aic_gives_nan_weight(self):
        aics = [10.0, float("nan"), 12.0]
        w = _compute_aic_weights(aics)
        assert np.isfinite(w[0])
        assert math.isnan(w[1])
        assert np.isfinite(w[2])

    def test_all_nan_returns_all_nan(self):
        aics = [float("nan"), float("nan")]
        w = _compute_aic_weights(aics)
        assert all(math.isnan(x) for x in w)

    def test_finite_weights_sum_to_one_when_some_nan(self):
        aics = [10.0, float("nan"), 12.0]
        w = _compute_aic_weights(aics)
        assert np.isclose(w[np.isfinite(w)].sum(), 1.0)


class TestFindStatsJson:

    def test_returns_none_when_no_file(self, tmp_path):
        assert _find_stats_json(str(tmp_path)) is None

    def test_returns_path_when_file_exists(self, tmp_path):
        (tmp_path / "prefit_calib_stats.json").write_text("{}")
        result = _find_stats_json(str(tmp_path))
        assert result is not None
        assert result.endswith("_calib_stats.json")

    def test_returns_most_recent_when_multiple(self, tmp_path):
        p1 = tmp_path / "run1_calib_stats.json"
        p2 = tmp_path / "run2_calib_stats.json"
        p1.write_text("{}")
        p2.write_text("{}")
        import time; time.sleep(0.01)
        p2.touch()
        result = _find_stats_json(str(tmp_path))
        assert os.path.basename(result) == "run2_calib_stats.json"


class TestReadComponents:

    def test_reads_all_three_keys(self, tmp_path):
        """Components are nested under a 'components' key (write_configuration format)."""
        cfg = {
            "components": {
                "condition_growth": "linear",
                "growth_transition": "instant",
                "theta_rescale": "passthrough",
            },
            "other_key": "ignored",
        }
        (tmp_path / "run_config.yaml").write_text(yaml.dump(cfg))
        cg, gt, tr = _read_components(str(tmp_path))
        assert cg == "linear"
        assert gt == "instant"
        assert tr == "passthrough"

    def test_returns_none_when_yaml_missing(self, tmp_path):
        assert _read_components(str(tmp_path)) == (None, None, None)

    def test_returns_none_for_missing_keys(self, tmp_path):
        (tmp_path / "run_config.yaml").write_text(yaml.dump({}))
        cg, gt, tr = _read_components(str(tmp_path))
        assert cg is None and gt is None and tr is None

    def test_returns_none_for_empty_components_dict(self, tmp_path):
        (tmp_path / "run_config.yaml").write_text(yaml.dump({"components": {}}))
        cg, gt, tr = _read_components(str(tmp_path))
        assert cg is None and gt is None and tr is None


class TestSummarizeGridCalibration:

    def _make_grid(self, tmp_path, runs):
        for r in runs:
            _make_completed_run(
                tmp_path,
                r["cg"], r["gt"], r["tr"],
                r.get("stats", _minimal_stats()),
            )
        return tmp_path

    def test_writes_summary_csv(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
            {"cg": "power", "gt": "memory", "tr": "logit"},
        ])
        summarize_grid_calibration(str(tmp_path),
                                   out_prefix=str(tmp_path / "summary"))
        assert (tmp_path / "summary.csv").exists()

    def test_default_out_prefix_is_grid_summary(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
        ])
        summarize_grid_calibration(str(tmp_path))
        assert (tmp_path / "grid_summary.csv").exists()

    def test_csv_has_expected_columns(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        for col in ("subdir", "condition_growth", "growth_transition",
                    "theta_rescale", "aic", "aic_weight", "rmse", "n_params",
                    "n_obs"):
            assert col in df.columns, f"Missing column: {col}"

    def test_one_row_per_completed_run(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
            {"cg": "power", "gt": "memory", "tr": "logit"},
            {"cg": "saturation", "gt": "baranyi", "tr": "passthrough"},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert len(df) == 3

    def test_sorted_by_aic_weight_descending(self, tmp_path):
        # Give each run a different rmse so AICs differ.
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough",
             "stats": _minimal_stats(rmse=1.0, n_params=10)},
            {"cg": "power", "gt": "memory", "tr": "logit",
             "stats": _minimal_stats(rmse=0.2, n_params=10)},
            {"cg": "saturation", "gt": "baranyi", "tr": "passthrough",
             "stats": _minimal_stats(rmse=2.0, n_params=10)},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        weights = df["aic_weight"].dropna().tolist()
        assert weights == sorted(weights, reverse=True)

    def test_best_fit_is_first_row(self, tmp_path):
        """The run with the lowest RMSE (and same n_params) should rank first."""
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough",
             "stats": _minimal_stats(rmse=1.0, n_params=5)},
            {"cg": "power", "gt": "memory", "tr": "logit",
             "stats": _minimal_stats(rmse=0.1, n_params=5)},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert df.iloc[0]["condition_growth"] == "power"

    def test_aic_weights_sum_to_one(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough",
             "stats": _minimal_stats(rmse=0.5)},
            {"cg": "power", "gt": "memory", "tr": "logit",
             "stats": _minimal_stats(rmse=1.0)},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert np.isclose(df["aic_weight"].sum(), 1.0)

    def test_skips_subdir_without_stats_json(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
        ])
        # Create an incomplete subdir with no stats JSON.
        incomplete = tmp_path / "power__memory__logit"
        incomplete.mkdir()
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert len(df) == 1

    def test_nan_n_params_produces_nan_aic(self, tmp_path):
        stats = _minimal_stats()
        stats["n_params"] = None
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough",
             "stats": stats},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert math.isnan(df.iloc[0]["aic"])

    def test_missing_grid_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Grid directory not found"):
            summarize_grid_calibration(str(tmp_path / "does_not_exist"))

    def test_empty_grid_returns_empty_dataframe(self, tmp_path):
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_component_names_in_output(self, tmp_path):
        self._make_grid(tmp_path, [
            {"cg": "saturation", "gt": "baranyi_k", "tr": "logit"},
        ])
        df = summarize_grid_calibration(str(tmp_path),
                                        out_prefix=str(tmp_path / "s"))
        assert df.iloc[0]["condition_growth"] == "saturation"
        assert df.iloc[0]["growth_transition"] == "baranyi_k"
        assert df.iloc[0]["theta_rescale"] == "logit"

    def test_main_cli_runs(self, tmp_path, mocker):
        """main() should invoke summarize_grid_calibration via generalized_main."""
        self._make_grid(tmp_path, [
            {"cg": "linear", "gt": "instant", "tr": "passthrough"},
        ])
        out = str(tmp_path / "out")
        mocker.patch("sys.argv",
                     ["tfs-summarize-grid-calibration", str(tmp_path),
                      "--out_prefix", out])
        summarize_main()
        assert os.path.exists(f"{out}.csv")
