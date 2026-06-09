"""
Tests for run_prefit_calibration.py.

The pre-fit script builds an in-process calibration ModelOrchestrator,
runs MAP, computes Hessian-based per-site sigmas, and then writes
in-place updates (with .bak backups) into the production priors and
guesses CSVs.  These tests exercise the helper functions directly and
mock the heavy machinery (read_configuration, ModelOrchestrator, RunInference)
when testing the orchestration in run_prefit_calibration / main.
"""
import dataclasses
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
import jax
import jax.numpy as jnp

import flax.struct as fstruct

from tfscreen.tfmodel.scripts.run_prefit_calibration_cli import (
    _apply_guesses_updates,
    _apply_priors_updates,
    _build_calibration_model,
    _build_csv_updates,
    _compute_growth_pred_std,
    _compute_theta_values,
    _csv_row_name,
    _identify_field_mapping,
    _inject_calibration_priors,
    _intersect_data,
    _run_calibration_diagnostics,
    _make_correlation_plot,
    _resolve_csv_paths,
    _write_calibration_stats,
    main,
    run_prefit_calibration,
    _CALIBRATION_OVERRIDES,
    _PINNED_COMPONENTS,
)


# ---------------------------------------------------------------------------
# Lightweight stand-ins for the flax pytree dataclasses used by ModelPriors.
#
# The real component-prior dataclasses are flax.struct.dataclass instances
# that expose a .replace() method.  We define a few minimal substitutes
# here so the unit tests don't need to instantiate a full ModelOrchestrator.
# ---------------------------------------------------------------------------

@fstruct.dataclass
class _FakeCondGrowth:
    growth_k_hyper_loc: float = 0.0
    k_hyper_loc_scale: float = 1.0
    growth_k_hyper_scale_loc: float = 0.5
    growth_m_hyper_loc: float = 0.0
    m_hyper_loc_scale: float = 1.0
    growth_m_hyper_scale_loc: float = 0.5
    pinned: dict = fstruct.field(default_factory=dict, pytree_node=False)


@fstruct.dataclass
class _FakeGrowthTransition:
    pre_t_hyper_loc: float = 0.0
    pre_t_hyper_loc_scale: float = 1.0
    pinned: dict = fstruct.field(default_factory=dict, pytree_node=False)


@fstruct.dataclass
class _FakeGTNoPinned:
    pre_t_hyper_loc: float = 0.0
    pre_t_hyper_loc_scale: float = 1.0
    # No pinned field; mirrors the "instant" growth_transition variant.


@fstruct.dataclass
class _FakeActivity:
    hyper_loc_loc: float = 0.0
    hyper_scale_loc: float = 1.0
    pinned: dict = fstruct.field(default_factory=dict, pytree_node=False)


@fstruct.dataclass
class _FakeDkGeno:
    hyper_loc_loc: float = 0.0
    hyper_scale_loc: float = 1.0
    hyper_shift_loc: float = 0.5
    pinned: dict = fstruct.field(default_factory=dict, pytree_node=False)


@fstruct.dataclass
class _FakeLnCfu0:
    # Array fields (one element per library class) matching the real ModelPriors.
    ln_cfu0_hyper_loc_locs: object = None
    ln_cfu0_hyper_scale_locs: object = None
    pinned: dict = fstruct.field(default_factory=dict, pytree_node=False)


@fstruct.dataclass
class _FakeTheta:
    theta_values: object = None  # array placeholder


@fstruct.dataclass
class _FakeGrowth:
    condition_growth: _FakeCondGrowth
    growth_transition: _FakeGrowthTransition
    activity: _FakeActivity
    dk_geno: _FakeDkGeno
    ln_cfu0: _FakeLnCfu0


@fstruct.dataclass
class _FakePriors:
    growth: _FakeGrowth
    theta: _FakeTheta


def _make_fake_priors(gt_has_pinned=True):
    gt = _FakeGrowthTransition() if gt_has_pinned else _FakeGTNoPinned()
    return _FakePriors(
        growth=_FakeGrowth(
            condition_growth=_FakeCondGrowth(),
            growth_transition=gt,
            activity=_FakeActivity(),
            dk_geno=_FakeDkGeno(),
            ln_cfu0=_FakeLnCfu0(
                ln_cfu0_hyper_loc_locs=jnp.array([5.0]),
                ln_cfu0_hyper_scale_locs=jnp.array([1.0]),
            ),
        ),
        theta=_FakeTheta(),
    )


# ---------------------------------------------------------------------------
# _intersect_data
# ---------------------------------------------------------------------------

class TestIntersectData:

    def _basic_growth(self):
        return pd.DataFrame({
            "genotype": ["wt", "wt", "A1T", "G2P"],
            "titrant_name": ["IPTG"] * 4,
            "titrant_conc": [0.0, 1.0, 0.0, 0.0],
            "ln_cfu": [1.0, 2.0, 3.0, 4.0],
        })

    def _basic_binding(self):
        return pd.DataFrame({
            "genotype": ["wt", "wt", "A1T", "M3L"],
            "titrant_name": ["IPTG"] * 4,
            "titrant_conc": [0.0, 1.0, 0.0, 0.0],
            "theta_obs": [0.1, 0.5, 0.2, 0.3],
            "theta_std": [0.05, 0.05, 0.05, 0.05],
        })

    def test_intersection_keeps_only_shared_rows(self):
        g, b = _intersect_data(self._basic_growth(), self._basic_binding())
        # G2P is only in growth, M3L is only in binding; both must drop.
        assert set(g["genotype"].unique()) == {"wt", "A1T"}
        assert set(b["genotype"].unique()) == {"wt", "A1T"}

    def test_intersection_returns_copies_not_views(self):
        gdf = self._basic_growth()
        bdf = self._basic_binding()
        g, _ = _intersect_data(gdf, bdf)
        # Mutating the returned frame must not alter the original.
        g.loc[g.index[0], "ln_cfu"] = -999.0
        assert gdf.loc[0, "ln_cfu"] != -999.0

    def test_intersection_preserves_per_titrant_resolution(self):
        # (wt, IPTG, 1.0) and (wt, IPTG, 0.0) must each be evaluated
        # separately — we cannot just intersect on genotype.
        growth = pd.DataFrame({
            "genotype": ["wt", "wt"],
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
            "ln_cfu": [1.0, 2.0],
        })
        binding = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [1.0],
            "theta_obs": [0.5],
            "theta_std": [0.05],
        })
        g, b = _intersect_data(growth, binding)
        assert len(g) == 1
        assert g["titrant_conc"].iloc[0] == 1.0

    def test_empty_intersection_raises(self):
        growth = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
            "ln_cfu": [1.0],
        })
        binding = pd.DataFrame({
            "genotype": ["A1T"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
            "theta_obs": [0.5],
            "theta_std": [0.05],
        })
        with pytest.raises(ValueError, match="intersection is empty"):
            _intersect_data(growth, binding)

    def test_missing_column_in_growth_raises(self):
        growth = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"]})
        binding = self._basic_binding()
        with pytest.raises(ValueError, match="growth_df is missing"):
            _intersect_data(growth, binding)

    def test_missing_column_in_binding_raises(self):
        growth = self._basic_growth()
        binding = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["IPTG"]})
        with pytest.raises(ValueError, match="binding_df is missing"):
            _intersect_data(growth, binding)


# ---------------------------------------------------------------------------
# _compute_theta_values
# ---------------------------------------------------------------------------

class TestComputeThetaValues:

    def _make_orchestrator_cal(self, titrant_names, titrant_concs):
        """Return a MagicMock with the binding_tm shape used by the helper."""
        orchestrator = MagicMock()
        # Order matters; the helper reads names then concs by index lookup.
        orchestrator.binding_tm.tensor_dim_names = ["titrant_name", "titrant_conc"]
        orchestrator.binding_tm.tensor_dim_labels = [list(titrant_names),
                                                     list(titrant_concs)]
        return orchestrator

    def test_inverse_variance_weighted_mean(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0])
        # Two genotypes: theta=0.2 (sigma=0.1) and theta=0.8 (sigma=0.01)
        # Inverse-variance weights heavily favor the second, so the
        # consensus should be ~0.8 (not the unweighted 0.5).
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "theta_obs": [0.2, 0.8],
            "theta_std": [0.1, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 1)
        # Closer to 0.8 than to the midpoint 0.5
        assert theta[0, 0] > 0.7

    def test_falls_back_to_plain_mean_when_all_stds_zero(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "theta_obs": [0.3, 0.7],
            "theta_std": [0.0, 0.0],  # invalid weights → fallback to mean
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta[0, 0] == pytest.approx(0.5)

    def test_unobserved_cell_defaults_to_midpoint(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [0.0, 1.0])
        # Only the IPTG=1.0 cell is observed.
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1.0],
            "theta_obs": [0.9],
            "theta_std": [0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta[0, 0] == pytest.approx(0.5)
        assert theta[0, 1] > 0.85

    def test_clip_to_open_interval(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [0.0, 1.0])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
            "theta_obs": [0.0, 1.0],
            "theta_std": [0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        # Zero and one would blow up the downstream logit; expect clipping.
        assert theta[0, 0] > 0.0
        assert theta[0, 1] < 1.0




# ---------------------------------------------------------------------------
# _csv_row_name / _build_csv_updates
# ---------------------------------------------------------------------------

class TestCsvRowName:

    def test_basic_row_name(self):
        assert _csv_row_name("condition_growth", "growth_k_hyper_loc") == \
            "growth.condition_growth.growth_k_hyper_loc"


class TestBuildCsvUpdates:

    def test_normal_site_writes_loc_and_scale(self):
        field_mapping = {
            "growth_k_hyper": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "growth_k_hyper_loc",
                "scale_field": "k_hyper_loc_scale",
                "is_array": False,
            },
        }
        hessian_results = {
            "growth_k_hyper": {"map": np.float32(2.5),
                               "sigma": np.float32(0.7)},
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping,
                                                          hessian_results)
        # MAP → loc_field; sigma → scale_field
        assert prior_updates[
            "growth.condition_growth.growth_k_hyper_loc"] == pytest.approx(2.5)
        assert prior_updates[
            "growth.condition_growth.k_hyper_loc_scale"] == pytest.approx(0.7)
        assert guess_updates["growth_k_hyper"] == pytest.approx(2.5)

    def test_halfnormal_site_recenters_on_map(self):
        field_mapping = {
            "k_hyper_scale_loc": {
                "component": "condition_growth",
                "dist_class": "HalfNormal",
                "scale_field": "growth_k_hyper_scale_loc",
                "is_array": False,
            },
        }
        hessian_results = {
            "k_hyper_scale_loc": {"map": np.float32(0.42),
                                  "sigma": np.float32(0.09)},
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping,
                                                          hessian_results)
        # HalfNormal has only one parameter (its scale). We recenter the
        # prior on the MAP point — sigma is dropped because there is no
        # second parameter to consume it.
        assert prior_updates == {
            "growth.condition_growth.growth_k_hyper_scale_loc": pytest.approx(0.42),
        }
        assert guess_updates["k_hyper_scale_loc"] == pytest.approx(0.42)

    def test_skips_sites_without_hessian_result(self):
        field_mapping = {
            "missing_site": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "growth_k_hyper_loc",
                "scale_field": "k_hyper_loc_scale",
                "is_array": False,
            },
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping, {})
        assert prior_updates == {}
        assert guess_updates == {}

    def test_array_site_writes_locs_to_guesses(self):
        """Simple-prior array sites write per-condition MAP values to guesses."""
        field_mapping = {
            "condition_growth_k": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "k_loc",
                "scale_field": "k_scale",
                "is_array": True,
            },
        }
        hessian_results = {
            "condition_growth_k": {
                "map": np.array([1.0, 2.0, 3.0]),
                "sigma": np.array([0.1, 0.2, 0.3]),
            },
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping,
                                                          hessian_results)
        # Array sites do NOT update priors
        assert prior_updates == {}
        # Guess locs get the per-condition MAP array
        assert "condition_growth_k_locs" in guess_updates
        assert np.allclose(guess_updates["condition_growth_k_locs"],
                           [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# _apply_priors_updates / _apply_guesses_updates
# ---------------------------------------------------------------------------

class TestApplyPriorsUpdates:

    def _write_csv(self, tmp_path, rows):
        path = tmp_path / "priors.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_writes_bak_and_updates_only_matching_rows(self, tmp_path):
        path = self._write_csv(tmp_path, [
            {"parameter": "growth.condition_growth.growth_k_hyper_loc",
             "value": 1.0},
            {"parameter": "growth.activity.hyper_loc_loc",
             "value": 0.0},
        ])
        updates = {"growth.condition_growth.growth_k_hyper_loc": 9.0}
        _apply_priors_updates(path, updates)

        bak = pd.read_csv(path + ".bak")
        new = pd.read_csv(path)
        # Backup retains the original
        assert bak.set_index("parameter")["value"][
            "growth.condition_growth.growth_k_hyper_loc"] == 1.0
        # Live file has the updated value
        assert new.set_index("parameter")["value"][
            "growth.condition_growth.growth_k_hyper_loc"] == 9.0
        # Untouched rows are preserved verbatim
        assert new.set_index("parameter")["value"][
            "growth.activity.hyper_loc_loc"] == 0.0

    def test_no_updates_is_noop(self, tmp_path):
        path = self._write_csv(tmp_path, [
            {"parameter": "x", "value": 1.0},
        ])
        _apply_priors_updates(path, {})
        # No .bak, no rewrite.
        assert not os.path.exists(path + ".bak")

    def test_warns_on_missing_row(self, tmp_path, capsys):
        path = self._write_csv(tmp_path, [
            {"parameter": "x", "value": 1.0},
        ])
        _apply_priors_updates(path, {"y_does_not_exist": 5.0})
        err = capsys.readouterr().err
        assert "no matching row" in err
        assert "y_does_not_exist" in err

    def test_raises_on_malformed_csv(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required"):
            _apply_priors_updates(str(bad), {"x": 1.0})


class TestApplyGuessesUpdates:

    def _write_csv(self, tmp_path, rows):
        path = tmp_path / "guesses.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_writes_bak_and_updates_only_scalar_rows(self, tmp_path):
        # Two rows for the same parameter: one scalar (flat_index NaN),
        # one array entry (flat_index=0).  Only the scalar must be
        # overwritten.
        path = self._write_csv(tmp_path, [
            {"parameter": "growth_k_hyper", "value": 1.0,
             "flat_index": float("nan")},
            {"parameter": "growth_k_offset", "value": 0.0, "flat_index": 0.0},
            {"parameter": "growth_k_offset", "value": 0.0, "flat_index": 1.0},
        ])
        updates = {"growth_k_hyper": 7.7,
                   "growth_k_offset": 999.0}
        _apply_guesses_updates(path, updates)

        new = pd.read_csv(path)
        # Scalar row updated
        scalar_row = new[(new["parameter"] == "growth_k_hyper")
                         & (new["flat_index"].isna())]
        assert scalar_row["value"].iloc[0] == pytest.approx(7.7)
        # Array rows untouched (the update key matched no scalar row)
        offset_rows = new[new["parameter"] == "growth_k_offset"]
        assert (offset_rows["value"] == 0.0).all()

        # .bak written
        assert os.path.exists(path + ".bak")

    def test_no_updates_is_noop(self, tmp_path):
        path = self._write_csv(tmp_path, [
            {"parameter": "x", "value": 1.0, "flat_index": float("nan")},
        ])
        _apply_guesses_updates(path, {})
        assert not os.path.exists(path + ".bak")

    def test_updates_array_rows_by_flat_index(self, tmp_path):
        path = self._write_csv(tmp_path, [
            {"parameter": "condition_growth_k_locs", "value": 0.0, "flat_index": 0.0},
            {"parameter": "condition_growth_k_locs", "value": 0.0, "flat_index": 1.0},
            {"parameter": "condition_growth_k_locs", "value": 0.0, "flat_index": 2.0},
            {"parameter": "other_param", "value": 9.9, "flat_index": float("nan")},
        ])
        _apply_guesses_updates(path, {"condition_growth_k_locs": np.array([1.1, 2.2, 3.3])})
        new = pd.read_csv(path)
        k_rows = new[new["parameter"] == "condition_growth_k_locs"].sort_values("flat_index")
        assert k_rows["value"].tolist() == pytest.approx([1.1, 2.2, 3.3])
        # Unrelated row preserved
        assert new[new["parameter"] == "other_param"]["value"].iloc[0] == pytest.approx(9.9)
        assert os.path.exists(path + ".bak")

    def test_treats_all_rows_scalar_when_no_flat_index_column(self, tmp_path):
        path = self._write_csv(tmp_path, [
            {"parameter": "alpha", "value": 1.0},
        ])
        _apply_guesses_updates(path, {"alpha": 4.5})
        new = pd.read_csv(path)
        assert new.set_index("parameter")["value"]["alpha"] == 4.5

    def test_raises_on_malformed_csv(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required"):
            _apply_guesses_updates(str(bad), {"x": 1.0})


# ---------------------------------------------------------------------------
# _resolve_csv_paths
# ---------------------------------------------------------------------------

class TestResolveCsvPaths:

    def _write_yaml(self, path, body):
        with open(path, "w") as fh:
            yaml.dump(body, fh)

    def test_returns_resolved_paths(self, tmp_path):
        priors = tmp_path / "p.csv"
        guesses = tmp_path / "g.csv"
        priors.write_text("parameter,value\n")
        guesses.write_text("parameter,value\n")
        cfg = tmp_path / "cfg.yaml"
        self._write_yaml(cfg, {"priors_file": "p.csv",
                               "guesses_file": "g.csv"})
        p, g = _resolve_csv_paths(str(cfg))
        assert os.path.abspath(p) == str(priors)
        assert os.path.abspath(g) == str(guesses)

    def test_absolute_paths_are_honored(self, tmp_path):
        priors = tmp_path / "abs_p.csv"
        guesses = tmp_path / "abs_g.csv"
        priors.write_text("parameter,value\n")
        guesses.write_text("parameter,value\n")
        cfg = tmp_path / "cfg.yaml"
        self._write_yaml(cfg, {"priors_file": str(priors),
                               "guesses_file": str(guesses)})
        p, g = _resolve_csv_paths(str(cfg))
        assert p == str(priors)
        assert g == str(guesses)

    def test_missing_priors_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        self._write_yaml(cfg, {"guesses_file": "g.csv"})
        with pytest.raises(ValueError, match="priors_file"):
            _resolve_csv_paths(str(cfg))

    def test_missing_guesses_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        self._write_yaml(cfg, {"priors_file": "p.csv"})
        with pytest.raises(ValueError, match="guesses_file"):
            _resolve_csv_paths(str(cfg))

    def test_missing_priors_file_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        guesses = tmp_path / "g.csv"
        guesses.write_text("parameter,value\n")
        self._write_yaml(cfg, {"priors_file": "p_missing.csv",
                               "guesses_file": "g.csv"})
        with pytest.raises(FileNotFoundError, match="Priors file not found"):
            _resolve_csv_paths(str(cfg))

    def test_missing_guesses_file_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        priors = tmp_path / "p.csv"
        priors.write_text("parameter,value\n")
        self._write_yaml(cfg, {"priors_file": "p.csv",
                               "guesses_file": "g_missing.csv"})
        with pytest.raises(FileNotFoundError, match="Guesses file not found"):
            _resolve_csv_paths(str(cfg))


# ---------------------------------------------------------------------------
# _build_calibration_model — verify it forwards the right overrides
# ---------------------------------------------------------------------------

class TestBuildCalibrationModel:

    def test_overrides_replace_production_components(self):
        orchestrator_prod = MagicMock()
        orchestrator_prod.settings = {
            "theta": "categorical_geno",          # → simple
            "activity": "horseshoe_geno",         # → hierarchical
            "dk_geno": "fixed",              # → hierarchical
            "ln_cfu0": "fixed",              # → hierarchical
            "transformation": "logit_norm",  # → single
            "theta_growth_noise": "beta",    # → zero
            "theta_binding_noise": "beta",   # → zero
            "condition_growth": "linear",    # passthrough
            "growth_transition": "instant",  # passthrough
            "batch_size": 7,
            "spiked_genotypes": ["WT"],
        }

        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.ModelOrchestrator"
        ) as MockGM:
            MockGM.return_value = MagicMock()
            growth_df = pd.DataFrame()
            binding_df = pd.DataFrame()
            _build_calibration_model(orchestrator_prod, growth_df, binding_df)

        kwargs = MockGM.call_args.kwargs
        # Overrides applied
        for k, v in _CALIBRATION_OVERRIDES.items():
            assert kwargs[k] == v
        # Spiked genotypes dropped (calibration only sees the intersection)
        assert kwargs["spiked_genotypes"] is None
        # batch_size pulled out of settings and passed positionally
        assert kwargs["batch_size"] == 7
        # condition_growth and growth_transition flow through unchanged
        assert kwargs["condition_growth"] == "linear"
        assert kwargs["growth_transition"] == "instant"
        # binding_weight must be 1.0 regardless of production value so the
        # calibration MAP learns the binding→growth linkage without the
        # production upweighting drowning the binding signal.
        assert kwargs["binding_weight"] == 1.0

    def test_binding_weight_reset_from_large_production_value(self):
        # Reproduces the weighting bug: production YAML stores a large
        # binding_weight (N_growth / N_binding); the calibration model
        # must use 1.0 instead.
        orchestrator_prod = MagicMock()
        orchestrator_prod.settings = {
            "theta": "categorical_geno",
            "activity": "horseshoe_geno",
            "dk_geno": "fixed",
            "ln_cfu0": "fixed",
            "transformation": "logit_norm",
            "theta_growth_noise": "beta",
            "theta_binding_noise": "beta",
            "condition_growth": "linear",
            "growth_transition": "instant",
            "batch_size": None,
            "spiked_genotypes": None,
            "binding_weight": 250.0,  # typical large production value
        }

        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.ModelOrchestrator"
        ) as MockGM:
            MockGM.return_value = MagicMock()
            _build_calibration_model(orchestrator_prod, pd.DataFrame(), pd.DataFrame())

        assert MockGM.call_args.kwargs["binding_weight"] == 1.0

    def test_does_not_mutate_production_settings(self):
        # Ensure we work on a copy.
        orchestrator_prod = MagicMock()
        orchestrator_prod.settings = {"theta": "categorical_geno",
                            "activity": "horseshoe_geno",
                            "dk_geno": "fixed",
                            "ln_cfu0": "fixed",
                            "transformation": "logit_norm",
                            "theta_growth_noise": "beta",
                            "theta_binding_noise": "beta",
                            "spiked_genotypes": None,
                            "batch_size": None,
                            "binding_weight": 150.0}
        original_settings = dict(orchestrator_prod.settings)

        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.ModelOrchestrator"
        ) as MockGM:
            MockGM.return_value = MagicMock()
            _build_calibration_model(orchestrator_prod, pd.DataFrame(), pd.DataFrame())

        assert orchestrator_prod.settings == original_settings


# ---------------------------------------------------------------------------
# _inject_calibration_priors — end-to-end pin / copy behaviour
# ---------------------------------------------------------------------------

class TestInjectCalibrationPriors:

    def _make_models(self, gt_has_pinned=True):
        orchestrator_cal = MagicMock()
        orchestrator_prod = MagicMock()
        orchestrator_cal._priors_history = []

        cal_priors = _make_fake_priors(gt_has_pinned=gt_has_pinned)
        prod_priors = _make_fake_priors(gt_has_pinned=True)
        # Tweak production scalars so we can verify they get copied.
        prod_priors = prod_priors.replace(
            growth=prod_priors.growth.replace(
                condition_growth=prod_priors.growth.condition_growth.replace(
                    growth_k_hyper_loc=2.5,
                    k_hyper_loc_scale=0.4,
                ),
                growth_transition=prod_priors.growth.growth_transition.replace(
                    pre_t_hyper_loc=3.5,
                ),
            ),
        )
        orchestrator_cal.priors = cal_priors
        orchestrator_prod.priors = prod_priors

        # Capture all writes to _priors so the test can introspect them.
        def _setter(value):
            orchestrator_cal._priors_history.append(value)
        type(orchestrator_cal)._priors = property(
            lambda self: self._priors_history[-1] if self._priors_history else None,
            lambda self, v: self._priors_history.append(v),
        )
        return orchestrator_cal, orchestrator_prod

    def test_copies_production_condition_growth_and_clears_pinned(self):
        orchestrator_cal, orchestrator_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, theta_values)

        new_priors = orchestrator_cal._priors
        assert new_priors.growth.condition_growth.growth_k_hyper_loc == 2.5
        assert new_priors.growth.condition_growth.k_hyper_loc_scale == 0.4
        # pinned cleared regardless of production's contents
        assert new_priors.growth.condition_growth.pinned == {}

    def test_copies_growth_transition_and_clears_pinned(self):
        orchestrator_cal, orchestrator_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, theta_values)

        new_priors = orchestrator_cal._priors
        assert new_priors.growth.growth_transition.pre_t_hyper_loc == 3.5
        assert new_priors.growth.growth_transition.pinned == {}

    def test_handles_growth_transition_without_pinned(self):
        # The "instant" growth_transition has no pinned field at all.
        orchestrator_cal, orchestrator_prod = self._make_models(gt_has_pinned=False)
        theta_values = np.array([[0.5]])
        # Should not raise.
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, theta_values)
        new_priors = orchestrator_cal._priors
        assert new_priors.growth.growth_transition.pre_t_hyper_loc == 3.5

    def test_pins_activity_hyperparams_to_prior_locs(self):
        orchestrator_cal, orchestrator_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, theta_values)
        pinned = orchestrator_cal._priors.growth.activity.pinned
        # Both suffixes from _PINNED_COMPONENTS["activity"] populated.
        for suffix, _ in _PINNED_COMPONENTS["activity"]:
            assert suffix in pinned

    def test_pins_ln_cfu0_hyperparams_single_class(self):
        # Single-class case: array fields of length 1 must produce
        # "hyper_loc_0" and "hyper_scale_0" in the pinned dict, matching
        # the site suffix "{name}_hyper_loc_0" used by define_model in
        # the hierarchical ln_cfu0 component.
        orchestrator_cal, orchestrator_prod = self._make_models()
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, np.array([[0.5]]))
        pinned = orchestrator_cal._priors.growth.ln_cfu0.pinned
        assert "hyper_loc_0" in pinned, f"pinned keys: {list(pinned)}"
        assert "hyper_scale_0" in pinned, f"pinned keys: {list(pinned)}"
        assert pinned["hyper_loc_0"] == pytest.approx(5.0)
        assert pinned["hyper_scale_0"] == pytest.approx(1.0)

    def test_pins_ln_cfu0_hyperparams_multi_class(self):
        # Multi-class case: 2-element arrays must produce per-class entries
        # "hyper_loc_0", "hyper_loc_1", "hyper_scale_0", "hyper_scale_1".
        orchestrator_cal, orchestrator_prod = self._make_models()
        # Replace the ln_cfu0 component in orchestrator_cal.priors (the
        # MagicMock attribute that _inject_calibration_priors reads).
        two_class_ln_cfu0 = _FakeLnCfu0(
            ln_cfu0_hyper_loc_locs=jnp.array([5.0, 7.0]),
            ln_cfu0_hyper_scale_locs=jnp.array([1.0, 2.0]),
        )
        cal_priors = orchestrator_cal.priors
        new_growth = cal_priors.growth.replace(ln_cfu0=two_class_ln_cfu0)
        orchestrator_cal.priors = cal_priors.replace(growth=new_growth)

        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, np.array([[0.5]]))
        pinned = orchestrator_cal._priors.growth.ln_cfu0.pinned
        for key, expected in [
            ("hyper_loc_0", 5.0), ("hyper_loc_1", 7.0),
            ("hyper_scale_0", 1.0), ("hyper_scale_1", 2.0),
        ]:
            assert key in pinned, f"missing '{key}'; pinned keys: {list(pinned)}"
            assert pinned[key] == pytest.approx(expected)

    def test_dk_geno_not_in_pinned_components(self):
        # dk_geno uses the fixed component during calibration (all zeros).
        # Pinning hyperparameters is meaningless for a fixed component, so
        # dk_geno must not appear in _PINNED_COMPONENTS.
        assert "dk_geno" not in _PINNED_COMPONENTS

    def test_dk_geno_not_pinned_after_inject(self):
        # _inject_calibration_priors must leave dk_geno.pinned untouched
        # (empty) because the fixed component has no hyperparameters.
        orchestrator_cal, orchestrator_prod = self._make_models()
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, np.array([[0.5]]))
        assert orchestrator_cal._priors.growth.dk_geno.pinned == {}

    def test_theta_values_are_set(self):
        orchestrator_cal, orchestrator_prod = self._make_models()
        theta_values = np.array([[0.1, 0.9]])
        _inject_calibration_priors(orchestrator_cal, orchestrator_prod, theta_values)
        result = orchestrator_cal._priors.theta.theta_values
        assert np.allclose(np.asarray(result), theta_values)


# ---------------------------------------------------------------------------
# run_prefit_calibration — orchestration with everything mocked
# ---------------------------------------------------------------------------

class TestRunPrefitCalibrationOrchestration:

    def _write_yaml_and_csvs(self, tmp_path):
        priors = tmp_path / "p.csv"
        guesses = tmp_path / "g.csv"
        priors.write_text("parameter,value\n")
        guesses.write_text("parameter,value\n")
        cfg = tmp_path / "cfg.yaml"
        with open(cfg, "w") as fh:
            yaml.dump({
                "priors_file": "p.csv",
                "guesses_file": "g.csv",
                "data": {"growth": "growth.csv", "binding": "binding.csv"},
            }, fh)
        return str(cfg), str(priors), str(guesses)

    def _patch_pipeline(self, mocker, orchestrator_cal=None, params=None,
                        hessian_results=None, field_mapping=None):
        """Stub out every heavy callsite of run_prefit_calibration."""
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._intersect_data",
            return_value=(pd.DataFrame(), pd.DataFrame()),
        )
        orchestrator_cal = orchestrator_cal or MagicMock()
        orchestrator_cal.init_params = {}
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._build_calibration_model",
            return_value=orchestrator_cal,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._compute_theta_values",
            return_value=np.array([[0.5]]),
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._inject_calibration_priors",
        )
        mock_ri = MagicMock()
        mock_ri._iterations_per_epoch = 1
        mock_ri.compute_hessian_sigmas.return_value = hessian_results or {}
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.RunInference",
            return_value=mock_ri,
        )
        mock_run_map = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._run_calibration_map",
            return_value=("svi_state", params or {}, True),
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._identify_field_mapping",
            return_value=field_mapping or {},
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._compute_growth_pred_std",
            return_value=None,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._run_calibration_diagnostics",
        )
        return mock_ri, mock_run_map

    def test_raises_when_seed_and_checkpoint_both_none(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        with pytest.raises(ValueError, match="seed must be provided"):
            run_prefit_calibration(config_file=cfg)

    def test_returns_run_calibration_map_result(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        self._patch_pipeline(mocker, params={"alpha_auto_loc": 1.0})
        result = run_prefit_calibration(config_file=cfg, seed=42)
        assert result == ("svi_state", {"alpha_auto_loc": 1.0}, True)

    def test_writes_bak_files_when_updates_present(self, tmp_path, mocker):
        cfg, priors, guesses = self._write_yaml_and_csvs(tmp_path)
        # Seed the priors CSV with both the loc-field and scale-field rows
        # that the Normal-site update should target, plus an unrelated
        # row to verify we don't disturb it.
        pd.DataFrame({
            "parameter": [
                "growth.condition_growth.growth_k_hyper_loc",
                "growth.condition_growth.k_hyper_loc_scale",
                "growth.activity.hyper_loc_loc",
            ],
            "value": [1.0, 1.0, 0.0],
        }).to_csv(priors, index=False)
        pd.DataFrame({
            "parameter": ["growth_k_hyper"],
            "value": [1.0],
            "flat_index": [float("nan")],
        }).to_csv(guesses, index=False)

        self._patch_pipeline(
            mocker,
            params={"growth_k_hyper_auto_loc": 9.0},
            hessian_results={
                "growth_k_hyper": {"map": np.float32(9.0),
                                   "sigma": np.float32(0.5)},
            },
            field_mapping={
                "growth_k_hyper": {
                    "component": "condition_growth",
                    "dist_class": "Normal",
                    "loc_field": "growth_k_hyper_loc",
                    "scale_field": "k_hyper_loc_scale",
                    "is_array": False,
                },
            },
        )
        run_prefit_calibration(config_file=cfg, seed=1)

        # Both .bak files written, both live CSVs updated.
        assert os.path.exists(priors + ".bak")
        assert os.path.exists(guesses + ".bak")
        new_priors = pd.read_csv(priors).set_index("parameter")["value"]
        # MAP → loc-field row; Hessian sigma → scale-field row;
        # unrelated row preserved.
        assert new_priors["growth.condition_growth.growth_k_hyper_loc"] == 9.0
        assert new_priors[
            "growth.condition_growth.k_hyper_loc_scale"] == pytest.approx(0.5)
        assert new_priors["growth.activity.hyper_loc_loc"] == 0.0
        new_guesses = pd.read_csv(guesses)
        assert new_guesses.iloc[0]["value"] == 9.0

    def test_default_seed_used_when_resuming_from_checkpoint(self, tmp_path,
                                                             mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        # No seed but a checkpoint_file → should not raise.
        run_prefit_calibration(config_file=cfg, seed=None,
                               checkpoint_file="/tmp/resume.pkl")
        # checkpoint_file forwarded to _run_calibration_map
        kwargs = mock_run_map.call_args.kwargs
        assert kwargs["checkpoint_file"] == "/tmp/resume.pkl"

    def test_optimizer_kwargs_forwarded(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(
            config_file=cfg,
            seed=1,
            adam_step_size=2.5e-3,
            adam_final_step_size=2.5e-7,
            adam_clip_norm=2.0,
            elbo_num_particles=5,
            convergence_tolerance=1e-4,
            convergence_window=20,
            patience=4,
            convergence_check_interval=3,
            checkpoint_interval=25,
            max_num_epochs=500,
            init_param_jitter=0.0,
        )
        kwargs = mock_run_map.call_args.kwargs
        assert kwargs["adam_step_size"] == 2.5e-3
        assert kwargs["adam_final_step_size"] == 2.5e-7
        assert kwargs["adam_clip_norm"] == 2.0
        assert kwargs["elbo_num_particles"] == 5
        assert kwargs["convergence_tolerance"] == 1e-4
        assert kwargs["convergence_window"] == 20
        assert kwargs["patience"] == 4
        assert kwargs["convergence_check_interval"] == 3
        assert kwargs["checkpoint_interval"] == 25
        assert kwargs["max_num_epochs"] == 500
        assert kwargs["init_param_jitter"] == 0.0

    def test_default_out_prefix_is_prefit(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1)
        assert mock_run_map.call_args.kwargs["out_prefix"] == "tfs_prefit"

    def test_custom_out_prefix_is_honored(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1,
                               out_prefix="my_runA")
        assert mock_run_map.call_args.kwargs["out_prefix"] == "my_runA"

    def test_default_init_param_jitter_is_zero(self, tmp_path, mocker):
        """Pre-fit should be deterministic given a seed; default jitter is 0."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1)
        assert mock_run_map.call_args.kwargs["init_param_jitter"] == 0.0

    def test_compute_growth_pred_std_called_with_ri_and_params(self, tmp_path,
                                                               mocker):
        """_compute_growth_pred_std is called with the RunInference instance
        and the exact params dict returned by _run_calibration_map."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        mock_params = {"alpha_auto_loc": np.float32(2.0)}
        mock_ri, _ = self._patch_pipeline(mocker, params=mock_params)

        # Override the stub so we can inspect calls.
        mock_std_fn = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._compute_growth_pred_std",
            return_value=None,
        )
        run_prefit_calibration(config_file=cfg, seed=1)

        assert mock_std_fn.call_count == 1
        call_args = mock_std_fn.call_args
        assert call_args.args[0] is mock_ri      # first positional: ri
        assert call_args.args[1] is mock_params  # second positional: params

    def test_growth_pred_std_forwarded_to_run_calibration_diagnostics(
            self, tmp_path, mocker):
        """The growth_pred_std returned by _compute_growth_pred_std is passed
        as the growth_pred_std keyword argument to _run_calibration_diagnostics."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        sentinel = np.ones((1, 2, 1, 1, 1, 1, 2))
        self._patch_pipeline(mocker)

        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._compute_growth_pred_std",
            return_value=sentinel,
        )
        mock_diag = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._run_calibration_diagnostics",
        )
        run_prefit_calibration(config_file=cfg, seed=1)

        diag_kwargs = mock_diag.call_args.kwargs
        assert diag_kwargs.get("growth_pred_std") is sentinel


# ---------------------------------------------------------------------------
# main() CLI plumbing
# ---------------------------------------------------------------------------

class TestPrefitMainCLI:

    def test_main_invokes_run_prefit_with_required_args(self, tmp_path):
        """`main()` should drive the wrapper through generalized_main."""
        argv = ["cal.yaml", "--seed", "13"]
        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()

        assert mock_run.call_count == 1
        kwargs = mock_run.call_args.kwargs
        assert kwargs["config_file"] == "cal.yaml"
        assert kwargs["seed"] == 13

    def test_main_forwards_custom_out_prefix(self, tmp_path):
        argv = [
            "cal.yaml",
            "--seed", "0",
            "--out_prefix", "calibration_runA",
        ]
        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()
        assert mock_run.call_args.kwargs["out_prefix"] == "calibration_runA"

    def test_main_forwards_checkpoint_file(self, tmp_path):
        argv = [
            "cal.yaml",
            "--seed", "0",
            "--checkpoint_file", "/tmp/ck.pkl",
        ]
        with patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()
        assert mock_run.call_args.kwargs["checkpoint_file"] == "/tmp/ck.pkl"


# ---------------------------------------------------------------------------
# _compute_growth_pred_std
# ---------------------------------------------------------------------------

class TestComputeGrowthPredStd:
    """Tests for _compute_growth_pred_std.

    The function wraps heavy JAX machinery (Hessian, Cholesky, vmap, Predictive).
    Tests that require the Hessian/Predictive path stub those out via
    ``_patch_heavy_machinery``; the empty-params case is exercised directly.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_heavy_machinery(self, mocker, n_params, n_samples, pred_output):
        """Stub JAX/NumPyro operations so the function reaches Predictive.

        * ``jax.device_put``  → identity passthrough
        * ``jax.hessian``     → returns the n_params × n_params identity
        * ``jax.random.normal`` → zeros, so all Laplace samples equal MAP
        * ``trace`` / ``seed`` (in the module) → empty model trace, meaning
          no constrained transforms and no per-site bijections applied
        * ``numpyro.infer.Predictive`` → returns ``pred_output``
        """
        mocker.patch("jax.device_put", side_effect=lambda x: x)
        hess = jnp.eye(n_params)
        # jax.hessian(fn) returns a callable; that callable(flat_map) returns
        # the matrix.  Use a MagicMock so the two-call chain works correctly.
        mock_hess_fn = MagicMock(return_value=hess)
        mocker.patch("jax.hessian", return_value=mock_hess_fn)
        mocker.patch(
            "jax.random.normal",
            return_value=jnp.zeros((n_samples, n_params)),
        )
        mock_traced = MagicMock()
        mock_traced.get_trace.return_value = {}
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.trace",
            return_value=mock_traced,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.seed",
            return_value=MagicMock(),
        )
        mock_pred_inst = MagicMock(return_value=pred_output)
        mocker.patch("numpyro.infer.Predictive", return_value=mock_pred_inst)

    def _make_inputs(self, n_params=2):
        """Build (ri, orchestrator_cal, params) mocks for _compute_growth_pred_std."""
        ri = MagicMock()
        ri.get_key.return_value = jax.random.PRNGKey(0)
        orchestrator_cal = MagicMock()
        orchestrator_cal.data.num_genotype = 3
        orchestrator_cal.get_batch.return_value = MagicMock()
        orchestrator_cal.priors = MagicMock()
        params = {f"site{i}_auto_loc": np.float32(float(i))
                  for i in range(n_params)}
        return ri, orchestrator_cal, params

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_returns_none_when_params_empty(self):
        """No _auto_loc params → early-exit None without touching JAX."""
        ri = MagicMock()
        orchestrator_cal = MagicMock()
        orchestrator_cal.data.num_genotype = 2
        orchestrator_cal.get_batch.return_value = MagicMock()
        assert _compute_growth_pred_std(ri, {}, orchestrator_cal) is None

    def test_returns_none_when_growth_pred_absent(self, mocker):
        """When Predictive returns no growth_pred site, result must be None."""
        n_params, n_samples = 2, 4
        ri, orchestrator_cal, params = self._make_inputs(n_params)
        self._patch_heavy_machinery(
            mocker, n_params, n_samples,
            pred_output={"other_site": np.ones((n_samples, 3))},
        )
        assert _compute_growth_pred_std(ri, params, orchestrator_cal,
                                        n_samples=n_samples) is None

    def test_returns_array_matching_growth_pred_shape(self, mocker):
        """Output shape must equal the (R,T,CP,CS,TN,TC,G) tensor shape."""
        n_params, n_samples = 2, 4
        tensor_shape = (1, 3, 1, 1, 1, 1, 2)
        ri, orchestrator_cal, params = self._make_inputs(n_params)
        growth_pred = np.ones((n_samples,) + tensor_shape) * 5.0
        self._patch_heavy_machinery(
            mocker, n_params, n_samples,
            pred_output={"growth_pred": growth_pred},
        )
        result = _compute_growth_pred_std(ri, params, orchestrator_cal,
                                          n_samples=n_samples)
        assert result is not None
        assert result.shape == tensor_shape

    def test_std_is_zero_for_identical_samples(self, mocker):
        """All Laplace samples identical → elementwise std must be 0."""
        n_params, n_samples = 2, 6
        tensor_shape = (1, 2, 1, 1, 1, 1, 3)
        ri, orchestrator_cal, params = self._make_inputs(n_params)
        growth_pred = np.ones((n_samples,) + tensor_shape) * 7.0
        self._patch_heavy_machinery(
            mocker, n_params, n_samples,
            pred_output={"growth_pred": growth_pred},
        )
        result = _compute_growth_pred_std(ri, params, orchestrator_cal,
                                          n_samples=n_samples)
        assert np.allclose(result, 0.0)

    def test_std_matches_numpy_std_of_samples(self, mocker):
        """Output must equal np.std(growth_pred_samples, axis=0) exactly."""
        n_params, n_samples = 2, 10
        tensor_shape = (1, 2, 1, 1, 1, 1, 3)
        ri, orchestrator_cal, params = self._make_inputs(n_params)
        rng = np.random.default_rng(42)
        growth_pred = rng.normal(size=(n_samples,) + tensor_shape).astype(np.float32)
        self._patch_heavy_machinery(
            mocker, n_params, n_samples,
            pred_output={"growth_pred": growth_pred},
        )
        result = _compute_growth_pred_std(ri, params, orchestrator_cal,
                                          n_samples=n_samples)
        # Compare in float64 (the function casts before computing std).
        expected = growth_pred.astype(np.float64).std(axis=0)
        assert np.allclose(result, expected, atol=1e-5)

    def test_no_overflow_for_large_growth_pred_values(self, mocker):
        """float32 growth_pred values large enough to overflow when squared
        must produce finite std (not inf/nan) after float64 cast + clipping."""
        n_params, n_samples = 2, 4
        tensor_shape = (1, 2, 1, 1, 1, 1, 2)
        ri, orchestrator_cal, params = self._make_inputs(n_params)
        # Values near the float32 overflow boundary (~1e38); squaring these
        # in float32 would overflow to inf inside np.std.
        growth_pred = np.full((n_samples,) + tensor_shape, 1e20, dtype=np.float32)
        growth_pred[0] = -1e20  # introduce variance so std != 0
        self._patch_heavy_machinery(
            mocker, n_params, n_samples,
            pred_output={"growth_pred": growth_pred},
        )
        result = _compute_growth_pred_std(ri, params, orchestrator_cal,
                                          n_samples=n_samples)
        assert result is not None
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# _write_calibration_stats
# ---------------------------------------------------------------------------

def _make_stats_df(n=20, include_std=True, seed=7):
    """Return a minimal DataFrame suitable for _write_calibration_stats."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "genotype":      [f"g{i % 3}" for i in range(n)],
        "ln_cfu":        rng.normal(10.0, 1.0, n),
        "ln_cfu_pred":   rng.normal(10.0, 1.0, n),
    })
    if include_std:
        df["ln_cfu_pred_std"] = rng.uniform(0.1, 0.5, n)
    return df


class TestWriteCalibrationStats:

    def test_writes_json_with_expected_keys(self, tmp_path):
        df = _make_stats_df()
        _write_calibration_stats(df, str(tmp_path / "run"))
        json_path = tmp_path / "run_calib_stats.json"
        assert json_path.exists()
        import json
        stats = json.loads(json_path.read_text())
        expected_keys = {
            "pct_success", "rmse", "normalized_rmse", "pearson_r",
            "spearman_r", "r_squared", "mean_error", "coverage_prob",
            "residual_corr", "residual_corr_p_value", "bp_p_value",
            "n_params", "n_obs",
        }
        assert expected_keys == set(stats.keys())

    def test_all_values_are_python_float_int_or_null(self, tmp_path):
        """Values must be Python float, int, or null (no numpy scalars or NaN)."""
        df = _make_stats_df()
        _write_calibration_stats(df, str(tmp_path / "run"))
        import json
        stats = json.loads((tmp_path / "run_calib_stats.json").read_text())
        int_keys = {"n_params", "n_obs"}
        for k, v in stats.items():
            if k in int_keys:
                assert v is None or isinstance(v, int), \
                    f"Key '{k}' has non-int value {v!r}"
            else:
                assert v is None or isinstance(v, float), \
                    f"Key '{k}' has non-float value {v!r}"
                if isinstance(v, float):
                    assert np.isfinite(v), f"Key '{k}' is not finite: {v}"

    def test_n_params_and_n_obs_written_when_provided(self, tmp_path):
        """n_params and n_obs appear as integers in the JSON when supplied."""
        import json
        df = _make_stats_df(n=10)
        _write_calibration_stats(df, str(tmp_path / "run"), n_params=7, n_obs=10)
        stats = json.loads((tmp_path / "run_calib_stats.json").read_text())
        assert stats["n_params"] == 7
        assert stats["n_obs"] == 10
        assert isinstance(stats["n_params"], int)
        assert isinstance(stats["n_obs"], int)

    def test_n_params_and_n_obs_null_when_not_provided(self, tmp_path):
        """n_params and n_obs are null in the JSON when not supplied."""
        import json
        df = _make_stats_df()
        _write_calibration_stats(df, str(tmp_path / "run"))
        stats = json.loads((tmp_path / "run_calib_stats.json").read_text())
        assert stats["n_params"] is None
        assert stats["n_obs"] is None

    def test_returns_silently_when_std_column_missing(self, tmp_path):
        """No JSON written when ln_cfu_pred_std is absent."""
        df = _make_stats_df(include_std=False)
        _write_calibration_stats(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_stats.json").exists()

    def test_returns_silently_when_pred_column_missing(self, tmp_path):
        """No JSON written when ln_cfu_pred is absent."""
        df = _make_stats_df()
        df = df.drop(columns=["ln_cfu_pred"])
        _write_calibration_stats(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_stats.json").exists()

    def test_returns_silently_when_no_valid_rows(self, tmp_path):
        """No JSON written when all rows have NaN predictions."""
        df = _make_stats_df()
        df["ln_cfu_pred"] = np.nan
        _write_calibration_stats(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_stats.json").exists()

    def test_nan_stats_serialised_as_null(self, tmp_path):
        """Stats entries that are NaN (e.g. constant inputs) must become null."""
        # Constant ln_cfu makes pearson_r undefined → NaN.
        df = _make_stats_df()
        df["ln_cfu"] = 10.0
        _write_calibration_stats(df, str(tmp_path / "run"))
        import json
        stats = json.loads((tmp_path / "run_calib_stats.json").read_text())
        assert stats["pearson_r"] is None


# ---------------------------------------------------------------------------
# _make_correlation_plot
# ---------------------------------------------------------------------------

def _make_corr_df(n_genotypes=3, n_per=8, seed=11):
    """Return a minimal DataFrame for _make_correlation_plot."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_genotypes):
        obs  = rng.normal(10.0 + i, 1.0, n_per)
        pred = obs + rng.normal(0, 0.3, n_per)
        for o, p in zip(obs, pred):
            rows.append({"genotype": f"g{i}", "ln_cfu": o, "ln_cfu_pred": p})
    return pd.DataFrame(rows)


class TestMakeCorrelationPlot:

    def test_writes_pdf(self, tmp_path):
        df = _make_corr_df()
        _make_correlation_plot(df, str(tmp_path / "run"))
        assert (tmp_path / "run_calib_correlation.pdf").exists()

    def test_returns_silently_when_genotype_column_missing(self, tmp_path):
        df = _make_corr_df().drop(columns=["genotype"])
        _make_correlation_plot(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_correlation.pdf").exists()

    def test_returns_silently_when_no_valid_rows(self, tmp_path):
        df = _make_corr_df()
        df["ln_cfu_pred"] = np.nan
        _make_correlation_plot(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_correlation.pdf").exists()

    def test_colors_cycle_for_more_genotypes_than_prop_cycle(self, tmp_path,
                                                              mocker):
        """When n_genotypes > len(prop_cycle) the modulo wraps colours; no
        KeyError or IndexError must be raised."""
        # Default prop_cycle has 10 colours; use 15 genotypes.
        df = _make_corr_df(n_genotypes=15)
        # Should complete without error and produce a PDF.
        _make_correlation_plot(df, str(tmp_path / "run"))
        assert (tmp_path / "run_calib_correlation.pdf").exists()

    def test_matplotlib_unavailable_returns_silently(self, tmp_path, mocker):
        df = _make_corr_df()
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=fake_import)
        _make_correlation_plot(df, str(tmp_path / "run"))
        assert not (tmp_path / "run_calib_correlation.pdf").exists()

    # -----------------------------------------------------------------
    # Helpers shared by the _run_calibration_diagnostics tests below.
    # -----------------------------------------------------------------

    @staticmethod
    def _make_fake_predict_dfs(genotypes=("gA", "gB"), n_t=3):
        """Return (all_dfs, orchestrator_mock) for diagnostics tests."""
        rng = np.random.default_rng(0)
        rows = []
        for i, g in enumerate(genotypes):
            for t_i, t in enumerate(np.linspace(0, 10, n_t)):
                rows.append({
                    "replicate": "r1",
                    "condition_pre": "cp1",
                    "condition_sel": "cs1",
                    "titrant_name": "tn1",
                    "titrant_conc": 0.0,
                    "genotype": g,
                    "t_sel": t,
                    "ln_cfu": rng.uniform(8, 12),
                    "ln_cfu_std": 0.2,
                    "q05": rng.uniform(7, 9),
                    "median": rng.uniform(9, 11),
                    "q95": rng.uniform(11, 13),
                    "replicate_idx": 0,
                    "time_idx": t_i,
                    "condition_pre_idx": 0,
                    "condition_sel_idx": 0,
                    "titrant_name_idx": 0,
                    "titrant_conc_idx": 0,
                    "genotype_idx": i,
                })

        pred_df = pd.DataFrame(rows)

        ln_cfu0_rows = []
        for i, g in enumerate(genotypes):
            ln_cfu0_rows.append({
                "replicate": "r1",
                "condition_pre": "cp1",
                "genotype": g,
                "q05": 8.5,
                "median": 9.0,
                "q95": 9.5,
            })
        ln_cfu0_df = pd.DataFrame(ln_cfu0_rows)

        all_dfs = {"growth_pred": pred_df, "ln_cfu0": ln_cfu0_df}

        orch = MagicMock()
        orch.growth_df = pd.DataFrame({
            "replicate": ["r1"] * len(genotypes),
            "condition_pre": ["cp1"] * len(genotypes),
            "condition_sel": ["cs1"] * len(genotypes),
            "titrant_name": ["tn1"] * len(genotypes),
            "titrant_conc": [0.0] * len(genotypes),
            "genotype": list(genotypes),
            "t_pre": [5.0] * len(genotypes),
            "t_sel": [10.0] * len(genotypes),
            "ln_cfu": [10.0] * len(genotypes),
            "ln_cfu_std": [0.2] * len(genotypes),
        })
        orch.presplit_df = None
        orch.growth_tm.tensor_dim_names = [
            "replicate", "time", "condition_pre", "condition_sel",
            "titrant_name", "titrant_conc", "genotype",
        ]

        return all_dfs, orch

    def test_run_calibration_diagnostics_calls_both_helpers(self, tmp_path, mocker):
        """_run_calibration_diagnostics calls _write_calibration_stats and
        _make_correlation_plot with a DataFrame that has ln_cfu_pred."""
        mock_stats = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._write_calibration_stats",
        )
        mock_corr = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._make_correlation_plot",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.plot_geno_trajectory",
            return_value=MagicMock(),
        )
        all_dfs, orch = self._make_fake_predict_dfs()
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.predict",
            return_value=all_dfs,
        )

        params = {"alpha_auto_loc": np.array([0.5, 0.6])}
        _run_calibration_diagnostics(orch, params, out_prefix=str(tmp_path / "run"))

        assert mock_stats.call_count == 1
        assert mock_corr.call_count == 1
        df_arg_stats = mock_stats.call_args.args[0]
        df_arg_corr  = mock_corr.call_args.args[0]
        assert isinstance(df_arg_stats, pd.DataFrame)
        assert isinstance(df_arg_corr, pd.DataFrame)
        assert "ln_cfu_pred" in df_arg_stats.columns
        # n_params and n_obs forwarded as kwargs
        stats_kwargs = mock_stats.call_args.kwargs
        assert "n_params" in stats_kwargs and "n_obs" in stats_kwargs
        assert isinstance(stats_kwargs["n_params"], int)
        assert isinstance(stats_kwargs["n_obs"], int)

    def test_run_calibration_diagnostics_one_pdf_per_genotype(self, tmp_path, mocker):
        """One trajectory PDF must be written for each genotype."""
        genotypes = ("genoX", "genoY", "genoZ")
        all_dfs, orch = self._make_fake_predict_dfs(genotypes=genotypes)
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.predict",
            return_value=all_dfs,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._write_calibration_stats",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._make_correlation_plot",
        )
        # plot_geno_trajectory returns a real Figure so savefig works.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.plot_geno_trajectory",
            side_effect=lambda df, **kw: plt.subplots(1)[0].get_figure(),
        )

        prefix = str(tmp_path / "run")
        _run_calibration_diagnostics(orch, {}, out_prefix=prefix)

        pdfs = [f for f in tmp_path.iterdir() if f.suffix == ".pdf"
                and "trajectory" in f.name]
        assert len(pdfs) == len(genotypes), \
            f"expected {len(genotypes)} PDFs, got {len(pdfs)}: {sorted(pdfs)}"

    def test_run_calibration_diagnostics_pdf_named_by_genotype(self, tmp_path, mocker):
        """Each PDF filename must contain the genotype name."""
        all_dfs, orch = self._make_fake_predict_dfs(genotypes=("WT", "V44A"))
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.predict",
            return_value=all_dfs,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._write_calibration_stats",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._make_correlation_plot",
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.plot_geno_trajectory",
            side_effect=lambda df, **kw: plt.subplots(1)[0].get_figure(),
        )

        prefix = str(tmp_path / "run")
        _run_calibration_diagnostics(orch, {}, out_prefix=prefix)

        names = [f.name for f in tmp_path.iterdir() if f.suffix == ".pdf"]
        assert any("WT" in n for n in names)
        assert any("V44A" in n for n in names)

    def test_run_calibration_diagnostics_genotype_slash_sanitized(self, tmp_path, mocker):
        """Slashes in genotype names are replaced with underscores in filenames."""
        all_dfs, orch = self._make_fake_predict_dfs(genotypes=("a/b",))
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.predict",
            return_value=all_dfs,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._write_calibration_stats",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._make_correlation_plot",
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.plot_geno_trajectory",
            side_effect=lambda df, **kw: plt.subplots(1)[0].get_figure(),
        )

        prefix = str(tmp_path / "run")
        _run_calibration_diagnostics(orch, {}, out_prefix=prefix)

        names = [f.name for f in tmp_path.iterdir() if f.suffix == ".pdf"]
        assert any("a_b" in n for n in names), \
            f"Expected 'a_b' in a filename, got: {names}"
        assert not any("a/b" in n for n in names)

    def test_run_calibration_diagnostics_growth_pred_std_attached(self, tmp_path, mocker):
        """When growth_pred_std is supplied, ln_cfu_pred_std appears in the
        DataFrame passed to _write_calibration_stats."""
        all_dfs, orch = self._make_fake_predict_dfs(genotypes=("g0",))
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.predict",
            return_value=all_dfs,
        )
        mock_stats = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._write_calibration_stats",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli._make_correlation_plot",
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".run_prefit_calibration_cli.plot_geno_trajectory",
            return_value=MagicMock(),
        )

        # Shape (1,3,1,1,1,1,1) matches 1 replicate × 3 times × ... × 1 geno
        gp_std = np.full((1, 3, 1, 1, 1, 1, 1), 0.42)
        _run_calibration_diagnostics(
            orch, {}, out_prefix=str(tmp_path / "run"),
            growth_pred_std=gp_std,
        )

        df_arg = mock_stats.call_args.args[0]
        assert "ln_cfu_pred_std" in df_arg.columns
        assert np.allclose(df_arg["ln_cfu_pred_std"].dropna(), 0.42)
