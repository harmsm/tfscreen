"""
Tests for run_prefit_calibration.py.

The pre-fit script builds an in-process calibration GrowthModel,
runs MAP, computes Hessian-based per-site sigmas, and then writes
in-place updates (with .bak backups) into the production priors and
guesses CSVs.  These tests exercise the helper functions directly and
mock the heavy machinery (read_configuration, GrowthModel, RunInference)
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

import flax.struct as fstruct

from tfscreen.analysis.hierarchical.growth_model.scripts.run_prefit_calibration import (
    _apply_guesses_updates,
    _apply_priors_updates,
    _build_calibration_model,
    _build_csv_updates,
    _compute_theta_values,
    _csv_row_name,
    _identify_field_mapping,
    _inject_calibration_priors,
    _intersect_data,
    _resolve_csv_paths,
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
# here so the unit tests don't need to instantiate a full GrowthModel.
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
    ln_cfu0_hyper_loc_loc: float = 5.0
    ln_cfu0_hyper_scale_loc: float = 1.0
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
            ln_cfu0=_FakeLnCfu0(),
        ),
        theta=_FakeTheta(),
    )


# ---------------------------------------------------------------------------
# _intersect_data
# ---------------------------------------------------------------------------

class TestIntersectData:

    def _basic_growth(self):
        return pd.DataFrame({
            "genotype": ["A", "A", "B", "C"],
            "titrant_name": ["IPTG"] * 4,
            "titrant_conc": [0.0, 1.0, 0.0, 0.0],
            "ln_cfu": [1.0, 2.0, 3.0, 4.0],
        })

    def _basic_binding(self):
        return pd.DataFrame({
            "genotype": ["A", "A", "B", "D"],
            "titrant_name": ["IPTG"] * 4,
            "titrant_conc": [0.0, 1.0, 0.0, 0.0],
            "theta_obs": [0.1, 0.5, 0.2, 0.3],
            "theta_std": [0.05, 0.05, 0.05, 0.05],
        })

    def test_intersection_keeps_only_shared_rows(self):
        g, b = _intersect_data(self._basic_growth(), self._basic_binding())
        # Genotype C is only in growth, D is only in binding; both must drop.
        assert set(g["genotype"].unique()) == {"A", "B"}
        assert set(b["genotype"].unique()) == {"A", "B"}

    def test_intersection_returns_copies_not_views(self):
        gdf = self._basic_growth()
        bdf = self._basic_binding()
        g, _ = _intersect_data(gdf, bdf)
        # Mutating the returned frame must not alter the original.
        g.loc[g.index[0], "ln_cfu"] = -999.0
        assert gdf.loc[0, "ln_cfu"] != -999.0

    def test_intersection_preserves_per_titrant_resolution(self):
        # (A, IPTG, 1.0) and (A, IPTG, 0.0) must each be evaluated
        # separately — we cannot just intersect on genotype.
        growth = pd.DataFrame({
            "genotype": ["A", "A"],
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
            "ln_cfu": [1.0, 2.0],
        })
        binding = pd.DataFrame({
            "genotype": ["A"],
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
            "genotype": ["A"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
            "ln_cfu": [1.0],
        })
        binding = pd.DataFrame({
            "genotype": ["B"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.0],
            "theta_obs": [0.5],
            "theta_std": [0.05],
        })
        with pytest.raises(ValueError, match="intersection is empty"):
            _intersect_data(growth, binding)

    def test_missing_column_in_growth_raises(self):
        growth = pd.DataFrame({"genotype": ["A"], "titrant_name": ["IPTG"]})
        binding = self._basic_binding()
        with pytest.raises(ValueError, match="growth_df is missing"):
            _intersect_data(growth, binding)

    def test_missing_column_in_binding_raises(self):
        growth = self._basic_growth()
        binding = pd.DataFrame({"genotype": ["A"], "titrant_name": ["IPTG"]})
        with pytest.raises(ValueError, match="binding_df is missing"):
            _intersect_data(growth, binding)


# ---------------------------------------------------------------------------
# _compute_theta_values
# ---------------------------------------------------------------------------

class TestComputeThetaValues:

    def _make_gm_cal(self, titrant_names, titrant_concs):
        """Return a MagicMock with the binding_tm shape used by the helper."""
        gm = MagicMock()
        # Order matters; the helper reads names then concs by index lookup.
        gm.binding_tm.tensor_dim_names = ["titrant_name", "titrant_conc"]
        gm.binding_tm.tensor_dim_labels = [list(titrant_names),
                                           list(titrant_concs)]
        return gm

    def test_inverse_variance_weighted_mean(self):
        gm = self._make_gm_cal(["IPTG"], [1.0])
        # Two genotypes: theta=0.2 (sigma=0.1) and theta=0.8 (sigma=0.01)
        # Inverse-variance weights heavily favor the second, so the
        # consensus should be ~0.8 (not the unweighted 0.5).
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "theta_obs": [0.2, 0.8],
            "theta_std": [0.1, 0.01],
        })
        theta = np.asarray(_compute_theta_values(gm, binding_df))
        assert theta.shape == (1, 1)
        # Closer to 0.8 than to the midpoint 0.5
        assert theta[0, 0] > 0.7

    def test_falls_back_to_plain_mean_when_all_stds_zero(self):
        gm = self._make_gm_cal(["IPTG"], [1.0])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "theta_obs": [0.3, 0.7],
            "theta_std": [0.0, 0.0],  # invalid weights → fallback to mean
        })
        theta = np.asarray(_compute_theta_values(gm, binding_df))
        assert theta[0, 0] == pytest.approx(0.5)

    def test_unobserved_cell_defaults_to_midpoint(self):
        gm = self._make_gm_cal(["IPTG"], [0.0, 1.0])
        # Only the IPTG=1.0 cell is observed.
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1.0],
            "theta_obs": [0.9],
            "theta_std": [0.01],
        })
        theta = np.asarray(_compute_theta_values(gm, binding_df))
        assert theta[0, 0] == pytest.approx(0.5)
        assert theta[0, 1] > 0.85

    def test_clip_to_open_interval(self):
        gm = self._make_gm_cal(["IPTG"], [0.0, 1.0])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
            "theta_obs": [0.0, 1.0],
            "theta_std": [0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(gm, binding_df))
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
            },
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping, {})
        assert prior_updates == {}
        assert guess_updates == {}

    def test_skips_array_map_values(self):
        # Per-array sites would be a programming error here (the field
        # mapping is supposed to filter them) but defend anyway.
        field_mapping = {
            "k_offset": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "growth_k_hyper_loc",
                "scale_field": "k_hyper_loc_scale",
            },
        }
        hessian_results = {
            "k_offset": {"map": np.array([1.0, 2.0, 3.0]),
                         "sigma": np.array([0.1, 0.1, 0.1])},
        }
        prior_updates, guess_updates = _build_csv_updates(field_mapping,
                                                          hessian_results)
        assert prior_updates == {}
        assert guess_updates == {}


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
        gm_prod = MagicMock()
        gm_prod.settings = {
            "theta": "categorical",          # → simple
            "activity": "horseshoe",         # → hierarchical
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
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.GrowthModel"
        ) as MockGM:
            MockGM.return_value = MagicMock()
            growth_df = pd.DataFrame()
            binding_df = pd.DataFrame()
            _build_calibration_model(gm_prod, growth_df, binding_df)

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

    def test_does_not_mutate_production_settings(self):
        # Ensure we work on a copy.
        gm_prod = MagicMock()
        gm_prod.settings = {"theta": "categorical",
                            "activity": "horseshoe",
                            "dk_geno": "fixed",
                            "ln_cfu0": "fixed",
                            "transformation": "logit_norm",
                            "theta_growth_noise": "beta",
                            "theta_binding_noise": "beta",
                            "spiked_genotypes": None,
                            "batch_size": None}
        original_settings = dict(gm_prod.settings)

        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.GrowthModel"
        ) as MockGM:
            MockGM.return_value = MagicMock()
            _build_calibration_model(gm_prod, pd.DataFrame(), pd.DataFrame())

        assert gm_prod.settings == original_settings


# ---------------------------------------------------------------------------
# _inject_calibration_priors — end-to-end pin / copy behaviour
# ---------------------------------------------------------------------------

class TestInjectCalibrationPriors:

    def _make_models(self, gt_has_pinned=True):
        gm_cal = MagicMock()
        gm_prod = MagicMock()
        gm_cal._priors_history = []

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
        gm_cal.priors = cal_priors
        gm_prod.priors = prod_priors

        # Capture all writes to _priors so the test can introspect them.
        def _setter(value):
            gm_cal._priors_history.append(value)
        type(gm_cal)._priors = property(
            lambda self: self._priors_history[-1] if self._priors_history else None,
            lambda self, v: self._priors_history.append(v),
        )
        return gm_cal, gm_prod

    def test_copies_production_condition_growth_and_clears_pinned(self):
        gm_cal, gm_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(gm_cal, gm_prod, theta_values)

        new_priors = gm_cal._priors
        assert new_priors.growth.condition_growth.growth_k_hyper_loc == 2.5
        assert new_priors.growth.condition_growth.k_hyper_loc_scale == 0.4
        # pinned cleared regardless of production's contents
        assert new_priors.growth.condition_growth.pinned == {}

    def test_copies_growth_transition_and_clears_pinned(self):
        gm_cal, gm_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(gm_cal, gm_prod, theta_values)

        new_priors = gm_cal._priors
        assert new_priors.growth.growth_transition.pre_t_hyper_loc == 3.5
        assert new_priors.growth.growth_transition.pinned == {}

    def test_handles_growth_transition_without_pinned(self):
        # The "instant" growth_transition has no pinned field at all.
        gm_cal, gm_prod = self._make_models(gt_has_pinned=False)
        theta_values = np.array([[0.5]])
        # Should not raise.
        _inject_calibration_priors(gm_cal, gm_prod, theta_values)
        new_priors = gm_cal._priors
        assert new_priors.growth.growth_transition.pre_t_hyper_loc == 3.5

    def test_pins_activity_hyperparams_to_prior_locs(self):
        gm_cal, gm_prod = self._make_models()
        theta_values = np.array([[0.5]])
        _inject_calibration_priors(gm_cal, gm_prod, theta_values)
        pinned = gm_cal._priors.growth.activity.pinned
        # Both suffixes from _PINNED_COMPONENTS["activity"] populated.
        for suffix, _ in _PINNED_COMPONENTS["activity"]:
            assert suffix in pinned

    def test_pins_dk_geno_and_ln_cfu0(self):
        gm_cal, gm_prod = self._make_models()
        _inject_calibration_priors(gm_cal, gm_prod, np.array([[0.5]]))
        for comp in ("dk_geno", "ln_cfu0"):
            pinned = getattr(gm_cal._priors.growth, comp).pinned
            for suffix, _ in _PINNED_COMPONENTS[comp]:
                assert suffix in pinned

    def test_theta_values_are_set(self):
        gm_cal, gm_prod = self._make_models()
        theta_values = np.array([[0.1, 0.9]])
        _inject_calibration_priors(gm_cal, gm_prod, theta_values)
        result = gm_cal._priors.theta.theta_values
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

    def _patch_pipeline(self, mocker, gm_cal=None, params=None,
                        hessian_results=None, field_mapping=None):
        """Stub out every heavy callsite of run_prefit_calibration."""
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._intersect_data",
            return_value=(pd.DataFrame(), pd.DataFrame()),
        )
        gm_cal = gm_cal or MagicMock()
        gm_cal.init_params = {}
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._build_calibration_model",
            return_value=gm_cal,
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._compute_theta_values",
            return_value=np.array([[0.5]]),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._inject_calibration_priors",
        )
        mock_ri = MagicMock()
        mock_ri._iterations_per_epoch = 1
        mock_ri.compute_hessian_sigmas.return_value = hessian_results or {}
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.RunInference",
            return_value=mock_ri,
        )
        mock_run_map = mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._run_calibration_map",
            return_value=("svi_state", params or {}, True),
        )
        mocker.patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration._identify_field_mapping",
            return_value=field_mapping or {},
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

    def test_default_out_root_is_prefit(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1)
        assert mock_run_map.call_args.kwargs["out_root"] == "prefit"

    def test_custom_out_root_is_honored(self, tmp_path, mocker):
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1,
                               out_root="my_runA")
        assert mock_run_map.call_args.kwargs["out_root"] == "my_runA"

    def test_default_init_param_jitter_is_zero(self, tmp_path, mocker):
        """Pre-fit should be deterministic given a seed; default jitter is 0."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        _, mock_run_map = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1)
        assert mock_run_map.call_args.kwargs["init_param_jitter"] == 0.0


# ---------------------------------------------------------------------------
# main() CLI plumbing
# ---------------------------------------------------------------------------

class TestPrefitMainCLI:

    def test_main_invokes_run_prefit_with_required_args(self, tmp_path):
        """`main()` should drive the wrapper through generalized_main."""
        argv = ["cal.yaml", "--seed", "13"]
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()

        assert mock_run.call_count == 1
        kwargs = mock_run.call_args.kwargs
        assert kwargs["config_file"] == "cal.yaml"
        assert kwargs["seed"] == 13

    def test_main_forwards_custom_out_root(self, tmp_path):
        argv = [
            "cal.yaml",
            "--seed", "0",
            "--out_root", "calibration_runA",
        ]
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()
        assert mock_run.call_args.kwargs["out_root"] == "calibration_runA"

    def test_main_forwards_checkpoint_file(self, tmp_path):
        argv = [
            "cal.yaml",
            "--seed", "0",
            "--checkpoint_file", "/tmp/ck.pkl",
        ]
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.scripts"
            ".run_prefit_calibration.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()
        assert mock_run.call_args.kwargs["checkpoint_file"] == "/tmp/ck.pkl"
