"""
Tests for prefit_calibration_cli.py.

The pre-fit script builds an in-process calibration ModelOrchestrator,
runs MAP, and then writes in-place updates (with .bak backups) into the
production priors and guesses CSVs.  These tests exercise the helper
functions directly and mock the heavy machinery (read_configuration,
ModelOrchestrator, RunInference) when testing the orchestration in
run_prefit_calibration / main.
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

from tfscreen.tfmodel.scripts.prefit_calibration_cli import (
    _apply_guesses_updates,
    _apply_priors_updates,
    _build_calibration_model,
    _build_csv_updates,
    _build_hessian_scale_updates,
    _compute_theta_values,
    _csv_row_name,
    _identify_field_mapping,
    _inject_calibration_priors,
    _intersect_data,
    _resolve_csv_paths,
    main,
    run_prefit_calibration,
    _CALIBRATION_OVERRIDES,
    _DEFAULT_K_SCALE_CEILING,
    _DEFAULT_K_SCALE_FLOOR,
    _DEFAULT_M_SCALE_CEILING,
    _DEFAULT_M_SCALE_FLOOR,
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

    def _make_orchestrator_cal(self, titrant_names, titrant_concs, genotypes):
        """Return a MagicMock with the binding_tm shape used by the helper."""
        orchestrator = MagicMock()
        # Order matters; the helper reads names, concs, and genotypes by index.
        orchestrator.binding_tm.tensor_dim_names = [
            "titrant_name", "titrant_conc", "genotype"
        ]
        orchestrator.binding_tm.tensor_dim_labels = [
            list(titrant_names),
            list(titrant_concs),
            list(genotypes),
        ]
        return orchestrator

    def test_per_genotype_theta_extracted_correctly(self):
        """Each genotype must get its own theta value, not the cross-genotype average."""
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0], ["wt", "mut"])
        # wt has theta=0.8, mut has theta=0.2; output must preserve the distinction.
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "genotype": ["wt", "mut"],
            "theta_obs": [0.8, 0.2],
            "theta_std": [0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 1, 2)
        assert theta[0, 0, 0] > 0.75  # wt ≈ 0.8
        assert theta[0, 0, 1] < 0.25  # mut ≈ 0.2

    def test_inverse_variance_weighting_within_genotype(self):
        """Multiple rows for the same genotype/cell must be IVW-averaged."""
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0], ["wt"])
        # Two replicate measurements: theta=0.2 (noisy) and theta=0.8 (precise).
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "genotype": ["wt", "wt"],
            "theta_obs": [0.2, 0.8],
            "theta_std": [0.1, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 1, 1)
        # Inverse-variance weighting heavily favours the precise observation.
        assert theta[0, 0, 0] > 0.7

    def test_falls_back_to_plain_mean_when_all_stds_zero(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0], ["wt"])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0],
            "genotype": ["wt", "wt"],
            "theta_obs": [0.3, 0.7],
            "theta_std": [0.0, 0.0],  # invalid weights → fallback to mean
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta[0, 0, 0] == pytest.approx(0.5)

    def test_unobserved_genotype_cell_defaults_to_midpoint(self):
        """A genotype present in labels but absent from binding_df gets 0.5."""
        orchestrator = self._make_orchestrator_cal(["IPTG"], [1.0], ["wt", "missing"])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1.0],
            "genotype": ["wt"],
            "theta_obs": [0.9],
            "theta_std": [0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 1, 2)
        assert theta[0, 0, 0] > 0.85   # wt observed
        assert theta[0, 0, 1] == pytest.approx(0.5)  # "missing" → midpoint

    def test_unobserved_concentration_cell_defaults_to_midpoint(self):
        """A concentration present in labels but absent from binding_df gets 0.5."""
        orchestrator = self._make_orchestrator_cal(["IPTG"], [0.0, 1.0], ["wt"])
        # Only the IPTG=1.0 cell is observed.
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG"],
            "titrant_conc": [1.0],
            "genotype": ["wt"],
            "theta_obs": [0.9],
            "theta_std": [0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 2, 1)
        assert theta[0, 0, 0] == pytest.approx(0.5)  # conc=0.0 not observed
        assert theta[0, 1, 0] > 0.85                 # conc=1.0 observed

    def test_output_shape(self):
        """Output shape must be (n_titrant_name, n_titrant_conc, n_genotype)."""
        orchestrator = self._make_orchestrator_cal(
            ["IPTG", "arabinose"], [0.0, 0.01, 1.0], ["wt", "A1T", "G2P"]
        )
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG"] * 3,
            "titrant_conc": [0.0, 0.01, 1.0],
            "genotype": ["wt", "A1T", "G2P"],
            "theta_obs": [0.9, 0.5, 0.1],
            "theta_std": [0.01, 0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (2, 3, 3)

    def test_clip_to_open_interval(self):
        orchestrator = self._make_orchestrator_cal(["IPTG"], [0.0, 1.0], ["wt"])
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
            "genotype": ["wt", "wt"],
            "theta_obs": [0.0, 1.0],
            "theta_std": [0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 2, 1)
        # Zero and one would blow up the downstream logit; expect clipping.
        assert theta[0, 0, 0] > 0.0
        assert theta[0, 1, 0] < 1.0

    def test_genotype_ordering_matches_labels(self):
        """Values must be placed at the index matching the genotype label order."""
        orchestrator = self._make_orchestrator_cal(
            ["IPTG"], [1.0], ["wt", "A1T", "G2P"]
        )
        binding_df = pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG", "IPTG"],
            "titrant_conc": [1.0, 1.0, 1.0],
            # Deliberately supply rows in a different order than the labels.
            "genotype": ["G2P", "wt", "A1T"],
            "theta_obs": [0.1, 0.9, 0.5],
            "theta_std": [0.01, 0.01, 0.01],
        })
        theta = np.asarray(_compute_theta_values(orchestrator, binding_df))
        assert theta.shape == (1, 1, 3)
        # wt is index 0 in labels → theta ≈ 0.9
        assert theta[0, 0, 0] > 0.85
        # A1T is index 1 → theta ≈ 0.5
        assert 0.45 < theta[0, 0, 1] < 0.55
        # G2P is index 2 → theta ≈ 0.1
        assert theta[0, 0, 2] < 0.15


# ---------------------------------------------------------------------------
# _csv_row_name / _build_csv_updates
# ---------------------------------------------------------------------------

class TestCsvRowName:

    def test_basic_row_name(self):
        assert _csv_row_name("condition_growth", "growth_k_hyper_loc") == \
            "growth.condition_growth.growth_k_hyper_loc"


class TestBuildCsvUpdates:

    def test_normal_site_writes_loc_to_priors_and_guesses(self):
        field_mapping = {
            "condition_growth_growth_k_hyper": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "growth_k_hyper_loc",
                "scale_field": "k_hyper_loc_scale",
                "is_array": False,
            },
        }
        params = {"condition_growth_growth_k_hyper_auto_loc": np.float32(2.5)}
        prior_updates, guess_updates = _build_csv_updates(field_mapping, params)
        # MAP → loc_field in priors
        assert prior_updates[
            "growth.condition_growth.growth_k_hyper_loc"] == pytest.approx(2.5)
        # scale_field is NOT updated (no Hessian sigma)
        assert "growth.condition_growth.k_hyper_loc_scale" not in prior_updates
        # MAP → scalar guess
        assert guess_updates["condition_growth_growth_k_hyper"] == pytest.approx(2.5)

    def test_halfnormal_site_writes_map_to_scale_field(self):
        field_mapping = {
            "condition_growth_k_hyper_scale": {
                "component": "condition_growth",
                "dist_class": "HalfNormal",
                "scale_field": "growth_k_hyper_scale_loc",
                "is_array": False,
            },
        }
        params = {"condition_growth_k_hyper_scale_auto_loc": np.float32(0.42)}
        prior_updates, guess_updates = _build_csv_updates(field_mapping, params)
        assert prior_updates == {
            "growth.condition_growth.growth_k_hyper_scale_loc": pytest.approx(0.42),
        }
        assert guess_updates["condition_growth_k_hyper_scale"] == pytest.approx(0.42)

    def test_skips_sites_without_auto_loc_in_params(self):
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
        params = {"condition_growth_k_auto_loc": np.array([1.0, 2.0, 3.0])}
        prior_updates, guess_updates = _build_csv_updates(field_mapping, params)
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
            ".prefit_calibration_cli.ModelOrchestrator"
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
            ".prefit_calibration_cli.ModelOrchestrator"
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
            ".prefit_calibration_cli.ModelOrchestrator"
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
        # theta_values is now (T, C, G); _inject_calibration_priors is shape-agnostic.
        theta_values = np.array([[[0.1, 0.9], [0.3, 0.7]]])  # shape (1, 2, 2)
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
                        field_mapping=None):
        """Stub out every heavy callsite of run_prefit_calibration."""
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli.read_configuration",
            return_value=(MagicMock(), {}),
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._intersect_data",
            return_value=(pd.DataFrame(), pd.DataFrame()),
        )
        orchestrator_cal = orchestrator_cal or MagicMock()
        orchestrator_cal.init_params = {}
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._build_calibration_model",
            return_value=orchestrator_cal,
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._compute_theta_values",
            return_value=np.array([[[0.5]]]),  # shape (T=1, C=1, G=1)
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._inject_calibration_priors",
        )
        mock_ri = MagicMock()
        mock_ri._iterations_per_epoch = 1
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli.RunInference",
            return_value=mock_ri,
        )
        mock_run_map = mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._run_calibration_map",
            return_value=("svi_state", params or {}, True),
        )
        mocker.patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli._identify_field_mapping",
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
        # Seed the priors CSV with a loc-field row that the Normal-site
        # update should target, plus an unrelated row to verify we don't
        # disturb it.
        pd.DataFrame({
            "parameter": [
                "growth.condition_growth.growth_k_hyper_loc",
                "growth.activity.hyper_loc_loc",
            ],
            "value": [1.0, 0.0],
        }).to_csv(priors, index=False)
        pd.DataFrame({
            "parameter": ["condition_growth_growth_k_hyper"],
            "value": [1.0],
            "flat_index": [float("nan")],
        }).to_csv(guesses, index=False)

        params = {"condition_growth_growth_k_hyper_auto_loc": np.float32(9.0)}
        self._patch_pipeline(
            mocker,
            params=params,
            field_mapping={
                "condition_growth_growth_k_hyper": {
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
        # MAP → loc-field row; unrelated row preserved.
        assert new_priors["growth.condition_growth.growth_k_hyper_loc"] == 9.0
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

    def test_hessian_called_after_map(self, tmp_path, mocker):
        """compute_hessian_sigmas must be called exactly once after MAP."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        mock_ri, _ = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1)
        mock_ri.compute_hessian_sigmas.assert_called_once()

    def test_hessian_chunk_size_forwarded(self, tmp_path, mocker):
        """The hessian_chunk_size kwarg must be passed to compute_hessian_sigmas."""
        cfg, _, _ = self._write_yaml_and_csvs(tmp_path)
        mock_ri, _ = self._patch_pipeline(mocker)
        run_prefit_calibration(config_file=cfg, seed=1, hessian_chunk_size=32)
        call_kwargs = mock_ri.compute_hessian_sigmas.call_args.kwargs
        assert call_kwargs.get("hessian_chunk_size") == 32


# ---------------------------------------------------------------------------
# _build_hessian_scale_updates
# ---------------------------------------------------------------------------

class TestBuildHessianScaleUpdates:
    """Tests for the Hessian-with-floor scale update builder."""

    def _make_field_mapping(self, sites):
        """Build a minimal field_mapping for the given list of (site, loc_field) pairs."""
        out = {}
        for site_name, loc_field in sites:
            suffix = loc_field.replace("_loc", "")
            out[site_name] = {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": loc_field,
                "scale_field": f"{suffix}_scale",
                "is_array": True,
            }
        return out

    def _make_hessian(self, site_name, sigmas):
        return {site_name: {"map": np.zeros_like(sigmas), "sigma": np.array(sigmas)}}

    # --- floor application ---

    def test_k_sigma_above_floor_is_kept(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.01, 0.02])
        g, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        np.testing.assert_allclose(g["condition_growth_k_scales"], [0.01, 0.02])

    def test_k_sigma_below_floor_is_floored(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.0001, 0.0005])
        g, _ = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert np.all(g["condition_growth_k_scales"] == pytest.approx(0.002))

    def test_m_sigma_uses_m_floor_not_k_floor(self):
        fm = self._make_field_mapping([("condition_growth_m", "m_loc")])
        hr = self._make_hessian("condition_growth_m", [0.0003, 0.0008])
        g, _ = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        np.testing.assert_allclose(g["condition_growth_m_scales"], [0.001, 0.001])

    def test_floor_applied_elementwise(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.001, 0.005])
        g, _ = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.003, m_scale_floor=0.001,
                                              k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        np.testing.assert_allclose(g["condition_growth_k_scales"], [0.003, 0.005])

    # --- guess output ---

    def test_k_scales_in_guess_updates(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.005, 0.007])
        g, _ = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert "condition_growth_k_scales" in g

    def test_m_scales_in_guess_updates(self):
        fm = self._make_field_mapping([("condition_growth_m", "m_loc")])
        hr = self._make_hessian("condition_growth_m", [0.003, 0.004])
        g, _ = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert "condition_growth_m_scales" in g

    # --- prior output ---

    def test_k_prior_update_uses_max_across_conditions(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.003, 0.007])
        _, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert p["growth.condition_growth.k_scale"] == pytest.approx(0.007)

    def test_k_prior_update_respects_floor(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.0001, 0.0002])
        _, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert p["growth.condition_growth.k_scale"] == pytest.approx(0.002)

    def test_m_prior_goes_to_m_scale_plus_not_m_scale(self):
        fm = self._make_field_mapping([("condition_growth_m", "m_loc")])
        hr = self._make_hessian("condition_growth_m", [0.004, 0.006])
        _, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert "growth.condition_growth.m_scale_plus" in p
        assert "growth.condition_growth.m_scale" not in p

    def test_m_scale_plus_is_max_floored(self):
        fm = self._make_field_mapping([("condition_growth_m", "m_loc")])
        hr = self._make_hessian("condition_growth_m", [0.009, 0.002])
        _, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert p["growth.condition_growth.m_scale_plus"] == pytest.approx(0.009)

    # --- edge cases ---

    def test_site_missing_from_hessian_skipped(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        g, p = _build_hessian_scale_updates(fm, {}, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert g == {} and p == {}

    def test_non_array_site_skipped(self):
        fm = {
            "condition_growth_k_hyper": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "k_hyper_loc",
                "scale_field": "k_hyper_scale",
                "is_array": False,
            }
        }
        hr = {"condition_growth_k_hyper": {"map": np.array(0.02), "sigma": np.array(0.0001)}}
        g, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert g == {} and p == {}

    def test_unknown_loc_field_skipped(self):
        fm = {
            "condition_growth_tau": {
                "component": "condition_growth",
                "dist_class": "Normal",
                "loc_field": "tau_loc",
                "scale_field": "tau_scale",
                "is_array": True,
            }
        }
        hr = {"condition_growth_tau": {"map": np.zeros(2), "sigma": np.ones(2) * 0.1}}
        g, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert g == {} and p == {}

    def test_joint_k_and_m_both_updated(self):
        fm = self._make_field_mapping([
            ("condition_growth_k", "k_loc"),
            ("condition_growth_m", "m_loc"),
        ])
        hr = {
            "condition_growth_k": {"map": np.zeros(2), "sigma": np.array([0.005, 0.008])},
            "condition_growth_m": {"map": np.zeros(2), "sigma": np.array([0.004, 0.002])},
        }
        g, p = _build_hessian_scale_updates(fm, hr, k_scale_floor=0.002, m_scale_floor=0.001, k_scale_ceiling=0.1, m_scale_ceiling=0.01)
        assert "condition_growth_k_scales" in g
        assert "condition_growth_m_scales" in g
        assert "growth.condition_growth.k_scale" in p
        assert "growth.condition_growth.m_scale_plus" in p

    # --- ceiling application ---

    def test_k_sigma_above_ceiling_is_capped(self):
        """A degenerate Hessian giving sigma >> ceiling must not loosen the prior."""
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [23.5, 31.6])
        g, p = _build_hessian_scale_updates(
            fm, hr, k_scale_floor=0.002, m_scale_floor=0.001,
            k_scale_ceiling=0.1, m_scale_ceiling=0.01,
        )
        assert np.all(g["condition_growth_k_scales"] == pytest.approx(0.1))
        assert p["growth.condition_growth.k_scale"] == pytest.approx(0.1)

    def test_m_sigma_above_ceiling_is_capped(self):
        fm = self._make_field_mapping([("condition_growth_m", "m_loc")])
        hr = self._make_hessian("condition_growth_m", [15.0, 22.0])
        g, p = _build_hessian_scale_updates(
            fm, hr, k_scale_floor=0.002, m_scale_floor=0.001,
            k_scale_ceiling=0.1, m_scale_ceiling=0.01,
        )
        assert np.all(g["condition_growth_m_scales"] == pytest.approx(0.01))
        assert p["growth.condition_growth.m_scale_plus"] == pytest.approx(0.01)

    def test_sigma_between_floor_and_ceiling_is_unchanged(self):
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.004, 0.006])
        g, p = _build_hessian_scale_updates(
            fm, hr, k_scale_floor=0.002, m_scale_floor=0.001,
            k_scale_ceiling=0.1, m_scale_ceiling=0.01,
        )
        np.testing.assert_allclose(g["condition_growth_k_scales"], [0.004, 0.006])
        assert p["growth.condition_growth.k_scale"] == pytest.approx(0.006)

    def test_ceiling_below_floor_raises_or_clips_to_ceiling(self):
        """When ceiling < floor (degenerate config), clip still works (clip clamps to ceiling)."""
        fm = self._make_field_mapping([("condition_growth_k", "k_loc")])
        hr = self._make_hessian("condition_growth_k", [0.05])
        g, _ = _build_hessian_scale_updates(
            fm, hr, k_scale_floor=0.05, m_scale_floor=0.001,
            k_scale_ceiling=0.05, m_scale_ceiling=0.01,
        )
        # floor == ceiling == 0.05 → result must be 0.05
        assert g["condition_growth_k_scales"][0] == pytest.approx(0.05)

    # --- default constant sanity ---

    def test_default_floors_are_sensible_constants(self):
        """Floor constants must be positive and k_floor >= m_floor."""
        assert _DEFAULT_K_SCALE_FLOOR > 0
        assert _DEFAULT_M_SCALE_FLOOR > 0
        assert _DEFAULT_K_SCALE_FLOOR >= _DEFAULT_M_SCALE_FLOOR

    def test_default_ceilings_above_floors(self):
        """Ceilings must be strictly above floors so the valid range is non-empty."""
        assert _DEFAULT_K_SCALE_CEILING > _DEFAULT_K_SCALE_FLOOR
        assert _DEFAULT_M_SCALE_CEILING > _DEFAULT_M_SCALE_FLOOR

    def test_default_ceilings_match_linear_defaults(self):
        """Ceilings should match the linear component's default prior scales."""
        from tfscreen.tfmodel.generative.components.growth.linear import get_hyperparameters
        h = get_hyperparameters()
        assert _DEFAULT_K_SCALE_CEILING == pytest.approx(h["k_scale"])
        assert _DEFAULT_M_SCALE_CEILING == pytest.approx(h["m_scale_plus"])


# ---------------------------------------------------------------------------
# main() CLI plumbing
# ---------------------------------------------------------------------------

class TestPrefitMainCLI:

    def test_main_invokes_run_prefit_with_required_args(self, tmp_path):
        """`main()` should drive the wrapper through generalized_main."""
        argv = ["cal.yaml", "--seed", "13"]
        with patch(
            "tfscreen.tfmodel.scripts"
            ".prefit_calibration_cli.run_prefit_calibration",
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
            ".prefit_calibration_cli.run_prefit_calibration",
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
            ".prefit_calibration_cli.run_prefit_calibration",
            autospec=True,
        ) as mock_run, \
             patch("sys.argv", ["tfs-prefit-calibration"] + argv):
            main()
        assert mock_run.call_args.kwargs["checkpoint_file"] == "/tmp/ck.pkl"
