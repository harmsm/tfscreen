import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
import warnings
from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.prediction import (
    copy_orchestrator,
    _convert_map_params,
    _align_site_to_tm_dims,
    _build_population_references,
    predict,
)


def _feasible_draw(site):
    """
    One-draw fake posterior value for a traced sample site: zeros for
    real-support sites, a valid in-support value (e.g. ones) for constrained
    ones. Plain zeros put positive-support sites (noise sigmas, scales) on
    the boundary, which numpyro >= 0.20 rejects as an invalid scale.
    """
    value = site["fn"].support.feasible_like(site["value"])
    return np.asarray(value)[None, ...]

@pytest.fixture
def dummy_orchestrator():
    """Create a dummy ModelOrchestrator instance for testing."""
    growth_df = pd.DataFrame({
        "library": ["lib"] * 4,
        "genotype": ["wt", "wt", "M42V", "M42V"],
        "titrant_name": ["tit1", "tit1", "tit1", "tit1"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "condition_pre": ["pre-1", "pre-1", "pre-1", "pre-1"],
        "condition_sel": ["sel+1", "sel+1", "sel+1", "sel+1"],
        "t_pre": [10.0, 10.0, 10.0, 10.0],
        "t_sel": [0.0, 20.0, 0.0, 20.0],
        "ln_cfu": [0.0, 5.0, 0.0, 3.0],
        "ln_cfu_std": [0.1, 0.1, 0.1, 0.1],
        "replicate": [1, 1, 1, 1]
    })

    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42V"],
        "titrant_name": ["tit1", "tit1"],
        "titrant_conc": [0.5, 0.5],
        "theta_obs": [0.5, 0.2],
        "theta_std": [0.01, 0.01]
    })

    # transformation='mixture' (rather than the 'single' default) is load-
    # bearing for TestPredictPopulationReferenceThreading below, which relies
    # on NEEDS_POPULATION=True.
    return ModelOrchestrator(
        growth_df, binding_df,
        transformation="mixture", transformation_lambda=(0.36, 0.05),
    )

def test_copy_orchestrator_defaults(dummy_orchestrator):
    """Test copy_orchestrator with all None inputs."""
    new_orchestrator = copy_orchestrator(dummy_orchestrator)
    # Categorical combinations (1 rep * 1 cp * 1 cs * 1 tn * 2 geno) = 2
    # Quantitative (1 t_pre * 2 t_sel * 2 conc) = 4
    # Total = 2 * 4 = 8
    assert len(new_orchestrator.growth_df) == 8
    assert new_orchestrator.settings == dummy_orchestrator.settings

def test_copy_orchestrator_expansion(dummy_orchestrator):
    """Test expanding quantitative inputs."""
    new_orchestrator = copy_orchestrator(
        dummy_orchestrator,
        t_sel=[0.0, 10.0, 20.0],
        titrant_conc=[0.0, 0.5, 1.0]
    )
    # 2 (genotypes) * 3 (t_sel) * 3 (conc) = 18
    assert len(new_orchestrator.growth_df) == 18
    assert set(new_orchestrator.growth_df["t_sel"]) == {0.0, 10.0, 20.0}
    assert set(new_orchestrator.growth_df["titrant_conc"]) == {0.0, 0.5, 1.0}

def test_copy_orchestrator_list_inputs(dummy_orchestrator):
    """Test passing single values as non-lists."""
    new_orchestrator = copy_orchestrator(
        dummy_orchestrator,
        t_sel=30.0,
        titrant_conc=2.0
    )
    assert 30.0 in new_orchestrator.growth_df["t_sel"].values
    assert 2.0 in new_orchestrator.growth_df["titrant_conc"].values

def test_copy_orchestrator_quantitative_errors(dummy_orchestrator):
    """Test validation of quantitative inputs."""
    with pytest.raises(ValueError, match="t_pre must be >= 0"):
        copy_orchestrator(dummy_orchestrator, t_pre=-1.0)
    
    with pytest.raises(ValueError, match="t_sel must be >= 0"):
        copy_orchestrator(dummy_orchestrator, t_sel=[-1.0, 1.0])
        
    with pytest.raises(ValueError, match="titrant_conc must be >= 0"):
        copy_orchestrator(dummy_orchestrator, titrant_conc=[1.0, -0.5])

def test_copy_orchestrator_t_pre_expansion(dummy_orchestrator):
    """Test expanding t_pre."""
    new_orchestrator = copy_orchestrator(
        dummy_orchestrator,
        t_pre=[10.0, 20.0]
    )
    # 2 (genotypes) * 2 (t_pre) * 2 (t_sel) * 2 (conc) = 16
    assert len(new_orchestrator.growth_df) == 16
    assert set(new_orchestrator.growth_df["t_pre"]) == {10.0, 20.0}

def test_copy_orchestrator_no_subsetting_binding(dummy_orchestrator):
    """Test that binding_df is NOT subsetted during copy."""
    # Original binding_df has wt and M42V
    assert len(dummy_orchestrator.binding_df) == 2

    new_orchestrator = copy_orchestrator(dummy_orchestrator)

    # New binding_df should still have all genotypes
    assert len(new_orchestrator.binding_df) == 2


def test_copy_orchestrator_genotypes_subset(dummy_orchestrator):
    """Requesting a subset of genotypes restricts growth_df rows."""
    new_orch = copy_orchestrator(dummy_orchestrator, genotypes=["wt"])
    assert set(new_orch.growth_df["genotype"]) == {"wt"}


def test_copy_orchestrator_genotypes_subset_tm_labels(dummy_orchestrator):
    """TM genotype labels match the requested subset."""
    new_orch = copy_orchestrator(dummy_orchestrator, genotypes=["wt"])
    labels = new_orch.growth_tm.tensor_dim_labels[-1].tolist()
    assert labels == ["wt"]


def test_copy_orchestrator_genotypes_subset_row_count(dummy_orchestrator):
    """Row count matches the single-genotype cross-product."""
    new_orch = copy_orchestrator(dummy_orchestrator, genotypes=["wt"])
    # 1 genotype * 1 t_pre * 2 t_sel * 2 titrant_conc = 4
    assert len(new_orch.growth_df) == 4


def test_copy_orchestrator_genotypes_all_explicit(dummy_orchestrator):
    """Passing all genotypes explicitly is equivalent to genotypes=None."""
    new_all = copy_orchestrator(dummy_orchestrator)
    new_explicit = copy_orchestrator(dummy_orchestrator, genotypes=["wt", "M42V"])
    assert set(new_explicit.growth_df["genotype"]) == set(new_all.growth_df["genotype"])


def test_copy_orchestrator_genotypes_unknown_raises(dummy_orchestrator):
    """Requesting an unknown genotype raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        copy_orchestrator(dummy_orchestrator, genotypes=["no_such_geno"])


def test_copy_orchestrator_genotypes_partial_unknown_raises(dummy_orchestrator):
    """A mix of valid and unknown genotypes still raises."""
    with pytest.raises(ValueError, match="not found"):
        copy_orchestrator(dummy_orchestrator, genotypes=["wt", "no_such_geno"])


def test_copy_orchestrator_genotypes_tm_labels_differ_from_original(dummy_orchestrator):
    """Subset TM labels must differ from the original so slicing fires in predict()."""
    new_orch = copy_orchestrator(dummy_orchestrator, genotypes=["wt"])
    orig_labels = dummy_orchestrator.growth_tm.tensor_dim_labels[-1].tolist()
    new_labels = new_orch.growth_tm.tensor_dim_labels[-1].tolist()
    assert orig_labels != new_labels


@pytest.fixture
def spiked_orchestrator():
    """ModelOrchestrator with one spiked genotype (A1G) and two library genotypes."""
    growth_df = pd.DataFrame({
        "library": ["lib"] * 6,
        "genotype": ["wt", "wt", "M42V", "M42V", "A1G", "A1G"],
        "titrant_name": ["tit1"] * 6,
        "titrant_conc": [0.0, 1.0] * 3,
        "condition_pre": ["pre-1"] * 6,
        "condition_sel": ["sel+1"] * 6,
        "t_pre": [10.0] * 6,
        "t_sel": [0.0, 20.0] * 3,
        "ln_cfu": [0.0, 5.0, 0.0, 3.0, 0.0, 4.0],
        "ln_cfu_std": [0.1] * 6,
        "replicate": [1] * 6,
    })
    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42V", "A1G"],
        "titrant_name": ["tit1"] * 3,
        "titrant_conc": [0.5] * 3,
        "theta_obs": [0.5, 0.2, 0.9],
        "theta_std": [0.01] * 3,
    })
    return ModelOrchestrator(growth_df, binding_df, spiked_genotypes=["A1G"])


def test_copy_orchestrator_spiked_outside_subset_dropped(spiked_orchestrator):
    """Spiked genotypes not in the requested subset are removed from settings."""
    new_orch = copy_orchestrator(spiked_orchestrator, genotypes=["wt", "M42V"])
    assert new_orch.settings.get("spiked_genotypes") == []


def test_copy_orchestrator_spiked_inside_subset_retained(spiked_orchestrator):
    """Spiked genotypes that are in the requested subset are kept in settings."""
    new_orch = copy_orchestrator(spiked_orchestrator, genotypes=["wt", "A1G"])
    assert "A1G" in (new_orch.settings.get("spiked_genotypes") or [])


def test_copy_orchestrator_spiked_partial_overlap():
    """Only spiked genotypes present in the subset survive."""
    # Build a fresh orchestrator with two spiked genotypes: A1G and S5T
    growth_df = pd.DataFrame({
        "library": ["lib"] * 8,
        "genotype": ["wt", "wt", "M42V", "M42V", "A1G", "A1G", "S5T", "S5T"],
        "titrant_name": ["tit1"] * 8,
        "titrant_conc": [0.0, 1.0] * 4,
        "condition_pre": ["pre-1"] * 8,
        "condition_sel": ["sel+1"] * 8,
        "t_pre": [10.0] * 8,
        "t_sel": [0.0, 20.0] * 4,
        "ln_cfu": [0.0, 5.0, 0.0, 3.0, 0.0, 4.0, 0.0, 4.5],
        "ln_cfu_std": [0.1] * 8,
        "replicate": [1] * 8,
    })
    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42V", "A1G"],
        "titrant_name": ["tit1"] * 3,
        "titrant_conc": [0.5] * 3,
        "theta_obs": [0.5, 0.2, 0.9],
        "theta_std": [0.01] * 3,
    })
    orch2 = ModelOrchestrator(growth_df, binding_df, spiked_genotypes=["A1G", "S5T"])
    new_orch = copy_orchestrator(orch2, genotypes=["wt", "A1G"])
    spiked_kept = new_orch.settings.get("spiked_genotypes") or []
    assert "A1G" in spiked_kept
    assert "S5T" not in spiked_kept


def test_copy_orchestrator_no_spiked_in_settings_unaffected(dummy_orchestrator):
    """When spiked_genotypes is absent from settings, subsetting doesn't break."""
    assert not dummy_orchestrator.settings.get("spiked_genotypes")
    new_orch = copy_orchestrator(dummy_orchestrator, genotypes=["wt"])
    assert set(new_orch.growth_df["genotype"]) == {"wt"}


def test_copy_orchestrator_spiked_no_subset_unchanged(spiked_orchestrator):
    """Without genotype subsetting, spiked_genotypes in settings are unchanged."""
    new_orch = copy_orchestrator(spiked_orchestrator)
    assert new_orch.settings.get("spiked_genotypes") == \
        spiked_orchestrator.settings.get("spiked_genotypes")


# ---------------------------------------------------------------------------
# _convert_map_params
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_trace():
    """
    Minimal model trace for testing _convert_map_params.

    Includes:
    - an unconstrained site (Normal → real support)
    - a positive-constrained site (HalfNormal → positive support)
    - an observed site (must be passed through with raw value)
    - a deterministic site (must be ignored — no _auto_loc key)
    """
    return {
        "real_site": {
            "type": "sample",
            "is_observed": False,
            "fn": dist.Normal(0.0, 1.0),
        },
        "positive_site": {
            "type": "sample",
            "is_observed": False,
            "fn": dist.HalfNormal(1.0),
        },
        "observed_site": {
            "type": "sample",
            "is_observed": True,
            "fn": dist.Normal(0.0, 1.0),
        },
        "det_site": {
            "type": "deterministic",
        },
    }


class TestConvertMapParams:
    """Unit tests for _convert_map_params."""

    def test_keys_are_renamed(self, fake_trace):
        """_auto_loc suffix is stripped from output keys."""
        map_params = {"real_site_auto_loc": np.array(0.5)}
        out = _convert_map_params(map_params, fake_trace)
        assert "real_site" in out
        assert "real_site_auto_loc" not in out

    def test_leading_sample_dim_added(self, fake_trace):
        """Each output value has a leading sample dimension of size 1."""
        map_params = {
            "real_site_auto_loc": np.array(0.5),
            "positive_site_auto_loc": np.array(-1.0),
        }
        out = _convert_map_params(map_params, fake_trace)
        for site_name, val in out.items():
            assert val.shape[0] == 1, (
                f"{site_name}: expected leading dim 1, got shape {val.shape}"
            )

    def test_identity_for_unconstrained_site(self, fake_trace):
        """Normal (real support) site value is passed through unchanged."""
        raw_val = np.array(-2.3)
        map_params = {"real_site_auto_loc": raw_val}
        out = _convert_map_params(map_params, fake_trace)
        np.testing.assert_allclose(
            np.asarray(out["real_site"][0]),
            raw_val,
            rtol=1e-5,
        )

    def test_positive_site_passes_through_unchanged(self, fake_trace):
        """HalfNormal (positive support) site value is passed through as-is.

        RunInference.write_params() calls svi.get_params(), which applies
        constrain_fn before saving, so stored _auto_loc values are already in
        constrained (positive) space.  _convert_map_params must NOT apply a
        second bijection.

        Regression: a previous version incorrectly called biject_to(positive)
        (== ExpTransform) on the stored value, mapping e.g. 2.09 → exp(2.09)
        = 8.09 instead of leaving it as 2.09.  This corrupted ln_cfu0 for
        library genotypes (which use a sampled HalfNormal scale) while leaving
        wt predictions unaffected (wt uses a fixed prior scale).
        """
        # Use the real-world value from prefit9 that exposed the bug.
        raw_val = np.array(2.09)   # already constrained (positive)
        map_params = {"positive_site_auto_loc": raw_val}
        out = _convert_map_params(map_params, fake_trace)
        constrained = float(out["positive_site"][0])

        # Value must pass through unchanged — no bijection should be applied.
        np.testing.assert_allclose(constrained, float(raw_val), rtol=1e-5,
                                   err_msg="positive site value should not be re-transformed")

        # Explicit guard against the old bug: exp(2.09) ≈ 8.09, not 2.09.
        assert abs(constrained - float(np.exp(raw_val))) > 1.0, (
            f"Output ({constrained:.4f}) looks like exp(input) "
            f"({float(np.exp(raw_val)):.4f}) — biject_to was probably re-introduced"
        )

    def test_observed_site_ignored(self, fake_trace):
        """Keys whose site is marked is_observed are not converted."""
        map_params = {
            "real_site_auto_loc": np.array(0.0),
            "observed_site_auto_loc": np.array(1.0),
        }
        out = _convert_map_params(map_params, fake_trace)
        # Both sites should appear with their raw values and a leading sample dim.
        assert "real_site" in out
        assert "observed_site" in out
        assert out["observed_site"].shape[0] == 1

    def test_non_auto_loc_keys_skipped(self, fake_trace):
        """Keys without _auto_loc suffix are not included in the output."""
        map_params = {
            "real_site_auto_loc": np.array(0.0),
            "some_other_key": np.array(99.0),
        }
        out = _convert_map_params(map_params, fake_trace)
        assert "some_other_key" not in out
        assert "real_site" in out

    def test_array_site_shape_preserved(self, fake_trace):
        """Multi-element sites get shape (1, N) not (N,)."""
        map_params = {"real_site_auto_loc": np.array([1.0, 2.0, 3.0])}
        out = _convert_map_params(map_params, fake_trace)
        assert out["real_site"].shape == (1, 3)

    def test_empty_input_returns_empty(self, fake_trace):
        out = _convert_map_params({}, fake_trace)
        assert out == {}

    def test_unknown_site_passes_value_through(self, fake_trace):
        """A key whose site is absent from the trace is still converted (no bijection)."""
        map_params = {"mystery_site_auto_loc": np.array(5.0)}
        out = _convert_map_params(map_params, fake_trace)
        assert "mystery_site" in out
        np.testing.assert_allclose(float(out["mystery_site"][0]), 5.0)


# ---------------------------------------------------------------------------
# _align_site_to_tm_dims
# ---------------------------------------------------------------------------

# TM dim order: rep=2, t=3, cp=2, cs=1, tn=1, tc=8, geno=473
_TM_SIZES = (2, 3, 2, 1, 1, 8, 473)


class TestAlignSiteToTmDims:
    def test_full_rank_identity(self):
        """A 7-D site matching TM exactly maps to [0,1,2,3,4,5,6]."""
        spatial = (2, 3, 2, 1, 1, 8, 473)
        assert _align_site_to_tm_dims(spatial, _TM_SIZES) == [0, 1, 2, 3, 4, 5, 6]

    def test_tail_aligned_1d(self):
        """A 1-D site of size 473 should map to the last TM dim (geno)."""
        assert _align_site_to_tm_dims((473,), _TM_SIZES) == [6]

    def test_tail_aligned_2d(self):
        """(8, 473) should map to TM dims [5, 6] (titrant_conc, geno)."""
        assert _align_site_to_tm_dims((8, 473), _TM_SIZES) == [5, 6]

    def test_broadcast_left_padding(self):
        """theta_growth_pred shape (1,1,1,1,1,8,473) — non-broadcast tail match."""
        spatial = (1, 1, 1, 1, 1, 8, 473)
        result = _align_site_to_tm_dims(spatial, _TM_SIZES)
        # Size-1 dims use right-to-left (modulo-safe); non-broadcast dims are [5, 6].
        assert result[5] == 5
        assert result[6] == 6

    def test_ln_cfu0_shape(self):
        """ln_cfu0 shape (2, 2, 473) should map to TM dims [0, 2, 6] = rep, cp, geno."""
        assert _align_site_to_tm_dims((2, 2, 473), _TM_SIZES) == [0, 2, 6]

    def test_ln_cfu0_tube_offset_shape(self):
        """(n_rep, n_cp) = (2, 2) maps to TM dims [0, 2]."""
        assert _align_site_to_tm_dims((2, 2), _TM_SIZES) == [0, 2]

    def test_single_geno_dim(self):
        """1-D (473,) still maps to geno regardless of other TM sizes."""
        tm = (5, 3, 7, 473)
        assert _align_site_to_tm_dims((473,), tm) == [3]

    def test_fallback_to_right_to_left_on_no_match(self):
        """If subsequence matching fails, right-to-left is returned."""
        # spatial size 999 is not in any TM dim
        result = _align_site_to_tm_dims((999,), _TM_SIZES)
        assert result == [6]  # right-to-left: len=7, n_sp=1 → 7-1+0=6


# ---------------------------------------------------------------------------
# predict() — priors propagation
# ---------------------------------------------------------------------------

class TestPredictPriorsPropagate:
    """
    Verify that predict() uses orchestrator.priors (the original calibration
    priors, which may contain pinned hyperpriors) when calling Predictive,
    not new_orchestrator.priors (the copy's default/reset priors).

    Regression test for the bug where copy_orchestrator() always creates
    priors with default values (e.g. theta_values=[[0.5]]) which would be
    used instead of the fitted calibration priors.
    """

    def test_predict_passes_original_priors_to_predictive(self, dummy_orchestrator, mocker):
        """
        predict() must forward orchestrator.priors (not copy's priors) to
        the Predictive call.  We patch Predictive to capture what priors
        argument was passed and assert it is the ORIGINAL orchestrator's
        priors object.
        """
        original_priors = dummy_orchestrator.priors
        # Sanity-check: copy_orchestrator creates an orchestrator whose priors
        # differ from the original (defaults are reset on copy).
        copy_orch = copy_orchestrator(dummy_orchestrator)
        # Priors are not the same object (copy builds fresh defaults).
        assert copy_orch.priors is not original_priors

        # Capture the priors argument that reach Predictive.__call__.
        captured = {}

        real_predictive_class = __import__(
            "numpyro.infer", fromlist=["Predictive"]
        ).Predictive

        class CapturingPredictive:
            """Thin wrapper that records the priors kwarg then delegates."""
            def __init__(self, *args, **kwargs):
                self._inner = real_predictive_class(*args, **kwargs)

            def __call__(self, rng_key, **kwargs):
                captured["priors"] = kwargs.get("priors")
                return self._inner(rng_key, **kwargs)

        mocker.patch(
            "tfscreen.tfmodel.analysis.prediction.Predictive",
            CapturingPredictive,
        )

        # Build a trivial 1-sample posterior dict (MAP-style, no _auto_loc).
        # Just need the latent sites that exist in the dummy model trace.
        from numpyro.handlers import seed, trace as nptrace
        seeded = seed(dummy_orchestrator.jax_model, rng_seed=0)
        model_tr = nptrace(seeded).get_trace(
            data=dummy_orchestrator.data,
            priors=dummy_orchestrator.priors,
        )
        fake_posteriors = {
            name: _feasible_draw(site)
            for name, site in model_tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        # The mixture transformation also reads library-wide deterministic
        # sites from the posterior.
        for site in ("theta_growth_pred", "dk_geno", "activity"):
            fake_posteriors[site] = np.asarray(
                model_tr[site]["value"])[np.newaxis, ...]

        predict(
            dummy_orchestrator,
            fake_posteriors,
            predict_sites=["growth_pred"],
            num_samples=None,
        )

        assert "priors" in captured, "CapturingPredictive was never called"
        assert captured["priors"] is original_priors, (
            "predict() passed the copy's priors to Predictive instead of "
            "the original orchestrator's priors.  This regression means "
            "pinned hyperpriors (e.g. activity_hyper_loc/scale in the "
            "calibration model) will be randomly re-sampled from the "
            "default prior, producing wildly incorrect predictions."
        )


# ---------------------------------------------------------------------------
# _build_population_references
# ---------------------------------------------------------------------------

def _fake_posteriors(orchestrator, num_draws=1):
    """
    Fake posterior with ``num_draws`` identical draws: every latent sample
    site and the theta_growth_pred/dk_geno/activity deterministic sites, all
    from one full-batch prior trace (library order), so they are mutually
    consistent.
    """
    from numpyro.handlers import seed, trace as nptrace

    data = orchestrator.get_batch(
        orchestrator.data, jnp.arange(orchestrator.data.num_genotype))
    model_tr = nptrace(seed(orchestrator.jax_model, rng_seed=0)).get_trace(
        data=data, priors=orchestrator.priors)
    posteriors = {
        name: np.repeat(np.asarray(site["value"])[np.newaxis, ...],
                        num_draws, axis=0)
        for name, site in model_tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }
    for site in ("theta_growth_pred", "dk_geno", "activity"):
        posteriors[site] = np.repeat(
            np.asarray(model_tr[site]["value"])[np.newaxis, ...],
            num_draws, axis=0)
    return posteriors


def _pred_growth(orchestrator, **kwargs):
    new = copy_orchestrator(orchestrator, **kwargs)
    return new.get_batch(new.data,
                         jnp.arange(new.data.num_genotype)).growth


class TestBuildPopulationReferences:

    def test_shapes_cover_the_whole_library(self, dummy_orchestrator):
        refs = _build_population_references(
            dummy_orchestrator, _fake_posteriors(dummy_orchestrator),
            _pred_growth(dummy_orchestrator, genotypes=["wt"]))
        num_genotype = dummy_orchestrator.data.growth.num_genotype
        for field in ("external_theta_population", "external_dk_population",
                      "external_activity_population"):
            assert refs[field].shape[-1] == num_genotype

    def test_theta_at_fit_grid_matches_stored_theta(self, dummy_orchestrator):
        """On the fit's own grid, theta rebuilt from the parameters equals
        the stored theta_growth_pred."""
        posteriors = _fake_posteriors(dummy_orchestrator)
        refs = _build_population_references(
            dummy_orchestrator, posteriors, _pred_growth(dummy_orchestrator))
        np.testing.assert_allclose(
            np.asarray(refs["external_theta_population"]),
            posteriors["theta_growth_pred"][0], rtol=1e-5)

    def test_theta_follows_the_prediction_grid(self, dummy_orchestrator):
        """Predicting at new concentrations needs co-resident theta at those
        concentrations, not the fit's."""
        pred_growth = _pred_growth(dummy_orchestrator,
                                   titrant_conc=[0.0, 0.25, 0.5, 2.0])
        refs = _build_population_references(
            dummy_orchestrator, _fake_posteriors(dummy_orchestrator),
            pred_growth)
        assert refs["external_theta_population"].shape[-2] == 4

    def test_median_over_first_max_samples(self, dummy_orchestrator):
        posteriors = _fake_posteriors(dummy_orchestrator, num_draws=10)
        rng = np.random.default_rng(0)
        posteriors["dk_geno"] = rng.normal(size=posteriors["dk_geno"].shape)
        refs = _build_population_references(
            dummy_orchestrator, posteriors, _pred_growth(dummy_orchestrator),
            max_samples=4)
        np.testing.assert_allclose(
            np.asarray(refs["external_dk_population"]),
            np.median(posteriors["dk_geno"][:4], axis=0), rtol=1e-6)

    @pytest.mark.parametrize("site", ["dk_geno", "activity"])
    def test_missing_site_raises(self, dummy_orchestrator, site):
        """A raw MAP checkpoint has no deterministic sites: refuse, pointing
        to tfs-sample-posterior, rather than predict without a background."""
        posteriors = _fake_posteriors(dummy_orchestrator)
        posteriors.pop(site)
        with pytest.raises(ValueError, match="tfs-sample-posterior"):
            _build_population_references(dummy_orchestrator, posteriors,
                                         _pred_growth(dummy_orchestrator))

    def test_predict_refuses_raw_map_checkpoint(self, dummy_orchestrator):
        posteriors = _fake_posteriors(dummy_orchestrator)
        posteriors.pop("dk_geno")
        with pytest.raises(ValueError, match="tfs-sample-posterior"):
            predict(dummy_orchestrator, posteriors,
                    predict_sites=["growth_pred"], num_samples=None)

    def test_wrong_genotype_count_raises(self, dummy_orchestrator):
        posteriors = _fake_posteriors(dummy_orchestrator)
        posteriors["dk_geno"] = posteriors["dk_geno"][..., :1]
        with pytest.raises(ValueError, match="whole library"):
            _build_population_references(dummy_orchestrator, posteriors,
                                         _pred_growth(dummy_orchestrator))


# ---------------------------------------------------------------------------
# predict() threading of the population references through genotype subsets
# ---------------------------------------------------------------------------

class TestPredictPopulationReferenceThreading:
    """
    predict() must thread library-wide theta/dk_geno/activity references and
    the fitted model's co-resident sets into pred_data whenever the
    transformation needs them (the mixture), and a genotype-subset request
    must still get references sized to the whole library and co-resident
    indices into that library, not the requested subset.
    """

    def _capture_pred_data(self, mocker):
        """Patch Predictive to record the `data` kwarg passed to it, without
        changing its behavior."""
        captured = {}
        real_predictive_class = __import__(
            "numpyro.infer", fromlist=["Predictive"]
        ).Predictive

        class CapturingPredictive:
            def __init__(self, *args, **kwargs):
                self._inner = real_predictive_class(*args, **kwargs)

            def __call__(self, rng_key, **kwargs):
                captured["data"] = kwargs.get("data")
                return self._inner(rng_key, **kwargs)

        mocker.patch(
            "tfscreen.tfmodel.analysis.prediction.Predictive",
            CapturingPredictive,
        )
        return captured

    def _fake_posteriors(self, orchestrator):
        return _fake_posteriors(orchestrator)

    def test_genotype_subset_gets_library_references_and_sets(
            self, dummy_orchestrator, mocker):
        fake_posteriors = self._fake_posteriors(dummy_orchestrator)
        captured = self._capture_pred_data(mocker)

        predict(dummy_orchestrator, fake_posteriors,
                predict_sites=["growth_pred"], num_samples=None,
                genotypes=["wt"])

        growth = captured["data"].growth
        num_genotype = dummy_orchestrator.data.growth.num_genotype
        assert num_genotype > 1
        for field in ("external_theta_population", "external_dk_population",
                      "external_activity_population"):
            assert getattr(growth, field).shape[-1] == num_genotype

        labels = list(dummy_orchestrator.growth_tm.tensor_dim_labels[-1])
        wt_row = labels.index("wt")
        np.testing.assert_array_equal(
            np.asarray(growth.coresident_idx),
            np.asarray(dummy_orchestrator.data.growth.coresident_idx)[[wt_row]])

    def test_subset_prediction_matches_full_prediction(self, dummy_orchestrator):
        """A genotype's prediction must not depend on which other genotypes
        were requested."""
        fake_posteriors = self._fake_posteriors(dummy_orchestrator)
        full = predict(dummy_orchestrator, fake_posteriors,
                       predict_sites=["growth_pred"], num_samples=None)
        subset = predict(dummy_orchestrator, fake_posteriors,
                         predict_sites=["growth_pred"], num_samples=None,
                         genotypes=["wt"])

        keys = ["genotype", "titrant_conc", "t_sel"]
        full_wt = (full[full["genotype"] == "wt"]
                   .sort_values(keys).reset_index(drop=True))
        subset = subset.sort_values(keys).reset_index(drop=True)
        q_cols = [c for c in subset.columns if c.startswith("q")]
        np.testing.assert_allclose(subset[q_cols].to_numpy(dtype=float),
                                   full_wt[q_cols].to_numpy(dtype=float),
                                   rtol=1e-5)

    def test_no_population_reference_when_transformation_does_not_need_one(
            self, mocker):
        """With transformation='single' (NEEDS_POPULATION=False), predict()
        must not build or thread population references at all -- they stay
        None, and nothing is required of the posterior dict."""
        growth_df = pd.DataFrame({
            "library": ["lib"] * 4,
            "genotype": ["wt", "wt", "M42V", "M42V"],
            "titrant_name": ["tit1", "tit1", "tit1", "tit1"],
            "titrant_conc": [0.0, 1.0, 0.0, 1.0],
            "condition_pre": ["pre-1", "pre-1", "pre-1", "pre-1"],
            "condition_sel": ["sel+1", "sel+1", "sel+1", "sel+1"],
            "t_pre": [10.0, 10.0, 10.0, 10.0],
            "t_sel": [0.0, 20.0, 0.0, 20.0],
            "ln_cfu": [0.0, 5.0, 0.0, 3.0],
            "ln_cfu_std": [0.1, 0.1, 0.1, 0.1],
            "replicate": [1, 1, 1, 1],
        })
        binding_df = pd.DataFrame({
            "genotype": ["wt", "M42V"],
            "titrant_name": ["tit1", "tit1"],
            "titrant_conc": [0.5, 0.5],
            "theta_obs": [0.5, 0.2],
            "theta_std": [0.01, 0.01],
        })
        single_orchestrator = ModelOrchestrator(growth_df, binding_df, transformation="single")

        from numpyro.handlers import seed, trace as nptrace
        seeded = seed(single_orchestrator.jax_model, rng_seed=0)
        model_tr = nptrace(seeded).get_trace(
            data=single_orchestrator.data, priors=single_orchestrator.priors,
        )
        fake_posteriors = {
            name: _feasible_draw(site)
            for name, site in model_tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }

        captured = self._capture_pred_data(mocker)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            predict(
                single_orchestrator,
                fake_posteriors,
                predict_sites=["growth_pred"],
                num_samples=None,
                genotypes=["wt"],
            )

        assert captured["data"].growth.external_theta_population is None


# ---------------------------------------------------------------------------
# predict() output alignment for condition_rep-plated sites
# ---------------------------------------------------------------------------

class TestPredictConditionRepSites:
    """
    condition_growth_k/m (growth/linear.py) are plated on condition_rep -- a
    TensorManager map_group that pools condition_pre and condition_sel
    labels into one shared index space, never a tensor_dim_names pivot
    axis. Verify predict() reports genuinely distinct per-condition values
    for such sites instead of silently broadcasting the first
    condition_rep's value to every row (the bug fixed by
    _is_condition_rep_site / _condition_rep_output_df in prediction.py).
    """

    @pytest.fixture
    def multi_condition_orchestrator(self):
        """Two genotypes, both spanning two distinct condition_sel values
        (same condition_pre throughout), giving num_condition_rep == 3
        ('pre-1', 'sel+1', 'sel+2')."""
        growth_df = pd.DataFrame({
            "library": ["lib"] * 8,
            "genotype": ["wt", "wt", "wt", "wt",
                         "M42V", "M42V", "M42V", "M42V"],
            "titrant_name": ["tit1"] * 8,
            "titrant_conc": [0.0, 1.0, 0.0, 1.0] * 2,
            "condition_pre": ["pre-1"] * 8,
            "condition_sel": ["sel+1", "sel+1", "sel+2", "sel+2"] * 2,
            "t_pre": [10.0] * 8,
            "t_sel": [0.0, 20.0, 0.0, 20.0] * 2,
            "ln_cfu": [0.0, 5.0, 0.0, 3.0] * 2,
            "ln_cfu_std": [0.1] * 8,
            "replicate": [1] * 8,
        })
        binding_df = pd.DataFrame({
            "genotype": ["wt", "M42V"],
            "titrant_name": ["tit1", "tit1"],
            "titrant_conc": [0.5, 0.5],
            "theta_obs": [0.5, 0.2],
            "theta_std": [0.01, 0.01],
        })
        return ModelOrchestrator(growth_df, binding_df, transformation="single")

    def _fake_posteriors(self, orchestrator, condition_growth_k_values):
        from numpyro.handlers import seed, trace as nptrace

        seeded = seed(orchestrator.jax_model, rng_seed=0)
        model_tr = nptrace(seeded).get_trace(
            data=orchestrator.data, priors=orchestrator.priors,
        )
        fake_posteriors = {
            name: _feasible_draw(site)
            for name, site in model_tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        assert fake_posteriors["condition_growth_k"].shape[1] == len(condition_growth_k_values)
        fake_posteriors["condition_growth_k"] = np.asarray([condition_growth_k_values])
        return fake_posteriors

    def test_condition_growth_k_differs_by_condition_rep(
            self, multi_condition_orchestrator):
        # num_condition_rep == 3, sorted alphabetically: pre-1, sel+1, sel+2
        true_values = [0.111, 0.222, 0.333]
        fake_posteriors = self._fake_posteriors(
            multi_condition_orchestrator, true_values
        )

        result = predict(
            multi_condition_orchestrator,
            fake_posteriors,
            predict_sites=["condition_growth_k"],
            num_samples=None,
            num_marginal_samples=1,
            genotypes=["wt"],
        )

        assert len(result) == 3
        assert set(result["condition_rep"]) == {"pre-1", "sel+1", "sel+2"}

        by_cond = result.set_index("condition_rep")["q0.5"]
        assert np.isclose(by_cond["pre-1"], 0.111)
        assert np.isclose(by_cond["sel+1"], 0.222)
        assert np.isclose(by_cond["sel+2"], 0.333)

        # The historical bug reported every row with the first
        # condition_rep's value -- guard against regressing to that.
        assert len(set(np.round(by_cond.values, 6))) == 3

    def test_condition_growth_k_and_m_together(
            self, multi_condition_orchestrator):
        """Requesting condition_growth_k and condition_growth_m together
        (as in the originally reported bug) must independently align each
        site to its own condition_rep values."""
        fake_posteriors = self._fake_posteriors(
            multi_condition_orchestrator, [0.111, 0.222, 0.333]
        )
        fake_posteriors["condition_growth_m"] = np.asarray([[0.01, 0.02, 0.03]])

        result = predict(
            multi_condition_orchestrator,
            fake_posteriors,
            predict_sites=["condition_growth_k", "condition_growth_m"],
            num_samples=None,
            num_marginal_samples=1,
            genotypes=["wt"],
        )

        k_df = result["condition_growth_k"].set_index("condition_rep")["q0.5"]
        m_df = result["condition_growth_m"].set_index("condition_rep")["q0.5"]

        assert np.isclose(k_df["sel+1"], 0.222)
        assert np.isclose(k_df["sel+2"], 0.333)
        assert np.isclose(m_df["sel+1"], 0.02)
        assert np.isclose(m_df["sel+2"], 0.03)
