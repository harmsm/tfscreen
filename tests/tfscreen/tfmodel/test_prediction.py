import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.prediction import copy_orchestrator, _convert_map_params, _align_site_to_tm_dims, predict

@pytest.fixture
def dummy_orchestrator():
    """Create a dummy ModelOrchestrator instance for testing."""
    growth_df = pd.DataFrame({
        "genotype": ["wt", "wt", "M42V", "M42V"],
        "titrant_name": ["tit1", "tit1", "tit1", "tit1"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "condition_pre": ["pre1", "pre1", "pre1", "pre1"],
        "condition_sel": ["sel1", "sel1", "sel1", "sel1"],
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

    return ModelOrchestrator(growth_df, binding_df)

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
            name: np.zeros((1,) + site["value"].shape)
            for name, site in model_tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }

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
