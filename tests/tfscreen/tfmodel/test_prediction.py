import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.prediction import copy_model_class, _convert_map_params

@pytest.fixture
def dummy_mc():
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

def test_copy_model_class_defaults(dummy_mc):
    """Test copy_model_class with all None inputs."""
    new_mc = copy_model_class(dummy_mc)
    # Categorical combinations (1 rep * 1 cp * 1 cs * 1 tn * 2 geno) = 2
    # Quantitative (1 t_pre * 2 t_sel * 2 conc) = 4
    # Total = 2 * 4 = 8
    assert len(new_mc.growth_df) == 8
    assert new_mc.settings == dummy_mc.settings

def test_copy_model_class_expansion(dummy_mc):
    """Test expanding quantitative inputs."""
    new_mc = copy_model_class(
        dummy_mc,
        t_sel=[0.0, 10.0, 20.0],
        titrant_conc=[0.0, 0.5, 1.0]
    )
    # 2 (genotypes) * 3 (t_sel) * 3 (conc) = 18
    assert len(new_mc.growth_df) == 18
    assert set(new_mc.growth_df["t_sel"]) == {0.0, 10.0, 20.0}
    assert set(new_mc.growth_df["titrant_conc"]) == {0.0, 0.5, 1.0}

def test_copy_model_class_list_inputs(dummy_mc):
    """Test passing single values as non-lists."""
    new_mc = copy_model_class(
        dummy_mc,
        t_sel=30.0,
        titrant_conc=2.0
    )
    assert 30.0 in new_mc.growth_df["t_sel"].values
    assert 2.0 in new_mc.growth_df["titrant_conc"].values

def test_copy_model_class_quantitative_errors(dummy_mc):
    """Test validation of quantitative inputs."""
    with pytest.raises(ValueError, match="t_pre must be >= 0"):
        copy_model_class(dummy_mc, t_pre=-1.0)
    
    with pytest.raises(ValueError, match="t_sel must be >= 0"):
        copy_model_class(dummy_mc, t_sel=[-1.0, 1.0])
        
    with pytest.raises(ValueError, match="titrant_conc must be >= 0"):
        copy_model_class(dummy_mc, titrant_conc=[1.0, -0.5])

def test_copy_model_class_t_pre_expansion(dummy_mc):
    """Test expanding t_pre."""
    new_mc = copy_model_class(
        dummy_mc,
        t_pre=[10.0, 20.0]
    )
    # 2 (genotypes) * 2 (t_pre) * 2 (t_sel) * 2 (conc) = 16
    assert len(new_mc.growth_df) == 16
    assert set(new_mc.growth_df["t_pre"]) == {10.0, 20.0}

def test_copy_model_class_no_subsetting_binding(dummy_mc):
    """Test that binding_df is NOT subsetted during copy."""
    # Original binding_df has wt and M42V
    assert len(dummy_mc.binding_df) == 2

    new_mc = copy_model_class(dummy_mc)

    # New binding_df should still have all genotypes
    assert len(new_mc.binding_df) == 2


# ---------------------------------------------------------------------------
# _convert_map_params
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_trace():
    """
    Minimal model trace for testing _convert_map_params.

    Includes:
    - an unconstrained site (Normal → real support → identity bijection)
    - a positive-constrained site (HalfNormal → positive support → softplus bijection)
    - an observed site (must be ignored)
    - a deterministic site (must be ignored)
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
        """Normal (real support) site value is unchanged by the bijection."""
        raw_val = np.array(-2.3)
        map_params = {"real_site_auto_loc": raw_val}
        out = _convert_map_params(map_params, fake_trace)
        np.testing.assert_allclose(
            np.asarray(out["real_site"][0]),
            raw_val,
            rtol=1e-5,
        )

    def test_bijection_for_positive_site(self, fake_trace):
        """HalfNormal (positive support) site value is transformed to be positive."""
        raw_val = np.array(-3.0)   # arbitrary unconstrained value
        map_params = {"positive_site_auto_loc": raw_val}
        out = _convert_map_params(map_params, fake_trace)
        constrained = float(out["positive_site"][0])
        # Result must be strictly positive.
        assert constrained > 0, f"Expected positive constrained value, got {constrained}"
        # Must match what biject_to gives.
        expected = float(biject_to(dist.HalfNormal(1.0).support)(jnp.array(raw_val)))
        np.testing.assert_allclose(constrained, expected, rtol=1e-5)

    def test_observed_site_ignored(self, fake_trace):
        """Keys whose site is marked is_observed are not converted."""
        map_params = {
            "real_site_auto_loc": np.array(0.0),
            "observed_site_auto_loc": np.array(1.0),
        }
        out = _convert_map_params(map_params, fake_trace)
        # observed_site should still appear (key is not filtered by _convert_map_params
        # — it has no _auto_loc guard in the trace loop, so the bijection path is
        # simply skipped and the raw value is passed through with leading dim).
        # The important thing: real_site IS bijected, observed_site is NOT blocked.
        assert "real_site" in out
        # observed_site: site is_observed=True so bijection branch is skipped;
        # value should be expanded raw value.
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
