import pytest
import pandas as pd
import numpy as np
import jax
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from numpyro.handlers import seed, trace

@pytest.fixture
def test_setup():
    """Create a dummy ModelClass and matching posteriors."""
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
    mc = ModelClass(growth_df, binding_df)
    
    # Trace to get shapes
    seeded_model = seed(mc.jax_model, rng_seed=0)
    traced_model = trace(seeded_model)
    model_trace = traced_model.get_trace(data=mc.data, priors=mc.priors)
    
    num_samples = 10
    posteriors = {}
    for name, site in model_trace.items():
        if site["type"] in ["sample", "deterministic"]:
            # Ensure we have the sample dimension
            posteriors[name] = np.zeros((num_samples,) + site["value"].shape)
            
    return mc, posteriors

def test_predict_basic(test_setup):
    """Test basic functionality."""
    mc, posteriors = test_setup
    df = predict(mc, posteriors, num_samples=5)
    
    # 8 rows expected
    assert len(df) == 8
    assert "median" in df.columns
    assert not df["median"].isna().any()

def test_predict_subset(test_setup):
    """Test categorical subsetting and basic mapping."""
    mc, posteriors = test_setup
    
    # Find genotype dim for ln_cfu0_offset
    label_list = list(mc.growth_tm.tensor_dim_labels[-1])
    wt_idx = label_list.index("wt")
    
    # Fill wt with unique value in offset
    # replicate, cp, genotype
    posteriors["ln_cfu0_offset"][..., wt_idx] = 200.0
    posteriors["ln_cfu0_hyper_loc"][:] = 0.0
    posteriors["ln_cfu0_hyper_scale"][:] = 1.0
    
    if "ln_cfu0" in posteriors:
        posteriors.pop("ln_cfu0")

    # Subset to only wt
    df = predict(mc, posteriors, genotypes=["wt"], num_samples=1)
    
    # 4 rows expected
    assert len(df) == 4
    assert (df["genotype"] == "wt").all()
    # Confirm it correctly mapped to index 1 (wt) and pulled 200.0
    assert np.isclose(df["median"].mean(), 200.0)

def test_predict_expansion(test_setup):
    """Test quantitative expansion."""
    mc, posteriors = test_setup
    # Expand time points
    df = predict(mc, posteriors, t_sel=[0.0, 10.0, 20.0], num_samples=5)
    
    # 12 rows expected
    assert len(df) == 12
    assert set(df["t_sel"]) == {0.0, 10.0, 20.0}

def test_predict_multiple_sites(test_setup):
    """Test predicting multiple sites."""
    mc, posteriors = test_setup
    
    # Predict growth_pred AND some other site (e.g. ln_cfu0)
    sites = ["growth_pred", "ln_cfu0"]
    results = predict(mc, posteriors, predict_sites=sites, num_samples=2)
    
    assert isinstance(results, dict)
    assert set(results.keys()) == set(sites)
    for s in sites:
        assert isinstance(results[s], pd.DataFrame)
        assert "median" in results[s].columns

def test_predict_broadcasting(test_setup):
    """Test predicting a global site (broadcasting)."""
    mc, posteriors = test_setup
    
    # theta_mu_hyper_loc is a scalar in this model (if configured as such)
    # Let's check a site we know is likely scalar or small
    # For this dummy, let's just use something global
    sites = ["ln_cfu0_hyper_loc"]
    df = predict(mc, posteriors, predict_sites=sites, num_samples=1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert "median" in df.columns

def test_predict_error_handling(test_setup):
    """Test error handling."""
    mc, posteriors = test_setup
    
    # Invalid genotype
    with pytest.raises(ValueError, match="not found in model_class.growth_df"):
        predict(mc, posteriors, genotypes=["bad_geno"])
    
    # Invalid posterior type
    with pytest.raises(ValueError, match="posteriors should be"):
        predict(mc, [1, 2, 3])
