import pytest
import pandas as pd
import numpy as np
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.prediction import copy_model_class

@pytest.fixture
def dummy_mc():
    """Create a dummy ModelClass instance for testing."""
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

    return ModelClass(growth_df, binding_df)

def test_copy_model_class_defaults(dummy_mc):
    """Test copy_model_class with all None inputs."""
    new_mc = copy_model_class(dummy_mc)
    # t_pre(10.0) * t_sel(0.0, 20.0) * conc(0.0, 1.0) * geno(wt, M42V) * cp(1) * cs(1) * tn(1) * rep(1) = 8
    assert len(new_mc.growth_df) == 8
    assert new_mc.settings == dummy_mc.settings

def test_copy_model_class_expansion(dummy_mc):
    """Test expanding quantitative inputs."""
    new_mc = copy_model_class(
        dummy_mc,
        t_sel=[0.0, 10.0, 20.0],
        titrant_conc=[0.0, 0.5, 1.0],
        genotypes=["wt"]
    )
    # 1 (t_pre) * 3 (t_sel) * 3 (conc) * 1 (geno) * 1 (cp) * 1 (cs) * 1 (tn) * 1 (rep) = 9
    assert len(new_mc.growth_df) == 9
    assert set(new_mc.growth_df["t_sel"]) == {0.0, 10.0, 20.0}
    assert set(new_mc.growth_df["titrant_conc"]) == {0.0, 0.5, 1.0}

def test_copy_model_class_subset(dummy_mc):
    """Test subsetting categorical inputs."""
    new_mc = copy_model_class(
        dummy_mc,
        genotypes=["M42V"],
        condition_pre=["pre1"],
        condition_sel=["sel1"],
        titrant_name=["tit1"],
        replicate=[1],
        t_sel=[0.0]
    )
    # 1(t_pre)*1(t_sel)*2(conc)*1*1*1*1*1 = 2
    assert len(new_mc.growth_df) == 2
    assert (new_mc.growth_df["genotype"] == "M42V").all()

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

def test_copy_model_class_categorical_errors(dummy_mc):
    """Test validation of categorical inputs."""
    with pytest.raises(ValueError, match="genotype values were not found"):
        copy_model_class(dummy_mc, genotypes=["bad_geno"])
    
    with pytest.raises(ValueError, match="condition_pre values were not found"):
        copy_model_class(dummy_mc, condition_pre=["bad_cond"])

def test_copy_model_class_t_pre_checks(dummy_mc):
    """Test t_pre single value constraints."""
    # Passing multiple t_pre should fail
    with pytest.raises(ValueError, match="t_pre must have exactly one value"):
        copy_model_class(dummy_mc, t_pre=[10.0, 20.0])
    
    # If original has multiple t_pre and none passed, it should fail
    growth_df_raw = pd.DataFrame({
        "genotype": ["wt", "wt", "M42V", "M42V"],
        "titrant_name": ["tit1", "tit1", "tit1", "tit1"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "condition_pre": ["pre1", "pre1", "pre1", "pre1"],
        "condition_sel": ["sel1", "sel1", "sel1", "sel1"],
        "t_pre": [10.0, 5.0, 10.0, 5.0], 
        "t_sel": [0.0, 20.0, 0.0, 20.0],
        "ln_cfu": [0.0, 5.0, 0.0, 3.0],
        "ln_cfu_std": [0.1, 0.1, 0.1, 0.1],
        "replicate": [1, 1, 1, 1]
    })
    binding_df_raw = pd.DataFrame({
        "genotype": ["wt", "M42V"],
        "titrant_name": ["tit1", "tit1"],
        "titrant_conc": [0.5, 0.5],
        "theta_obs": [0.5, 0.2],
        "theta_std": [0.01, 0.01]
    })
    mc_bad = ModelClass(growth_df_raw, binding_df_raw)
    with pytest.raises(ValueError, match="t_pre must have exactly one value"):
        copy_model_class(mc_bad)

def test_copy_model_class_binding_subsetting(dummy_mc):
    """Test that binding_df is correctly subsetted."""
    # Original binding_df has wt and M42V
    assert len(dummy_mc.binding_df) == 2
    
    # Subset growth to only wt
    new_mc = copy_model_class(dummy_mc, genotypes=["wt"])
    
    # New binding_df should only have wt
    assert len(new_mc.binding_df) == 1
    assert (new_mc.binding_df["genotype"] == "wt").all()
