import pytest
import os
import pandas as pd
import numpy as np
from tfscreen.analysis.hierarchical.growth_model import GrowthModel as ModelClass
from tfscreen.analysis.hierarchical.growth_model.prediction import predict
from tfscreen.analysis.hierarchical.run_inference import RunInference

@pytest.mark.slow
def test_prediction_smoke(growth_smoke_csv, 
                           binding_smoke_csv, 
                           tmpdir):
    """
    Smoke test for the simplified prediction logic.
    """
    out_root = os.path.join(tmpdir, "test_predict_smoke")
    
    # 1. Initialize and run a very short optimization to get samples
    model = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        theta="hill",
        transformation="logit_norm"
    )
    
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=1,
        out_root=out_root
    )
    
    inference.get_posteriors(
        svi=svi,
        svi_state=svi_state,
        out_root=out_root,
        num_posterior_samples=5
    )
    posterior_file = f"{out_root}_posterior.h5"
    
    # 2. Test prediction with quantitative expansion and genotype subsetting
    all_genotypes = model.growth_tm.tensor_dim_labels[-1]
    subset_genotypes = [all_genotypes[0]] # Just the first one
    
    new_t_sel = [10.0, 30.0, 50.0]
    new_titrant_conc = [0.0, 0.1, 1.0] # These must be in the original data for 'congression'
    
    pred_df = predict(
        model,
        posterior_file,
        predict_sites=["growth_pred"],
        t_sel=new_t_sel,
        titrant_conc=new_titrant_conc,
        genotypes=subset_genotypes,
        num_samples=None # Verify None works
    )
    
    assert isinstance(pred_df, pd.DataFrame)
    
    # Check that genotypes are subsetted correctly
    assert set(pred_df["genotype"]) == set(subset_genotypes)
    
    # Check that quantitative variables are expanded correctly
    assert set(pred_df["t_sel"]) == set(new_t_sel)
    assert set(pred_df["titrant_conc"]) == set(new_titrant_conc)
    
    # Check for quantile columns (defaults from load_posteriors)
    assert "median" in pred_df.columns
    assert "lower_95" in pred_df.columns
    assert "upper_95" in pred_df.columns

    # 3. Test expansion restriction for plated dimensions
    # 'logit_norm' plates on titrant_conc, so expanding it should fail.
    with pytest.raises(ValueError, match="is plated on .* and cannot be expanded"):
        predict(
            model,
            posterior_file,
            titrant_conc=[999.9] # Not in original data
        )

    with pytest.raises(ValueError, match="is plated on .* and cannot be expanded"):
        
        # 4. Test categorical theta restriction (as requested)
        # Create model with categorical theta
        model_cat = ModelClass(
            growth_df=growth_smoke_csv,
            binding_df=binding_smoke_csv,
            theta="categorical"
        )
        
        # Generate a valid posterior file for model_cat
        inference_cat = RunInference(model=model_cat, seed=42)
        svi_cat = inference_cat.setup_svi(adam_step_size=1e-3)
        svi_state_cat, _, _ = inference_cat.run_optimization(
            svi=svi_cat, max_num_epochs=1, out_root=f"{out_root}_cat"
        )
        inference_cat.get_posteriors(
            svi=svi_cat, svi_state=svi_state_cat, 
            out_root=f"{out_root}_cat", num_posterior_samples=5
        )
        posterior_file_cat = f"{out_root}_cat_posterior.h5"

        predict(
            model_cat,
            posterior_file_cat,
            titrant_conc=[999.9]
        )
