import pytest
import os
import jax.numpy as jnp
import numpy as np
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.run_inference import RunInference
import h5py

@pytest.mark.slow
@pytest.mark.parametrize("epistasis_mode", ["genotype", "none","horseshoe","spikeslab"])
def test_checkpoint_and_posterior_smoke(growth_smoke_csv, 
                                        binding_smoke_csv, 
                                        epistasis_mode,
                                        tmpdir):
    """
    Test the full workflow: Run SVI, save checkpoint, restore, and generate posteriors.
    """
    out_root = os.path.join(tmpdir, "test_posteriors")
    
    # 1. Initialize and run a short optimization
    model = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        condition_growth="hierarchical",
        transformation="congression",
        theta="hill",
        epistasis_mode=epistasis_mode,
        batch_size=None
    )
    
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    
    # Run for 10 steps and ensure a checkpoint is written
    # Run for 2 epochs and ensure a checkpoint is written. 
    # current_step must reach checkpoint_interval_steps.
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=2,
        out_root=out_root,
        checkpoint_interval=1 
    )
    
    checkpoint_file = f"{out_root}_checkpoint.pkl"
    assert os.path.exists(checkpoint_file)
    
    # 2. Restore from checkpoint in a new RunInference instance
    new_inference = RunInference(model=model, seed=43) # Different seed
    restored_state = new_inference._restore_checkpoint(checkpoint_file)
    
    assert restored_state is not None
    # We could theoretically check if parameters match, but the state structure 
    # is internal to numpyro/optax.
    
    # 3. Generate posteriors from the restored state
    # We use a very small number of samples for speed
    new_inference.get_posteriors(
        svi=svi,
        svi_state=restored_state,
        out_root=out_root,
        num_posterior_samples=10,
        sampling_batch_size=5,
        forward_batch_size=10
    )
    
    posterior_file = f"{out_root}_posterior.h5"
    assert os.path.exists(posterior_file)
    
    # Load and check the saved file
    with h5py.File(posterior_file, 'r') as data:
        assert "growth_pred" in data
        assert data["growth_pred"].shape[0] == 10 # num_posterior_samples

@pytest.mark.slow
@pytest.mark.parametrize("epistasis_mode", ["genotype", "none","horseshoe","spikeslab"])
def test_extract_parameters_smoke(growth_smoke_csv, 
                                  binding_smoke_csv, 
                                  epistasis_mode,
                                  tmpdir):
    """
    Test extracting parameter DataFrames from generated posteriors.
    """
    out_root = os.path.join(tmpdir, "test_extract")
    
    model = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        theta="hill",
        transformation="congression",
        epistasis_mode=epistasis_mode
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
    
    # Test extraction
    param_dfs = model.extract_parameters(posterior_file)
    assert "hill_n" in param_dfs
    assert "activity" in param_dfs
    assert "lam" in param_dfs # for congression
    
    # Verify they are DataFrames
    import pandas as pd
    for df in param_dfs.values():
        assert isinstance(df, pd.DataFrame)
