import pytest
import os
import shutil
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry

# Define configurations to test
# We want to test a representative subset of the registry
CONDITION_GROWTH_OPTS = ["independent", "hierarchical"]
TRANSFORMATION_OPTS = ["congression", "single"]
THETA_OPTS = ["hill"] # categorical is also possible but hill is the most complex
EPISTASIS_MODE_OPTS = ["genotype", "none","horseshoe","spikeslab"]

@pytest.mark.slow
@pytest.mark.parametrize("condition_growth", CONDITION_GROWTH_OPTS)
@pytest.mark.parametrize("transformation", TRANSFORMATION_OPTS)
@pytest.mark.parametrize("theta", THETA_OPTS)
@pytest.mark.parametrize("epistasis_mode", EPISTASIS_MODE_OPTS)
def test_model_svi_smoke(growth_smoke_csv, 
                         binding_smoke_csv, 
                         condition_growth, 
                         transformation, 
                         theta, 
                         epistasis_mode,
                         tmpdir):
    """
    Perform a very short SVI run for different model configurations.
    
    This is a smoke test to ensure that the model components can be 
    initialized and a few steps of optimization can be performed without error.
    """
    out_root = os.path.join(tmpdir, f"smoke_{condition_growth}_{transformation}_{theta}")
    
    # Initialize ModelClass
    model = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        condition_growth=condition_growth,
        transformation=transformation,
        theta=theta,
        epistasis_mode=epistasis_mode,
        batch_size=None # Use all data for smoke test
    )
    
    # Initialize RunInference
    inference = RunInference(model=model, seed=42)
    
    # Setup SVI
    svi = inference.setup_svi(adam_step_size=1e-3)
    
    # Run a very short optimization
    # We only run for a few steps to verify it doesn't crash
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_root=out_root,
        checkpoint_interval=10 # Skip checkpointing for this quick test
    )
    
    assert svi_state is not None
    assert isinstance(params, dict)

def test_registry_coverage():
    """
    Ensure that we are testing the keys that exist in the registry.
    This is a meta-test to make sure the smoke tests stay relevant.
    """
    expected_keys = [
        "condition_growth", "ln_cfu0", "dk_geno", "activity", 
        "transformation", "theta", "theta_growth_noise", "theta_binding_noise"
    ]
    for key in expected_keys:
        assert key in model_registry
