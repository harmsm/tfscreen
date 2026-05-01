import pytest
import os
import shutil
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry

# Define configurations to test
# We want to test a representative subset of the registry
# Define configurations to test. Each configuration is a dictionary of 
# ModelClass arguments. We want to make sure every entry in the 
# model_registry is tested at least once. 
SMOKE_CONFIGS = [
    # Default-ish config
    {
        "condition_growth":"linear",
        "transformation":"empirical",
        "theta":"hill",
        "growth_transition":"instant",
        "dk_geno":"hierarchical",
        "activity":"horseshoe",
        "ln_cfu0":"hierarchical",
        "theta_growth_noise":"zero",
        "theta_binding_noise":"zero"
    },
    # test other condition growth and transformation
    {
        "condition_growth":"linear",
        "transformation":"logit_norm",
        "theta":"categorical",
        "growth_transition":"memory",
        "dk_geno":"fixed",
        "activity":"hierarchical",
        "theta_growth_noise":"beta",
        "theta_binding_noise":"beta"
    },
    # test other transition and transformation
    {
        "condition_growth":"linear_fixed",
        "transformation":"logit_norm",
        "theta":"hill",
        "growth_transition":"baranyi",
        "activity":"fixed",
    },
    # test power and saturation
    {
        "condition_growth":"power",
        "transformation":"single",
    },
    {
        "condition_growth":"saturation",
    },
    # test mutation-decomposed Hill theta
    {
        "theta":"hill_mut",
        "dk_geno":"hierarchical_mut",
        "activity":"hierarchical_mut",
        "epistasis":False,
    },
    # test lac_dimer_lnK_mut theta (partition-function model)
    {
        "theta":"lac_dimer_lnK_mut",
        "epistasis":False,
    },
]

@pytest.mark.slow
@pytest.mark.parametrize("config", SMOKE_CONFIGS)
def test_model_svi_smoke(growth_smoke_csv, 
                         binding_smoke_csv, 
                         config,
                         tmpdir):
    """
    Perform a very short SVI run for different model configurations.
    
    This is a smoke test to ensure that the model components can be 
    initialized and a few steps of optimization can be performed without error.
    """
    out_root = os.path.join(tmpdir, "smoke_test")
    
    # Initialize ModelClass
    model = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None, # Use all data for smoke test
        **config
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
