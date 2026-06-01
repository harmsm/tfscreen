import pytest
import os
import shutil
import jax.numpy as jnp
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.generative.registry import model_registry

# Define configurations to test
# We want to test a representative subset of the registry
# Define configurations to test. Each configuration is a dictionary of 
# ModelOrchestrator arguments. We want to make sure every entry in the 
# model_registry is tested at least once. 
SMOKE_CONFIGS = [
    # Default-ish config
    {
        "condition_growth":"linear",
        "transformation":"empirical",
        "theta":"hill_geno",
        "growth_transition":"instant",
        "dk_geno":"hierarchical_geno",
        "activity":"horseshoe_geno",
        "ln_cfu0":"hierarchical",
        "theta_growth_noise":"zero",
        "theta_binding_noise":"zero"
    },
    # test other condition growth and transformation
    {
        "condition_growth":"linear",
        "transformation":"logit_norm",
        "theta":"categorical_geno",
        "growth_transition":"memory",
        "dk_geno":"fixed",
        "activity":"hierarchical_geno",
        "theta_growth_noise":"beta",
        "theta_binding_noise":"beta"
    },
    # test other transition and transformation
    {
        "condition_growth":"power",
        "transformation":"logit_norm",
        "theta":"hill_geno",
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
        "dk_geno":"hierarchical_geno",
        "activity":"hierarchical_mut",
        "epistasis":False,
    },
    # test horseshoe_mut activity (mut-decomposed horseshoe)
    {
        "theta":"hill_mut",
        "dk_geno":"hierarchical_geno",
        "activity":"horseshoe_mut",
        "epistasis":False,
    },
    # test lac_dimer_lnK_mut theta (partition-function model)
    {
        "theta":"thermo.O2_C4_K3_U0_a.PK",
        "epistasis":False,
    },
    # test hierarchical_factored ln_cfu0, logit theta_rescale, normal_kt growth_noise
    {
        "ln_cfu0":"hierarchical_factored",
        "theta_rescale":"logit",
        "growth_noise":"normal_kt",
    },
    # test logit_normal theta_growth_noise, normal sample_offset, baranyi_k transition
    {
        "theta_growth_noise":"logit_normal",
        "sample_offset":"normal",
        "growth_transition":"baranyi_k",
    },
    # test baranyi_tau growth_transition
    {
        "growth_transition":"baranyi_tau",
    },
    # test two_pop growth_transition
    {
        "growth_transition":"two_pop",
    },
    # test lac-dimer unfolded (U1) PK theta (no struct data required)
    {
        "theta":"thermo.O2_C4_K3_U1_a.PK",
        "epistasis":False,
    },
    # test MWC-dimer (O2_C12_K5) U0 PK theta
    {
        "theta":"thermo.O2_C12_K5_U0_a.PK",
        "epistasis":False,
    },
    # test MWC-dimer unfolded (U1) PK theta
    {
        "theta":"thermo.O2_C12_K5_U1_a.PK",
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
    out_prefix = os.path.join(tmpdir, "smoke_test")
    
    # Initialize ModelOrchestrator
    model = ModelOrchestrator(
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
        out_prefix=out_prefix,
        checkpoint_interval=10 # Skip checkpointing for this quick test
    )
    
    assert svi_state is not None
    assert isinstance(params, dict)

@pytest.mark.slow
def test_model_svi_smoke_lnK_nn_prior(growth_smoke_csv,
                                       binding_smoke_csv,
                                       struct_smoke_h5_path,
                                       tmpdir):
    """SVI smoke test for lac_dimer_lnK_nn_prior (requires thermo_data)."""
    out_prefix = os.path.join(tmpdir, "smoke_nn_prior")

    model = ModelOrchestrator(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None,
        theta="thermo.O2_C4_K3_U0_a.PnnC",
        thermo_data=struct_smoke_h5_path,
        epistasis=False,
    )

    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=10,
    )

    assert svi_state is not None
    assert isinstance(params, dict)


@pytest.mark.slow
@pytest.mark.parametrize("theta_key", [
    "thermo.O2_C4_K3_U1_a.PnnC",
])
def test_model_svi_smoke_lac_U1_nn_prior(growth_smoke_csv,
                                          binding_smoke_csv,
                                          struct_smoke_h5_path,
                                          theta_key,
                                          tmpdir):
    """SVI smoke test for lac-dimer U1 PnnC (uses same HDF5 fixture as U0)."""
    out_prefix = os.path.join(tmpdir, "smoke_lac_u1_nn")
    model = ModelOrchestrator(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None,
        theta=theta_key,
        thermo_data=struct_smoke_h5_path,
        epistasis=False,
    )
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=10,
    )
    assert svi_state is not None
    assert isinstance(params, dict)


@pytest.mark.slow
@pytest.mark.parametrize("theta_key", [
    "thermo.O2_C4_K3_U0_a.PddG",
    "thermo.O2_C4_K3_U1_a.PddG",
])
def test_model_svi_smoke_lac_ddG_prior(growth_smoke_csv,
                                        binding_smoke_csv,
                                        ddg_prior_csv_lac,
                                        theta_key,
                                        tmpdir):
    """SVI smoke test for lac-dimer PddG variants (per-mutation per-structure ΔΔG prior)."""
    out_prefix = os.path.join(tmpdir, f"smoke_{theta_key.replace('.', '_')}")
    model = ModelOrchestrator(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None,
        theta=theta_key,
        thermo_data=ddg_prior_csv_lac,
        epistasis=False,
    )
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=10,
    )
    assert svi_state is not None
    assert isinstance(params, dict)


@pytest.mark.slow
@pytest.mark.parametrize("theta_key", [
    "thermo.O2_C12_K5_U0_a.PnnC",
    "thermo.O2_C12_K5_U1_a.PnnC",
])
def test_model_svi_smoke_mwc_nn_prior(growth_smoke_csv,
                                       binding_smoke_csv,
                                       struct_smoke_h5_path_mwc,
                                       theta_key,
                                       tmpdir):
    """SVI smoke test for MWC-dimer PnnC variants (requires 6-structure HDF5)."""
    out_prefix = os.path.join(tmpdir, f"smoke_{theta_key.replace('.', '_')}")
    model = ModelOrchestrator(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None,
        theta=theta_key,
        thermo_data=struct_smoke_h5_path_mwc,
        epistasis=False,
    )
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=10,
    )
    assert svi_state is not None
    assert isinstance(params, dict)


@pytest.mark.slow
@pytest.mark.parametrize("theta_key", [
    "thermo.O2_C12_K5_U0_a.PddG",
    "thermo.O2_C12_K5_U1_a.PddG",
])
def test_model_svi_smoke_mwc_ddG_prior(growth_smoke_csv,
                                        binding_smoke_csv,
                                        ddg_prior_csv_mwc,
                                        theta_key,
                                        tmpdir):
    """SVI smoke test for MWC-dimer PddG variants (requires 6-structure ΔΔG CSV)."""
    out_prefix = os.path.join(tmpdir, f"smoke_{theta_key.replace('.', '_')}")
    model = ModelOrchestrator(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        batch_size=None,
        theta=theta_key,
        thermo_data=ddg_prior_csv_mwc,
        epistasis=False,
    )
    inference = RunInference(model=model, seed=42)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=10,
    )
    assert svi_state is not None
    assert isinstance(params, dict)


def test_registry_coverage():
    """
    Ensure that the model_registry contains all expected component axes.
    This is a meta-test to make sure smoke tests stay relevant as the registry grows.
    """
    expected_keys = [
        "condition_growth", "ln_cfu0", "dk_geno", "activity",
        "transformation", "theta_rescale", "theta",
        "theta_growth_noise", "theta_binding_noise",
        "growth_noise", "sample_offset", "growth_transition",
    ]
    for key in expected_keys:
        assert key in model_registry
