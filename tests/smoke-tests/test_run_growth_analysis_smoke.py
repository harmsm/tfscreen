import pytest
import os
import h5py
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.configuration_io import write_configuration
from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import run_growth_analysis


def _build_config(gm, tmpdir, growth_smoke_csv, binding_smoke_csv):
    """Write a config file for gm and return its path."""
    config_root = os.path.join(tmpdir, "smoke")
    write_configuration(
        gm=gm,
        out_root=config_root,
        growth_df_path=str(growth_smoke_csv),
        binding_df_path=str(binding_smoke_csv),
    )
    return f"{config_root}_config.yaml"

@pytest.mark.slow
def test_run_growth_analysis_smoke(growth_smoke_csv, binding_smoke_csv, tmpdir):
    """
    Smoke test for run_growth_analysis script.
    """
    out_root = os.path.join(tmpdir, "run_smoke")
    
    # 1. Initialize a model and write its configuration
    gm = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        condition_growth="linear",
        transformation="logit_norm",
        theta="hill",
        batch_size=None
    )
    
    config_root = os.path.join(tmpdir, "smoke")
    write_configuration(
        gm=gm,
        out_root=config_root,
        growth_df_path=str(growth_smoke_csv),
        binding_df_path=str(binding_smoke_csv)
    )
    
    config_file = f"{config_root}_config.yaml"
    assert os.path.exists(config_file)
    
    # 2. Run the analysis using the script entry point
    # We use very small epochs and samples for speed
    run_growth_analysis(
        config_file=config_file,
        seed=42,
        analysis_method="svi",
        out_root=out_root,
        max_num_epochs=1,
        num_posterior_samples=5,
        sampling_batch_size=5,
        forward_batch_size=10,
        always_get_posterior=True
    )
    
    # 3. Verify outputs
    posterior_file = f"{out_root}_posterior.h5"
    assert os.path.exists(posterior_file)
    
    # Load and check the saved file
    with h5py.File(posterior_file, 'r') as data:
        assert "growth_pred" in data
        assert data["growth_pred"].shape[0] == 5


@pytest.mark.slow
def test_run_growth_analysis_nuts_smoke(growth_smoke_csv, binding_smoke_csv, tmpdir):
    """
    Smoke test for run_growth_analysis with analysis_method='nuts'.

    Runs a tiny NUTS chain and verifies that:
    - a posterior HDF5 is written with the expected keys
    - a checkpoint pkl is written containing 'mcmc_samples'
    """
    gm = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        condition_growth="linear",
        transformation="logit_norm",
        theta="hill",
        batch_size=None,
    )
    config_file = _build_config(gm, tmpdir, growth_smoke_csv, binding_smoke_csv)
    assert os.path.exists(config_file)

    out_root = os.path.join(tmpdir, "nuts_smoke")

    _, mcmc_samples, converged = run_growth_analysis(
        config_file=config_file,
        seed=42,
        analysis_method="nuts",
        out_root=out_root,
        nuts_num_warmup=5,
        nuts_num_samples=10,
        nuts_num_chains=1,
        forward_batch_size=10,
    )

    # NUTS always returns converged=True
    assert converged is True

    # mcmc_samples dict is returned
    assert isinstance(mcmc_samples, dict)
    assert len(mcmc_samples) > 0

    # Checkpoint written with mcmc_samples key
    import dill
    chk_path = f"{out_root}_checkpoint.pkl"
    assert os.path.exists(chk_path)
    with open(chk_path, "rb") as f:
        chk = dill.load(f)
    assert "mcmc_samples" in chk

    # Posterior HDF5 written with growth_pred
    posterior_file = f"{out_root}_posterior.h5"
    assert os.path.exists(posterior_file)
    with h5py.File(posterior_file, "r") as hf:
        assert "growth_pred" in hf
        assert hf["growth_pred"].shape[0] == 10


@pytest.mark.slow
def test_run_growth_analysis_nuts_posterior_smoke(growth_smoke_csv, binding_smoke_csv, tmpdir):
    """
    Smoke test for analysis_method='posterior' applied to a saved NUTS checkpoint.

    Runs a tiny NUTS chain, then re-generates posteriors from the checkpoint
    and verifies that the output HDF5 is recreated correctly.
    """
    gm = ModelClass(
        growth_df=growth_smoke_csv,
        binding_df=binding_smoke_csv,
        condition_growth="linear",
        transformation="logit_norm",
        theta="hill",
        batch_size=None,
    )
    config_file = _build_config(gm, tmpdir, growth_smoke_csv, binding_smoke_csv)

    # Step 1: run NUTS to produce a checkpoint
    nuts_root = os.path.join(tmpdir, "nuts_for_post")
    run_growth_analysis(
        config_file=config_file,
        seed=42,
        analysis_method="nuts",
        out_root=nuts_root,
        nuts_num_warmup=5,
        nuts_num_samples=10,
        nuts_num_chains=1,
        forward_batch_size=10,
    )
    chk_path = f"{nuts_root}_checkpoint.pkl"
    assert os.path.exists(chk_path)

    # Step 2: re-generate posteriors from the NUTS checkpoint
    post_root = os.path.join(tmpdir, "nuts_post")
    run_growth_analysis(
        config_file=config_file,
        seed=0,          # seed is unused for NUTS posterior mode but required by API
        analysis_method="posterior",
        checkpoint_file=chk_path,
        out_root=post_root,
        forward_batch_size=10,
    )

    posterior_file = f"{post_root}_posterior.h5"
    assert os.path.exists(posterior_file)
    with h5py.File(posterior_file, "r") as hf:
        assert "growth_pred" in hf
        assert hf["growth_pred"].shape[0] == 10
