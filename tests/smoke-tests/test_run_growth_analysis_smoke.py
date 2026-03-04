import pytest
import os
import h5py
from tfscreen.analysis.hierarchical.growth_model import GrowthModel as ModelClass
from tfscreen.analysis.hierarchical.growth_model.configuration_io import write_configuration
from tfscreen.analysis.hierarchical.growth_model.scripts.run_growth_analysis import run_growth_analysis

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
        transformation="congression",
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
