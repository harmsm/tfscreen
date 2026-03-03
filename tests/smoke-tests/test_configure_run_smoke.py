import pytest
import pandas as pd
import os
import yaml
from tfscreen.analysis.hierarchical.configure_growth_analysis import configure_growth_analysis
from tfscreen.analysis.hierarchical.run_growth_analysis import run_growth_analysis

@pytest.mark.slow
def test_configure_run_pipeline_smoke(tmpdir):
    """
    Smoke test for the full configuration -> running pipeline.
    """
    # Create mock data
    growth_df = pd.DataFrame({
        "replicate": ["R1", "R1"],
        "condition_pre": ["C1", "C1"],
        "condition_sel": ["C2", "C2"],
        "genotype": ["A123B", "C456D"],
        "t": [0, 10],
        "t_sel": [0, 10],
        "t_pre": [12, 12],
        "ln_cfu": [1.0, 2.0],
        "ln_cfu_std": [0.1, 0.1],
        "titrant_name": ["T1", "T1"],
        "titrant_conc": [0.1, 0.1]
    })
    growth_path = os.path.join(tmpdir, "test_growth.csv")
    growth_df.to_csv(growth_path, index=False)

    binding_df = pd.DataFrame({
        "genotype": ["A123B", "C456D"],
        "titrant_name": ["T1", "T1"],
        "titrant_conc": [0.1, 0.1],
        "theta_obs": [0.5, 0.6],
        "theta_std": [0.05, 0.05]
    })
    binding_path = os.path.join(tmpdir, "test_binding.csv")
    binding_df.to_csv(binding_path, index=False)

    out_root = os.path.join(tmpdir, "test_tfs")

    # Run configuration
    configure_growth_analysis(growth_df=growth_path,
                              binding_df=binding_path,
                              out_root=out_root)

    config_file = f"{out_root}_config.yaml"
    priors_file = f"{out_root}_priors.csv"
    guesses_file = f"{out_root}_guesses.csv"

    # Verify files exist
    assert os.path.exists(config_file)
    assert os.path.exists(priors_file)
    assert os.path.exists(guesses_file)

    # Verify YAML content
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    assert "priors_file" in config
    assert "guesses_file" in config
    assert "priors" not in config
    assert "init_params" not in config

    # Run analysis (smoke test)
    # We use max_num_epochs=1 to make it fast
    # We use always_get_posterior=True to trigger summarize_posteriors
    run_growth_analysis(config_file=config_file,
                        seed=42,
                        max_num_epochs=1,
                        num_posterior_samples=10,
                        sampling_batch_size=10,
                        always_get_posterior=True,
                        out_root=os.path.join(tmpdir, "test_tfs_out"))

    # Verify outputs from summarize_posteriors
    assert os.path.exists(os.path.join(tmpdir, "test_tfs_out_growth_pred.csv"))
    assert os.path.exists(os.path.join(tmpdir, "test_tfs_out_hill_n.csv"))
    assert os.path.exists(os.path.join(tmpdir, "test_tfs_out_posterior.h5"))
