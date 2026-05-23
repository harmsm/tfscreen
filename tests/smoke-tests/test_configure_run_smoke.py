import pytest
import pandas as pd
import os
import yaml
from tfscreen.analysis.hierarchical.growth_model.scripts.configure_model_cli import configure_model
from tfscreen.analysis.hierarchical.growth_model.scripts.fit_model_cli import fit_model
from tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors_cli import summarize_posteriors

@pytest.mark.slow
def test_configure_run_binding_only_smoke(tmpdir):
    """
    Smoke test for the binding-only pipeline (no growth data).

    Verifies that configure -> run -> summarize completes without error when
    theta is inferred directly from observed theta values rather than from
    bacterial growth data.
    """
    # Minimal binding dataset: two genotypes, two titrant concentrations
    binding_df = pd.DataFrame({
        "genotype": ["wt", "wt", "A123B", "A123B"],
        "titrant_name": ["T1", "T1", "T1", "T1"],
        "titrant_conc": [0.1, 1.0, 0.1, 1.0],
        "theta_obs": [0.8, 0.3, 0.7, 0.2],
        "theta_std": [0.05, 0.05, 0.05, 0.05],
    })
    binding_path = os.path.join(tmpdir, "test_binding.csv")
    binding_df.to_csv(binding_path, index=False)

    out_prefix = os.path.join(tmpdir, "test_tfs_binding_only")

    configure_model(growth_df=None,
                              binding_df=binding_path,
                              binding_only=True,
                              theta_model="categorical",
                              out_prefix=out_prefix)

    config_file = f"{out_prefix}_config.yaml"
    priors_file = f"{out_prefix}_priors.csv"
    guesses_file = f"{out_prefix}_guesses.csv"

    assert os.path.exists(config_file)
    assert os.path.exists(priors_file)
    assert os.path.exists(guesses_file)

    # Confirm the YAML records no growth data path
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    assert "growth" not in config["data"]
    assert config["components"]["binding_only"] is True

    out_prefix = os.path.join(tmpdir, "test_tfs_binding_only_out")
    fit_model(config_file=config_file,
                        seed=42,
                        max_num_epochs=1,
                        num_posterior_samples=10,
                        sampling_batch_size=10,
                        always_get_posterior=True,
                        out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


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

    out_prefix = os.path.join(tmpdir, "test_tfs")

    # Run configuration
    configure_model(growth_df=growth_path,
                              binding_df=binding_path,
                              out_prefix=out_prefix)

    config_file = f"{out_prefix}_config.yaml"
    priors_file = f"{out_prefix}_priors.csv"
    guesses_file = f"{out_prefix}_guesses.csv"

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
    # We use always_get_posterior=True to ensure the posterior file is written
    out_prefix = os.path.join(tmpdir, "test_tfs_out")
    fit_model(config_file=config_file,
                        seed=42,
                        max_num_epochs=1,
                        num_posterior_samples=10,
                        sampling_batch_size=10,
                        always_get_posterior=True,
                        out_prefix=out_prefix)

    assert os.path.exists(os.path.join(tmpdir, "test_tfs_out_posterior.h5"))

    # Summarize posteriors as a separate step
    summarize_posteriors(config_file=config_file,
                         posterior_file=f"{out_prefix}_posterior.h5",
                         out_prefix=out_prefix)

    assert os.path.exists(os.path.join(tmpdir, "test_tfs_out_hill_n.csv"))
