import pytest
import os
import yaml
import numpy as np
import pandas as pd
import sys
from unittest.mock import MagicMock, patch
from tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors import summarize_posteriors, main

@pytest.fixture
def mock_config():
    return {
        "settings": {
            "condition_growth": "hierarchical",
            "ln_cfu0": "hierarchical",
            "dk_geno": "hierarchical",
            "activity": "hierarchical",
            "theta": "hill",
            "transformation": "none",
            "theta_growth_noise": "none",
            "theta_binding_noise": "none",
            "spiked_genotypes": None
        },
        "growth_df": "growth.csv",
        "binding_df": "binding.csv"
    }

def test_summarize_posteriors_npz(tmpdir, mock_config):
    config_file = os.path.join(tmpdir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    posterior_file = os.path.join(tmpdir, "post.npz")
    np.savez(posterior_file, a=np.array([1]))

    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.GrowthModel") as MockGM, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.extract_parameters") as mock_extract_params, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.read_configuration") as mock_read:

        mock_extract_params.return_value = {"param1": pd.DataFrame({"x": [1]})}
        gm = MockGM.return_value
        gm.settings = {"theta": "hill"}
        mock_read.return_value = (gm, {})

        out_prefix = os.path.join(tmpdir, "tfs")
        summarize_posteriors(config_file, posterior_file, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_param1.csv")
    assert not os.path.exists(f"{out_prefix}_growth_pred.csv")
    assert not os.path.exists(f"{out_prefix}_theta_curves.csv")

def test_summarize_posteriors_h5(tmpdir, mock_config):
    import h5py
    config_file = os.path.join(tmpdir, "config_h5.yaml")
    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    posterior_file = os.path.join(tmpdir, "post.h5")
    with h5py.File(posterior_file, 'w') as f:
        f.create_dataset("a", data=np.array([1]))

    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.GrowthModel") as MockGM, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.extract_parameters") as mock_extract_params, \
         patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.read_configuration") as mock_read:

        mock_extract_params.return_value = {"param1": pd.DataFrame({"x": [1]})}
        gm = MockGM.return_value
        gm.settings = {"theta": "hill"}
        mock_read.return_value = (gm, {})

        out_prefix = os.path.join(tmpdir, "tfs_h5")
        summarize_posteriors(config_file, posterior_file, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_param1.csv")

def test_summarize_posteriors_errors():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        summarize_posteriors("nonexistent.yaml", "p.npz")

    with patch("os.path.exists", side_effect=lambda x: x == "config.yaml"):
        with patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.read_configuration") as mock_read:
            mock_read.return_value = (MagicMock(), {})
            with pytest.raises(FileNotFoundError, match="Posterior file not found"):
                summarize_posteriors("config.yaml", "missing.npz")

def test_main():
    with patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.generalized_main") as mock_gen:
        main()
        mock_gen.assert_called_once_with(summarize_posteriors)

def test_entry_point():
    with patch.object(sys, 'argv', ['summarize_posteriors', '--help']):
        with patch("tfscreen.analysis.hierarchical.growth_model.scripts.summarize_posteriors.generalized_main") as mock_gen:
            try:
                main()
            except SystemExit:
                pass
            mock_gen.assert_called_once()
