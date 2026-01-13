import jax.numpy as jnp
import numpyro
import pytest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from tfscreen.analysis.hierarchical.analyze_theta import analyze_theta

def create_mock_data():
    # Create simple growth data
    # 3 genotypes: wt, A1B, C2D, A1B/C2D
    # Required columns in _read_growth_df: 
    # ["genotype", "titrant_name", "condition_pre", "condition_sel", "titrant_conc", 
    #  "ln_cfu", "ln_cfu_std", "replicate", "t_pre", "t_sel"]
    growth_data = {
        "replicate": ["rep1"] * 8,
        "t_pre": [0.0, 12.0] * 4,
        "t_sel": [12.0, 24.0] * 4, # dummy sel times
        "condition_pre": ["pre"] * 8,
        "condition_sel": ["sel"] * 8,
        "titrant_name": ["titrant1"] * 8,
        "titrant_conc": [0.0] * 8,
        "genotype": ["wt", "wt", "A1B", "A1B", "C2D", "C2D", "A1B/C2D", "A1B/C2D"],
        "cfu": [100.0, 1000.0, 100.0, 800.0, 100.0, 600.0, 100.0, 400.0],
        "cfu_std": [10.0, 100.0, 10.0, 80.0, 10.0, 60.0, 10.0, 40.0]
    }
    growth_df = pd.DataFrame(growth_data)
    growth_df["ln_cfu"] = np.log(growth_df["cfu"])
    growth_df["ln_cfu_std"] = growth_df["cfu_std"] / growth_df["cfu"]
    
    # Create simple binding data
    # Required columns in _read_binding_df:
    # ["genotype", "titrant_name", "theta_obs", "theta_std", "titrant_conc"]
    binding_data = {
        "titrant_name": ["titrant1"] * 8,
        "titrant_conc": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "genotype": ["wt", "wt", "A1B", "A1B", "C2D", "C2D", "A1B/C2D", "A1B/C2D"],
        "theta_obs": [0.1, 0.5, 0.1, 0.4, 0.1, 0.3, 0.1, 0.2],
        "theta_std": [0.01] * 8
    }
    binding_df = pd.DataFrame(binding_data)
    
    return growth_df, binding_df

@pytest.mark.parametrize("mode", ["genotype", "horseshoe", "spikeslab", "none"])
def test_epistasis_modes_smoke(mode):
    growth_df, binding_df = create_mock_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = os.path.join(tmpdir, "test")
        
        # Run analyze_theta in the specified mode
        # Using very few epochs for a smoke test
        svi_state, svi_params, converged = analyze_theta(
            growth_df=growth_df,
            binding_df=binding_df,
            seed=0,
            epistasis_mode=mode,
            max_num_epochs=5,
            batch_size=1024,
            out_root=out_root,
            analysis_method="svi"
        )
        
        assert svi_params is not None
        # Check for model-specific parameters
        param_names = list(svi_params.keys())
        if mode == "horseshoe":
            assert any("epi_tau" in k for k in param_names)
        elif mode == "spikeslab":
            assert any("epi_prob" in k for k in param_names)
        elif mode == "none":
            # In 'none' mode, we shouldn't see epi parameters for dk_geno or internal samplers
            assert not any("epi_tau" in k for k in param_names)
            assert not any("epi_prob" in k for k in param_names)

if __name__ == "__main__":
    pytest.main([__file__])
