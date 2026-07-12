"""Tests for tfscreen.simulate.binding_data.generate_binding_df."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.binding_data import generate_binding_df


def test_generate_binding_data_uses_precomputed_theta():
    """generate_binding_df must use theta_true from binding_theta_df,
    not re-sample from the prior."""
    binding_cfg = {
        "genotypes": ["wt", "M1A"],
        "titrant_name": "iptg",
        "titrant_conc": [0.0, 1.0],
        "noise": 0.0,
    }
    rng = np.random.default_rng(0)
    binding_theta_df = pd.DataFrame([
        {"genotype": "wt",  "titrant_conc": 0.0, "theta_true": 0.1},
        {"genotype": "wt",  "titrant_conc": 1.0, "theta_true": 0.9},
        {"genotype": "M1A", "titrant_conc": 0.0, "theta_true": 0.2},
        {"genotype": "M1A", "titrant_conc": 1.0, "theta_true": 0.8},
    ])

    result = generate_binding_df(binding_cfg, rng, binding_theta_df)

    assert set(result.columns) >= {"genotype", "titrant_name", "titrant_conc",
                                   "theta_obs", "theta_std"}
    wt_row = result[(result["genotype"] == "wt") & (result["titrant_conc"] == 1.0)]
    assert float(wt_row["theta_obs"].iloc[0]) == pytest.approx(0.9)


def test_generate_binding_data_missing_genotype_raises():
    """Raises ValueError when a genotype/conc pair is absent from binding_theta_df."""
    binding_cfg = {
        "genotypes": ["wt", "A2V"],   # A2V is valid but not in binding_theta_df
        "titrant_name": "iptg",
        "titrant_conc": [1.0],
        "noise": 0.0,
    }
    rng = np.random.default_rng(0)
    binding_theta_df = pd.DataFrame([
        {"genotype": "wt", "titrant_conc": 1.0, "theta_true": 0.5},
    ])

    with pytest.raises(ValueError, match="No pre-computed theta"):
        generate_binding_df(binding_cfg, rng, binding_theta_df)
