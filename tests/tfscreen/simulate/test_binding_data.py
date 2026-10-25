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


def test_generate_binding_data_noise_unclipped_by_default():
    """
    Noisy theta_obs is not clipped to [0, 1] unless clip_theta_obs is set.

    The fit's binding likelihood is an unclipped Normal; clipped anchors
    near 0 and 1 are biased (SVI-overconfidence fix grid, 2026-09-29).
    """
    binding_theta_df = pd.DataFrame(
        [{"genotype": "wt", "titrant_conc": c, "theta_true": t}
         for c, t in [(0.0, 0.999), (0.1, 0.5), (1.0, 0.001)]])
    cfg = {"titrant_name": "iptg", "titrant_conc": [0.0, 0.1, 1.0],
           "noise": 0.3}
    obs = []
    for clip in (False, True):
        res = generate_binding_df({**cfg, "clip_theta_obs": clip},
                                  np.random.default_rng(1), binding_theta_df)
        obs.append(res["theta_obs"].to_numpy())
    assert ((obs[0] < 0) | (obs[0] > 1)).any()
    np.testing.assert_allclose(obs[1], np.clip(obs[0], 0, 1))
