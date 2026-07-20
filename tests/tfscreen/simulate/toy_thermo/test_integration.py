"""End-to-end: toy_thermo output feeds cat_response and extract_epistasis."""

import numpy as np

from tfscreen.simulate.toy_thermo import (
    ThermoModel, sample_effects, build_titration_df,
)
from tfscreen.analysis.cat_response.cat_response import cat_response
from tfscreen.analysis.extract_epistasis import extract_epistasis


def _df():
    model = ThermoModel(ln_K_conf=0.0, ln_K_dna=18.0, ln_K_eff=14.0,
                        protein_total=1e-6, dna_total=1e-9)
    grid = np.logspace(-8, -2, 13)
    # XsiteY names so extract_epistasis can parse the genotypes
    eff = sample_effects(["A1V", "A2V", "A3V"],
                         effect_sd={"HD": 1.5, "H": 0.5, "L": 0.5, "LE": 1.5},
                         epistasis_sd={"HD": 0.3, "LE": 0.3},
                         rng=np.random.default_rng(1))
    return build_titration_df(model, grid, effects=eff, observable_std=0.02)


def test_cat_response_consumes_titration_df():
    df = _df()
    res, pred, assess, delta = cat_response(
        df, x_obs="titrant_conc", y_obs="observable",
        y_std="observable_std", progress=False)
    # one row per genotype (wt + 3 singles + 3 doubles)
    assert len(res) == 7
    assert "best_model" in res.columns
    assert "fittable" in res.columns


def test_extract_epistasis_consumes_titration_df():
    df = _df()
    ep = extract_epistasis(df, y_obs="observable", y_std="observable_std",
                           group_by="titrant_conc")
    assert {"genotype", "titrant_conc", "ep_obs", "ep_std"} <= set(ep.columns)
    # 3 double mutants x 13 concentrations
    assert len(ep) == 3 * 13
    assert np.all(np.isfinite(ep["ep_obs"]))
