import pytest
import numpy as np
import pandas as pd

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.prior_predictive import draw_prior, growth_df_from_prior


@pytest.fixture
def dummy_mc():
    growth_df = pd.DataFrame({
        "genotype":      ["wt", "wt", "M42V", "M42V"],
        "titrant_name":  ["tit1", "tit1", "tit1", "tit1"],
        "titrant_conc":  [0.0, 1.0, 0.0, 1.0],
        "condition_pre": ["pre1", "pre1", "pre1", "pre1"],
        "condition_sel": ["sel1", "sel1", "sel1", "sel1"],
        "t_pre":         [10.0, 10.0, 10.0, 10.0],
        "t_sel":         [0.0, 20.0, 0.0, 20.0],
        "ln_cfu":        [0.0, 5.0, 0.0, 3.0],
        "ln_cfu_std":    [0.1, 0.1, 0.1, 0.1],
        "replicate":     [1, 1, 1, 1],
    })
    binding_df = pd.DataFrame({
        "genotype":     ["wt", "M42V"],
        "titrant_name": ["tit1", "tit1"],
        "titrant_conc": [0.5, 0.5],
        "theta_obs":    [0.5, 0.2],
        "theta_std":    [0.01, 0.01],
    })
    return ModelOrchestrator(growth_df, binding_df)


# ---------------------------------------------------------------------------
# draw_prior
# ---------------------------------------------------------------------------

def test_draw_prior_returns_dicts(dummy_mc):
    predictions, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    assert isinstance(predictions, dict)
    assert isinstance(latent_params, dict)


def test_draw_prior_contains_growth_pred(dummy_mc):
    predictions, _ = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    assert "growth_pred" in predictions


def test_draw_prior_growth_pred_shape(dummy_mc):
    predictions, _ = draw_prior(dummy_mc, rng_key=0, num_draws=3)
    gp = predictions["growth_pred"]
    assert gp.shape[0] == 3, "First dim should equal num_draws"
    assert gp.ndim > 1, "growth_pred should have spatial dims beyond num_draws"


def test_draw_prior_latent_params_shape(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=3)
    for name, val in latent_params.items():
        assert val.shape[0] == 3, f"Parameter '{name}' first dim should equal num_draws"


def test_draw_prior_no_observed_in_latent(dummy_mc):
    # Observed sites (obs= in model) should not appear in latent_params.
    # The growth observation site is named 'final_binding_obs_growth_obs'.
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    assert "final_binding_obs_growth_obs" not in latent_params


def test_draw_prior_reproducible(dummy_mc):
    _, p1 = draw_prior(dummy_mc, rng_key=42, num_draws=1)
    _, p2 = draw_prior(dummy_mc, rng_key=42, num_draws=1)
    for k in p1:
        np.testing.assert_array_equal(p1[k], p2[k])


def test_draw_prior_different_seeds(dummy_mc):
    _, p1 = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    _, p2 = draw_prior(dummy_mc, rng_key=1, num_draws=1)
    # At least one parameter should differ across seeds.
    any_diff = any(not np.array_equal(p1[k], p2[k]) for k in p1 if k in p2)
    assert any_diff


def test_draw_prior_multiple_independent_draws(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=5)
    # Draws should not all be identical.
    first_key = next(iter(latent_params))
    v = latent_params[first_key]
    if v.shape[0] > 1 and v.size > v.shape[0]:
        assert not np.all(v[0] == v[1])


# ---------------------------------------------------------------------------
# growth_df_from_prior
# ---------------------------------------------------------------------------

def test_growth_df_from_prior_returns_dataframe(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    df = growth_df_from_prior(dummy_mc, latent_params)
    assert isinstance(df, pd.DataFrame)


def test_growth_df_from_prior_same_rows(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    df = growth_df_from_prior(dummy_mc, latent_params)
    assert len(df) == len(dummy_mc.growth_df)


def test_growth_df_from_prior_has_ln_cfu(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    df = growth_df_from_prior(dummy_mc, latent_params)
    assert "ln_cfu" in df.columns
    assert df["ln_cfu"].notna().all()


def test_growth_df_from_prior_preserves_metadata(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    df = growth_df_from_prior(dummy_mc, latent_params)
    for col in ["genotype", "titrant_conc", "ln_cfu_std", "replicate"]:
        assert col in df.columns


def test_growth_df_from_prior_noise_changes_ln_cfu(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=1)
    df_clean = growth_df_from_prior(dummy_mc, latent_params, noise_rng=None)
    rng = np.random.default_rng(7)
    df_noisy = growth_df_from_prior(dummy_mc, latent_params, noise_rng=rng)
    # Noisy and clean should differ (adding N(0, 0.1) noise almost certainly changes values).
    assert not np.allclose(df_clean["ln_cfu"].values, df_noisy["ln_cfu"].values)


def test_growth_df_from_prior_different_draws(dummy_mc):
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=5)
    df0 = growth_df_from_prior(dummy_mc, latent_params, draw_idx=0)
    df1 = growth_df_from_prior(dummy_mc, latent_params, draw_idx=1)
    # Different draws should generally produce different predictions.
    assert not np.allclose(df0["ln_cfu"].values, df1["ln_cfu"].values)


def test_latent_params_compatible_with_predict(dummy_mc):
    """latent_params from draw_prior can be passed to predict() without error."""
    from tfscreen.tfmodel.analysis.prediction import predict
    _, latent_params = draw_prior(dummy_mc, rng_key=0, num_draws=2)
    result = predict(dummy_mc, latent_params, predict_sites=["growth_pred"],
                     num_samples=None, num_marginal_samples=2)
    assert "median" in result.columns or any(c in result.columns
                                             for c in ["lower_95", "upper_95"])
