"""Tests for tfscreen.simulate.growth_parameters_output.generate_growth_parameters_df."""

import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.growth_parameters_output import generate_growth_parameters_df


def test_linear_model_maps_b_and_m():
    growth_cfg = {
        "kanR+kan": {"model": "linear", "b": 0.005, "m": -0.01},
    }
    result = generate_growth_parameters_df(growth_cfg).set_index("condition_rep")
    assert result.loc["kanR+kan", "growth_k"] == pytest.approx(0.005)
    assert result.loc["kanR+kan", "growth_m"] == pytest.approx(-0.01)


def test_linear_is_default_model_when_omitted():
    growth_cfg = {"kanR-kan": {"b": 0.02, "m": 0.001}}
    result = generate_growth_parameters_df(growth_cfg).set_index("condition_rep")
    assert result.loc["kanR-kan", "growth_k"] == pytest.approx(0.02)
    assert result.loc["kanR-kan", "growth_m"] == pytest.approx(0.001)


def test_power_model_maps_b_a_n():
    growth_cfg = {
        "M9+kan_hi": {"model": "power", "b": 0.001, "a": 0.04, "n": 2.0},
    }
    result = generate_growth_parameters_df(growth_cfg).set_index("condition_rep")
    assert result.loc["M9+kan_hi", "growth_k"] == pytest.approx(0.001)
    assert result.loc["M9+kan_hi", "growth_m"] == pytest.approx(0.04)
    assert result.loc["M9+kan_hi", "growth_n"] == pytest.approx(2.0)


def test_saturation_model_maps_kmin_kmax():
    growth_cfg = {
        "M9+kan": {"model": "saturation", "kmin": 0.001, "kmax": 0.04},
    }
    result = generate_growth_parameters_df(growth_cfg).set_index("condition_rep")
    assert result.loc["M9+kan", "growth_min"] == pytest.approx(0.001)
    assert result.loc["M9+kan", "growth_max"] == pytest.approx(0.04)


def test_multiple_conditions_one_row_each():
    growth_cfg = {
        "kanR+kan": {"model": "linear", "b": 0.005, "m": -0.01},
        "kanR-kan": {"model": "linear", "b": 0.02, "m": 0.001},
    }
    result = generate_growth_parameters_df(growth_cfg)
    assert set(result["condition_rep"]) == {"kanR+kan", "kanR-kan"}
    assert len(result) == 2


def test_mixed_models_nan_fill_unused_columns():
    """A run mixing linear and saturation conditions produces NaN for the
    columns that don't apply to a given condition's model."""
    growth_cfg = {
        "linear_cond": {"model": "linear", "b": 0.005, "m": -0.01},
        "sat_cond": {"model": "saturation", "kmin": 0.001, "kmax": 0.04},
    }
    result = generate_growth_parameters_df(growth_cfg).set_index("condition_rep")
    assert result.loc["linear_cond", "growth_k"] == pytest.approx(0.005)
    assert np.isnan(result.loc["linear_cond", "growth_min"])
    assert result.loc["sat_cond", "growth_min"] == pytest.approx(0.001)
    assert np.isnan(result.loc["sat_cond", "growth_k"])


def test_unknown_model_raises():
    growth_cfg = {"cond": {"model": "quadratic", "b": 1.0}}
    with pytest.raises(ValueError, match="Unknown growth model"):
        generate_growth_parameters_df(growth_cfg)


def test_missing_required_key_raises():
    growth_cfg = {"cond": {"model": "linear", "b": 0.005}}  # missing 'm'
    with pytest.raises(ValueError, match="missing required parameter"):
        generate_growth_parameters_df(growth_cfg)


def test_empty_growth_cfg_returns_empty_df():
    result = generate_growth_parameters_df({})
    assert len(result) == 0


def test_original_growth_cfg_not_mutated():
    """The 'model' key must not be popped from the caller's dict."""
    growth_cfg = {"cond": {"model": "linear", "b": 0.005, "m": -0.01}}
    generate_growth_parameters_df(growth_cfg)
    assert growth_cfg["cond"]["model"] == "linear"
    assert "b" in growth_cfg["cond"]
