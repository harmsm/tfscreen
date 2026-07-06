"""Tests for tfscreen.simulate.transformation_lam_output.generate_transformation_lam_df."""

import pytest

from tfscreen.simulate.transformation_lam_output import generate_transformation_lam_df


def test_configured_float_value():
    cf = {"transformation_poisson_lambda": 1.5}
    result = generate_transformation_lam_df(cf)
    assert len(result) == 1
    assert result.loc[0, "parameter"] == "lam"
    assert result.loc[0, "ref"] == pytest.approx(1.5)


def test_none_value_gives_zero_ref():
    cf = {"transformation_poisson_lambda": None}
    result = generate_transformation_lam_df(cf)
    assert result.loc[0, "ref"] == pytest.approx(0.0)


def test_zero_value_gives_zero_ref():
    cf = {"transformation_poisson_lambda": 0}
    result = generate_transformation_lam_df(cf)
    assert result.loc[0, "ref"] == pytest.approx(0.0)


def test_missing_key_gives_zero_ref():
    cf = {}
    result = generate_transformation_lam_df(cf)
    assert result.loc[0, "ref"] == pytest.approx(0.0)


def test_string_coercible_value():
    cf = {"transformation_poisson_lambda": "2.5"}
    result = generate_transformation_lam_df(cf)
    assert result.loc[0, "ref"] == pytest.approx(2.5)
    assert isinstance(result.loc[0, "ref"], float)


def test_output_columns():
    cf = {"transformation_poisson_lambda": 1.5}
    result = generate_transformation_lam_df(cf)
    assert set(result.columns) == {"parameter", "ref"}
