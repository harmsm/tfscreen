"""
Tests for resolve_obs_columns -- quantile-column defaults for y_obs / y_std.
"""
import pytest
import pandas as pd

from tfscreen.util import resolve_obs_columns


def _quantile_df():
    return pd.DataFrame({
        "genotype": ["wt", "A15G"],
        "q0.159": [0.4, 1.4],
        "q0.5": [0.5, 1.5],
        "q0.841": [0.6, 1.8],
    })


def test_explicit_names_pass_through_unchanged():
    df = pd.DataFrame({"genotype": ["wt"], "y": [1.0], "err": [0.1]})
    out_df, y_obs, y_std = resolve_obs_columns(df, y_obs="y", y_std="err")

    assert y_obs == "y"
    assert y_std == "err"
    # No sigma column added; original frame returned untouched.
    assert out_df is df
    assert "_sigma" not in out_df.columns


def test_y_obs_defaults_to_median_quantile():
    df = _quantile_df()
    out_df, y_obs, y_std = resolve_obs_columns(df)

    assert y_obs == "q0.5"
    assert y_std == "_sigma"


def test_sigma_is_symmetric_half_width():
    df = _quantile_df()
    out_df, y_obs, y_std = resolve_obs_columns(df)

    # (q0.841 - q0.159) / 2
    assert out_df["_sigma"].tolist() == pytest.approx([0.1, 0.2])


def test_original_frame_not_mutated_when_sigma_added():
    df = _quantile_df()
    out_df, _, _ = resolve_obs_columns(df)

    assert "_sigma" in out_df.columns
    assert "_sigma" not in df.columns  # copy-on-write, original untouched
    assert out_df is not df


def test_explicit_y_obs_but_default_y_std():
    df = _quantile_df()
    out_df, y_obs, y_std = resolve_obs_columns(df, y_obs="q0.5")

    assert y_obs == "q0.5"
    assert y_std == "_sigma"


def test_no_sigma_columns_leaves_y_std_none():
    df = pd.DataFrame({"genotype": ["wt"], "q0.5": [0.5]})
    out_df, y_obs, y_std = resolve_obs_columns(df)

    assert y_obs == "q0.5"
    assert y_std is None
    assert out_df is df


def test_only_one_sigma_column_leaves_y_std_none():
    # Only the upper bound present -> cannot form a half-width.
    df = pd.DataFrame({"genotype": ["wt"], "q0.5": [0.5], "q0.841": [0.6]})
    _, y_obs, y_std = resolve_obs_columns(df)

    assert y_obs == "q0.5"
    assert y_std is None


def test_missing_point_quantile_raises():
    df = pd.DataFrame({"genotype": ["wt"], "value": [0.5]})
    with pytest.raises(ValueError, match="No 'y_obs' column"):
        resolve_obs_columns(df)


def test_custom_quantiles():
    df = pd.DataFrame({
        "genotype": ["wt"],
        "q0.25": [0.3],
        "q0.75": [0.7],
    })
    out_df, y_obs, y_std = resolve_obs_columns(
        df, point_quantile=0.25, sigma_quantiles=(0.25, 0.75)
    )

    assert y_obs == "q0.25"
    assert y_std == "_sigma"
    assert out_df["_sigma"].iloc[0] == pytest.approx((0.7 - 0.3) / 2)


def test_custom_sigma_col_name():
    df = _quantile_df()
    out_df, _, y_std = resolve_obs_columns(df, sigma_col="theta_std")

    assert y_std == "theta_std"
    assert "theta_std" in out_df.columns
