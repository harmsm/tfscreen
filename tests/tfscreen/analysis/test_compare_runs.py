"""
Unit tests for tfscreen.analysis.compare_runs.
"""

import numpy as np
import pandas as pd
import pytest

from tfscreen.analysis.compare_runs import (
    compare_runs,
    aggregate_runs,
    detect_match_keys,
    detect_index_by,
    resolve_schema,
    shared_quantile_levels,
    _mixture_quantiles,
)

# Standard quantile ladder used by the aggregate builders/tests.
_LEVELS = [0.025, 0.159, 0.5, 0.841, 0.975]


# --- Builders ----------------------------------------------------------------

def make_estimate(theta_by_geno, concs=(0.0, 1.0), sigma=0.05,
                  titrant_name=None):
    """
    Build one estimate table with a minimal (q0.159, q0.5, q0.841) schema.

    Parameters
    ----------
    theta_by_geno : dict[str, sequence]
        genotype -> q0.5 values, one per concentration.
    concs : sequence
        Titrant concentrations (len must match each value sequence).
    sigma : float or dict[str, sequence]
        1-sigma half-width; scalar (shared) or per-genotype per-conc.
    titrant_name : str or None
        If given, a ``titrant_name`` column is added with this constant value.
    """
    rows = []
    for geno, thetas in theta_by_geno.items():
        for j, conc in enumerate(concs):
            th = thetas[j]
            sg = sigma[geno][j] if isinstance(sigma, dict) else sigma
            row = {"genotype": geno, "titrant_conc": conc,
                   "q0.159": th - sg, "q0.5": th, "q0.841": th + sg}
            if titrant_name is not None:
                row["titrant_name"] = titrant_name
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["genotype"]
    if titrant_name is not None:
        cols.append("titrant_name")
    cols += ["titrant_conc", "q0.159", "q0.5", "q0.841"]
    return df.loc[:, cols]


def make_estimate_ladder(theta_by_geno, concs=(0.0, 1.0), sigma=0.05,
                         titrant_name=None, levels=_LEVELS):
    """
    Build an estimate table with a full (Gaussian) quantile ladder.

    Each (genotype, conc) gets ``q0.5 = theta`` and the other levels placed at
    ``theta + z(level) * sigma`` (Normal), clamped to [0, 1].
    """
    from scipy.stats import norm
    zs = {lvl: norm.ppf(lvl) for lvl in levels}
    rows = []
    for geno, thetas in theta_by_geno.items():
        for j, conc in enumerate(concs):
            th = thetas[j]
            sg = sigma[geno][j] if isinstance(sigma, dict) else sigma
            row = {"genotype": geno, "titrant_conc": conc}
            if titrant_name is not None:
                row["titrant_name"] = titrant_name
            for lvl in levels:
                row[f"q{lvl}"] = float(np.clip(th + zs[lvl] * sg, 0.0, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def make_param_table(values, key_cols, sigma=0.1):
    """
    Build a parameter-file-shaped table (arbitrary identity columns, one value).

    Parameters
    ----------
    values : list of (dict, float)
        Each entry is (identity-column values, q0.5 value).
    key_cols : list of str
        Identity column order.
    """
    rows = []
    for keys, v in values:
        row = {c: keys[c] for c in key_cols}
        row.update({"q0.159": v - sigma, "q0.5": v, "q0.841": v + sigma})
        rows.append(row)
    return pd.DataFrame(rows)


def get_row(result, **keys):
    """Return the single result row matching all ``keys`` as a Series."""
    mask = np.ones(len(result), dtype=bool)
    for k, v in keys.items():
        mask &= (result[k] == v).to_numpy()
    sub = result.loc[mask]
    assert len(sub) == 1, f"expected 1 row for {keys}, got {len(sub)}"
    return sub.iloc[0]


# --- Key detection -----------------------------------------------------------

def test_detect_match_keys_theta_table():
    df = make_estimate({"wt": [0.1, 0.2]})
    assert detect_match_keys([df, df]) == ["genotype", "titrant_conc"]


def test_detect_match_keys_with_titrant_name():
    df = make_estimate({"wt": [0.1, 0.2]}, titrant_name="iptg")
    assert detect_match_keys([df, df]) == ["genotype", "titrant_name",
                                           "titrant_conc"]


def test_detect_match_keys_excludes_incidental_columns():
    df = make_estimate({"wt": [0.1, 0.2]})
    df = df.assign(in_training_data=1, in_regime=1)
    df["Unnamed: 0"] = range(len(df))
    assert detect_match_keys([df, df]) == ["genotype", "titrant_conc"]


def test_detect_match_keys_excludes_value_columns():
    df = pd.DataFrame({"genotype": ["wt"], "est": [0.1], "se": [0.01]})
    assert detect_match_keys([df, df],
                             value_columns=["est", "se"]) == ["genotype"]


def test_detect_match_keys_uses_only_shared_columns():
    a = make_estimate({"wt": [0.1, 0.2]}, titrant_name="iptg")
    b = make_estimate({"wt": [0.1, 0.2]})
    assert detect_match_keys([a, b]) == ["genotype", "titrant_conc"]


def test_detect_match_keys_override_validated():
    df = make_estimate({"wt": [0.1, 0.2]})
    assert detect_match_keys([df, df], match_by=["genotype"]) == ["genotype"]
    with pytest.raises(ValueError, match="not present in every"):
        detect_match_keys([df, df], match_by=["nope"])


def test_detect_match_keys_all_values_raises():
    df = pd.DataFrame({"q0.5": [0.1], "q0.159": [0.0], "q0.841": [0.2]})
    with pytest.raises(ValueError, match="Could not auto-detect any match-key"):
        detect_match_keys([df, df])


@pytest.mark.parametrize("keys,expected", [
    (["genotype", "titrant_conc"], "genotype"),
    (["parameter"], "parameter"),
    (["condition_rep"], "condition_rep"),
    (["genotype", "parameter"], "genotype"),
])
def test_detect_index_by_auto(keys, expected):
    assert detect_index_by(keys) == expected


def test_detect_index_by_ambiguous_raises():
    with pytest.raises(ValueError, match="Could not auto-detect the entity"):
        detect_index_by(["replicate", "condition_rep"])


def test_detect_index_by_override_must_be_a_match_key():
    assert detect_index_by(["replicate", "condition_rep"],
                           index_by="condition_rep") == "condition_rep"
    with pytest.raises(ValueError, match="is not one of the match-key"):
        detect_index_by(["replicate"], index_by="genotype")


def test_group_by_must_be_in_match_key():
    df = make_estimate({"wt": [0.1, 0.2]})
    with pytest.raises(ValueError, match="not part of the match key"):
        compare_runs([df, df.copy()], group_by=["nope"])


def test_group_by_may_not_repeat_index_by():
    df = make_estimate({"wt": [0.1, 0.2]})
    with pytest.raises(ValueError, match="must not repeat index_by"):
        compare_runs([df, df.copy()], group_by=["genotype"])


def test_resolve_schema_reports_all_three_key_sets():
    df = make_estimate({"wt": [0.1, 0.2]}, titrant_name="iptg")
    schema = resolve_schema([df, df.copy()], group_by=["titrant_name"])
    assert schema["match_by"] == ["genotype", "titrant_name", "titrant_conc"]
    assert schema["index_by"] == "genotype"
    assert schema["group_by"] == ["titrant_name"]
    assert schema["report_keys"] == ["genotype", "titrant_name"]
    assert schema["residual"] == ["titrant_conc"]
    assert schema["y_obs"] == "q0.5"
    assert schema["y_std"] == "_sigma"


# --- Real parameter-file shapes ----------------------------------------------

def test_dk_geno_shape_genotype_only():
    """*_params_dk_geno.csv: genotype is the only identity column."""
    runs = [make_param_table([({"genotype": "wt"}, 0.0 + 0.02 * i),
                              ({"genotype": "m1"}, -0.3)],
                             ["genotype"]) for i in range(3)]
    result = compare_runs(runs)
    assert list(result["genotype"]) == ["m1", "wt"]  # sorted by rms_sd
    row = get_row(result, genotype="wt")
    assert row["n_rows"] == 1
    # No residual axis -> dynamic_range undefined, not zero.
    assert np.isnan(row["dynamic_range"])


def test_hill_param_shape_genotype_and_titrant_name():
    """*_params_theta_log_hill_K.csv: genotype x titrant_name."""
    runs = []
    for i in range(3):
        runs.append(make_param_table(
            [({"genotype": g, "titrant_name": t}, v + 0.05 * i)
             for g, v in [("wt", -4.0), ("m1", -3.0)]
             for t in ("iptg", "onpf")],
            ["genotype", "titrant_name"],
        ))
    result = compare_runs(runs)
    assert set(result["genotype"]) == {"wt", "m1"}
    assert get_row(result, genotype="wt")["n_rows"] == 2

    # Breaking out by titrant gives one row per genotype x titrant.
    by_titrant = compare_runs(runs, group_by=["titrant_name"])
    assert len(by_titrant) == 4
    assert set(by_titrant.columns[:2]) == {"genotype", "titrant_name"}


def test_growth_param_shape_needs_explicit_index_by():
    """*_params_growth_k.csv: replicate x condition_rep, no genotype."""
    runs = [make_param_table(
        [({"replicate": r, "condition_rep": c}, 0.5 + 0.02 * i)
         for r in (1, 2) for c in ("kanR+kan", "pheS+4cp")],
        ["replicate", "condition_rep"],
    ) for i in range(3)]

    with pytest.raises(ValueError, match="Could not auto-detect the entity"):
        compare_runs(runs)

    result = compare_runs(runs, index_by="condition_rep")
    assert list(result.columns)[0] == "condition_rep"
    assert set(result["condition_rep"]) == {"kanR+kan", "pheS+4cp"}
    # Residual axis is `replicate`, so dynamic_range is defined.
    assert not np.isnan(get_row(result,
                                condition_rep="kanR+kan")["dynamic_range"])


def test_k_ref_shape_parameter_column():
    """*_params_k_ref.csv: a single `parameter` identity column."""
    runs = [make_param_table([({"parameter": "k_ref"}, 0.9 + 0.01 * i)],
                             ["parameter"]) for i in range(3)]
    result = compare_runs(runs)
    assert list(result["parameter"]) == ["k_ref"]
    assert result.iloc[0]["rms_sd"] == pytest.approx(0.01)


# --- Mean mode ---------------------------------------------------------------

def test_mean_mode_identical_runs_have_zero_spread():
    df = make_estimate({"wt": [0.1, 0.2], "m1": [0.5, 0.6]})
    result = compare_runs([df, df.copy(), df.copy()])
    assert (result["rms_sd"] == 0).all()
    assert (result["max_sd"] == 0).all()
    assert result["chi2"].to_numpy() == pytest.approx(0.0, abs=1e-12)


def test_mean_mode_half_range_for_small_n():
    # 3 runs (< _SMALL_N_CUTOFF) -> half-range estimator.
    runs = [make_estimate({"g": [v, v]}) for v in (0.10, 0.12, 0.14)]
    result = compare_runs(runs)
    assert result.iloc[0]["spread_estimator"] == "half_range"
    assert result.iloc[0]["rms_sd"] == pytest.approx((0.14 - 0.10) / 2)


def test_mean_mode_std_estimator_matches_ddof_one():
    """At uniform weights the weighted spread is exactly the ddof=1 std."""
    rng = np.random.default_rng(0)
    base = {"wt": [0.1, 0.2, 0.3], "m1": [0.5, 0.6, 0.7]}
    runs = [make_estimate({g: list(np.array(v) + rng.normal(0, 0.03, 3))
                           for g, v in base.items()},
                          concs=(0.0, 1.0, 10.0))
            for _ in range(6)]
    result = compare_runs(runs)
    assert result.iloc[0]["spread_estimator"] == "std"

    expected = (pd.concat(runs)
                .groupby(["genotype", "titrant_conc"])["q0.5"].std(ddof=1)
                .groupby("genotype")
                .apply(lambda s: np.sqrt(np.mean(s ** 2))))
    for geno, want in expected.items():
        assert get_row(result, genotype=geno)["rms_sd"] == pytest.approx(want)


def test_mean_value_and_dynamic_range():
    runs = [make_estimate({"g": [0.2, 0.8]}) for _ in range(3)]
    row = compare_runs(runs).iloc[0]
    assert row["mean_value"] == pytest.approx(0.5)
    assert row["dynamic_range"] == pytest.approx(0.6)


def test_flat_curve_has_zero_dynamic_range():
    runs = [make_estimate({"g": [0.5, 0.5]}) for _ in range(3)]
    assert compare_runs(runs).iloc[0]["dynamic_range"] == pytest.approx(0.0)


def test_result_sorted_by_rms_sd_ascending():
    runs = []
    for i in range(3):
        runs.append(make_estimate({"steady": [0.5, 0.5],
                                   "jumpy": [0.5 + 0.1 * i, 0.5]}))
    result = compare_runs(runs)
    assert list(result["genotype"]) == ["steady", "jumpy"]
    assert result["rms_sd"].is_monotonic_increasing


def test_mean_mode_requires_two_runs():
    df = make_estimate({"g": [0.1, 0.2]})
    with pytest.raises(ValueError, match="at least 2 estimate"):
        compare_runs([df])


def test_estimate_dfs_must_be_a_list():
    df = make_estimate({"g": [0.1, 0.2]})
    with pytest.raises(ValueError, match="must be a list"):
        compare_runs(df)


# --- Coverage ----------------------------------------------------------------

def test_n_present_and_n_runs_report_coverage():
    runs = [make_estimate({"a": [0.1, 0.2], "b": [0.5, 0.5]}) for _ in range(3)]
    runs[2] = runs[2].loc[runs[2]["genotype"] == "a"]
    result = compare_runs(runs)
    assert get_row(result, genotype="a")["n_present"] == 3
    assert get_row(result, genotype="b")["n_present"] == 2
    assert (result["n_runs"] == 3).all()


# --- group_by ----------------------------------------------------------------

def test_group_by_finest_gives_one_row_per_grid_point():
    runs = [make_estimate({"wt": [0.1 + 0.02 * i, 0.2], "m1": [0.5, 0.6]})
            for i in range(4)]
    result = compare_runs(runs, group_by=["titrant_conc"])
    assert len(result) == 4  # 2 genotypes x 2 concentrations
    assert (result["n_rows"] == 1).all()
    # No residual axis is left, so dynamic_range is undefined everywhere.
    assert result["dynamic_range"].isna().all()


def test_group_by_finest_dof_is_n_runs_minus_one():
    runs = [make_estimate({"wt": [0.1 + 0.02 * i, 0.2]}) for i in range(4)]
    result = compare_runs(runs, group_by=["titrant_conc"])
    assert (result["dof"] == 3).all()
    assert (result["n_eff"] == 4).all()


def test_pooled_dof_is_n_terms_minus_n_rows():
    runs = [make_estimate({"wt": [0.1 + 0.02 * i, 0.2]}) for i in range(4)]
    row = compare_runs(runs).iloc[0]
    assert row["n_rows"] == 2
    assert row["n_eff"] == 8       # 4 runs x 2 rows
    assert row["dof"] == 8 - 2     # one dof per row for its own mean


def test_group_by_intermediate_depth():
    """group_by titrant_name leaves titrant_conc as the residual axis."""
    runs = []
    for i in range(3):
        a = make_estimate({"g": [0.1 + 0.02 * i, 0.9]}, titrant_name="iptg")
        b = make_estimate({"g": [0.5, 0.5]}, titrant_name="onpf")
        runs.append(pd.concat([a, b], ignore_index=True))
    result = compare_runs(runs, group_by=["titrant_name"])
    assert len(result) == 2
    assert get_row(result, genotype="g",
                   titrant_name="onpf")["dynamic_range"] == pytest.approx(0.0)
    assert get_row(result, genotype="g",
                   titrant_name="iptg")["dynamic_range"] > 0.5


# --- Axis 2 ------------------------------------------------------------------

def test_overdispersion_near_one_when_spread_matches_sigma():
    # Runs disagree by exactly their reported sigma -> chi2/dof of order 1.
    rng = np.random.default_rng(3)
    sigma = 0.05
    runs = [make_estimate({"g": list(0.5 + rng.normal(0, sigma, 2))},
                          sigma=sigma) for _ in range(30)]
    row = compare_runs(runs).iloc[0]
    assert 0.5 < row["overdispersion"] < 2.0
    assert row["overdispersion_p"] > 0.01


def test_overdispersion_large_when_sigma_too_tight():
    # Big disagreement, tiny reported sigma -> overconfident.
    runs = [make_estimate({"g": [0.1 + 0.2 * i, 0.5]}, sigma=0.001)
            for i in range(3)]
    row = compare_runs(runs).iloc[0]
    assert row["overdispersion"] > 100
    assert row["overdispersion_p"] < 1e-6


def test_overdispersion_small_when_sigma_generous():
    runs = [make_estimate({"g": [0.1 + 0.001 * i, 0.5]}, sigma=0.5)
            for i in range(3)]
    row = compare_runs(runs).iloc[0]
    assert row["overdispersion"] < 0.1


def test_zero_sigma_observations_are_skipped():
    runs = [make_estimate({"g": [0.1 + 0.02 * i, 0.5]}, sigma=0.0)
            for i in range(3)]
    row = compare_runs(runs).iloc[0]
    # No valid chi-square terms at all -> Axis 2 is NaN, Axis 1 still reported.
    assert np.isnan(row["chi2"])
    assert np.isnan(row["overdispersion"])
    assert row["rms_sd"] > 0


def test_missing_sigma_reports_axis2_as_nan():
    runs = [pd.DataFrame({"genotype": ["wt", "m1"],
                          "q0.5": [0.1 + 0.02 * i, 0.5]}) for i in range(3)]
    result = compare_runs(runs)
    assert result["chi2"].isna().all()
    assert result["overdispersion"].isna().all()
    assert result["mean_reported_sigma"].isna().all()
    assert (result["rms_sd"] >= 0).all()


def test_overdispersion_q_is_bh_adjusted():
    rng = np.random.default_rng(11)
    genos = {f"g{i}": list(0.5 + rng.normal(0, 0.05, 2)) for i in range(20)}
    runs = [make_estimate({g: list(np.array(v) + rng.normal(0, 0.05, 2))
                           for g, v in genos.items()}, sigma=0.05)
            for _ in range(6)]
    result = compare_runs(runs)
    # BH q is always >= the raw p and bounded by 1.
    assert (result["overdispersion_q"] >= result["overdispersion_p"] - 1e-12).all()
    assert (result["overdispersion_q"] <= 1.0).all()


# --- Explicit y_obs / y_std --------------------------------------------------

def test_explicit_y_obs_and_y_std():
    runs = [pd.DataFrame({"genotype": ["wt", "m1"],
                          "est": [0.1 + 0.02 * i, 0.5],
                          "se": [0.03, 0.03]}) for i in range(3)]
    result = compare_runs(runs, y_obs="est", y_std="se")
    assert get_row(result, genotype="m1")["rms_sd"] == pytest.approx(0.0)
    assert get_row(result, genotype="wt")["rms_sd"] == pytest.approx(0.02)
    assert get_row(result, genotype="wt")["mean_reported_sigma"] == \
        pytest.approx(0.03)


def test_explicit_y_obs_is_not_treated_as_a_key():
    runs = [pd.DataFrame({"genotype": ["wt"], "est": [0.1 + 0.02 * i],
                          "se": [0.03]}) for i in range(3)]
    schema = resolve_schema(runs, y_obs="est", y_std="se")
    assert schema["match_by"] == ["genotype"]


def test_custom_point_and_sigma_quantiles():
    runs = [make_estimate_ladder({"g": [0.4 + 0.02 * i, 0.6]}, sigma=0.05)
            for i in range(3)]
    result = compare_runs(runs, point_quantile=0.5,
                          sigma_quantiles=(0.025, 0.975))
    # (q0.975 - q0.025)/2 is ~1.96 sigma, so the reported sigma is wider.
    assert result.iloc[0]["mean_reported_sigma"] > 0.05


def test_missing_point_quantile_raises():
    runs = [make_estimate({"g": [0.1, 0.2]}) for _ in range(2)]
    with pytest.raises(ValueError, match="q0.9"):
        compare_runs(runs, point_quantile=0.9)


# --- Uniqueness --------------------------------------------------------------

def test_non_unique_match_key_raises():
    runs = [pd.DataFrame({"genotype": ["wt", "wt"],
                          "q0.159": [0.0, 0.1], "q0.5": [0.1, 0.2],
                          "q0.841": [0.2, 0.3]}) for _ in range(2)]
    with pytest.raises(ValueError, match="does not uniquely identify a row"):
        compare_runs(runs)


def test_non_unique_match_key_fixed_by_match_by():
    runs = [pd.DataFrame({"genotype": ["wt", "wt"], "replicate": [1, 2],
                          "q0.159": [0.0, 0.1], "q0.5": [0.1, 0.2],
                          "q0.841": [0.2, 0.3]}) for _ in range(2)]
    result = compare_runs(runs)  # replicate is auto-detected as a key
    assert len(result) == 1
    assert result.iloc[0]["n_rows"] == 2


# --- Reference mode ----------------------------------------------------------

def test_reference_mode_pooled_deviation():
    ref = make_estimate({"g": [0.5, 0.5]})
    runs = [make_estimate({"g": [0.5 + d, 0.5]}) for d in (0.1, -0.1)]
    result = compare_runs(runs, reference_df=ref)
    row = result.iloc[0]
    assert row["mode"] == "reference"
    assert row["spread_estimator"] == "rms_dev"
    # conc 0: rms of (+0.1, -0.1) = 0.1; conc 1: 0. Pooled: sqrt(0.01/2).
    assert row["rms_sd"] == pytest.approx(np.sqrt(0.01 / 2))
    assert row["max_sd"] == pytest.approx(0.1)


def test_reference_mode_target_is_the_reference():
    ref = make_estimate({"g": [0.2, 0.8]})
    runs = [make_estimate({"g": [0.5, 0.5]}) for _ in range(2)]
    row = compare_runs(runs, reference_df=ref).iloc[0]
    assert row["mean_value"] == pytest.approx(0.5)      # mean of 0.2 and 0.8
    assert row["dynamic_range"] == pytest.approx(0.6)   # range of the reference


def test_reference_mode_dof_is_n_terms():
    ref = make_estimate({"g": [0.5, 0.5]})
    runs = [make_estimate({"g": [0.5, 0.5]}) for _ in range(3)]
    row = compare_runs(runs, reference_df=ref).iloc[0]
    assert row["dof"] == 6      # 3 runs x 2 rows; the target is not estimated
    assert row["n_eff"] == 6


def test_reference_mode_variance_includes_reference_sigma():
    ref = make_estimate({"g": [0.5, 0.5]}, sigma=0.1)
    runs = [make_estimate({"g": [0.6, 0.5]}, sigma=0.1) for _ in range(2)]
    row = compare_runs(runs, reference_df=ref).iloc[0]
    # dev = 0.1 at conc 0, var = 0.1^2 + 0.1^2 -> chi2 term = 0.5 each.
    assert row["chi2"] == pytest.approx(1.0)


def test_reference_mode_accepts_a_single_run():
    ref = make_estimate({"g": [0.5, 0.5]})
    run = make_estimate({"g": [0.6, 0.5]})
    assert len(compare_runs([run], reference_df=ref)) == 1


def test_reference_mode_drops_rows_absent_from_reference(capsys):
    ref = make_estimate({"a": [0.5, 0.5]})
    runs = [make_estimate({"a": [0.5, 0.5], "b": [0.2, 0.2]})
            for _ in range(2)]
    result = compare_runs(runs, reference_df=ref)
    assert list(result["genotype"]) == ["a"]
    assert "absent from the reference" in capsys.readouterr().out


# --- Robustness --------------------------------------------------------------

def test_non_finite_values_are_dropped_with_a_warning():
    runs = [make_estimate({"a": [0.5, 0.5], "b": [0.2, 0.2]})
            for _ in range(3)]
    runs[0].loc[runs[0]["genotype"] == "b", "q0.5"] = np.nan
    with pytest.warns(UserWarning, match="non-finite"):
        result = compare_runs(runs)
    assert get_row(result, genotype="b")["n_present"] == 2
    assert get_row(result, genotype="a")["n_present"] == 3


def test_disjoint_rows_do_not_error():
    """Runs need not share a grid; only the shared rows are compared."""
    a = make_estimate({"g": [0.5, 0.5]}, concs=(0.0, 1.0))
    b = make_estimate({"g": [0.5, 0.5, 0.5]}, concs=(0.0, 1.0, 10.0))
    result = compare_runs([a, b])
    assert len(result) == 1
    # The 10.0 row is present in only one run, so it contributes no spread and
    # is not counted in the pooling depth.
    assert result.iloc[0]["n_rows"] == 2
    assert result.iloc[0]["rms_sd"] == pytest.approx(0.0)


# --- _mixture_quantiles ------------------------------------------------------

def test_mixture_quantiles_identical_runs_return_input():
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    vals = 0.5 + norm.ppf(probs) * 0.05
    out = _mixture_quantiles(np.vstack([vals, vals, vals]), probs)
    np.testing.assert_allclose(out, vals, atol=1e-9)


def test_mixture_quantiles_between_spread_widens_interval():
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    z = norm.ppf(probs)
    v1, v2 = 0.4 + z * 0.05, 0.6 + z * 0.05
    out = _mixture_quantiles(np.vstack([v1, v2]), probs)
    hi, lo, med = (_LEVELS.index(x) for x in (0.841, 0.159, 0.5))
    assert (out[hi] - out[lo]) > (v1[hi] - v1[lo])
    assert 0.4 < out[med] < 0.6


def test_mixture_quantiles_point_mass_all_equal():
    probs = np.array(_LEVELS)
    out = _mixture_quantiles(np.full((3, probs.size), 0.0), probs)
    np.testing.assert_allclose(out, 0.0)


def test_mixture_quantiles_point_mass_mixed_is_monotone_between():
    probs = np.array([0.1, 0.5, 0.9])
    a, b = 0.2, 0.8
    out = _mixture_quantiles(np.vstack([np.full(3, a), np.full(3, b)]), probs)
    assert np.all(np.diff(out) >= 0)
    assert a <= out[1] <= b


def test_mixture_quantiles_uniform_weights_match_default():
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    z = norm.ppf(probs)
    V = np.vstack([0.4 + z * 0.05, 0.6 + z * 0.05])
    np.testing.assert_allclose(_mixture_quantiles(V, probs),
                               _mixture_quantiles(V, probs, weights=[1.0, 1.0]))


def test_mixture_quantiles_weights_shift_the_mixture():
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    z = norm.ppf(probs)
    V = np.vstack([0.4 + z * 0.05, 0.6 + z * 0.05])
    med = _LEVELS.index(0.5)
    heavy_low = _mixture_quantiles(V, probs, weights=[9.0, 1.0])[med]
    heavy_high = _mixture_quantiles(V, probs, weights=[1.0, 9.0])[med]
    assert heavy_low < heavy_high


def test_mixture_quantiles_rejects_bad_weight_shape():
    probs = np.array(_LEVELS)
    V = np.vstack([np.full(probs.size, 0.4), np.full(probs.size, 0.6)])
    with pytest.raises(ValueError, match="must have shape"):
        _mixture_quantiles(V, probs, weights=[1.0, 1.0, 1.0])


# --- aggregate_runs ----------------------------------------------------------

def agg_row(agg, genotype, conc):
    sub = agg[(agg["genotype"] == genotype) & (agg["titrant_conc"] == conc)]
    assert len(sub) == 1
    return sub.iloc[0]


def test_aggregate_identical_runs_matches_input():
    df = make_estimate_ladder({"wt": [0.2, 0.8], "m1": [0.5, 0.5]})
    agg = aggregate_runs([df, df.copy(), df.copy()])
    for lvl in _LEVELS:
        assert f"q{lvl}" in agg.columns
    row = agg_row(agg, "wt", 0.0)
    src = df[(df["genotype"] == "wt") & (df["titrant_conc"] == 0.0)].iloc[0]
    for lvl in _LEVELS:
        np.testing.assert_allclose(row[f"q{lvl}"], src[f"q{lvl}"], atol=1e-9)
    assert row["n_present"] == 3


def test_aggregate_between_variance_widens():
    r1 = make_estimate_ladder({"g": [0.4, 0.4]}, sigma=0.05)
    r2 = make_estimate_ladder({"g": [0.6, 0.6]}, sigma=0.05)
    agg = aggregate_runs([r1, r2])
    row = agg_row(agg, "g", 0.0)
    src = r1.iloc[0]
    assert (row["q0.841"] - row["q0.159"]) > (src["q0.841"] - src["q0.159"])
    np.testing.assert_allclose(row["q0.5"], 0.5, atol=0.05)


def test_aggregate_coverage_reported():
    base = {"a": [0.2, 0.8]}
    r3 = make_estimate_ladder({"a": [0.2, 0.8], "b": [0.5, 0.5]})
    agg = aggregate_runs([make_estimate_ladder(base),
                          make_estimate_ladder(base), r3])
    assert agg_row(agg, "a", 0.0)["n_present"] == 3
    assert agg_row(agg, "b", 0.0)["n_present"] == 1


def test_aggregate_keyed_on_parameter_shape():
    runs = [make_estimate_ladder({"g": [0.4, 0.6]}, titrant_name="iptg")
            for _ in range(2)]
    agg = aggregate_runs(runs)
    assert "titrant_name" in agg.columns
    assert (agg["titrant_name"] == "iptg").all()


def test_aggregate_respects_explicit_match_by():
    runs = [make_estimate_ladder({"g": [0.4, 0.6]}) for _ in range(2)]
    agg = aggregate_runs(runs, match_by=["genotype", "titrant_conc"])
    assert list(agg.columns[:2]) == ["genotype", "titrant_conc"]


def test_aggregate_requires_two_runs():
    with pytest.raises(ValueError, match="at least 2"):
        aggregate_runs([make_estimate_ladder({"g": [0.2, 0.8]})])


def test_aggregate_requires_two_quantile_levels():
    runs = [pd.DataFrame({"genotype": ["g"], "q0.5": [0.5]}) for _ in range(2)]
    with pytest.raises(ValueError, match="fewer than 2 quantile"):
        aggregate_runs(runs)


def test_shared_quantile_levels():
    a = make_estimate({"g": [0.1, 0.2]})                      # 3 levels
    b = make_estimate_ladder({"g": [0.1, 0.2]})               # 5 levels
    assert shared_quantile_levels([a, b]) == [0.159, 0.5, 0.841]
    thin = pd.DataFrame({"genotype": ["g"], "q0.5": [0.5]})
    assert shared_quantile_levels([a, thin]) == []
