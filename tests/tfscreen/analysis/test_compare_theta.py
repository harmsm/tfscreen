"""
Unit tests for tfscreen.analysis.compare_theta.
"""

import numpy as np
import pandas as pd
import pytest

from tfscreen.analysis.compare_theta import (
    compare_theta,
    aggregate_theta,
    stability_crosstabs,
    _detect_keys,
    _extract_run,
    _assign_tier,
    _mixture_quantiles,
)

# Standard quantile ladder used by the aggregate builders/tests.
_LEVELS = [0.025, 0.159, 0.5, 0.841, 0.975]


def make_estimate_ladder(theta_by_geno, concs=(0.0, 1.0), sigma=0.05,
                         titrant_name=None, levels=_LEVELS):
    """
    Build an estimate table with a full (Gaussian) quantile ladder.

    Each (genotype, conc) gets quantiles ``q0.5 = theta`` and the other levels
    placed at ``theta + z(level) * sigma`` (Normal), clamped to [0, 1].
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


# --- Fixtures / builders -----------------------------------------------------

def make_estimate(theta_by_geno, concs=(0.0, 1.0), sigma=0.05,
                  titrant_name=None):
    """
    Build one synthetic estimate table with the standard quantile schema.

    Parameters
    ----------
    theta_by_geno : dict[str, sequence]
        genotype -> theta (q0.5) values, one per concentration.
    concs : sequence
        Titrant concentrations (len must match each theta sequence).
    sigma : float or dict[str, sequence]
        1-sigma half-width; scalar (shared) or per-genotype per-conc.
    titrant_name : str or None
        If given, a ``titrant_name`` column is added with this constant value.

    Returns
    -------
    pandas.DataFrame
        Columns: genotype, [titrant_name,] titrant_conc, q0.159, q0.5, q0.841.
    """
    rows = []
    for geno, thetas in theta_by_geno.items():
        for j, conc in enumerate(concs):
            th = thetas[j]
            if isinstance(sigma, dict):
                sg = sigma[geno][j]
            else:
                sg = sigma
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


def get_row(result, genotype):
    """Return the single result row for a genotype as a Series."""
    sub = result.loc[result["genotype"] == genotype]
    assert len(sub) == 1, f"expected 1 row for {genotype}, got {len(sub)}"
    return sub.iloc[0]


# --- Key detection -----------------------------------------------------------

def test_detect_keys_without_name():
    df = make_estimate({"wt": [0.1, 0.2]})
    assert _detect_keys(df) == ["genotype", "titrant_conc"]


def test_detect_keys_with_name():
    df = make_estimate({"wt": [0.1, 0.2]}, titrant_name="iptg")
    assert _detect_keys(df) == ["genotype", "titrant_name", "titrant_conc"]


def test_detect_keys_missing_raises():
    df = pd.DataFrame({"genotype": ["wt"], "q0.5": [0.1]})
    with pytest.raises(ValueError, match="titrant_conc"):
        _detect_keys(df)


# --- sigma extraction --------------------------------------------------------

def test_extract_run_sigma():
    df = make_estimate({"wt": [0.3, 0.7]}, sigma=0.05)
    keys = _detect_keys(df)
    out = _extract_run(df, keys, 0.5, (0.159, 0.841))
    np.testing.assert_allclose(out["theta"].to_numpy(), [0.3, 0.7])
    np.testing.assert_allclose(out["sigma"].to_numpy(), [0.05, 0.05])


def test_extract_run_missing_quantile_raises():
    df = make_estimate({"wt": [0.3, 0.7]}).drop(columns=["q0.841"])
    keys = _detect_keys(df)
    with pytest.raises(ValueError, match="q0.841"):
        _extract_run(df, keys, 0.5, (0.159, 0.841))


# --- _assign_tier ------------------------------------------------------------

@pytest.mark.parametrize("rms,expected", [
    (0.0, "A"),
    (0.019, "A"),
    (0.02, "B"),
    (0.049, "B"),
    (0.05, "C"),
    (0.099, "C"),
    (0.10, "D"),
    (0.5, "D"),
])
def test_assign_tier_edges(rms, expected):
    tier = _assign_tier(rms, n_present=4, n_runs=4, min_coverage=0.5,
                        sd_tier_edges=(0.02, 0.05, 0.10))
    assert tier == expected


def test_assign_tier_low_coverage():
    # present in 1 of 4 runs, min_coverage 0.5 -> needs >=2
    tier = _assign_tier(0.0, n_present=1, n_runs=4, min_coverage=0.5,
                        sd_tier_edges=(0.02, 0.05, 0.10))
    assert tier == "low_coverage"


# --- Axis 1: reproducibility, mean mode --------------------------------------

def test_mean_mode_identical_runs():
    df = make_estimate({"wt": [0.2, 0.8], "m1": [0.1, 0.9]})
    result = compare_theta([df, df.copy(), df.copy()])
    assert (result["rms_sd"] == 0).all()
    assert (result["max_sd"] == 0).all()
    assert (result["tier"] == "A").all()
    assert (result["mode"] == "mean").all()


def test_mean_mode_known_offset_halfrange():
    # N=2 (<5) => estimator is half_range = (max-min)/2.
    d = 0.2
    r1 = make_estimate({"g": [0.5, 0.5]})
    r2 = make_estimate({"g": [0.5, 0.5 + d]})
    result = compare_theta([r1, r2])
    row = get_row(result, "g")
    assert row["spread_estimator"] == "half_range"
    # conc0 spread 0, conc1 spread d/2
    np.testing.assert_allclose(row["max_sd"], d / 2)
    np.testing.assert_allclose(row["rms_sd"], np.sqrt((0 + (d / 2) ** 2) / 2))


def test_mean_mode_std_estimator_for_large_n():
    # >=5 runs => sample std (ddof=1).
    thetas = [0.5, 0.5, 0.5, 0.5, 0.6]  # at a single conc across 5 runs
    runs = [make_estimate({"g": [t]}, concs=(0.0,)) for t in thetas]
    result = compare_theta(runs)
    row = get_row(result, "g")
    assert row["spread_estimator"] == "std"
    np.testing.assert_allclose(row["rms_sd"], np.std(thetas, ddof=1))


def test_flat_curve_is_tier_a_and_zero_range():
    # Genotype pinned flat across all concs and runs: perfectly stable, no
    # dynamic range. Must land in tier A (the whole point of absolute spread).
    flat = make_estimate({"flat": [1e-4, 1e-4, 1e-4]}, concs=(0.0, 1.0, 2.0))
    result = compare_theta([flat, flat.copy(), flat.copy()])
    row = get_row(result, "flat")
    assert row["tier"] == "A"
    np.testing.assert_allclose(row["dynamic_range"], 0.0)


# --- Axis 2: self-consistency / overdispersion -------------------------------

def test_axis2_wide_sigma_not_flagged():
    d = 0.2
    r1 = make_estimate({"g": [0.5, 0.5]}, sigma=0.5)
    r2 = make_estimate({"g": [0.5, 0.5 + d]}, sigma=0.5)
    result = compare_theta([r1, r2], overdispersion_threshold=2.0)
    row = get_row(result, "g")
    # chi2 = d^2/(2 sigma^2), dof = 4 terms - 2 grids = 2
    expected = (d ** 2 / (2 * 0.5 ** 2)) / 2
    np.testing.assert_allclose(row["overdispersion"], expected)
    assert not row["overdispersed"]


def test_axis2_tight_sigma_flagged():
    d = 0.2
    r1 = make_estimate({"g": [0.5, 0.5]}, sigma=0.01)
    r2 = make_estimate({"g": [0.5, 0.5 + d]}, sigma=0.01)
    result = compare_theta([r1, r2], overdispersion_threshold=2.0)
    row = get_row(result, "g")
    expected = (d ** 2 / (2 * 0.01 ** 2)) / 2
    np.testing.assert_allclose(row["overdispersion"], expected)
    assert row["overdispersed"]


def test_axis2_zero_sigma_skipped():
    # A zero-width interval must not blow up axis 2 (division by zero).
    r1 = make_estimate({"g": [0.5, 0.5]}, sigma=0.0)
    r2 = make_estimate({"g": [0.5, 0.6]}, sigma=0.0)
    result = compare_theta([r1, r2])
    row = get_row(result, "g")
    # All chi2 terms invalid -> overdispersion is NaN, not inf, and not flagged.
    assert np.isnan(row["overdispersion"])
    assert not row["overdispersed"]


# --- Reference mode ----------------------------------------------------------

def test_reference_mode_pooled_deviation():
    ref = make_estimate({"g": [0.5, 0.5]}, sigma=0.1)
    # two estimate runs deviating from reference by +a and -b at conc1
    a, b = 0.1, 0.3
    e1 = make_estimate({"g": [0.5, 0.5 + a]}, sigma=0.1)
    e2 = make_estimate({"g": [0.5, 0.5 - b]}, sigma=0.1)
    result = compare_theta([e1, e2], reference_df=ref)
    row = get_row(result, "g")
    assert row["mode"] == "reference"
    assert row["spread_estimator"] == "rms_dev"

    # per-grid spread: conc0 -> 0; conc1 -> sqrt(mean(a^2, b^2))
    conc1_spread = np.sqrt((a ** 2 + b ** 2) / 2)
    np.testing.assert_allclose(row["max_sd"], conc1_spread)
    # rms_sd pooled over all (grid, run) deviations = sqrt(mean(0,0,a^2,b^2))
    np.testing.assert_allclose(
        row["rms_sd"], np.sqrt((a ** 2 + b ** 2) / 4)
    )


def test_reference_mode_overdispersion_includes_ref_sigma():
    ref = make_estimate({"g": [0.5]}, concs=(0.0,), sigma=0.1)
    a = 0.2
    e1 = make_estimate({"g": [0.5 + a]}, concs=(0.0,), sigma=0.1)
    result = compare_theta([e1], reference_df=ref)
    row = get_row(result, "g")
    # single term: dev^2 / (sigma^2 + sigma_ref^2); dof = 1 term (fixed target)
    expected = a ** 2 / (0.1 ** 2 + 0.1 ** 2)
    np.testing.assert_allclose(row["overdispersion"], expected)


def test_reference_mode_drops_genotype_absent_from_reference(capsys):
    ref = make_estimate({"g": [0.5, 0.5]})
    e1 = make_estimate({"g": [0.5, 0.6], "extra": [0.1, 0.2]})
    result = compare_theta([e1, e1.copy()], reference_df=ref)
    assert "extra" not in set(result["genotype"])
    assert "g" in set(result["genotype"])
    out = capsys.readouterr().out
    assert "absent from the reference" in out


# --- Coverage ----------------------------------------------------------------

def test_low_coverage_tier_and_n_present():
    base = {"a": [0.1, 0.2], "b": [0.3, 0.4]}
    r1 = make_estimate(base)
    r2 = make_estimate(base)
    r3 = make_estimate(base)
    # 'c' appears in only 1 of 4 runs
    r4 = make_estimate({"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.9, 0.9]})
    result = compare_theta([r1, r2, r3, r4], min_coverage=0.5)
    c = get_row(result, "c")
    assert c["n_present"] == 1
    assert c["tier"] == "low_coverage"
    a = get_row(result, "a")
    assert a["n_present"] == 4


# --- Grid / key validation ---------------------------------------------------

def test_grid_mismatch_raises():
    r1 = make_estimate({"g": [0.1, 0.2]}, concs=(0.0, 1.0))
    r2 = make_estimate({"g": [0.1, 0.2]}, concs=(0.0, 2.0))
    with pytest.raises(ValueError, match="condition grid"):
        compare_theta([r1, r2])


def test_key_mismatch_raises():
    r1 = make_estimate({"g": [0.1, 0.2]}, titrant_name="iptg")
    r2 = make_estimate({"g": [0.1, 0.2]})  # no titrant_name
    with pytest.raises(ValueError, match="key columns"):
        compare_theta([r1, r2])


def test_mean_mode_requires_two_runs():
    r1 = make_estimate({"g": [0.1, 0.2]})
    with pytest.raises(ValueError, match="at least 2"):
        compare_theta([r1])


# --- titrant_name handling ---------------------------------------------------

def test_titrant_name_dynamic_range_is_per_name_max():
    # Genotype is flat in 'iptg' but responsive in 'atc'; dynamic_range should
    # reflect the responsive titrant, not be washed out by pooling.
    iptg = make_estimate({"g": [0.5, 0.5]}, titrant_name="iptg")
    atc = make_estimate({"g": [0.1, 0.9]}, titrant_name="atc")
    r1 = pd.concat([iptg, atc], ignore_index=True)
    result = compare_theta([r1, r1.copy()])
    row = get_row(result, "g")
    np.testing.assert_allclose(row["dynamic_range"], 0.8)
    # per-grid sd columns exist for both titrants
    sd_cols = [c for c in result.columns if c.startswith("sd[")]
    assert any("iptg" in c for c in sd_cols)
    assert any("atc" in c for c in sd_cols)


# --- Crosstabs ---------------------------------------------------------------

def test_stability_crosstabs_populates_cells():
    # Construct a mix: a stable+consistent, an unstable+overconfident.
    stable = make_estimate({"s": [0.1, 0.9]}, sigma=0.1)
    unstable_a = make_estimate({"u": [0.5, 0.5]}, sigma=0.001)
    unstable_b = make_estimate({"u": [0.5, 0.9]}, sigma=0.001)
    r1 = pd.concat([stable, unstable_a], ignore_index=True)
    r2 = pd.concat([stable.copy(), unstable_b], ignore_index=True)
    result = compare_theta([r1, r2])
    tabs = stability_crosstabs(result)
    assert "tier_vs_overdispersion" in tabs
    assert "tier_vs_dynamic_range" in tabs
    total = tabs["tier_vs_overdispersion"].to_numpy().sum()
    assert total == 2  # both genotypes graded


def test_result_sorted_by_rms_sd():
    r1 = make_estimate({"a": [0.5, 0.5], "b": [0.5, 0.5]})
    r2 = make_estimate({"a": [0.5, 0.55], "b": [0.5, 0.9]})
    result = compare_theta([r1, r2])
    assert list(result["rms_sd"]) == sorted(result["rms_sd"])


# --- aggregate_theta ---------------------------------------------------------

def agg_row(agg, genotype, conc):
    sub = agg[(agg["genotype"] == genotype) & (agg["titrant_conc"] == conc)]
    assert len(sub) == 1
    return sub.iloc[0]


def test_mixture_quantiles_identical_runs_return_input():
    # Mixing identical distributions returns that distribution.
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    vals = 0.5 + norm.ppf(probs) * 0.05
    V = np.vstack([vals, vals, vals])
    out = _mixture_quantiles(V, probs)
    np.testing.assert_allclose(out, vals, atol=1e-9)


def test_mixture_quantiles_between_spread_widens_interval():
    # Two runs, equal width but shifted medians -> the mixture is wider than
    # either input (captures between-run spread).
    from scipy.stats import norm
    probs = np.array(_LEVELS)
    z = norm.ppf(probs)
    v1 = 0.4 + z * 0.05
    v2 = 0.6 + z * 0.05
    V = np.vstack([v1, v2])
    out = _mixture_quantiles(V, probs)
    hi = _LEVELS.index(0.841)
    lo = _LEVELS.index(0.159)
    mix_width = out[hi] - out[lo]
    run_width = (v1[hi] - v1[lo])
    assert mix_width > run_width
    # median sits between the two run medians
    med = _LEVELS.index(0.5)
    assert 0.4 < out[med] < 0.6


def test_mixture_quantiles_point_mass_all_equal():
    probs = np.array(_LEVELS)
    V = np.full((3, probs.size), 0.0)  # all runs pinned at theta=0
    out = _mixture_quantiles(V, probs)
    np.testing.assert_allclose(out, 0.0)


def test_mixture_quantiles_point_mass_mixed_is_monotone_between():
    # Two point masses at a and b: mixture quantiles monotone and in [a, b].
    probs = np.array([0.1, 0.5, 0.9])
    a, b = 0.2, 0.8
    V = np.vstack([np.full(3, a), np.full(3, b)])
    out = _mixture_quantiles(V, probs)
    assert np.all(np.diff(out) >= 0)
    assert a <= out[1] <= b


def test_aggregate_identical_runs_matches_input():
    df = make_estimate_ladder({"wt": [0.2, 0.8], "m1": [0.5, 0.5]})
    agg = aggregate_theta([df, df.copy(), df.copy()])
    # schema: keys + ladder + n_present
    for lvl in _LEVELS:
        assert f"q{lvl}" in agg.columns
    assert "n_present" in agg.columns
    row = agg_row(agg, "wt", 0.0)
    src = df[(df["genotype"] == "wt") & (df["titrant_conc"] == 0.0)].iloc[0]
    for lvl in _LEVELS:
        np.testing.assert_allclose(row[f"q{lvl}"], src[f"q{lvl}"], atol=1e-9)
    assert row["n_present"] == 3


def test_aggregate_between_variance_widens():
    r1 = make_estimate_ladder({"g": [0.4, 0.4]}, sigma=0.05)
    r2 = make_estimate_ladder({"g": [0.6, 0.6]}, sigma=0.05)
    agg = aggregate_theta([r1, r2])
    row = agg_row(agg, "g", 0.0)
    agg_width = row["q0.841"] - row["q0.159"]
    single_width = (
        make_estimate_ladder({"g": [0.4, 0.4]}, sigma=0.05).iloc[0]["q0.841"]
        - make_estimate_ladder({"g": [0.4, 0.4]}, sigma=0.05).iloc[0]["q0.159"]
    )
    assert agg_width > single_width
    # symmetric shift -> median near the mean of the two medians
    np.testing.assert_allclose(row["q0.5"], 0.5, atol=0.05)


def test_aggregate_coverage_reported():
    base = {"a": [0.2, 0.8]}
    r1 = make_estimate_ladder(base)
    r2 = make_estimate_ladder(base)
    r3 = make_estimate_ladder({"a": [0.2, 0.8], "b": [0.5, 0.5]})
    agg = aggregate_theta([r1, r2, r3])
    assert agg_row(agg, "a", 0.0)["n_present"] == 3
    assert agg_row(agg, "b", 0.0)["n_present"] == 1


def test_aggregate_with_titrant_name():
    r1 = make_estimate_ladder({"g": [0.2, 0.8]}, titrant_name="iptg")
    agg = aggregate_theta([r1, r1.copy()])
    assert "titrant_name" in agg.columns
    assert (agg["titrant_name"] == "iptg").all()


def test_aggregate_requires_two_runs():
    r1 = make_estimate_ladder({"g": [0.2, 0.8]})
    with pytest.raises(ValueError, match="at least 2"):
        aggregate_theta([r1])


def test_aggregate_key_mismatch_raises():
    r1 = make_estimate_ladder({"g": [0.2, 0.8]}, titrant_name="iptg")
    r2 = make_estimate_ladder({"g": [0.2, 0.8]})
    with pytest.raises(ValueError, match="key columns"):
        aggregate_theta([r1, r2])
