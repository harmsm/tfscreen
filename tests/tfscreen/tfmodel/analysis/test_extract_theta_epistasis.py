import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.extraction import extract_theta_epistasis
from tfscreen.analysis.extract_epistasis import extract_epistasis


# A four-genotype mutant cycle: wt, two singles, and their double.  Groups are
# assigned so that compute_theta_samples indexes theta_* posterior columns by
# map_theta_group: wt->0, A10G->1, C20D->2, A10G/C20D->3.
_GENOTYPES = ["wt", "A10G", "C20D", "A10G/C20D"]


def _make_model(concs=(1.0,)):
    """Minimal orchestrator mock driving the real hill_geno component."""
    rows = []
    for g_idx, g in enumerate(_GENOTYPES):
        for c in concs:
            rows.append({"genotype": g,
                         "titrant_name": "iptg",
                         "titrant_conc": float(c),
                         "map_theta_group": g_idx})
    model = MagicMock(spec=ModelOrchestrator)
    model._theta = "hill_geno"
    tm = MagicMock()
    tm.df = pd.DataFrame(rows)
    model.training_tm = tm
    model.growth_tm = tm
    return model


def _flat_posteriors(theta_per_group):
    """
    Build posterior samples that make theta == theta_low for every genotype
    (theta_high == theta_low collapses the Hill curve to a constant), so the
    per-genotype theta is exactly ``theta_per_group``.

    Parameters
    ----------
    theta_per_group : np.ndarray, shape (S, 4)
        Desired theta for (wt, A10G, C20D, A10G/C20D) in each of S draws.
    """
    theta_per_group = np.asarray(theta_per_group, dtype=float)
    S = theta_per_group.shape[0]
    return {
        "theta_hill_n": np.ones((S, 4)),
        "theta_log_hill_K": np.zeros((S, 4)),
        "theta_theta_high": theta_per_group.copy(),
        "theta_theta_low": theta_per_group.copy(),
    }


def _logit(x):
    return np.log(x / (1.0 - x))


def test_epistasis_value_matches_hand_computation():
    """ep_q0.5 equals the logit difference-of-differences of the corner thetas."""
    # Constant across draws so the median is exact.
    S = 50
    theta = np.tile([0.5, 0.6, 0.7, 0.9], (S, 1))
    model = _make_model()
    post = _flat_posteriors(theta)

    result = extract_theta_epistasis(model, post, scale="logit")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["genotype"] == "A10G/C20D"

    # Quantile columns follow the library-wide bare-q convention (no ep_ prefix).
    assert "q0.5" in result.columns
    assert not any(c.startswith("ep_") for c in result.columns)

    # 00=wt(0.5), 01=A10G(0.6), 10=C20D(0.7), 11=double(0.9)
    expected = (_logit(0.9) - _logit(0.7)) - (_logit(0.6) - _logit(0.5))
    assert np.isclose(row["q0.5"], expected, atol=1e-9)


def test_joint_covariance_shrinks_uncertainty_vs_marginal():
    """
    Perfectly correlated corners -> joint epistasis is exactly zero in every
    draw (zero-width posterior), while the marginal path (which assumes the
    four corners are independent) reports a substantial nonzero ep_std.
    """
    rng = np.random.default_rng(0)
    S = 2000
    # Every corner shares the SAME theta in each draw -> logit ddd == 0 exactly.
    shared = rng.uniform(0.2, 0.8, size=S)
    theta = np.repeat(shared[:, None], 4, axis=1)

    model = _make_model()
    post = _flat_posteriors(theta)

    joint = extract_theta_epistasis(model, post, q_to_get=[0.5, 0.159, 0.841],
                                    scale="logit")
    joint_row = joint.iloc[0]

    # Joint: ep is identically zero -> point estimate 0, zero-width interval.
    assert np.isclose(joint_row["q0.5"], 0.0, atol=1e-9)
    joint_width = joint_row["q0.841"] - joint_row["q0.159"]
    assert np.isclose(joint_width, 0.0, atol=1e-9)

    # Marginal path on the SAME samples: build a per-genotype theta table with a
    # median and a std from the same posterior draws, then run the independent
    # error-propagation path.
    theta_med = np.quantile(theta, 0.5, axis=0)
    theta_std = (np.quantile(theta, 0.841, axis=0)
                 - np.quantile(theta, 0.159, axis=0)) / 2.0
    marg_df = pd.DataFrame({
        "genotype": _GENOTYPES,
        "titrant_name": "iptg",
        "titrant_conc": 1.0,
        "theta": theta_med,
        "theta_std": theta_std,
    })
    marg = extract_epistasis(marg_df, y_obs="theta", y_std="theta_std",
                             group_by=["titrant_name", "titrant_conc"],
                             scale="logit")

    # The marginal path cannot see the correlation, so it reports real spread.
    assert marg.iloc[0]["ep_std"] > 0.1
    assert marg.iloc[0]["ep_std"] > joint_width


def test_in_regime_column_present_and_int():
    S = 20
    theta = np.tile([0.5, 0.6, 0.7, 0.9], (S, 1))
    result = extract_theta_epistasis(_make_model(), _flat_posteriors(theta))
    assert "in_regime" in result.columns
    assert result["in_regime"].dtype.kind in ("i", "u")
    # every corner comfortably inside [0.01, 0.99] -> in regime
    assert result.iloc[0]["in_regime"] == 1


def test_in_regime_false_when_a_corner_saturates():
    # Double mutant pinned at 0.999 -> above 1 - eps=0.99 -> out of regime.
    S = 20
    theta = np.tile([0.5, 0.6, 0.7, 0.999], (S, 1))
    result = extract_theta_epistasis(_make_model(), _flat_posteriors(theta))
    assert result.iloc[0]["in_regime"] == 0


def test_in_regime_uses_interval_not_median():
    # A corner whose posterior MEDIAN is in-band but whose upper tail crosses the
    # boundary is out of regime (the flag checks the central interval).
    rng = np.random.default_rng(1)
    S = 4000
    wide = rng.uniform(0.95, 0.9999, size=S)     # median ~0.975 (< 0.99), q0.975 > 0.99
    theta = np.column_stack([np.full(S, 0.5), np.full(S, 0.6),
                             np.full(S, 0.7), wide])
    assert np.median(wide) < 0.99                # median is inside the band
    result = extract_theta_epistasis(_make_model(), _flat_posteriors(theta),
                                     regime_ci=0.95)
    assert result.iloc[0]["in_regime"] == 0


def test_in_regime_flags_tight_but_saturated_epistasis():
    # The subtle case: all four corners perfectly correlated near saturation.
    # Epistasis is identically zero (a *tight* posterior), yet the corners are
    # saturated -> in_regime must flag it as model-conditional, not data-backed.
    rng = np.random.default_rng(2)
    S = 3000
    shared = rng.uniform(0.98, 0.9999, size=S)
    theta = np.repeat(shared[:, None], 4, axis=1)
    result = extract_theta_epistasis(_make_model(), _flat_posteriors(theta),
                                     q_to_get=[0.5, 0.159, 0.841])
    row = result.iloc[0]
    assert np.isclose(row["q0.5"], 0.0, atol=1e-9)                  # tight
    assert np.isclose(row["q0.841"] - row["q0.159"], 0.0, atol=1e-9)
    assert row["in_regime"] == 0                                   # but flagged


def test_in_regime_respects_regime_eps():
    # A corner at 0.995 is out of the default band but inside a looser eps=0.001.
    S = 20
    theta = np.tile([0.5, 0.6, 0.7, 0.995], (S, 1))
    strict = extract_theta_epistasis(_make_model(), _flat_posteriors(theta),
                                     regime_eps=0.01)
    loose = extract_theta_epistasis(_make_model(), _flat_posteriors(theta),
                                    regime_eps=0.001)
    assert strict.iloc[0]["in_regime"] == 0
    assert loose.iloc[0]["in_regime"] == 1


def test_in_regime_map_single_draw_is_point_check():
    # One "draw" (MAP): the interval collapses to the point value.
    theta = np.array([[0.5, 0.6, 0.7, 0.999]])   # S=1, double saturated
    result = extract_theta_epistasis(_make_model(), _flat_posteriors(theta),
                                     q_to_get=[0.5])
    assert result.iloc[0]["in_regime"] == 0


@pytest.mark.parametrize("bad", [-0.1, 0.5, 0.7])
def test_bad_regime_eps_raises(bad):
    with pytest.raises(ValueError, match="regime_eps"):
        extract_theta_epistasis(_make_model(), _flat_posteriors(
            np.tile([0.5, 0.6, 0.7, 0.9], (5, 1))), regime_eps=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
def test_bad_regime_ci_raises(bad):
    with pytest.raises(ValueError, match="regime_ci"):
        extract_theta_epistasis(_make_model(), _flat_posteriors(
            np.tile([0.5, 0.6, 0.7, 0.9], (5, 1))), regime_ci=bad)


def test_no_complete_cycle_returns_empty():
    """Without a double mutant, no cycle exists; output is empty but well-formed."""
    model = _make_model()
    # Drop the double mutant from the training data.
    model.training_tm.df = model.training_tm.df[
        model.training_tm.df["genotype"] != "A10G/C20D"
    ].reset_index(drop=True)

    result = extract_theta_epistasis(model, _flat_posteriors(
        np.tile([0.5, 0.6, 0.7], (5, 1))), scale="logit")

    assert result.empty
    assert "genotype" in result.columns
    assert any(c.startswith("q") for c in result.columns)
    assert "in_regime" in result.columns


def test_double_with_missing_single_is_dropped():
    """A double whose single parent has no theta row yields no cycle (not a crash)."""
    model = _make_model()
    # Remove one single parent; the double + wt + other single remain.
    keep = model.training_tm.df["genotype"] != "C20D"
    model.training_tm.df = model.training_tm.df[keep].reset_index(drop=True)

    # 3 groups remain (wt=0, A10G=1, A10G/C20D=3 in the original ids -> but the
    # trimmed frame carries its own map_theta_group values); use a matching post.
    S = 5
    n_groups = model.training_tm.df["map_theta_group"].nunique()
    theta = np.tile(np.linspace(0.4, 0.8, n_groups), (S, 1))
    # Remap groups to a contiguous 0..n-1 range so compute_theta_samples indexes
    # cleanly into the posterior columns.
    remap = {g: i for i, g in
             enumerate(sorted(model.training_tm.df["map_theta_group"].unique()))}
    model.training_tm.df["map_theta_group"] = \
        model.training_tm.df["map_theta_group"].map(remap)

    result = extract_theta_epistasis(model, _flat_posteriors(theta), scale="logit")
    assert result.empty


def test_wrong_theta_component_raises():
    model = _make_model()
    model._theta = "no_such_component"  # not registered -> module is None
    with pytest.raises(ValueError, match="build_calc_df"):
        extract_theta_epistasis(model, _flat_posteriors(
            np.tile([0.5, 0.6, 0.7, 0.9], (5, 1))))


def test_mult_scale_constant_rejected():
    model = _make_model()
    with pytest.raises(ValueError, match="scale_constant"):
        extract_theta_epistasis(model, _flat_posteriors(
            np.tile([0.5, 0.6, 0.7, 0.9], (5, 1))),
            scale="mult", scale_constant=2.0)
