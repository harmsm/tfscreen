"""Tests for tfscreen.analysis.calibration_grid (tfs-summarize-calibration)."""

import json
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tfscreen.analysis.calibration_grid import (
    calibration_metrics,
    genotype_strata,
    paired_differences,
    parse_baseline,
    quantile_columns,
    read_quantities,
    summarize_calibration,
    summarize_grid_runs,
    summarize_run,
)


LEVELS = np.array([0.001, 0.01, 0.025, 0.05, 0.1, 0.159, 0.25, 0.5,
                   0.75, 0.841, 0.9, 0.95, 0.975, 0.99, 0.999])


def _posterior(rng, n, sigma, spread):
    """Truth, and quantile ladders of a posterior with std sigma*spread.

    With spread == 1 the intervals are calibrated: truth ~ N(center, sigma).
    """
    center = rng.uniform(0.2, 0.8, size=n)
    ref = center + rng.normal(0.0, sigma, size=n)
    qmat = center[:, None] + norm.ppf(LEVELS)[None, :] * sigma * spread
    return ref, qmat


# ---------------------------------------------------------------------------
# calibration_metrics
# ---------------------------------------------------------------------------

def test_calibrated_posterior_has_nominal_coverage():
    ref, qmat = _posterior(np.random.default_rng(0), 20000, 0.05, 1.0)
    m = calibration_metrics(ref, qmat, LEVELS)
    assert m["n"] == 20000
    assert m["coverage_0.95"] == pytest.approx(0.95, abs=0.01)
    assert m["coverage_0.5"] == pytest.approx(0.5, abs=0.015)
    assert m["calibration_error"] < 0.015
    assert m["width_0.95"] == pytest.approx(2 * 1.96 * 0.05, rel=0.01)
    assert m["rmse"] == pytest.approx(0.05, rel=0.03)


def test_overconfident_posterior_undercovers_with_negative_bias():
    ref, qmat = _posterior(np.random.default_rng(1), 20000, 0.05, 0.25)
    m = calibration_metrics(ref, qmat, LEVELS)
    assert m["coverage_0.95"] < 0.5
    assert m["calibration_bias"] < -0.3
    assert m["ks_stat"] > 0.2


def test_nonfinite_rows_are_dropped():
    ref = np.array([0.5, np.nan, 0.4])
    qmat = np.tile(np.linspace(0.0, 1.0, len(LEVELS)), (3, 1))
    qmat[2, 3] = np.nan
    assert calibration_metrics(ref, qmat, LEVELS)["n"] == 1


def test_no_rows_gives_nan_metrics():
    m = calibration_metrics(np.array([np.nan]),
                            np.full((1, len(LEVELS)), 0.5), LEVELS)
    assert m["n"] == 0 and np.isnan(m["coverage_0.95"])


def test_quantile_columns_sorted_numerically():
    df = pd.DataFrame(columns=["q0.975", "genotype", "q0.025", "q0.5", "ref"])
    cols, levels = quantile_columns(df)
    assert cols == ["q0.025", "q0.5", "q0.975"]
    assert list(levels) == [0.025, 0.5, 0.975]


# ---------------------------------------------------------------------------
# A fake grid
# ---------------------------------------------------------------------------

GENOTYPES = [f"G{i}V" for i in range(40)]
SPIKED = {"G0V", "G1V"}
ALSO_BULK = {"G1V"}                       # G1V: spiked and in the bulk (mixed)
BINDING = {"G0V", "G1V", "G2V", "G3V"}   # two spiked + two bulk with binding


def _write_run(run_dir, combo, spread, seed, with_summary=True):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "combo.json"), "w") as fh:
        json.dump(combo, fh)

    rows = ([("spiked", g) for g in GENOTYPES if g in SPIKED]
            + [("double-1-2", g) for g in GENOTYPES
               if g not in SPIKED or g in ALSO_BULK])
    pd.DataFrame(rows, columns=["library_origin", "genotype"]).to_csv(
        os.path.join(run_dir, "tfs_sim_library.csv"), index=False)
    pd.DataFrame({"genotype": sorted(BINDING)}).to_csv(
        os.path.join(run_dir, "tfs_sim_binding.csv"), index=False)

    if not with_summary:
        return
    summary = os.path.join(run_dir, "summary")
    os.makedirs(summary, exist_ok=True)
    rng = np.random.default_rng(seed)
    concs = np.arange(25)
    genos = np.repeat(GENOTYPES, len(concs))
    ref, qmat = _posterior(rng, len(genos), 0.05, spread)
    ref[:len(concs)] = 0.999            # G0V saturated
    theta = pd.DataFrame(qmat, columns=[f"q{lv:g}" for lv in LEVELS])
    theta.insert(0, "genotype", genos)
    theta.insert(1, "titrant_conc", np.tile(concs, len(GENOTYPES)))
    theta["ref"] = ref
    theta.to_csv(os.path.join(summary, "tfs_summarize_theta_corr_test.csv"), index=False)

    ref_m, qmat_m = _posterior(rng, 30, 0.1, spread)
    mut = pd.DataFrame(qmat_m, columns=[f"q{lv:g}" for lv in LEVELS])
    mut.insert(0, "mutation", [f"M{i}" for i in range(30)])
    mut["ref"] = ref_m
    mut.to_csv(os.path.join(summary, "tfs_summarize_params_d_logit_low.csv"), index=False)
    # Calibration by-products written next to params files must be ignored.
    pd.DataFrame({"nominal": [0.5], "empirical": [0.5]}).to_csv(
        os.path.join(summary, "tfs_summarize_params_d_logit_low_calibration_curve.csv"),
        index=False)


@pytest.fixture
def grid_dir(tmp_path):
    """component (overconfident) and auto_normal (calibrated) x 2 sim seeds."""
    grid = tmp_path / "grid"
    for guide, rank, spread in (("component", None, 0.3), ("auto_normal", None, 1.0)):
        for sim_seed in (1, 2):
            combo = {"simulate": {"seed": sim_seed},
                     "template": {"guide_type": guide, "guide_rank": rank,
                                  "batch_size": 160, "fit_seed": 1}}
            _write_run(str(grid / f"run_{guide}_sim{sim_seed}"), combo, spread,
                       seed=sim_seed * 10 + (spread == 1.0))
    _write_run(str(grid / "run_unfinished"),
               {"simulate": {"seed": 3},
                "template": {"guide_type": "auto_normal", "guide_rank": None,
                             "batch_size": 160, "fit_seed": 1}},
               spread=1.0, seed=0, with_summary=False)
    return str(grid)


def test_genotype_strata(grid_dir):
    strata = genotype_strata(os.path.join(grid_dir, "run_component_sim1")).set_index("genotype")
    assert len(strata) == len(GENOTYPES)
    assert strata.loc["G0V", "purity"] == "spike"
    assert strata.loc["G1V", "purity"] == "mixed"
    assert strata.loc["G2V", "purity"] == "bulk"
    assert strata.loc["G2V", "has_binding"] == "yes"
    assert strata.loc["G9V", "has_binding"] == "no"


def test_read_quantities_skips_calibration_byproducts(grid_dir):
    summary = os.path.join(grid_dir, "run_component_sim1", "summary")
    assert [q for q, _ in read_quantities(summary)] == ["theta_test", "d_logit_low"]


def test_summarize_run_strata_rows(grid_dir):
    rows, problem = summarize_run(os.path.join(grid_dir, "run_auto_normal_sim1"))
    assert problem is None
    theta = pd.DataFrame([r for r in rows if r["quantity"] == "theta_test"])
    pooled = theta[(theta.has_binding == "all") & (theta.purity == "all")
                   & (theta.theta_regime == "all")]
    assert pooled["n"].item() == 40 * 25
    cross = theta[(theta.has_binding == "yes") & (theta.purity == "bulk")]
    assert cross["n"].item() == 2 * 25        # G2V, G3V
    saturated = theta[theta.theta_regime == "saturated"]
    assert saturated["n"].item() >= 25        # at least G0V
    # Mutation-level params have no genotype, so only a pooled row.
    mut = [r for r in rows if r["quantity"] == "d_logit_low"]
    assert len(mut) == 1 and mut[0]["has_binding"] == "all"


def test_unfinished_run_reported_in_status(grid_dir):
    runs_df, status_df, grid_vars = summarize_grid_runs(grid_dir)
    status = status_df.set_index("run")["status"]
    assert status["run_unfinished"] != "ok"
    assert (status.drop("run_unfinished") == "ok").all()
    assert set(grid_vars) == {"seed", "guide_type", "guide_rank", "batch_size", "fit_seed"}


@pytest.mark.parametrize("text,expected", [
    (["guide_type=component", "guide_rank=None"], {"guide_type": "component", "guide_rank": None}),
    (["batch_size=160", "eps=0.5"], {"batch_size": 160, "eps": 0.5}),
])
def test_parse_baseline(text, expected):
    assert parse_baseline(text) == expected


def test_parse_baseline_requires_key_value():
    with pytest.raises(ValueError, match="key=value"):
        parse_baseline(["component"])


def test_paired_differences_match_same_simulation(grid_dir):
    runs_df, _, grid_vars = summarize_grid_runs(grid_dir)
    paired = paired_differences(runs_df, grid_vars,
                                {"guide_type": "component", "guide_rank": None})
    pooled = paired[(paired.quantity == "theta_test") & (paired.has_binding == "all")
                    & (paired.purity == "all") & (paired.theta_regime == "all")]
    assert sorted(pooled["seed"]) == [1, 2]            # one pair per simulation
    assert (pooled["delta_calibration_error"] < 0).all()  # calibrated beats baseline
    assert (pooled["delta_width_0.95"] > 0).all()


def test_paired_differences_fail_fast(grid_dir):
    runs_df, _, grid_vars = summarize_grid_runs(grid_dir)
    with pytest.raises(ValueError, match="not grid variables"):
        paired_differences(runs_df, grid_vars, {"nope": 1})
    with pytest.raises(ValueError, match="No runs match"):
        paired_differences(runs_df, grid_vars, {"guide_type": "missing"})


def test_summarize_calibration_writes_outputs(grid_dir, tmp_path):
    out_prefix = str(tmp_path / "out" / "calib")
    summarize_calibration(grid_dir, out_prefix=out_prefix,
                          baseline=["guide_type=component", "guide_rank=None"],
                          facet_by=["batch_size"])
    for suffix in ("_runs.csv", "_run_status.csv", "_arms.csv", "_paired.csv",
                   "_paired_summary.csv", "_theta_test_calibration_curves.pdf",
                   "_metadata.json"):
        assert os.path.exists(out_prefix + suffix), suffix

    arms = pd.read_csv(out_prefix + "_arms.csv")
    pooled = arms[(arms.quantity == "theta_test") & (arms.has_binding == "all")
                  & (arms.purity == "all") & (arms.theta_regime == "all")]
    assert set(pooled["n_runs"]) == {2}               # 2 finished sims per arm
    cov = pooled.set_index("guide_type")["coverage_0.95_mean"]
    assert cov["auto_normal"] == pytest.approx(0.95, abs=0.03)
    assert cov["component"] < 0.7

    paired_summary = pd.read_csv(out_prefix + "_paired_summary.csv")
    row = paired_summary[(paired_summary.quantity == "theta_test")
                         & (paired_summary.has_binding == "all")
                         & (paired_summary.purity == "all")
                         & (paired_summary.theta_regime == "all")]
    assert row["n_pairs"].item() == 2
    assert row["delta_calibration_error_mean"].item() < 0

    meta = json.load(open(out_prefix + "_metadata.json"))
    assert meta["replicate_keys"] == ["seed", "fit_seed"]
    assert meta["n_runs"] == 5 and meta["n_runs_ok"] == 4


def test_unfinished_baseline_skips_paired_outputs(grid_dir, tmp_path):
    """Mid-grid: the baseline exists but has not finished -> warn, no paired files."""
    out_prefix = str(tmp_path / "mid")
    with pytest.warns(UserWarning, match="No finished run matches baseline"):
        summarize_calibration(grid_dir, out_prefix=out_prefix,
                              baseline=["guide_type=auto_normal", "seed=3"])
    assert os.path.exists(out_prefix + "_arms.csv")
    assert not os.path.exists(out_prefix + "_paired.csv")


def test_baseline_absent_from_grid_raises(grid_dir, tmp_path):
    """A baseline matching no run at all (e.g. a typo) fails before writing."""
    with pytest.raises(ValueError, match="No run in"):
        summarize_calibration(grid_dir, out_prefix=str(tmp_path / "typo"),
                              baseline=["guide_type=componet"])
    with pytest.raises(ValueError, match="not grid variables"):
        summarize_calibration(grid_dir, out_prefix=str(tmp_path / "typo"),
                              baseline=["guide=component"])


def test_facet_by_must_be_arm_variable(grid_dir, tmp_path):
    with pytest.raises(ValueError, match="facet_by"):
        summarize_calibration(grid_dir, out_prefix=str(tmp_path / "c"),
                              facet_by=["seed"])


def test_main_registers_list_flags():
    from unittest.mock import patch
    from tfscreen.analysis.scripts import summarize_calibration_cli as cli
    with patch.object(cli, "generalized_main") as gm:
        cli.main()
    args, kwargs = gm.call_args
    assert args[0] is summarize_calibration
    assert kwargs["manual_arg_nargs"] == {"replicate_keys": "+", "baseline": "+",
                                          "facet_by": "+"}
