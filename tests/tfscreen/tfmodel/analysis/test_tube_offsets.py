"""The per-tube sample-offset diagnostic (offset mode)."""
import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.analysis.tube_offsets import (
    format_tube_offset_summary,
    tube_offset_diagnostic,
)

CONCS = [0.0, 0.01, 0.1, 1.0]
TIMES = [0.0, 60.0, 120.0]


def tube_table(offset_fn, seed=0, noise=0.17):
    """3 replicates x 3 times x 4 IPTG in two selection conditions."""
    rng = np.random.default_rng(seed)
    rows = []
    for cond in ("kanR+kan", "pheS+4CP"):
        for rep in (1, 2, 3):
            for t in TIMES:
                for i, c in enumerate(CONCS):
                    rows.append(dict(replicate=rep, library="lib",
                                     condition_pre="pre", condition_sel=cond,
                                     titrant_name="iptg", titrant_conc=c,
                                     t_pre=30.0, t_sel=t,
                                     q0_5=offset_fn(cond, i, t)
                                     + rng.normal(0, noise)))
    return pd.DataFrame(rows).rename(columns={"q0_5": "q0.5"})


def test_noise_offsets_are_not_structured():
    _, trends, summary = tube_offset_diagnostic(
        tube_table(lambda c, i, t: 0.0), sigma=0.17)
    assert not summary["structured"]
    assert summary["n_tubes"] == 72
    assert summary["n_conditions"] == 2
    assert summary["sd_over_sigma"] == pytest.approx(1.0, abs=0.3)
    assert set(trends["condition_sel"]) == {"kanR+kan", "pheS+4CP"}


def test_iptg_pattern_is_flagged():
    """The dev-data offset mode: offsets follow IPTG, opposite by marker."""
    def mode(cond, i, t):
        return (2.0 - 2.0 * i) if cond == "kanR+kan" else (-1.0 + i)
    tubes, trends, summary = tube_offset_diagnostic(tube_table(mode),
                                                    sigma=0.17)
    assert summary["structured"]
    assert summary["n_structured"] == 2
    t = trends.set_index("condition_sel")
    assert t.loc["kanR+kan", "rho_titrant"] < -0.8
    assert t.loc["pheS+4CP", "rho_titrant"] > 0.8
    assert t.loc["kanR+kan", "r2_titrant"] > 0.9
    assert summary["abs_z_max"] > 10
    np.testing.assert_allclose(tubes["z"], tubes["offset"] / 0.17)
    text = format_tube_offset_summary(summary, trends)
    assert "offset mode" in text


def test_time_trend_is_flagged():
    _, trends, summary = tube_offset_diagnostic(
        tube_table(lambda c, i, t: 0.01 * t if c == "kanR+kan" else 0.0),
        sigma=0.17)
    t = trends.set_index("condition_sel")
    assert t.loc["kanR+kan", "structured"]
    assert t.loc["kanR+kan", "q_time"] < 0.05
    assert not t.loc["pheS+4CP", "structured"]


def test_without_sigma_reports_ln_units():
    tubes, trends, summary = tube_offset_diagnostic(
        tube_table(lambda c, i, t: 0.0), sigma=None)
    assert "z" not in tubes.columns
    assert summary["sigma"] is None and summary["sd_over_sigma"] is None
    assert trends["mean_z"].isna().all()
    assert "prior SD" not in format_tube_offset_summary(summary, trends)


def test_constant_titrant_gives_nan_trend():
    df = tube_table(lambda c, i, t: 0.0)
    df["titrant_conc"] = 0.0
    _, trends, summary = tube_offset_diagnostic(df, sigma=0.17)
    assert trends["rho_titrant"].isna().all()
    assert trends["r2_titrant"].isna().all()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="t_sel"):
        tube_offset_diagnostic(tube_table(lambda c, i, t: 0.0)
                               .drop(columns="t_sel"))
    with pytest.raises(ValueError, match="q0.5"):
        tube_offset_diagnostic(tube_table(lambda c, i, t: 0.0)
                               .drop(columns="q0.5"))
