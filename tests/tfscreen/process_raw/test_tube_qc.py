"""Tests for the tube composition QC (process_raw/tube_qc.py, tfs-check-tubes)."""
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tfscreen.process_raw import tube_qc as Q
from tfscreen.process_raw.scripts.check_tubes_cli import check_tubes, main

CONCS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1.0]


def _simulate(num_genotypes=300, replicates=(1, 2), times=(100.0, 150.0, 200.0),
              concs=CONCS, condition_sel="pheS+4CP", titrant_name="iptg",
              seed=0, depth=2e6, tube_noise=0.1):
    """
    Long-form counts: each genotype's log abundance grows at
    k_g + m_g * c / (c + K) for t_sel, from a per-replicate starting pool,
    then each tube gets log-normal jitter per genotype (PCR, sampling) and
    reads are drawn as Poisson.
    """
    rng = np.random.default_rng(seed)
    k = rng.normal(0, 0.003, num_genotypes)
    m = rng.normal(0, 0.01, num_genotypes)
    rows = []
    for rep in replicates:
        base = rng.normal(0, 1, num_genotypes)
        for t in times:
            for c in concs:
                ln = (base + t * (k + m * c / (c + 0.01))
                      + rng.normal(0, tube_noise, num_genotypes))
                p = np.exp(ln - ln.max())
                lam = depth * p / p.sum()
                counts = rng.poisson(lam)
                sample = f"{condition_sel}_{titrant_name}_r{rep}_t{t:g}_c{c:g}"
                for g in range(num_genotypes):
                    rows.append((f"A{g + 1}G", counts[g], rep, condition_sel,
                                 titrant_name, t, c, sample))
    return pd.DataFrame(rows, columns=["genotype", "counts", "replicate",
                                       "condition_sel", "titrant_name",
                                       "t_sel", "titrant_conc", "sample"])


def _swap(df, rep, t, c1, c2):
    """Swap the titrant labels (and sample names) of two tubes."""
    df = df.copy()
    a = (df["replicate"] == rep) & (df["t_sel"] == t) & (df["titrant_conc"] == c1)
    b = (df["replicate"] == rep) & (df["t_sel"] == t) & (df["titrant_conc"] == c2)
    sa, sb = df.loc[a, "sample"].iloc[0], df.loc[b, "sample"].iloc[0]
    df.loc[a, ["titrant_conc", "sample"]] = [c2, sb]
    df.loc[b, ["titrant_conc", "sample"]] = [c1, sa]
    return df


@pytest.fixture(scope="module")
def clean():
    return _simulate()


@pytest.fixture(scope="module")
def clean_scores(clean):
    return Q.score_tubes(clean)


# --- helpers ---

def test_isotonic_pools_violators():
    np.testing.assert_allclose(Q._isotonic([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    np.testing.assert_allclose(Q._isotonic([0, 1, 2]), [0, 1, 2])


def test_order_violations_monotone_is_zero():
    np.testing.assert_array_equal(Q.order_violations([0, 0.1, 0.5, 1.0]), 0)
    np.testing.assert_array_equal(Q.order_violations([0.3]), 0)


def test_order_violations_signs():
    v = Q.order_violations([0.0, 0.9, 0.3, 0.6, 1.0])
    assert v[1] > 0          # above the fit at the next concentration
    assert v[0] == 0 and v[-1] == 0
    v = Q.order_violations([0.0, 0.3, 0.6, 0.1])
    assert v[-1] < 0         # the top concentration below its neighbor


def test_deviations_leave_the_tube_out():
    expected = np.array([0.0, 0.5, 0.8, 1.0])
    score = 1.2 * expected
    dv, sd = Q.deviations(score, expected)
    np.testing.assert_allclose(dv, 0, atol=1e-12)
    # The low tube's expectation hardly depends on the estimated scale.
    assert sd[0] == 1.0 and 1.0 < sd[1] < sd[-1]
    score = score.copy()
    score[-1] = 0.0          # the top tube cannot set its own scale
    dv, _ = Q.deviations(score, expected)
    assert dv[-1] == pytest.approx(-1.2)


def test_row_scale_ignores_low_concentrations():
    # The low tube's ratio (-4) would dominate a plain ratio median.
    assert Q._row_scale(np.array([-0.1, 0.9]), np.array([0.025, 1.0])) == 0.9
    assert Q._row_scale(np.array([0.1, 0.2]), np.array([0.05, 0.2])) == 1.0


# --- scoring ---

def test_clean_data_not_flagged(clean_scores):
    s = clean_scores
    assert not s["flagged"].any()
    assert len(s) == 2 * 3 * len(CONCS)
    # Endpoints of the expected profile are 0 and 1 by construction.
    exp = s.groupby("titrant_conc")["expected"].first()
    assert exp.iloc[0] == pytest.approx(0, abs=1e-12)
    assert exp.iloc[-1] == pytest.approx(1)
    assert (np.diff(exp.to_numpy()) > -0.02).all()
    # Every row rises with concentration.
    for _, g in s.groupby(["replicate", "t_sel"]):
        assert g.sort_values("titrant_conc")["score"].is_monotonic_increasing


def test_rows_of_any_scale_start_at_zero(clean_scores):
    # The effect grows with time; an additive row effect alone shifted the
    # early and late rows by about +/-0.17 here.
    s = clean_scores
    low = s[s["titrant_conc"] == 0.0]
    assert low["score"].abs().max() < 0.03
    assert s["deviation_z"].abs().max() < 4


def test_row_scale_grows_with_time(clean_scores):
    b = clean_scores.groupby("t_sel")["row_scale"].mean()
    assert b.is_monotonic_increasing
    assert b.iloc[0] < 0.8 < 1.2 < b.iloc[-1]


@pytest.mark.parametrize("c1,c2", [(1e-4, 1.0), (0.0, 3e-2), (1e-3, 1e-1)])
def test_swapped_pair_is_flagged(clean, c1, c2):
    s = Q.score_tubes(_swap(clean, 2, 150.0, c1, c2))
    flagged = s[s["flagged"]]
    assert len(flagged) == 2
    assert set(flagged["replicate"]) == {2}
    assert set(flagged["t_sel"]) == {150.0}
    assert set(flagged["titrant_conc"]) == {c1, c2}
    hi = flagged[flagged["titrant_conc"] == min(c1, c2)]
    assert hi["score"].iloc[0] > hi["expected"].iloc[0]


def test_missing_concentrations_not_shifted(clean):
    # Replicate 1 at t 200 keeps only the three lowest concentrations; row
    # centering would push them toward each other's mean (the study's bias).
    keep = ~((clean["replicate"] == 1) & (clean["t_sel"] == 200.0)
             & (clean["titrant_conc"] > 1e-3))
    s = Q.score_tubes(clean[keep])
    assert not s["flagged"].any()
    row = s[(s["replicate"] == 1) & (s["t_sel"] == 200.0)]
    assert len(row) == 3
    full = Q.score_tubes(clean)
    ref = full[(full["replicate"] == 1) & (full["t_sel"] == 200.0)
               & (full["titrant_conc"] <= 1e-3)]
    np.testing.assert_allclose(row["score"].to_numpy(),
                               ref["score"].to_numpy(), atol=0.03)


def test_swap_in_incomplete_row(clean):
    keep = ~((clean["replicate"] == 1) & (clean["t_sel"] == 100.0)
             & clean["titrant_conc"].isin([1e-3, 1e-2]))
    s = Q.score_tubes(_swap(clean[keep], 1, 100.0, 1e-4, 1e-1))
    flagged = s[s["flagged"]]
    assert set(flagged["titrant_conc"]) == {1e-4, 1e-1}
    assert set(flagged["t_sel"]) == {100.0}


def test_tube_totals_cancel(clean, clean_scores):
    scaled = clean.copy()
    one = scaled["sample"] == scaled["sample"].iloc[0]
    scaled.loc[one, "counts"] *= 10
    s = Q.score_tubes(scaled)
    np.testing.assert_allclose(s["score"], clean_scores["score"], atol=1e-12)


def test_panels_are_separate_and_any_titrant(clean):
    other = _simulate(condition_sel="kanR+kan", titrant_name="atc", seed=1)
    other = _swap(other, 1, 100.0, 1e-4, 1.0)
    s = Q.score_tubes(pd.concat([clean, other], ignore_index=True))
    assert set(s["titrant_name"]) == {"iptg", "atc"}
    flagged = s[s["flagged"]]
    assert set(flagged["condition_sel"]) == {"kanR+kan"}
    assert len(flagged) == 2


def test_unknown_genotype_ignored(clean, clean_scores):
    extra = clean.drop_duplicates("sample").assign(genotype="__unknown__",
                                                   counts=10 ** 6)
    s = Q.score_tubes(pd.concat([clean, extra], ignore_index=True))
    np.testing.assert_allclose(s["score"], clean_scores["score"], atol=1e-12)


def test_without_sample_column(clean, clean_scores):
    s = Q.score_tubes(clean.drop(columns="sample"))
    np.testing.assert_allclose(s["score"], clean_scores["score"], atol=1e-12)


# --- errors ---

def test_duplicate_tube_raises(clean):
    dup = clean.copy()
    m = (dup["replicate"] == 1) & (dup["t_sel"] == 100.0) & (dup["titrant_conc"] == 0.0)
    dup.loc[m, "titrant_conc"] = 1e-4
    with pytest.raises(ValueError, match="one tube per concentration"):
        Q.score_tubes(dup)


def test_single_concentration_raises(clean):
    with pytest.raises(ValueError, match="only one"):
        Q.score_tubes(clean[clean["titrant_conc"] == 0.0])


def test_too_few_genotypes_raises(clean):
    with pytest.raises(ValueError, match="genotypes have at least"):
        Q.score_tubes(clean, min_counts=10 ** 9)


def test_missing_column_raises(clean):
    with pytest.raises(Exception):
        Q.score_tubes(clean.drop(columns="counts"))


# --- CLI ---

def test_check_tubes_writes_table(tmp_path, clean, capsys):
    f = tmp_path / "growth.csv"
    _swap(clean, 1, 200.0, 1e-4, 1.0).to_csv(f, index=False)
    out = str(tmp_path / "qc")
    check_tubes(str(f), out_prefix=out)
    s = pd.read_csv(out + ".csv")
    assert s["flagged"].sum() == 2
    assert "2 flagged tube(s)" in capsys.readouterr().out


def test_main_parses_flags(tmp_path, clean):
    f = tmp_path / "growth.csv"
    clean.to_csv(f, index=False)
    out = str(tmp_path / "qc")
    with patch.object(sys, "argv", ["tfs-check-tubes", str(f), "--out_prefix",
                                    out, "--flag_z", "100", "--min_counts", "2"]):
        main()
    assert not pd.read_csv(out + ".csv")["flagged"].any()


# --- internals ---

def test_deviations_of_a_single_tube():
    dv, sd = Q.deviations([0.7], [1.0])
    np.testing.assert_array_equal(dv, [0.0])
    np.testing.assert_array_equal(sd, [1.0])


def _additive_grid(seed=0, G=6, R=2, C=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (G, R))
    d = rng.normal(0, 1, (G, C))
    return a, d, a[:, :, None] + d[:, None, :]


def test_project_defaults_to_unit_row_scale():
    a, d, L = _additive_grid()
    s0, e0 = Q._project(L, a, d)
    s1, e1 = Q._project(L, a, d, np.ones(L.shape[1]))
    np.testing.assert_allclose(s0, s1)
    np.testing.assert_allclose(e0, e1)
    # An exact additive fit scores every row at its expectation.
    np.testing.assert_allclose(s0, np.broadcast_to(e0, s0.shape), atol=1e-12)
    assert e0[0] == pytest.approx(0.0) and e0[-1] == pytest.approx(1.0)


def test_project_without_an_axis_raises():
    a, d, L = _additive_grid()
    d[:, -1] = d[:, 0]
    with pytest.raises(ValueError, match="no axis to score"):
        Q._project(L, a, d)


def test_fit_profiles_stops_once_scales_settle():
    a, d, L = _additive_grid()
    with patch.object(Q, "_project", wraps=Q._project) as proj:
        fa, fd, fb = Q._fit_profiles(L, n_iter=20)
    # Converged on the first pass instead of running all 20.
    assert proj.call_count < 20
    np.testing.assert_allclose(fb, 1.0)
    np.testing.assert_allclose(fa[:, :, None] + fd[:, None, :], L, atol=1e-8)


def test_panel_without_titrant_effect_raises(clean):
    flat = clean.copy()
    # Every tube of a replicate x time row has the same composition.
    first = flat[flat["titrant_conc"] == 0.0].set_index(
        ["genotype", "replicate", "t_sel"])["counts"]
    key = pd.MultiIndex.from_frame(flat[["genotype", "replicate", "t_sel"]])
    flat["counts"] = first.reindex(key).to_numpy()
    with pytest.raises(ValueError, match=r"Panel .*no axis to score"):
        Q.score_tubes(flat)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_cli_module_runs_as_script(monkeypatch):
    import runpy
    monkeypatch.setattr(sys, "argv", ["tfs-check-tubes", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("tfscreen.process_raw.scripts.check_tubes_cli",
                         run_name="__main__")
    assert exc.value.code == 0
