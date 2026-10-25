import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel import two_stage as ts


# --- stage 1 genotypes ---

def test_stage1_genotypes_keeps_wt_spikes_and_singles():
    lib = pd.DataFrame({
        "genotype": ["wt", "A1B", "C2D", "A1B/C2D", "E3F/G4H", "H5I/J6K/L7M"],
        "in_spiked_origin": [True, True, False, False, True, True]})
    have = ["wt", "A1B", "C2D", "A1B/C2D", "E3F/G4H", "N8P", "__unknown__"]
    out = ts.stage1_genotypes(lib, have)
    # the spiked double comes along; the unspiked double, absent spikes and
    # __unknown__ do not; a single not in the library table still counts
    assert out == sorted(["wt", "A1B", "C2D", "E3F/G4H", "N8P"])


def test_is_single():
    assert ts.is_single("A1B")
    assert not ts.is_single("wt")
    assert not ts.is_single("A1B/C2D")
    assert not ts.is_single("__unknown__")


# --- jackknife subsamples ---

def _growth(n_rep=3, concs=(0.0, 0.1, 1.0), conds=("kanR+kan", "pheS+4CP"),
            times=(100, 150, 200), genotypes=("wt", "A1B")):
    rows = []
    for c in conds:
        for x in concs:
            for r in range(n_rep):
                for t in times:
                    sample = f"{c}-{x}-{r}-{t}"
                    for g in genotypes:
                        rows.append({"sample": sample, "library": c[:4],
                                     "condition_sel": c, "titrant_conc": x,
                                     "replicate": r, "t_sel": t,
                                     "genotype": g, "counts": 10})
    return pd.DataFrame(rows)


def test_tube_drops_are_stratified_and_leave_one():
    tubes = ts.tube_table(_growth())
    drop = ts.choose_tube_drops(tubes, 0.2, seed=1)
    kept = tubes[~tubes["sample"].isin(drop)]
    per = tubes.groupby(list(ts.STRATA)).size()
    kept_per = kept.groupby(list(ts.STRATA)).size()
    assert (kept_per >= 1).all()
    # 9 tubes per stratum, 20% -> 2 dropped from each
    assert ((per - kept_per) == 2).all()
    assert ts.choose_tube_drops(tubes, 0.2, seed=1) == drop
    assert ts.choose_tube_drops(tubes, 0.2, seed=2) != drop
    # a stratum of one tube is never emptied
    one = tubes.drop_duplicates(list(ts.STRATA))
    assert ts.choose_tube_drops(one, 0.9, seed=1) == set()
    with pytest.raises(ValueError):
        ts.choose_tube_drops(tubes, 1.0, seed=1)


def test_jackknife_subsample_drops_whole_tubes():
    g = _growth()
    sub, n, d, dropped = ts.jackknife_subsample(g, 0.2, seed=3)
    assert n == g["sample"].nunique()
    assert d == len(dropped) == 12
    assert not sub["sample"].isin(dropped).any()
    assert len(sub) == len(g) - d * 2      # two genotypes per tube
    with pytest.raises(ValueError, match="tube column"):
        ts.tube_table(g.drop(columns="library"))


# --- jackknife k and m ---

def _km(k, m, conds=("a", "b", "c")):
    return pd.DataFrame({"k": k, "m": m}, index=pd.Index(conds,
                                                         name="condition_rep"))


def test_km_table_from_extract_params():
    k = pd.DataFrame({"condition_rep": ["a", "a", "b"], "q0.5": [1.0, 3.0, 5.0]})
    m = pd.DataFrame({"condition_rep": ["a", "b"], "q0.5": [0.1, 0.2]})
    out = ts.km_table(k, m)
    assert out.loc["a", "k"] == 2.0 and out.loc["b", "m"] == 0.2
    with pytest.raises(ValueError, match="different conditions"):
        ts.km_table(k, m.iloc[:1])


def test_jackknife_km_scales_and_finds_the_main_direction():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1, 200)
    est = [_km([0.01 + 0.002 * c, 0.02 + 0.004 * c, 0.03 + 0.001 * e],
               [-0.014, 0.0, 0.014])
           for c, e in zip(common, rng.normal(0, 0.01, 200))]
    out = ts.jackknife_km(est, n=100, d=20)
    assert out["factor"] == 4.0
    raw_sd = np.std([e.loc["a", "k"] for e in est])
    assert out["se"].loc["a", "k_se"] == pytest.approx(2.0 * raw_sd)
    assert out["se"].loc["a", "m_se"] == pytest.approx(0.0)
    assert out["k_share"] > 0.99
    step = out["k_step"]
    # the step follows the common move (a:b = 1:2) and its largest entry is +
    assert step["b"] == pytest.approx(2 * step["a"], rel=0.01)
    assert step["b"] > 0
    assert np.hypot(step["a"], step["b"]) == pytest.approx(
        np.sqrt(out["k_cov"].loc["a", "a"] + out["k_cov"].loc["b", "b"]),
        rel=0.02)


def test_jackknife_km_refuses_bad_inputs():
    a = _km([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="two subsamples"):
        ts.jackknife_km([a], 10, 2)
    with pytest.raises(ValueError, match="0 < d < n"):
        ts.jackknife_km([a, a], 10, 0)
    with pytest.raises(ValueError, match="different conditions"):
        ts.jackknife_km([a, _km([1, 2], [0, 0], ("a", "b"))], 10, 2)


# --- pinned priors ---

def test_pinned_growth_priors_and_shift_tables():
    km = _km([0.015, 0.03, 0.005], [-0.014, 0.0, 0.014])
    t = ts.pinned_growth_priors(km)
    assert list(t.columns) == ["condition_rep", "k_loc", "k_scale", "m_loc",
                               "m_scale", "k_pinned", "m_pinned"]
    assert (t["k_pinned"] == 1).all() and (t["m_pinned"] == 1).all()
    one = ts.pinned_growth_priors(km, conditions=["a"])
    assert list(one["condition_rep"]) == ["a"]
    step = pd.Series({"a": 0.004, "b": 0.008, "c": 0.002})
    plus, minus = ts.k_shift_priors(km, step)
    assert np.allclose(plus["k_loc"] - minus["k_loc"], 2 * step.to_numpy())
    assert np.allclose(plus["m_loc"], km["m"])
    with pytest.raises(ValueError, match="every condition"):
        ts.k_shift_priors(km, step.drop("c"))


# --- the shift table ---

def test_shift_table_half_difference_and_wt_zero():
    rows = pd.DataFrame({"genotype": ["wt", "wt", "a", "a"],
                         "titrant_name": "iptg",
                         "titrant_conc": [0.0, 1.0, 0.0, 1.0]})
    plus = rows.assign(**{"q0.5": [1.0, 0.0, 0.50, 0.20]})
    minus = rows.assign(**{"q0.5": [1.0, 0.0, 0.56, 0.28]})
    out = ts.shift_table(plus, minus)
    a = out[out["genotype"] == "a"].set_index("titrant_conc")["shift"]
    assert a[0.0] == pytest.approx(-0.03) and a[1.0] == pytest.approx(-0.04)
    assert (out.loc[out["genotype"] == "wt", "shift"] == 0).all()
    with pytest.raises(ValueError, match="different rows"):
        ts.shift_table(plus, minus.iloc[:3])


# --- chunked growth tables ---

def test_growth_genotypes_and_filter_growth_csv(tmp_path):
    g = _growth()
    src = tmp_path / "g.csv"
    g.to_csv(src, index=False)
    assert ts.growth_genotypes(str(src), chunksize=7) == {"wt", "A1B"}
    dst = tmp_path / "o.csv"
    n = ts.filter_growth_csv(str(src), str(dst), genotypes=["A1B"], chunksize=7)
    out = pd.read_csv(dst)
    assert n == len(out) == (g["genotype"] == "A1B").sum()
    assert set(out["genotype"]) == {"A1B"}
    n = ts.filter_growth_csv(str(src), str(dst), library="kanR", chunksize=7)
    assert set(pd.read_csv(dst)["library"]) == {"kanR"}
    with pytest.raises(ValueError, match="no rows"):
        ts.filter_growth_csv(str(src), str(dst), library="nope")


# --- marker consistency ---

def _pred(x, sd, genotypes, concs=(0.0, 1.0)):
    rows = [{"genotype": g, "titrant_name": "iptg", "titrant_conc": c,
             "q0.5": x[g][i], "q0.025": x[g][i] - 1.96 * sd,
             "q0.975": x[g][i] + 1.96 * sd}
            for g in genotypes for i, c in enumerate(concs)]
    return pd.DataFrame(rows)


def test_marker_consistency_flags_a_level_offset_after_the_affine_map():
    rng = np.random.default_rng(0)
    genos = ["wt"] + [f"A{i}B" for i in range(200)]
    both = {g: rng.uniform(0, 1, 2) for g in genos}
    # one library alone: stretched and shifted, small noise, one outlier
    one = {g: 0.1 + 1.1 * v + rng.normal(0, 0.01, 2) for g, v in both.items()}
    one["A7B"] = one["A7B"] + 0.3
    out = ts.marker_consistency(_pred(both, 0.05, genos), _pred(one, 0.05, genos))
    assert "wt" not in out.index
    intercept, slope = out.attrs["affine"]
    assert slope == pytest.approx(1.1, abs=0.02)
    assert out.loc["A7B", "inconsistent"]
    assert out["inconsistent"].sum() == 1
    assert out.loc["A7B", "z"] == pytest.approx(
        0.3 / np.hypot(0.05, 0.05), rel=0.15)


# --- k and m from a MAP when extraction cannot give them ---

def _priors_csv(path, k_pinned, m_pinned):
    conds = ["a", "b", "c"]
    rows = [("growth.condition_growth.m_scale", 0.01, None, None)]
    for i, c in enumerate(conds):
        rows += [("growth.condition_growth.k_loc", 0.01 * (i + 1), i, c),
                 ("growth.condition_growth.m_loc", -0.001 * (i + 1), i, c),
                 ("growth.condition_growth.k_pinned", float(k_pinned[i]), i, c),
                 ("growth.condition_growth.m_pinned", float(m_pinned[i]), i, c)]
    pd.DataFrame(rows, columns=["parameter", "value", "flat_index",
                                "condition_rep"]).to_csv(path, index=False)


def test_km_from_map_pinned_free_and_partial(tmp_path):
    pri, npz = tmp_path / "p.csv", tmp_path / "m.npz"
    # k pinned everywhere, m free everywhere
    _priors_csv(pri, [1, 1, 1], [0, 0, 0])
    np.savez(npz, condition_growth_m_auto_loc=np.array([-0.5, 0.0, 0.5]))
    out = ts.km_from_map(str(npz), str(pri))
    assert list(out.index) == ["a", "b", "c"]
    assert np.allclose(out["k"], [0.01, 0.02, 0.03])
    assert np.allclose(out["m"], [-0.5, 0.0, 0.5])
    # m pinned in b only: the free entries come from {site}_free
    _priors_csv(pri, [1, 1, 1], [0, 1, 0])
    np.savez(npz, condition_growth_m_free_auto_loc=np.array([7.0, 9.0]))
    out = ts.km_from_map(str(npz), str(pri))
    assert np.allclose(out["m"], [7.0, -0.002, 9.0])
    # a free parameter missing from the MAP is an error
    _priors_csv(pri, [0, 1, 1], [1, 1, 1])
    np.savez(npz, other=np.zeros(1))
    with pytest.raises(ValueError, match="no k"):
        ts.km_from_map(str(npz), str(pri))


# --- the k-shift refits ---

def test_dk_geno_hypers_and_pins(tmp_path):
    npz = tmp_path / "p.npz"
    np.savez(npz, dk_geno_hyper_loc_auto_loc=np.array([-2.0]),
             dk_geno_hyper_scale_auto_loc=np.array(1.5),
             dk_geno_hyper_shift_auto_loc=np.array(0.07),
             other_auto_loc=np.zeros(3))
    held = ts.dk_geno_hypers(str(npz))
    assert held == pytest.approx({"hyper_loc": -2.0, "hyper_scale": 1.5,
                                  "hyper_shift": 0.07})
    pri = tmp_path / "priors.csv"
    pd.DataFrame({"parameter": ["growth.dk_geno.hyper_loc_loc",
                                "growth.dk_geno.pinned.hyper_loc"],
                  "value": ["-3.5", "9"]}).to_csv(pri, index=False)
    ts.pin_dk_geno_hypers(str(pri), held)
    ts.pin_dk_geno_hypers(str(pri), held)        # idempotent
    df = pd.read_csv(pri)
    assert len(df) == 4
    rows = dict(zip(df["parameter"], df["value"]))
    assert rows["growth.dk_geno.pinned.hyper_loc"] == pytest.approx(-2.0)
    assert rows["growth.dk_geno.hyper_loc_loc"] == pytest.approx(-3.5)


def test_dk_geno_hypers_absent(tmp_path):
    npz = tmp_path / "p.npz"
    np.savez(npz, other_auto_loc=np.zeros(3))
    assert ts.dk_geno_hypers(str(npz)) == {}
    pri = tmp_path / "priors.csv"
    pd.DataFrame({"parameter": ["a"], "value": [1]}).to_csv(pri, index=False)
    ts.pin_dk_geno_hypers(str(pri), {})
    assert len(pd.read_csv(pri)) == 1


def test_refit_drift():
    def pred(x):
        return pd.DataFrame({"genotype": ["wt", "A1B", "C2D"],
                             "titrant_name": "iptg", "titrant_conc": 0.0,
                             "q0.5": [1.0, x, x + 0.1]})
    out = ts.refit_drift(pred(0.5), {"plus": pred(0.45), "zero": pred(0.5)})
    out = out.set_index("refit")
    assert out.loc["plus", "median_change"] == pytest.approx(-0.05)
    assert out.loc["zero", "median_change"] == pytest.approx(0.0)


# --- the summary table ---

def test_genotype_reads_in_chunks(tmp_path):
    g = pd.DataFrame({"genotype": ["wt", "A1B", "wt", "A1B", "C2D"],
                      "library": ["kanR", "kanR", "pheS", "pheS", "kanR"],
                      "counts": [10, 3, 5, 2, 7]})
    g.to_csv(tmp_path / "g.csv", index=False)
    r = ts.genotype_reads(str(tmp_path / "g.csv"), chunksize=2)
    assert r.loc["wt"].tolist() == [15, 10, 5]
    assert list(r.columns) == ["reads", "reads_kanR", "reads_pheS"]
    assert r.loc["C2D", "reads_pheS"] == 0


def test_num_mutations():
    assert [ts.num_mutations(g) for g in ("wt", "A1B", "A1B/C2D/E3F")] == [0, 1, 3]


def test_x_summary_joins():
    pred = pd.DataFrame({"genotype": ["wt", "A1B", "A1B"], "titrant_name": "iptg",
                         "titrant_conc": [0.0, 0.0, 1.0], "q0.025": 0.1,
                         "q0.5": 0.5, "q0.975": 0.9, "in_training_data": 1,
                         "theta_scale": "X"})
    reads = pd.DataFrame({"reads": [9], "reads_kanR": [9]},
                         index=pd.Index(["A1B"], name="genotype"))
    cons = pd.DataFrame({"z": [2.5], "inconsistent": [True]},
                        index=pd.Index(["A1B"], name="genotype"))
    shift = pred[["genotype", "titrant_name", "titrant_conc"]].assign(shift=[0, -0.03, -0.02])
    x = ts.x_summary(pred, reads, cons, shift)
    assert x["reads"].tolist() == [0, 9, 9]
    assert x["k_shift"].tolist() == [0, -0.03, -0.02]
    assert x["inconsistent"].iloc[1] == True and pd.isna(x["inconsistent"].iloc[0])
    assert x.columns[-2:].tolist() == ["consistency_z", "inconsistent"]
    bare = ts.x_summary(pred, reads)
    assert bare["k_shift"].isna().all() and bare["consistency_z"].isna().all()


def test_flag_reads_strings_and_numbers():
    assert ts._flag(" True ") and ts._flag("1.0") and ts._flag("1")
    assert not ts._flag("0") and not ts._flag("false")
    assert ts._flag(1.0) and not ts._flag(0)


def test_km_from_map_scalar_rows_and_defaults(tmp_path):
    pri, npz = tmp_path / "p.csv", tmp_path / "m.npz"
    # k_loc per condition; m_loc and m_pinned scalar rows (broadcast);
    # no k_pinned rows at all (k free everywhere by default)
    rows = [("growth.condition_growth.m_loc", -0.004, None, None),
            ("growth.condition_growth.m_pinned", 1.0, None, None)]
    rows += [("growth.condition_growth.k_loc", 0.01 * (i + 1), i, c)
             for i, c in enumerate("abc")]
    pd.DataFrame(rows, columns=["parameter", "value", "flat_index",
                                "condition_rep"]).to_csv(pri, index=False)
    np.savez(npz, condition_growth_k_auto_loc=np.array([0.1, 0.2, 0.3]))
    out = ts.km_from_map(str(npz), str(pri))
    assert list(out.index) == ["a", "b", "c"]
    assert np.allclose(out["k"], [0.1, 0.2, 0.3])
    assert np.allclose(out["m"], -0.004)


def test_km_from_map_needs_per_condition_rows(tmp_path):
    pri, npz = tmp_path / "p.csv", tmp_path / "m.npz"
    pd.DataFrame({"parameter": ["growth.condition_growth.k_loc"],
                  "value": [0.01], "flat_index": [None],
                  "condition_rep": [None]}).to_csv(pri, index=False)
    np.savez(npz, other=np.zeros(1))
    with pytest.raises(ValueError, match="no per-condition growth rows"):
        ts.km_from_map(str(npz), str(pri))


def test_marker_consistency_needs_shared_non_wt_rows():
    both = _pred({"wt": [1.0, 0.0], "A1B": [0.5, 0.4]}, 0.05, ["wt", "A1B"])
    one = _pred({"wt": [1.0, 0.0], "C2D": [0.5, 0.4]}, 0.05, ["wt", "C2D"])
    with pytest.raises(ValueError, match="no non-wt rows in common"):
        ts.marker_consistency(both, one)
